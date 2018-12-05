from __future__ import division
from utils import AverageMeter, BaseOptions, Recorder, Logger, time_string, convert_secs2time, eval_cmc_map, \
    reset_state_dict, extract_features, create_stat_string, save_checkpoint, adjust_learning_rate, accuracy, \
    partition_params
import dataloader
import img_root_txt

import os
import time
import math
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils import model_zoo
import resnet_mmd

cudnn.benchmark = True

def data_load(args):
    """
    the source loader contains images from multiple source domains. source labels of these domains are not mixed.
    """
    root_path, sources, target, seed, batch_size, batch_size_t= \
        args.img_root, args.sources, args.target, args.seed, args.batch_size, args.batch_size_t

    src_txt = ''
    for iSource in range(len(sources)):
        src_txt += sources[iSource] + '_'
    src_txt += 'train.txt'
    tgt_p_txt = target + '_prob.txt'
    tgt_g_txt = target + '_gallery.txt'

    if not os.path.exists(root_path + src_txt):
        img_root_txt.merge_source_txt(sources)
    if not os.path.exists(root_path + tgt_p_txt):
        img_root_txt.make_target_txt(target)

    kwargs = {'num_workers': 2, 'pin_memory': True}

    source_loader, num_class = dataloader.load_training(root_path, src_txt, batch_size, kwargs)
    target_prob_loader, _ = dataloader.load_testing(root_path, tgt_p_txt, batch_size, kwargs)
    target_gallery_loader, _ = dataloader.load_testing(root_path, tgt_g_txt, batch_size, kwargs)

    return source_loader, target_prob_loader, target_gallery_loader, num_class


def load_pretrain(model):
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if not "cls_fc" in k:
            model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    model.load_state_dict(model_dict)
    return model

def main():
    opts = BaseOptions()
    args = opts.parse()

    args.save_path = os.path.join(args.save_path, args.method)

    logger = Logger(args.save_path)
    opts.print_options(logger)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    Asource_loader, Atarget_loader, Bsource_loader, Btarget_loader, target_test_loader, num_class = data_load(args)
    unknownID = num_class - 1

    net = resnet_mmd.CDANet(num_classes=num_class)
    if args.cuda:
        net = net.cuda()
    load_pretrain(net)
    logger.print_log('loaded pre-trained feature net')

    optimizer = torch.optim.SGD([
            {'params': net.sharedNet.parameters()},
            {'params': net.cls_fc.parameters(), 'lr': args.lr},
        ], lr=args.lr / 10, momentum=args.momentum, weight_decay=args.wd)

    train_stats = ('total_acc', 'loss', 'loss_cls', 'loss_mmd')
    val_stats = ('total_acc', 'known_acc', 'unknown_acc')
    recorder = Recorder(args.epochs, val_stats[0], train_stats, val_stats)
    logger.print_log('observing training stats: {} \nvalidation stats: {}'.format(train_stats, val_stats))

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.print_log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #recorder = checkpoint['recorder']
            #start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.print_log("=> no checkpoint found at '{}'".format(args.resume))

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(start_epoch, args.epochs):

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        logger.print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, args.epochs, need_time))

        lr, _ = adjust_learning_rate(optimizer, (args.lr, args.lr / 10), epoch, args.epochs, args.lr_strategy)
        print("   lr:{}".format(lr))

        reachHighest = train(epoch, net, optimizer,
                             Asource_loader, Atarget_loader, Bsource_loader, Btarget_loader, args, recorder, logger)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if epoch % args.evaluate_freq == 0:
            reachHighest = evaluate(net, target_test_loader, unknownID, epoch, recorder, logger, args)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, reachHighest, args.save_path, 'checkpoint.pth.tar')
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))


def train(epoch, model, optimizer, Asource_loader, Atarget_loader, Bsource_loader, Btarget_loader, args, recorder, logger):
    batch_time_meter = AverageMeter()
    stats = recorder.train_stats
    meters = {stat: AverageMeter() for stat in stats}

    model.train()

    len_Asource_loader, len_Bsource_loader, len_Atarget_loader, len_Btarget_loader \
        = len(Asource_loader), len(Bsource_loader), len(Atarget_loader), len(Btarget_loader)
    iter_Asource, iter_Bsource = iter(Asource_loader), iter(Bsource_loader)
    iter_Atarget, iter_Btarget = iter(Atarget_loader), iter(Btarget_loader)

    len_loader_max = max(len_Atarget_loader, len_Btarget_loader)

    end = time.time()
    for i in range(1, len_loader_max+1):

        # use images iteratively
        if i % len_Asource_loader == 0:
            iter_Asource = iter(Asource_loader)
        if i % len_Bsource_loader == 0:
            iter_Bsource = iter(Bsource_loader)
        if i % len_Atarget_loader == 0:
            iter_Atarget = iter(Atarget_loader)
        if i % len_Btarget_loader == 0:
            iter_Btarget = iter(Btarget_loader)


        data_Asource, label_Asource, _ = iter_Asource.next()
        data_Atarget, _, _ = iter_Atarget.next()
        data_Bsource, label_Bsource, _ = iter_Bsource.next()
        data_Btarget, _, _ = iter_Btarget.next()

        data_source_var = autograd.Variable(torch.cat( (data_Asource, data_Bsource))).cuda()
        label_source_var = autograd.Variable(torch.cat((label_Asource, label_Bsource))).cuda()
        domain_source_var = torch.cat((torch.zeros(args.batch_size), torch.ones(args.batch_size)))
        data_target_var = autograd.Variable(torch.cat((data_Atarget, data_Btarget))).cuda()
        domain_target_var = torch.cat((torch.zeros(args.batch_size_t), torch.ones(args.batch_size_t)))

        # get classifier loss and mmd loss
        optimizer.zero_grad()

        label_source_pred, loss_mmd, _, _ = model(data_source_var, label_source_var, domain_source_var, data_target_var, domain_target_var)
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source_var)
        acc = accuracy(label_source_pred.data, label_source_var.data, topk=(1,))
        gamma = 2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1
        loss = loss_cls + gamma * loss_mmd
        loss.backward()
        optimizer.step()

        # update meters
        meters['total_acc'].update(acc[0].item(), args.batch_size)
        meters['loss'].update(loss, args.batch_size)
        meters['loss_cls'].update(loss_cls, args.batch_size)
        meters['loss_mmd'].update(loss_mmd, args.batch_size)

        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        freq = args.batch_size / batch_time_meter.avg
        end = time.time()

        if i % args.print_freq == 0:
            logger.print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   Freq {:.1f}   '.format(
                epoch, i, len_loader_max, freq) + create_stat_string(meters) + time_string())

    logger.print_log('  **Train**  ' + create_stat_string(meters))

    return recorder.update(epoch=epoch, is_train=True, meters=meters)

def evaluate(model, target_test_loader, unknownID, epoch, recorder, logger, args):
    stats = recorder.val_stats
    meters = {stat: AverageMeter() for stat in stats}
    model.eval()

    test_loss = 0
    len_target_dataset = len(target_test_loader.dataset)
    iter_target = iter(target_test_loader)
    for j in range(0, len_target_dataset):
        try:
            data, label, _ = iter_target.next()
        except StopIteration:
            break
        if args.cuda == 1:
            data, label = data.cuda(), label.cuda()
        with torch.no_grad():
            data_var = autograd.Variable(data)
        t_output, _, _, _ = model(data_var, None, None, data_var, None)
        test_loss += F.nll_loss(F.log_softmax(t_output, dim = 1), label, size_average=False).data.item() # sum up batch loss

        pred = t_output.data.max(1)[1] # get the index of the max log-probability

        print(pred)
        print(label)

        known_idx = [i for i in range(0, len(label)) if not label[i] == unknownID]
        unknown_idx = [i for i in range(0, len(label)) if label[i] == unknownID]
        total_acc = 100. * pred.eq(label.view_as(pred)).cpu().sum() / len(label)
        meters['total_acc'].update(total_acc, args.batch_size)

        if not len(known_idx) == 0:
            known_acc = 100. * pred[known_idx].eq(label[known_idx].view_as(pred[known_idx])).cpu().sum() / len(known_idx)
            meters['known_acc'].update(known_acc, args.batch_size)
        if not len(unknown_idx) == 0:
            unknown_acc = 100. * pred[unknown_idx].eq(label[unknown_idx].view_as(pred[unknown_idx])).cpu().sum() / len(unknown_idx)
            meters['unknown_acc'].update(unknown_acc, args.batch_size)

    logger.print_log('  **Test**  ' + create_stat_string(meters))
    recorder.update(epoch=epoch, is_train=False, meters=meters)

if __name__ == '__main__':
    main()
