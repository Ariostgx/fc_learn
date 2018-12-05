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
import resnet

cudnn.benchmark = True

def data_load(args):
    """
    both source and target images are composed of domain A and B
    """
    root_path, combination, source_num, plus_cnum, seed, batch_size = \
        args.img_root, args.domain_com, args.source_num, args.plus_num, args.seed, args.batch_size

    txt_name = root_path + '{}_srcn_{}_plusn_{}_seed_{}'.format(combination, source_num, plus_cnum, seed)
    if not os.path.exists(root_path+txt_name +'_s.txt'):
        img_root_txt.make_img_root_txt(combination, source_num, plus_cnum, seed)

    kwargs = {'num_workers': 2, 'pin_memory': True}

    source_loader, _ = dataloader.load_training(root_path, txt_name +'_s.txt', batch_size, kwargs)
    target_train_loader, _ = dataloader.load_training(root_path, txt_name +'_t.txt', batch_size * 5, kwargs)
    target_test_loader, num_class = dataloader.load_testing(root_path, txt_name +'_t.txt', batch_size, kwargs)

    return source_loader, target_train_loader, target_test_loader, num_class


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
    logger = Logger(args.save_path)
    opts.print_options(logger)

    source_loader, target_train_loader, target_test_loader, num_class = data_load(args)
    unknownID = num_class - 1

    net = resnet.CDANet(num_classes=num_class)
    load_pretrain(net)
    logger.print_log('loaded pre-trained feature net')

    optimizer = torch.optim.SGD([
            {'params': net.sharedNet.parameters()},
            {'params': net.cls_fc.parameters(), 'lr': args.lr},
        ], lr=args.lr / 10, momentum=args.momentum, weight_decay=args.wd)

    train_stats = ('total_acc', 'loss', 'loss_clr', 'loss_mdd')
    val_stats = ('total_acc', 'known_acc', 'unknown_acc')
    recorder = Recorder(args.epochs, val_stats[0], train_stats, val_stats)
    logger.print_log('observing training stats: {} \nvalidation stats: {}'.format(train_stats, val_stats))

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.print_log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.print_log("=> no checkpoint found at '{}'".format(args.resume))

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(start_epoch, args.epochs+1):

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        logger.print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, args.epochs, need_time))

        lr, _ = adjust_learning_rate(optimizer, (args.lr, args.lr / 10), epoch, args.epochs, args.lr_strategy)
        print("   lr:{}".format(lr))

        reachHighest = train(epoch, net, optimizer,
                             source_loader, target_train_loader, args, recorder, logger)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if epoch + 1 % args.evaluate_freq == 0:
            reachHighest = evaluate(net, target_test_loader, unknownID, epoch, recorder, logger, args)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, reachHighest, args.save_path, 'checkpoint.pth.tar')
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))


def train(epoch, model, optimizer, source_loader, target_train_loader, args, recorder, logger):
    batch_time_meter = AverageMeter()
    stats = recorder.train_stats
    meters = {stat: AverageMeter() for stat in stats}

    model.train()

    len_source_dataset, len_source_loader, len_target_loader\
        = len(source_loader.dataset), len(source_loader), len(target_train_loader)
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)

    end = time.time()
    for i in range(1, len_target_loader):
        # use target images iteratively
        if i % len_source_loader == 0:
            iter_source = iter(source_loader)
        data_source, label_source, domain_source = iter_source.next()
        data_target, _, domain_target = iter_target.next()
        data_source_var, label_source_var, domain_source_var = autograd.Variable(data_source), autograd.Variable(label_source), autograd.Variable(domain_source)
        data_target_var, domain_target_var = autograd.Variable(data_target), autograd.Variable(domain_target)

        # get classifier loss and mmd loss
        optimizer.zero_grad()
        label_source_pred, loss_mmd, _, _ = model(data_source_var, label_source_var, domain_source_var, data_target_var, domain_target_var)
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source_var)
        acc = accuracy(label_source_pred.data, label_source, topk=(1,))
        gamma = 2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1
        loss = loss_cls + gamma * loss_mmd
        loss.backward()
        optimizer.step()

        # update meters
        meters['total_acc'].update(acc[0][0], args.batch_size)
        meters['loss'].update(loss, args.batch_size)
        meters['loss_cls'].update(loss_cls, args.batch_size)
        meters['loss_mmd'].update(loss_mmd, args.batch_size)

        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        freq = args.batch_size / batch_time_meter.avg
        end = time.time()

        if i % args.print_freq == 0:
            logger.print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   Freq {:.1f}   '.format(
                epoch, i, len_source_loader, freq) + create_stat_string(meters) + time_string())

    logger.print_log('  **Train**  ' + create_stat_string(meters))

    return recorder.update(epoch=epoch, is_train=True, meters=meters)

def evaluate(model, target_test_loader, unknownID, epoch, recorder, logger, args):
    stats = recorder.val_stats
    meters = {stat: AverageMeter() for stat in stats}
    model.eval()

    test_loss = 0
    correct = 0
    len_target_dataset = len(target_test_loader.dataset)

    for data, label in target_test_loader:
        data, label = data, label
        data_var = autograd.Variable(data, volatile=True)
        t_output, _, _, _ = model(data_var, None, None, data_var, None)
        test_loss += F.nll_loss(F.log_softmax(t_output, dim = 1), label, size_average=False).data[0] # sum up batch loss

        pred = t_output.data.max(1)[1] # get the index of the max log-probability

        known_idx = [i for i in range(0, len(label)) if not label[i] == unknownID]
        unknown_idx = [i for i in range(0, len(label)) if label[i] == unknownID]
        total_acc = pred.eq(label.view_as(pred)).cpu().sum() / len(label)
        known_acc = pred[known_idx].eq(label[known_idx].view_as(known_idx)).cpu().sum() / len(known_idx)
        unknown_acc = pred[known_idx].eq(label[unknown_idx].view_as(unknown_idx)).cpu().sum() / len(unknown_idx)

        meters['total_acc'].update(total_acc, args.batch_size)
        meters['known_acc'].update(known_acc, args.batch_size)
        meters['unknown_acc'].update(unknown_acc, args.batch_size)


    logger.print_log('  **Test**  ' + create_stat_string(meters))
    recorder.update(epoch=epoch, is_train=False, meters=meters)

if __name__ == '__main__':
    main()