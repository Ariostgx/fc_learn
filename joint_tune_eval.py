from __future__ import division
from utils import AverageMeter, BaseOptions, Recorder, Logger, time_string, convert_secs2time, eval_cmc_map, \
    reset_state_dict, extract_features, create_stat_string, save_checkpoint, adjust_learning_rate, accuracy, \
    partition_params
import dataloader
import img_root_txt
from ReIDdatasets import Market

import os
import time
import math
import numpy as np

import torch
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import resnet_mmd
from scipy.spatial.distance import cdist

cudnn.benchmark = True


def data_load(args):
    """
    the source loader contains images from multiple source domains. source labels of these domains are not mixed.
    """
    root_path, sources, target, seed, batch_size= \
        args.img_root, args.sources, args.target, args.seed, args.batch_size

    src_txt = ''
    for iSource in range(len(sources)):
        src_txt += sources[iSource] + '_'
    src_txt += 'train.txt'
    tgt_g_txt = target[0] + "_gallery.txt"
    tgt_p_txt = target[0] + "_prob.txt"

    if not os.path.exists(root_path + src_txt):
        img_root_txt.merge_source_txt(sources)

    kwargs = {'num_workers': 2, 'pin_memory': True}


    source_loader, num_class = dataloader.load_training(root_path, src_txt, batch_size, kwargs)
    gallery_loader, _ = dataloader.load_testing(root_path, tgt_g_txt, batch_size, kwargs)
    probe_loader, _ = dataloader.load_testing(root_path, tgt_p_txt, batch_size, kwargs)

    return source_loader, probe_loader, gallery_loader, num_class


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

    torch.cuda.manual_seed(args.seed)

    source_loader, target_prob_loader, target_gallery_loader, num_class = data_load(args)

    net = resnet_mmd.jointNet(num_classes=num_class)
    net = net.cuda()
    load_pretrain(net)
    logger.print_log('loaded pre-trained feature net')

    criterion_CE = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD([
        {'params': net.sharedNet.parameters()},
        {'params': net.cls_fc.parameters(), 'lr': args.lr},
    ], lr=args.lr / 10, momentum=args.momentum, weight_decay=args.wd)

    # bn_params, conv_params = partition_params(net.sharedNet, 'bn')

    # optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
    # {'params': conv_params}, {'params': net.cls_fc.parameters()}], lr=args.lr, momentum=0.9, weight_decay=args.wd)

    train_stats = ('acc', 'loss')
    val_stats = ('acc', 'map')
    recorder = Recorder(args.epochs, val_stats[0], train_stats, val_stats)
    logger.print_log('observing training stats: {} \nvalidation stats: {}'.format(train_stats, val_stats))

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.print_log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # recorder = checkpoint['recorder']
            #start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.print_log("=> no checkpoint found at '{}'".format(args.resume))

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(start_epoch + 1, args.epochs + 1):

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        logger.print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, args.epochs, need_time))

        lr = adjust_learning_rate(optimizer, (args.lr, args.lr, args.lr), epoch - 1, args.epochs, args.lr_strategy)
        # lr = adjust_learning_rate(optimizer, (args.lr, args.lr), epoch - 1, args.epochs, args.lr_strategy)
        print("   lr:{}".format(lr[0]))

        train(source_loader, net,
              criterion_CE,
              optimizer, epoch, recorder, logger, args)

        # evaluate(target_gallery_loader, target_prob_loader, net,
        #         epoch, recorder, logger)

        if epoch % args.record_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }, False, args.save_path, 'epoch_{}_checkpoint.pth.tar'.format(epoch))
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if epoch % args.evaluate_freq == 0 or epoch == 1:
            evaluate(target_gallery_loader, target_prob_loader, net,
                     epoch, recorder, logger)


def train(train_loader, net,
          criterion_CE,
          optimizer, epoch, recorder, logger, args):
    batch_time_meter = AverageMeter()
    stats = recorder.train_stats
    meters = {stat: AverageMeter() for stat in stats}

    net.train()

    end = time.time()
    for i, (imgs, labels, _, _) in enumerate(train_loader):
        # for i, (imgs, labels, _,) in enumerate(train_loader):
        imgs_var = torch.autograd.Variable(imgs.cuda())
        labels_var = torch.autograd.Variable(labels.cuda())

        _, predictions = net(imgs_var)

        optimizer.zero_grad()
        softmax = criterion_CE(predictions, labels_var)
        softmax.backward()
        acc = accuracy(predictions.data, labels.cuda(), topk=(1,))
        optimizer.step()

        # update meters
        meters['acc'].update(acc[0][0], args.batch_size)
        meters['loss'].update(softmax.data.mean(), args.batch_size)

        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        freq = args.batch_size / batch_time_meter.avg
        end = time.time()

        if i % args.print_freq == 0:
            logger.print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   Freq {:.1f}   '.format(
                epoch, i, len(train_loader), freq) + create_stat_string(meters) + time_string())

    logger.print_log('  **Train**  ' + create_stat_string(meters))

    recorder.update(epoch=epoch, is_train=True, meters=meters)


def evaluate(gallery_loader, probe_loader, net,
             epoch, recorder, logger):
    stats = recorder.val_stats
    meters = {stat: AverageMeter() for stat in stats}
    net.eval()

    gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, net, index_feature=0,
                                                                       require_views=True)
    probe_features, probe_labels, probe_views = extract_features(probe_loader, net, index_feature=0, require_views=True)
    dist = cdist(gallery_features, probe_features, metric='euclidean')
    CMC, MAP = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views)
    rank1 = CMC[0]
    meters['acc'].update(rank1, 1)
    meters['map'].update(MAP, 1)

    logger.print_log('  **Test**  ' + create_stat_string(meters))
    recorder.update(epoch=epoch, is_train=False, meters=meters)


if __name__ == '__main__':
    main()
