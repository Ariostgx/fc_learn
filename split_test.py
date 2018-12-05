from __future__ import division
from utils import AverageMeter, BaseOptions, Recorder, Logger, time_string, convert_secs2time, eval_cmc_map, \
    reset_state_dict, extract_split, create_stat_string, save_checkpoint, adjust_learning_rate, accuracy, \
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
from resnet_mmd import Bottleneck, block_networks, block_configs
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

    if not os.path.exists(root_path + src_txt):
        img_root_txt.merge_source_txt(sources)

    kwargs = {'num_workers': 2, 'pin_memory': True}

    mean = np.array([0.485, 0.406, 0.456])
    std = np.array([0.229, 0.224, 0.225])

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    test_path = os.path.join(args.market_root, "{}.mat".format(args.target))
    #train_data = Market(test_path, state='train', transform=train_transform)
    gallery_data = Market(test_path, state='gallery', transform=test_transform)
    probe_data = Market(test_path, state='probe', transform=test_transform)
    #num_class = train_data.return_num_class()

    source_loader, num_class = dataloader.load_training(root_path, src_txt, batch_size, kwargs)
#    source_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
#                                               num_workers=2, pin_memory=True, drop_last=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=True)
    probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=2, pin_memory=True)

    return source_loader, probe_loader, gallery_loader, num_class

def load_pretrain(nets):
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    pretrained_dict = model_zoo.load_url(url)
    for _, subnet in nets.items():
        model_dict = subnet.module.state_dict()
        for k, _ in model_dict.items():
            if not "cls" in k:
                model_dict[k] = pretrained_dict[k[k.find(".")+1:]]
        subnet.module.load_state_dict(model_dict)
    return nets

def load_resume(nets, state_dict):
    for name, subnet in nets.items():
        subnet.load_state_dict(state_dict[name])

    return nets

def load_feaNet(nets, pretrained_dict):
    netName = 'sharedNet'
    for _, subnet in nets.items():
        model_dict = subnet.module.state_dict()
        for k, _ in model_dict.items():
            if not "cls" in k:
                model_dict[k] = pretrained_dict[netName + '.' + k[k.find(".")+1:]]
        subnet.module.load_state_dict(model_dict)

    return nets

def get_device_ids(gpu):
    gpu_list = list(map(int, gpu.split(',')))

    return gpu_list

def splitSourceDomain(dataset):
    class_numbers, class_maps, domains = {}, {}, []
    domains = np.unique(dataset.img_domain)
    imgLabels = np.array(dataset.img_label)
    for domain in domains:
        isDomain = dataset.img_domain == domain
        domLabels = imgLabels[isDomain]
        domClasses = np.unique(domLabels)
        class_numbers[domain] = len(domClasses)
        class_maps[domain] = { label : idx for idx, label in enumerate(domClasses)}

    return class_numbers, class_maps, domains

def main():
    opts = BaseOptions()
    args = opts.parse()

    args.save_path = os.path.join(args.save_path, args.method)

    logger = Logger(args.save_path)
    opts.print_options(logger)
    block_num = args.num_share_block

    torch.cuda.manual_seed(args.seed)

    source_loader, target_prob_loader, target_gallery_loader, num_class = data_load(args)

    class_numbers, class_map, domains = splitSourceDomain(source_loader.dataset)

    nets = {}
    nets['sharedNet'] = resnet_mmd.shareNet(block_num)
    nets['dClassNet'] = resnet_mmd.subNet(block_num, len(domains))
    for domain in domains:
        nets[domain] = resnet_mmd.subNet(block_num, len(class_map[domain]))

    gpu_list = get_device_ids(args.gpu)
    for name in nets.keys():
        nets[name].cuda()
        nets[name] = nn.DataParallel(nets[name], device_ids=gpu_list)

    load_pretrain(nets)
    logger.print_log('loaded pre-trained feature net')

    criterion_CE = nn.CrossEntropyLoss().cuda()

    init_lr = []
    parameters = [{'params': nets['sharedNet'].module.parameters(),'lr':0}]
    init_lr.append(0)
    parameters.append({'params': nets['dClassNet'].module.parameters()})
    init_lr.append(args.lr)
    for domain in domains:
        parameters.append({'params': nets[domain].module.parameters()})
        init_lr.append(args.lr)

    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    train_stats = ('acc', 'loss', 'domain_acc')
    val_stats = ('acc','map')
    recorder = Recorder(args.epochs, val_stats[0], train_stats, val_stats)
    logger.print_log('observing training stats: {} \nvalidation stats: {}'.format(train_stats, val_stats))

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.print_log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.preFeaNet:
                nets = load_feaNet(nets, checkpoint['state_dict'])
            else:
                nets = load_resume(nets, checkpoint['state_dict'])
                start_epoch = checkpoint['epoch']
                recorder = checkpoint['recorder']
                optimizer.load_state_dict(checkpoint['optimizer'])
            logger.print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.print_log("=> no checkpoint found at '{}'".format(args.resume))

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(start_epoch+1, args.epochs+1):

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        logger.print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, args.epochs, need_time))

        lr = adjust_learning_rate(optimizer, init_lr, epoch, args.epochs, args.lr_strategy)
        print("   lr:{}".format(lr[0]))

        train(source_loader, nets, criterion_CE, domains, class_map,
             optimizer, epoch, recorder, logger, args)

        if epoch % args.record_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': {},
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }
            for name, subnet in nets.items():
                checkpoint['state_dict'][name] = subnet.state_dict()

            save_checkpoint(checkpoint, False, args.save_path, 'epoch_{}_checkpoint.pth.tar'.format(epoch))

        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if epoch % args.evaluate_freq == 0 or epoch == 1:
            evaluate(target_gallery_loader, target_prob_loader, nets,
                     epoch, recorder, logger, domains)


def train(train_loader, nets, criterion_CE, totalDomains, class_map,
          optimizer, epoch, recorder, logger, args):

    batch_time_meter = AverageMeter()
    stats = recorder.train_stats
    meters = {stat: AverageMeter() for stat in stats}

    for _, subnet in nets.items():
        subnet.train()

    domain_map = {domain:idx for idx, domain in enumerate(totalDomains)}

    end = time.time()
    for i, (imgs, labels, _, domains) in enumerate(train_loader):
        imgs_var = torch.autograd.Variable(imgs.cuda())
        domains_var = torch.autograd.Variable(domains.cuda())

        optimizer.zero_grad()
        acc = 0
        loss = 0
        clsSoftmax = torch.tensor(0.0).cuda()

        # get image feature from shared net
        img_feature = nets['sharedNet'](imgs_var)

        # get label prediction for each domain
        for domain in totalDomains:
            isDomain = domains_var == int(domain)
            if torch.equal(torch.sum(isDomain).cpu(), torch.tensor([0])):
                continue
            domLabels = labels[isDomain]
            domLabels = torch.tensor([class_map[domain][int(i)] for i in domLabels])
            labels_var = torch.autograd.Variable(domLabels.cuda())

            _, pred = nets[domain](img_feature[isDomain,:])

            softmax = criterion_CE(pred, labels_var)
            domain_size = len(pred)
            acc += sum(accuracy(pred.data, labels_var.data, topk=(1,))) * domain_size
            clsSoftmax += softmax

        acc /= args.batch_size

        # get domain prediction
        _, domPrediction = nets['dClassNet'](img_feature)
        domains_var = torch.autograd.Variable(torch.tensor([domain_map[int(i)] for i in domains_var]).cuda())
        domSoftmax = criterion_CE(domPrediction, domains_var)
        domAcc = sum(accuracy(domPrediction.data, domains_var.data, topk=(1,)))

        loss = clsSoftmax + domSoftmax
        loss.backward()

        optimizer.step()

        # update meters
        meters['acc'].update(acc, args.batch_size)
        meters['loss'].update(loss, args.batch_size)
        meters['domain_acc'].update(domAcc, args.batch_size)

        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        freq = args.batch_size / batch_time_meter.avg
        end = time.time()

        if i % args.print_freq == 0:
            logger.print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   Freq {:.1f}   '.format(
                epoch, i, len(train_loader), freq) + create_stat_string(meters) + time_string())

    logger.print_log('  **Train**  ' + create_stat_string(meters))

    recorder.update(epoch=epoch, is_train=True, meters=meters)

def evaluate(gallery_loader, probe_loader, nets,
             epoch, recorder, logger, domains):

    stats = recorder.val_stats
    meters = {stat: AverageMeter() for stat in stats}

    for _, subnet in nets.items():
        subnet.eval()

    gallery_features, gallery_labels, gallery_views = extract_split(gallery_loader, nets, domains, index_feature=0, require_views=True)
    probe_features, probe_labels, probe_views = extract_split(probe_loader, nets, domains, index_feature=0, require_views=True)
    dist = cdist(gallery_features, probe_features, metric='euclidean')
    CMC, MAP = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views)
    rank1 = CMC[0]
    meters['acc'].update(rank1, 1)
    meters['map'].update(MAP, 1)

    logger.print_log('  **Test**  ' + create_stat_string(meters))
    recorder.update(epoch=epoch, is_train=False, meters=meters)

if __name__ == '__main__':
    main()
