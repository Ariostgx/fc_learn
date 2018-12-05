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

def load_pretrain(model):
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()
    for k, _ in model_dict.items():
        if not "cls" in k:
            model_dict[k] = pretrained_dict[k[k.find(".")+1:]]
    model.load_state_dict(model_dict)

    for key, value in model.splitNets.items():
        splitNetOne = value
        break
    model_dict = splitNetOne.state_dict()
    for k, _ in model_dict.items():
        model_dict[k] = pretrained_dict[k]
    for domain in model.splitNets.keys():
        model.splitNets[domain].load_state_dict(model_dict)

    return model

def load_resume(net, state_dict):
    net.load_state_dict(state_dict['main'])

    for domain in net.splitNets.keys():
        net.splitNets[domain].load_state_dict(state_dict[str(domain)])

    return net

def load_feaNet(model, pretrained_dict):
    subnet = 'sharedNet'
    model_dict = model.state_dict()
    for k, _ in model_dict.items():
        if not "cls" in k:
            model_dict[k] = pretrained_dict[subnet + '.' + k[k.find(".")+1:]]
    model.load_state_dict(model_dict)

    for key, value in model.splitNets.items():
        splitNetOne = value
        break
    model_dict = splitNetOne.state_dict()
    for k, _ in model_dict.items():
        model_dict[k] = pretrained_dict[subnet + '.' + k]
    for domain in model.splitNets.keys():
        model.splitNets[domain].load_state_dict(model_dict)

    return model

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

    torch.cuda.manual_seed(args.seed)

    source_loader, target_prob_loader, target_gallery_loader, num_class = data_load(args)

    class_numbers, class_map, domains = splitSourceDomain(source_loader.dataset)

    net = resnet_mmd.splitNet_oneCls(class_numbers, domains, args.num_share_block)


    net = net.cuda()
    for domain, subnet in net.splitNets.items():
        subnet = subnet.cuda()
    load_pretrain(net)
    logger.print_log('loaded pre-trained feature net')

    criterion_CE = nn.CrossEntropyLoss().cuda()

    init_lr = []

    if args.lockShare == 1:
        share_lr = 0
        print("lock share!")
    else:
        share_lr = args.lr / 100
        print('share low lr rate!')

    parameters = [{'params': net.sharedNet.parameters(), 'lr':share_lr}]
    init_lr.append(share_lr)
    parameters.append({'params': net.shared_cls.parameters(), 'lr': args.lr})
    init_lr.append(args.lr)
    parameters.append({'params': net.domain_feaNet.parameters()})
    init_lr.append(args.lr/10)
    parameters.append({'params': net.domain_cls.parameters(), 'lr': args.lr})
    init_lr.append(args.lr)
    for domain in domains:
        parameters.append({'params': net.splitNets[domain].parameters()})
        init_lr.append(args.lr/10)

    optimizer = torch.optim.SGD(parameters, lr=args.lr/10, momentum=args.momentum, weight_decay=args.wd)

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
                net = load_feaNet(net, checkpoint['state_dict'])
            else:
                net = load_resume(net, checkpoint['state_dict'])
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

        #if epoch == 1:
        #    evaluate(target_gallery_loader, target_prob_loader, net,
        #             epoch, recorder, logger)

        train(source_loader, net, criterion_CE, domains, class_map,
             optimizer, epoch, recorder, logger, args)

        #evaluate(target_gallery_loader, target_prob_loader, net,
                 #epoch, recorder, logger)

        if epoch % args.record_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': {'main':net.state_dict()},
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }
            for domain, subnet in net.splitNets.items():
                checkpoint['state_dict'][str(domain)] = subnet.state_dict()

            save_checkpoint(checkpoint, False, args.save_path, 'epoch_{}_checkpoint.pth.tar'.format(epoch))

        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if epoch % args.evaluate_freq == 0 or epoch == 1:
            evaluate(target_gallery_loader, target_prob_loader, net,
                     epoch, recorder, logger)


def train(train_loader, net, criterion_CE, totalDomains, class_map,
          optimizer, epoch, recorder, logger, args):

    batch_time_meter = AverageMeter()
    stats = recorder.train_stats
    meters = {stat: AverageMeter() for stat in stats}

    net.train()
    domain_map = {domain:idx for idx, domain in enumerate(totalDomains)}

    end = time.time()
    for i, (imgs, labels, _, domains) in enumerate(train_loader):
        imgs_var = torch.autograd.Variable(imgs.cuda())
        domains_var = torch.autograd.Variable(domains.cuda())

        optimizer.zero_grad()
        acc = 0
        loss = 0
        clsSoftmax = torch.tensor(0.0).cuda()
        source_pred, domPrediction = net(imgs_var, domains_var)
        for domain in totalDomains:
            isDomain = domains_var==int(domain)
            if torch.equal(torch.sum(isDomain).cpu(), torch.tensor([0])):
                continue
            domLabels = labels[isDomain]
            labels_var = torch.autograd.Variable(domLabels.cuda())
            pred = source_pred[domain]
            softmax = criterion_CE(pred, labels_var)
            domain_size = len(pred)
            acc += sum(accuracy(pred.data, labels_var.data, topk=(1,))) * domain_size
            loss += domain_size * softmax.data.mean()
            clsSoftmax += softmax

        acc /= args.batch_size
        loss /= args.batch_size

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

def evaluate(gallery_loader, probe_loader, net,
             epoch, recorder, logger):

    stats = recorder.val_stats
    meters = {stat: AverageMeter() for stat in stats}
    net.eval()

    gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, net, index_feature=0, require_views=True)
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
