import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from resnet import ResNet
from resnet_split import ResNet_S1, ResNet_S2, ResNet_S3, ResNet_P1, ResNet_P2, ResNet_P3

import torch
import scipy.io as sio


__all__ = ['ResNet', 'resnet50']

block_networks = {1:{'share':ResNet_S1, 'split':ResNet_P1}, 2:{'share':ResNet_S2, 'split':ResNet_P2}, 3:{'share':ResNet_S3, 'split':ResNet_P3}}
block_configs = [3,4,6,3]

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class jointNet(nn.Module):
    def __init__(self, num_classes=31):
        super(jointNet, self).__init__()
        self.sharedNet = resnet50(False)
        self.num_classes = num_classes
        self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source):
        source = self.sharedNet(source)
        source_pred = self.cls_fc(source)

        return source, source_pred

class splitNet(nn.Module):
    def __init__(self, class_numbers, domains, shared_blocks):
        super(splitNet, self).__init__()
        self.domains = domains
        networks = block_networks[shared_blocks]
        self.sharedNet = networks['share'](Bottleneck, block_configs)
        self.splitNets = {}
        for domain in domains:
            self.splitNets[domain] = {}
            self.splitNets[domain]['feaNet'] = networks['split'](Bottleneck, block_configs)
            self.splitNets[domain]['cls_fc'] = nn.Linear(2048, class_numbers[domain])
        self.domain_feaNet = networks['split'](Bottleneck, block_configs)
        self.domain_cls = nn.Linear(2048, len(domains))

    def forward(self, source, domainIds=None):
        source = self.sharedNet(source)

        if self.training:
            source_pred = {}
            for domain in self.domains:
                isDomain = domainIds == int(domain)
                if torch.equal(torch.sum(isDomain).cpu(), torch.tensor([0])):
                    continue
                self.splitNets[domain]['feaNet'].train()
                self.splitNets[domain]['cls_fc'].train()
                feature = self.splitNets[domain]['feaNet'](source[isDomain,:])
                prediction = self.splitNets[domain]['cls_fc'](feature)
                source_pred[domain] = prediction

            domFeature = self.domain_feaNet(source)
            domPrediction = self.domain_cls(domFeature)
            return source_pred, domPrediction

        else:
            domPrediction = self.domain_cls(self.domain_feaNet(source))
            m = nn.Softmax()
            domPrediction = m(domPrediction)

            if domainIds == None or domainIds == -1:
                feature = torch.zeros([source.shape[0], 2048]).cuda()

                for idx, domain in enumerate(self.domains):
                    domScore = torch.tensor([[domPrediction[i][idx]] for i in range(source.shape[0])])
                    domScore = domScore.repeat(1, 2048)
                    self.splitNets[domain].eval()
                    fea = self.splitNets[domain](source)
                    feature += domScore.cuda() * fea
            else:
                self.splitNets[domainIds].eval()
                feature = self.splitNets[domainIds](source)

            return feature, domPrediction

class splitNet_oneCls(nn.Module):
    def __init__(self, class_numbers, domains, shared_blocks):
        super(splitNet_oneCls, self).__init__()
        self.domains = domains
        networks = block_networks[shared_blocks]
        self.sharedNet = networks['share'](Bottleneck, block_configs)
        total_class_number = sum([class_numbers[domain] for domain in domains])
        self.shared_cls = nn.Linear(2048, total_class_number)

        self.splitNets = {}
        for domain in domains:
            self.splitNets[domain] = {}
            self.splitNets[domain] = networks['split'](Bottleneck, block_configs)

        self.domain_feaNet = networks['split'](Bottleneck, block_configs)
        self.domain_cls = nn.Linear(2048, len(domains))

    def forward(self, source, domainIds=None):
        source = self.sharedNet(source)

        if self.training:
            source_pred = {}
            for domain in self.domains:
                isDomain = domainIds == int(domain)
                if torch.equal(torch.sum(isDomain).cpu(), torch.tensor([0])):
                    continue
                self.splitNets[domain].train()
                feature = self.splitNets[domain](source[isDomain,:])
                prediction = self.shared_cls(feature)
                source_pred[domain] = prediction

            domFeature = self.domain_feaNet(source)
            domPrediction = self.domain_cls(domFeature)
            return source_pred, domPrediction

        else:
            domPrediction = self.domain_cls(self.domain_feaNet(source))
            m = nn.Softmax()
            domPrediction = m(domPrediction)

            if domainIds == None or domainIds == -1:
                feature = torch.zeros([source.shape[0], 2048]).cuda()

                for idx, domain in enumerate(self.domains):
                    domScore = torch.tensor([[domPrediction[i][idx]] for i in range(source.shape[0])])
                    domScore = domScore.repeat(1, 2048)
                    self.splitNets[domain].eval()
                    fea = self.splitNets[domain](source)
                    feature += domScore.cuda() * fea
            else:
                self.splitNets[domainIds].eval()
                feature = self.splitNets[domainIds](source)

            return feature, domPrediction




class shareNet(nn.Module):
    def __init__(self, shared_blocks):
        super(shareNet, self).__init__()
        self.feaNet = block_networks[shared_blocks]['share'](Bottleneck, block_configs)

    def forward(self, source):
        feature = self.feaNet(source)

        return feature

class subNet(nn.Module):
    def __init__(self, shared_blocks, num_class):
        super(subNet, self).__init__()
        self.feaNet = block_networks[shared_blocks]['split'](Bottleneck, block_configs)
        self.cls = nn.Linear(2048, len(num_class))

    def forward(self, source):
        feature = self.feaNet(source)
        pred = self.cls(feature)

        return feature, pred

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
