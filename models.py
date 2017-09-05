import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from torch.autograd import Variable
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


def classification_layer_init(tensor, pi=0.01):

    fill_constant = - math.log((1 - pi) / pi)

    if isinstance(tensor, Variable):
        classification_layer_init(tensor.data)

    return tensor.fill_(fill_constant)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv1x1(in_channels, out_channels, **kwargs):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
    nn.init.normal(layer.weight.data, std=0.01)
    nn.init.constant(layer.bias.data, val=0)

    return layer


class BasicBlockFeatures(BasicBlock):

    def forward(self, x):

        if isinstance(x, tuple):
            x = x[0]

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        conv2_rep = out
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, conv2_rep


class BottleneckFeatures(Bottleneck):
    '''
    A Bottleneck that returns its last conv layer features.
    '''

    def forward(self, x):

        if isinstance(x, tuple):
            x = x[0]

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        conv3_rep = out
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, conv3_rep


class ResNetFeatures(ResNet):
    '''
    A ResNet that returns features instead of classification.
    '''

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, c2 = self.layer1(x)
        x, c3 = self.layer2(x)
        x, c4 = self.layer3(x)
        x, c5 = self.layer4(x)

        return c2, c3, c4, c5


def resnet18_features(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BasicBlockFeatures, [2, 2, 2, 2], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model


def resnet34_features(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BasicBlockFeatures, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

    return model


def resnet50_features(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model


def resnet101_features(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 23, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

    return model


def resnet152_features(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BottleneckFeatures, [3, 8, 36, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

    return model


class SmartFeaturePyramid(nn.Module):
    '''
    Super smart feature pyramid
    '''

    def __init__(self, resnet):
        super(SmartFeaturePyramid, self).__init__()

        self.resnet = resnet
        _temporary_input = Variable(torch.Tensor(1, 3, 224, 224))
        self.feature_sizes = [output.size(1) for output in self.resnet(_temporary_input)]

        self.pyramid_transformation_6 = nn.Conv2d(self.feature_sizes[-1], 256, kernel_size=3)
        self.pyramid_transformation_7 = nn.Conv2d(256, 256, kernel_size=3)

        self.pyramid_transformations = nn.ModuleList([nn.Conv2d(feature_size, 256, kernel_size=1)
                                                      # ignore the first output (c2)
                                                      for feature_size in self.feature_sizes[1:]])

    def forward(self, x):
        convolutional_features = [*self.resnet(x)]

        pyramid_feature_6 = self.pyramid_transformation_6(convolutional_features[-1])
        pyramid_feature_7 = self.pyramid_transformation_7(F.relu(pyramid_feature_6))

        pyramid_features = []

        for current_transformation, pyramid_transformation in enumerate(self.pyramid_transformations):
            pyramid_features.append(
                pyramid_transformation(convolutional_features[1 + current_transformation]))


class SubNet(nn.Module):

    def __init__(self, mode, anchors=9, classes=2, depth=4, activation=F.relu):
        self.mode = mode
        self.anchors = anchors
        self.classes = classes
        self.depth = depth
        self.activation = activation

        self.subnet_base = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3)
                                          for _ in range(depth)])

        if mode == 'boxes':
            self.subnet_output = nn.Conv2d(256, 4 * self.anchors, kernel_size=3)
        elif mode == 'classes':
            self.subnet_output = nn.Conv2d(256, self.classes * self.anchors, kernel_size=3)

    def forward(self, x):

        for layer in self.subnet_base:
            x = self.activation(layer(x))

        return self.subnet_output(x)


class RetinaNet(nn.Module):

    def __init__(self, resnet):
        self.resnet_features = resnet()
        self.subnet_boxes = SubNet(mode='boxes')
        self.subnet_classes = SubNet(mode='classes')

    def forward(self, x):
        pass
