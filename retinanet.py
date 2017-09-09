import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from utilities.layers import conv1x1, conv3x3


class SubNet(nn.Module):
    '''
    '''

    def __init__(self, mode, anchors=9, classes=80, depth=4, base_activation=F.relu,
                 output_activation=F.sigmoid):
        super(SubNet, self).__init__()
        self.anchors = anchors
        self.classes = classes
        self.depth = depth
        self.activation = activation

        self.subnet_base = nn.ModuleList([conv3x3(256, 256, padding=1)
                                          for _ in range(depth)])

        if mode == 'boxes':
            self.subnet_output = conv3x3(256, 4 * self.anchors, padding=1)
        elif mode == 'classes':
            self.subnet_output = conv3x3(256, self.classes * self.anchors, padding=1)

        self.classification_layer_init(self.subnet_output.bias.data)

    def classification_layer_init(self, tensor, pi=0.01):
        fill_constant = - math.log((1 - pi) / pi)

        if isinstance(tensor, Variable):
            self.classification_layer_init(tensor.data)

        return tensor.fill_(fill_constant)

    def forward(self, x):
        for layer in self.subnet_base:
            x = self.activation(layer(x))

        box_predictions = box_predictions.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        return F.sigmoid(self.subnet_output(x))


class RetinaNet(nn.Module):
    '''
    '''

    def __init__(self, classes=80):
        '''
        '''
        super(RetinaNet, self).__init__()
        self.classes = classes

        self.resnet = resnet50_features(pretrained=True)
        self.feature_pyramid = FeaturePyramid(self.resnet)

        self.subnet_boxes = SubNet(mode='boxes')
        self.subnet_classes = SubNet(mode='classes')

    def forward(self, x):
        '''
        '''

        boxes = []
        classes = []

        features = self.feature_pyramid(x)

        for feature in features:
            box_predictions = self.subnet_boxes(feature)
            class_predictions = self.subnet_classes(feature)
            class_predictions = class_predictions.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_of_classes)

            boxes.append(box_predictions)
            classes.append(class_predictions)

        return torch.cat(boxes, 1), torch.cat(classes, 1)
