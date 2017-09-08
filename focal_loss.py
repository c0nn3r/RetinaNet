import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class FocalLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):

        cross_entropy = F.cross_entropy(output, target)
        cross_entropy_log = torch.log(cross_entropy)

        focal_loss = -((1 - cross_entropy) ** self.focusing_param) * cross_entropy_log

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


def test_focal_loss():
    loss = FocalLoss()

    input = Variable(torch.randn(3, 5), requires_grad=True)
    target = Variable(torch.LongTensor(3).random_(5))

    print(input)
    print(target)

    output = loss(input, target)
    print(output)
    output.backward()
