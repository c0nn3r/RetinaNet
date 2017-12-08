import torch
import argparse

import torch.nn as nn

from tqdm import tqdm
from torch import optim
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler

from loss import FocalLoss
from retinanet import RetinaNet
from datasets import CocoDetection


train_loader = torch.utils.data.DataLoader(
    CocoDetection(root="./datasets/COCO/train2017",
                  annFile="./datasets/COCO/annotations/instances_train2017.json",
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      # normalized because of the pretrained imagenet
                      transforms.Normalize((0.1307,), (0.3081,))
                  ])),
    # batch size should be 16
    batch_size=1, shuffle=True)

model = RetinaNet(classes=80)
model.eval()

optimizer = optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.9,
                      weight_decay=0.0001)

scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     # Milestones are set assuming batch size is 16:
                                     # 60000 / batch_size = 3750
                                     # 80000 / batch_size = 5000
                                     milestones=[3750, 5000],
                                     gamma=0.1)


criterion = FocalLoss(80)


def train(model, cuda=False):

    average_loss = 0

    if cuda:
        model.cuda()
        model = nn.DataParallel(model)

    for current_batch, (images, box_targets, class_targets) in enumerate(
            tqdm(train_loader, desc='Training on COCO', unit='epoch')):

        scheduler.step()

        optimizer.zero_grad()

        if cuda:
            images.cuda()
            box_targets.cuda()
            class_targets.cuda()

        images = Variable(images)
        # box_predictions = Variable(box_targets)
        # class_predictions = Variable(class_targets)
        box_predictions, classes_predictions = model(images)
        
        loss = criterion(box_predictions, box_targets, class_predictions, class_targets)
        # loss.backwards()
        loss.backward()

        average_loss += loss[0]

        # boxes, classes = model(images)

        optimizer.step()

        print(f'Batch: {current_batch}, Loss: {loss[0]}, Average Loss: {average_loss / current_batch + 1}')


if __name__ == '__main__':
    train(model)
