import torch

from datasets import CocoDetection
from torchvision import transforms

from tqdm import tqdm
from torch import optim
from torch.optim import lr_scheduler

train_loader = torch.utils.data.DataLoader(
    CocoDetection(root="./datasets/COCO/train2017",
                  annFile="./datasets/COCO/annotations/instances_train2017.json",
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      # transforms.Normalize((0.1307,), (0.3081,))
                  ])),
    batch_size=2, shuffle=True)

# optimizer = optim.SGD(lr=0.01, momentum=0.9, weight_decay=0.0001)

# scheduler = lr_scheduler.MultiStepLR(optimizer,
#                                      # 60000 / batch_size (16) = 3750
#                                      # 80000 / batch_size (16) = 5000
#                                      milestones=[3750, 5000],
#                                      gamma=0.1)


def train():
    for sample, target in tqdm(train_loader):
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    train()
