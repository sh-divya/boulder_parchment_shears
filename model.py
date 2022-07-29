import os
import torch
import random
import torchvision
from torch import nn
from torchvision.io.image import read_image
from torch.utils.data import DataLoader, Dataset

class BPSDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.labels = {
            'rock': 0,
            'paper': 1,
            'scissors': 2
        }

        self.imgs = []
        for rps in os.listdir(root):
            for img in os.listdir(os.path.join(root, rps)):
                self.imgs.append(os.path.join(
                                 os.path.join(root, rps),
                                 img)
                )

        random.shuffle(self.imgs)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = read_image(self.imgs[idx], mode=torchvision.io.image.ImageReadMode.RGB).float()
        label = self.labels[self.imgs[idx].split('/')[-2]]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # print(label)
        label = torch.tensor([i==label for i in range(3)], dtype=float)

        return image, label

class CNN(nn.Module):
    def __init__(self, convs=[], channels=[], lin_layers=[]):
        super().__init__()
        self.conv_layers = []
        self.size = 300
        channel = 3

        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU()
        self.batch_norms = []
        for c, conv in enumerate(convs):
            k, s = conv
            # print(channels[c])
            # print(c)
            layer = nn.Conv2d(channel, channels[c], k, s)
            channel = channels[c]
            self.size = ((self.size - k) / s) + 1
            # print(self.size)
            # pool = nn.MaxPool2d(2, 2)
            self.conv_layers += [layer]
            self.size = ((self.size - 2) / 2) + 1
            # print(self.size)
            # self.batch_norms.append(nn.BatchNorm2d(channel).cuda())
        self.conv_layers = nn.ModuleList(self.conv_layers)
        # print(self.size)
        # print(channel)
        self.length = self.size * self.size * channel
        # print(self.length)
        self.lin_layers = []
        for lin in lin_layers:
            self.lin_layers += [nn.Linear(int(self.length), int(lin))]
            self.length = lin
        self.lin_layers = nn.ModuleList(self.lin_layers)

    def forward(self, x):
        for c, conv in enumerate(self.conv_layers):
            # print(x.shape)
            x = self.pool(self.act(conv(x)))
            # x = self.batch_norms[c](self.pool(self.act(conv(x))))
            # conv(x)
            # break

        # x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)

        for l, lin in enumerate(self.lin_layers):
            x = lin(x)
            if l == len(self.lin_layers) - 1:
                break
            else:
                x = self.act(x)

        return x

if __name__ == '__main__':
    device = torch.device('cuda:0')
    temp = BPSDataset('/home/minion/Documents/boulder_parchment_shears/rps')

    # print(temp[:2000])
    for i in range(2520):
        print(temp[i][0].to(device))
    net = CNN([(5, 1), (5, 1)], (6, 16), (24 * 24,  12 * 12, 3)).to(device)
    # net(temp[0][0].float())

