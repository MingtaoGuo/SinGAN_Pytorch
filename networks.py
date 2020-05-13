import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, nums_block):
        super(Generator, self).__init__()
        self.nums_block = nums_block
        self.padding = nn.ZeroPad2d(5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, nums_block, 3, 1),
            nn.BatchNorm2d(nums_block),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nums_block, nums_block, 3, 1),
            nn.BatchNorm2d(nums_block),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nums_block, nums_block, 3, 1),
            nn.BatchNorm2d(nums_block),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(nums_block, nums_block, 3, 1),
            nn.BatchNorm2d(nums_block),
            nn.LeakyReLU(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(nums_block, 3, 3, 1),
            nn.Tanh()
        )

    def forward(self, z, x):
        temp = x
        x = self.padding(x+z)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        res = self.conv5(x)
        return res + temp


class Discriminator(nn.Module):
    def __init__(self, nums_block):
        super(Discriminator, self).__init__()
        self.nums_block = nums_block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, nums_block, 3, 1),
            nn.BatchNorm2d(nums_block),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nums_block, nums_block, 3, 1),
            nn.BatchNorm2d(nums_block),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nums_block, nums_block, 3, 1),
            nn.BatchNorm2d(nums_block),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(nums_block, nums_block, 3, 1),
            nn.BatchNorm2d(nums_block),
            nn.LeakyReLU(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(nums_block, 1, 3, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x