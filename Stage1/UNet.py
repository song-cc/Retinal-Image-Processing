import torch
import torch.nn as nn

from Stage1.DropBlock import DropBlock2D
from Stage1.SA_block import SA


class down_s(nn.Module):
    def __init__(self, c_in, c_out):
        super(down_s, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            DropBlock2D(),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
        self.Maxpool = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            SA()
        )

    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.Maxpool(out_1)

        return out_1, out_2


class up_s(nn.Module):
    def __init__(self, c_in, c_out):
        super(up_s, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            DropBlock2D(),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=c_out, out_channels=int(c_out / 2), kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(int(c_out / 2)),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Conv2d(c_out, 1, 3, padding=1),
            DropBlock2D(),
            nn.Sigmoid()
        )

    def forward(self, x0, x1):
        '''
            x0:下一层的输入
            x1：x0对应下采样层的输入
        '''
        y = self.conv2(x0)
        y0 = self.out(y)
        y = self.up(y)
        y = torch.cat((y, x1), dim=1)

        return y0, y


class conv(nn.Module):
    def __init__(self):
        super(conv, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            DropBlock2D(),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            # nn.Softmax()
        )

    def forward(self, x):
        y = self.conv3(x)
        return y


class UNet(nn.Module):
    def __init__(self, down_chanels, up_chanels):
        super(UNet, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.d1 = down_s(down_chanels[0], down_chanels[1])
        self.d2 = down_s(down_chanels[1], down_chanels[2])
        self.d3 = down_s(down_chanels[2], down_chanels[3])
        self.d4 = down_s(down_chanels[3], down_chanels[4])

        self.u1 = up_s(up_chanels[0], up_chanels[1])
        self.u2 = up_s(up_chanels[1], up_chanels[2])
        self.u3 = up_s(up_chanels[2], up_chanels[3])
        self.u4 = up_s(up_chanels[3], up_chanels[4])

        self.conv = conv()

    def forward(self, x):
        x = self.norm(x)
        y0, y1 = self.d1(x)
        y2, y3 = self.d2(y1)
        y4, y5 = self.d3(y3)
        y6, y7 = self.d4(y5)
        y_8, y = self.u1(y7, y6)
        y_9, y = self.u2(y, y4)
        y_10, y = self.u3(y, y2)
        y_11, y = self.u4(y, y0)
        y = self.conv(y)

        return y_8, y_9, y_10, y_11, y