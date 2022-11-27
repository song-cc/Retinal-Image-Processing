import torch
import torch.nn as nn
import multiprocessing as mp
import multiprocessing as mp

import torch
import torch.nn as nn


class Conv1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

    def forward(self,x):
        return self.conv(x)

class Bottlneck(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,downsampling=False,expansion=4):
        super(Bottlneck, self).__init__()
        self.downsampling = downsampling
        self.expansion = expansion
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel * self.expansion,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channel * self.expansion)
        )

        if self.downsampling:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_channel,out_channel * self.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channel * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,blocks,num_classes=2,expansion=4):
        super(ResNet, self).__init__()
        self.normal = nn.BatchNorm2d(3)
        self.expansion = expansion

        self.conv1 = Conv1(3,64)
        self.layer1 = self.make_layer(64,64,blocks[0],stride=1)
        self.layer2 = self.make_layer(256,128,blocks[0],stride=2)
        self.layer3 = self.make_layer(512,256,blocks[2],stride=2)
        self.layer4 = self.make_layer(1024,512,blocks[3],stride=2)

        self.avgpool = nn.AvgPool2d(7,stride=1)
        #self.linear = nn.Linear(8192,num_classes)
        self.linear = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64,8),
            nn.ReLU(inplace=True),
            nn.Linear(8,1),
            nn.Sigmoid()
                                    )
        self.softMax = nn.Softmax()

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal(m.weight,mode="fan_out",nonlinearity="relu")
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant(m.weight,1)
                nn.init.constant(m.bias,0)

    def make_layer(self,in_channel,out_channel,block,stride):
        layers = []
        layers.append(Bottlneck(in_channel,out_channel,stride,downsampling=True))

        for i in range(1,block):
            layers.append(Bottlneck(out_channel * self.expansion,out_channel))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.normal(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0),-1)
        x = self.linear(x)

        #x = self.softMax(x)
        return x

def ResNet50():
    return ResNet([3,4,6,3])

def ResNet101():
    return ResNet([3,4,23,3])

if __name__ == '__main__':

    mp.set_start_method('spawn')
    model = ResNet50()
    inpurt = torch.randn(1,3,224,224)
    out = model(inpurt)
    print(out.shape)
