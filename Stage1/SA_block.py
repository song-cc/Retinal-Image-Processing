import torch
import torch.nn as nn

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7, 7),padding=3),
            nn.ReLU()
        )

    def forward(self,x):
        x1 = torch.max(x,dim=1).values
        x2 = torch.mean(x,dim=1)
        x1 = torch.unsqueeze(x1,dim=1)
        x2 = torch.unsqueeze(x2,dim=1)
        y = torch.cat((x1,x2),dim=1)
        y = self.conv(y)
        y = torch.multiply(y,x)

        return y
