"""Compact two-channel U-Net used by the RIM-ONE baseline."""
import torch
from torch import nn


def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU())


class UNet(nn.Module):
    def __init__(self):
        super().__init__(); self.e1=block(3,16); self.e2=block(16,32); self.e3=block(32,64); self.mid=block(64,128)
        self.pool=nn.MaxPool2d(2); self.d3=block(128+64,64); self.d2=block(64+32,32); self.d1=block(32+16,16); self.head=nn.Conv2d(16,2,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); x=self.mid(self.pool(c))
        x=self.d3(torch.cat((nn.functional.interpolate(x,size=c.shape[-2:]),c),1))
        x=self.d2(torch.cat((nn.functional.interpolate(x,size=b.shape[-2:]),b),1))
        return self.head(self.d1(torch.cat((nn.functional.interpolate(x,size=a.shape[-2:]),a),1)))
