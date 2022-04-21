import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import _my_decorator



@_my_decorator.set()
class ConvLayer(nn.Module):
    def __init__(self, cfg = None):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(cfg['in_channels'], 32, 3, padding = 1 )


        if cfg['activation'] =='sigmoid':
            self.act = nn.Sigmoid()
        elif cfg['activation'] =='relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

        self.outdim = 28*28*32

    def forward(self,x):
        x = self.act(self.conv(x))
        return x