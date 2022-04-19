import torch
import torch.nn as nn

from .build import _my_decorator

@_my_decorator.set()
class DenseLayer(nn.Module):
    def __init__(self, cfg = None):
        super(DenseLayer, self).__init__()
        self.flatten = nn.Flatten()
        self.d0 = nn.Linear(28*28,256)

        if cfg['activation'] == 'sigmoid':
            self.act = nn.Sigmoid()
        elif cfg['activation'] == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

        self.outdim = 256

    def forward(self,x):
        x = self.flatten(x)
        x = self.act(self.d0(x))
        return x
