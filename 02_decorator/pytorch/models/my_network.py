import torch
import torch.nn as nn
from layers.build import get_layer

class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.pre_layer = get_layer(cfg['prelayer'], cfg)

        self.d1 = nn.Linear(self.pre_layer.outdim,128)
        self.d2 = nn.Linear(128,10)

        if cfg['activation'] == 'sigmoid':
            self.act = nn.Sigmoid()
        elif cfg['activation'] == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.act(self.d1(x))
        return self.d2(x)
