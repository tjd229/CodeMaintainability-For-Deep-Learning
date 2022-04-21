import torch
import torch.nn as nn
from layers.build import get_layer

class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        _cfg = {k:val for k,val in cfg.items()}
        _cfg['in_channels'] = 1
        self.layer1 = get_layer(cfg['layer1'], _cfg)
        _cfg['in_channels'] = 32
        self.layer2 = get_layer(cfg['layer2'], _cfg)
        _cfg['activation'] = 'none'
        self.layer3 = get_layer(cfg['layer3'], _cfg)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.d1 = nn.Linear(self.layer3.outdim,128)
        self.d2 = nn.Linear(128,10)

        self.fwd_method = cfg['forward_method']

    def vanilla_fwd(self, x):
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y3 = self.relu(y3)
        y3 = self.flatten(y3)
        d1 = self.relu(self.d1(y3))
        return self.d2(d1)
    def residual_fwd(self,x):
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y3 = self.relu(y1+y3)
        y3 = self.flatten(y3)
        d1 = self.relu(self.d1(y3))
        return self.d2(d1)

    _forward_catalog = {
        "vanilla": vanilla_fwd,
        "residual": residual_fwd,
    }

    def forward(self, x):
        return self._forward_catalog[self.fwd_method].__get__(self)(x)
