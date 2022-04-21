import torch
import torch.nn as nn

class MyModel(nn.Module):


    def __init__(self, cfg):
        super(MyModel, self).__init__()
        _cfg = {k: val for k, val in cfg.items()}

        self.conv1 = nn.Conv2d(1,16,3, padding = 1)
        _cfg["in_channels"] = 16
        self.after1 = self.after_layer(_cfg)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        _cfg["in_channels"] = 32
        self.after2 = self.after_layer(_cfg)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        _cfg["in_channels"] = 64
        self.after3 = self.after_layer(_cfg)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.after4 = self.relu(_cfg)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(14 * 14 * 128, 10)


    def relu(self, cfg):
        return nn.ReLU()
    def bn(self,cfg):
        return nn.BatchNorm2d(cfg["in_channels"])

    _afterlayer_catalog = {
        "relu": relu,
        "bn": bn,
    }
    def after_layer(self, cfg):
        seq = cfg['after_layer']
        return nn.Sequential( *[self._afterlayer_catalog[name].__get__(self)(cfg) for name in seq] )

    def forward(self, x):
        x = self.conv1(x)
        x = self.after1(x)
        x = self.conv2(x)
        x = self.after2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.after3(x)
        x = self.conv4(x)
        x = self.after4(x)
        x = self.flatten(x)
        return self.dense(x)
