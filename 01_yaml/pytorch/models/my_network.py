import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, activation='sigmoid'):
        super(MyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.d0 = nn.Linear(28*28,256)
        self.d1 = nn.Linear(256,128)
        self.d2 = nn.Linear(128,10)

        if activation =='sigmoid':
            self.act = nn.Sigmoid()
        elif activation =='relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.flatten(x)
        x = self.act(self.d0(x))
        x = self.act(self.d1(x))
        return self.d2(x)
