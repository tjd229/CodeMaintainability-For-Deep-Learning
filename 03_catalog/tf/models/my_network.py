import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from layers.build import get_layer

class MyModel(Model):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.layer1 = get_layer(cfg['layer1'], cfg)
        self.layer2 = get_layer(cfg['layer2'], cfg)
        self.layer3 = get_layer(cfg['layer3'], {'activation' : None})
        self.relu = tf.keras.layers.ReLU()
        self.flatten = Flatten()

        self.d1 = Dense(128, activation=cfg['activation'])
        self.d2 = Dense(10)

        self.fwd_method = cfg['forward_method']

    def vanilla_fwd(self, x):
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y3 = self.relu(y3)
        y3 = self.flatten(y3)
        d1 = self.d1(y3)
        return self.d2(d1)
    def residual_fwd(self,x):
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y3 = self.relu(y1+y3)
        y3 = self.flatten(y3)
        d1 = self.d1(y3)
        return self.d2(d1)

    _forward_catalog = {
        "vanilla": vanilla_fwd,
        "residual": residual_fwd,
    }

    def call(self, x):
        return self._forward_catalog[self.fwd_method].__get__(self)(x)
