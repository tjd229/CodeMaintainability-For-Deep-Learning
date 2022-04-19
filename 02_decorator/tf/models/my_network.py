import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from layers.build import get_layer

class MyModel(Model):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.pre_layer = get_layer(cfg['prelayer'], cfg)

        self.d1 = Dense(128, activation=cfg['activation'])
        self.d2 = Dense(10)

    def summary(self):
        super().summary()
        self.pre_layer.summary()

    def call(self, x):
        x = self.pre_layer(x)

        x = self.d1(x)
        return self.d2(x)
