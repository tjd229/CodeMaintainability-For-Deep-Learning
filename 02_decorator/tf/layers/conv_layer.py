import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten

from .build import _my_decorator

@_my_decorator.set()
class ConvLayer(tf.keras.Model):
    def __init__(self, cfg = None):
        super(ConvLayer, self).__init__()
        self.conv = Conv2D(32, 3, activation=cfg['activation'])
        self.flatten = Flatten()

    def call(self,x):
        x = self.conv(x)
        x = self.flatten(x)
        return x