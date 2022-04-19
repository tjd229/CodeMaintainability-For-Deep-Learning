import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten

from .build import _my_decorator

@_my_decorator.set()
class DenseLayer(tf.keras.Model):
    def __init__(self, cfg = None):
        super(DenseLayer, self).__init__()
        self.flatten = Flatten()
        self.d0 = Dense(256, activation=cfg['activation'])

    def call(self,x):
        x = self.flatten(x)
        x = self.d0(x)
        return x