import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self, activation='sigmoid'):
        super(MyModel, self).__init__()
        self.flatten = Flatten()
        self.d0 = Dense(256, activation=activation)
        self.d1 = Dense(128, activation=activation)
        self.d2 = Dense(10)

    def call(self, x):
        x = self.flatten(x)
        x = self.d0(x)
        x = self.d1(x)
        return self.d2(x)
