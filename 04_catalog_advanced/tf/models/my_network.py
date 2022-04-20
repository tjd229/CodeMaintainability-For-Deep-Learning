import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class MyModel(Model):
    def __init__(self, cfg):
        super(MyModel, self).__init__()

        self.conv1 = Conv2D(16, 3, padding = 'same')
        self.after1 = self.after_layer(cfg)
        self.conv2 = Conv2D(32, 3, padding = 'same')
        self.after2 = self.after_layer(cfg)
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2) )
        self.conv3 = Conv2D(64, 3, padding = 'same')
        self.after3 = self.after_layer(cfg)
        self.conv4 = Conv2D(128, 3, padding = 'same')
        self.after4 = self.relu()
        self.flatten = Flatten()
        self.dense = Dense( 10)

    def relu(self):
        return tf.keras.layers.ReLU()
    def bn(self):
        return tf.keras.layers.BatchNormalization()

    _afterlayer_catalog = {
        "relu": relu,
        "bn": bn,
    }
    def after_layer(self, cfg):
        seq = cfg['after_layer']
        layer = tf.keras.Sequential()
        for name in seq:
            layer.add(self._afterlayer_catalog[name].__get__(self)())
        return layer
    def summary(self):
        super().summary()
        self.after1.summary()
        self.after2.summary()
        self.after3.summary()
    def call(self, x):
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
