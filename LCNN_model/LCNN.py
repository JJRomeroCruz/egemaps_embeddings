import tensorflow as tf
from tensorflow.keras import layers, models

class PreEmphasis(layers.Layer):
    def __init__(self, coef=0.97, **kwargs):
        super().__init__(**kwargs)
        self.coef = coef
        a = tf.constant([-coef, 1.0], shape=(1,2,1))
        self.filter = tf.transpose(a, perm=[0,2,1])

    def call(self, inputs):
        x = tf.expand_dims(inputs, -1) # esto tiene shape de [batch, time, 1]
        x = tf.pad