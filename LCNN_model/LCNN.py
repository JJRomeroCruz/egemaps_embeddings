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
        x = tf.pad(x, [[0, 0], [1, 0], [0, 0]], mode='REFLECT')
        return tf.squeeze(tf.nn.conv1d(x, self.filter, stride=1, padding='VALID'), -1)
    

def mfm(x):
    # asumimos que la dimensión de los canales es lo ultimo
    c = tf.shape(x)[-1]
    assert c % 2 == 0
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    return tf.maximum(x1, x2)

def build_lcnn(num_classes=2):
    inp = layers.Input(shape=(None, ), name="waveform")
    x = PreEmphasis()(inp)
    x = tf.signal.stft(
        x, frame_length=400, frame_step=160, fft_lenght=512, window_fn=tf.signal.hamming_window
        )
    x = tf.abs(x)
    mel = tf.signal.linear_to_mel_weight_matrix(80, 257, 16000, 20.0, 7600.0)
    x = tf.tensordot(x, mel, axes=[-1, 0])
    x = tf.math.log(x + 1e-6)
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    x = x - mean
    x = tf.expand_dims(x, -1) # [batch, time, n_mels, 1]

    # bloques convolucionales
    def conv_mfm_block(y, filters, k, pool=False):
        y = layers.Conv2D(filters*2. k, padding='same')(y)
        y=mfm(y)
        if pool:
            y = layers.MaxPool2D(2)(y)
        return y
    
    x = conv_mfm_block(x, 64, 5, pool=True)
    x = conv_mfm_block(x, 64, 1)
    x = conv_mfm_block(x, 96, 3, pool=True)
    x = conv_mfm_block(x, 128, 3, pool=True)
    x = conv_mfm_block(x, 128, 1)
    x = conv_mfm_block(x, 64, 3)
    x = conv_mfm_block(x, 64, 1)
    x = conv_mfm_block(x, 64, 3, pool=True)

    x = layers.GlobalMaxPool2D()(x)
    x = layers.Dense(128*2)(x)
    x = mfm(x)
    emb = layers.BatchNormalization()(x)

    logits = layers.Dense(num_classes)(emb)
    out = layers.Activation('softmax')(logits)

    return models.Model(inp, [out, emb], name='LCNN_TF')