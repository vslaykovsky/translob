import numpy as np
import tensorflow as tf

from keras import Input, Model
from keras import backend as K
from keras import layers
from keras.layers import Layer

from python.LobTransformer import TransformerBlock


class PositionalEncodingLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, *args, **kwargs):
        steps, d_model = x.get_shape().as_list()[-2:]
        ps = np.zeros([steps, 1], dtype=K.floatx())
        for tx in range(steps):
            ps[tx, :] = [(2 / (steps - 1)) * tx - 1]

        ps_expand = K.expand_dims(K.constant(ps), axis=0)
        ps_tiled = K.tile(ps_expand, [K.shape(x)[0], 1, 1])

        x = K.concatenate([x, ps_tiled], axis=-1)
        return x


def explicit_model():
    inputs = Input(shape=(100, 40))
    x = layers.Conv1D(14, kernel_size=2, strides=1, activation='relu', padding='causal')(inputs)
    x = layers.Conv1D(14, kernel_size=2, dilation_rate=2, activation='relu', padding='causal')(x)
    x = layers.Conv1D(14, kernel_size=2, dilation_rate=4, activation='relu', padding='causal')(x)
    x = layers.Conv1D(14, kernel_size=2, dilation_rate=8, activation='relu', padding='causal')(x)
    x = layers.Conv1D(14, kernel_size=2, dilation_rate=16, activation='relu', padding='causal')(x)
    x = layers.LayerNormalization()(x)
    x = PositionalEncodingLayer()(x)
    lt = TransformerBlock('tb1', 3, True)
    blocks = 2
    for block in range(blocks):
        x = lt(x)
    x = (layers.Flatten())(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer='l2')(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=out)
    model.build(input_shape=(1, 100, 40))
    model.summary()

    X = tf.random.normal((100000, 100, 40))
    y = tf.random.uniform([100000], minval=0, maxval=2, dtype=tf.dtypes.int32)
    model.compile(tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        name="Adam",
    ), loss=tf.keras.losses.SparseCategoricalCrossentropy())
    model.fit(X, y, batch_size=32, epochs=2)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    explicit_model()
