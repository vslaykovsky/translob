import sys

import keras
import numpy as np
from keras import Sequential
from keras import backend as K
from keras import layers

from python.LobTransformer import TransformerBlock, LayerNormalization


class PositionalEncodingLayer(keras.layers.Layer):
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


def main(argv):
    """
    # train hyperparameters
    Batch size 32
    Adam β 1 0.9
    Adam β 2 0.999
    Learning rate 1 * 10 ^ (-4)
    """
    model = Sequential()
    model.add(layers.Conv1D(14, kernel_size=2, strides=1, activation='relu', padding='causal'))
    model.add(layers.Conv1D(14, kernel_size=2, dilation_rate=2, activation='relu', padding='causal'))
    model.add(layers.Conv1D(14, kernel_size=2, dilation_rate=4, activation='relu', padding='causal'))
    model.add(layers.Conv1D(14, kernel_size=2, dilation_rate=8, activation='relu', padding='causal'))
    model.add(layers.Conv1D(14, kernel_size=2, dilation_rate=16, activation='relu', padding='causal'))
    model.add(layers.LayerNormalization())
    model.add(PositionalEncodingLayer())
    lt = TransformerBlock('tb1', 3, True)
    model.add(lt)
    model.add(lt)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_regularizer='l2'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(3, activation='softmax'))
    model.build(input_shape=(1, 100, 40))
    model.summary()


if __name__ == '__main__':
    main(sys.argv)
