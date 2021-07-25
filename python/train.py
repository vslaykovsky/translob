import argparse
import math
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras import backend as K
from keras import layers
from keras.layers import Layer
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import classification_report

from LobTransformer import TransformerBlock


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


def translob_model(**kwargs):
    inputs = Input(shape=(kwargs.get('sequence_length', 100), 40))
    x = inputs
    max_conv_filters = kwargs.get('num_conv_filters', 14)
    max_conv_dilation = kwargs.get('max_conv_dilation', 16)
    for dilation in [2 ** v for v in list(range(math.ceil(math.log2(max_conv_dilation)) + 1))]:
        x = layers.Conv1D(
            max_conv_filters, kernel_size=2, dilation_rate=dilation, activation='relu', padding='causal'
        )(x)
    x = layers.LayerNormalization()(x)
    x = PositionalEncodingLayer()(x)
    tb = TransformerBlock('tb1', kwargs.get('num_attention_heads', 3), True)
    blocks = kwargs.get('num_transformer_blocks', 2)
    for block in range(blocks):
        if kwargs.get('transformer_blocks_share_weights', True):
            x = tb(x)
        else:
            x = TransformerBlock(f'transformer_block_{block}', kwargs.get('num_attention_heads', 3), True)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64,
                     activation='relu',
                     kernel_regularizer='l2',
                     kernel_initializer='glorot_uniform')(x)
    x = layers.Dropout(kwargs.get('dropout_rate', 0.1))(x)
    out = layers.Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=out)
    model.summary()

    model.compile(
        tf.keras.optimizers.Adam(
            learning_rate=kwargs.get('lr', 0.0001),
            beta_1=kwargs.get('adam_beta1', 0.9),
            beta_2=kwargs.get('adam_beta2', 0.999),
            name="Adam",
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['sparse_categorical_accuracy'],
    )
    return model


def train_translob(X_train, y_train, X_val, y_val, **kwargs):
    print('Train', X_train.shape, y_train.shape, 'Val', X_val.shape, y_val.shape)
    model = translob_model(**kwargs)

    length = kwargs.get('sequence_length', 100)
    train_gen = TimeseriesGenerator(X_train, y_train, length, shuffle=True, batch_size=kwargs.get('batch_size', 32))
    val_gen = TimeseriesGenerator(X_val, y_val, length, batch_size=kwargs.get('batch_size', 32))

    model.fit(
        train_gen,
        epochs=kwargs.get('epochs', 100),
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=(
                    "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', patience=10,
                                             min_delta=0.0002),
            #             ModelCheckpoint('mdl.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        ],
        validation_data=val_gen
    )
    return model


def gen_data(data, horizon):
    x = data[:40, :].T  # 40 == 10 price + volume asks + 10 price + volume bids
    y = data[-5 + horizon, :].T  # 5
    return x[:-1], (y[1:] - 1).astype(np.int32)  # shift y by 1


def load_dataset(dir, horizon):
    dec_data = np.loadtxt(f'{dir}/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt')
    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

    dec_test1 = np.loadtxt(f'{dir}/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt')
    dec_test2 = np.loadtxt(f'{dir}/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt')
    dec_test3 = np.loadtxt(f'{dir}/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt')
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

    return gen_data(dec_train, horizon), gen_data(dec_val, horizon), gen_data(dec_test, horizon)


def eval(model, X_test, y_test, **kwargs):
    ts = TimeseriesGenerator(X_test, y_test, kwargs.get('sequence_length', 100), batch_size=32, shuffle=False)
    y_true = np.concatenate([y for x, y in ts])
    y_pred = np.argmax(model.predict(ts), -1)
    print(classification_report(y_true, y_pred))
    return classification_report(y_true, y_pred, output_dict=True)['weighted avg']['f1-score']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransLOB model training/evaluation')
    parser.add_argument('--fi2010',
                        help='FI2010 dataset dir. Get it from https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649')
    parser.add_argument('--horizon', type=int, help='horizon to predict [0..4]', default=4)
    args = parser.parse_args(sys.argv[1:])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(args.fi2010, horizon=args.horizon)

    params = {
        # inputs
        'sequence_length': 100,
        # model
        'num_conv_filters': 14,
        'max_conv_dilation': 16,
        'num_attention_heads': 3,
        'num_transformer_blocks': 2,
        'transformer_blocks_share_weights': True,
        'dropout_rate': 0.1,
        # training
        'lr': 0.0001,
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'batch_size': 32,
        'epochs': 150
    }

    model = train_translob(X_train, y_train, X_val, y_val, **params)
    print('Validation performance')
    eval(model, X_val, y_val, **params)
    print('Test performance')
    eval(model, X_test, y_test, **params)
