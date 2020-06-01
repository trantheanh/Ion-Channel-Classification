import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from metrics.core import BinaryAccuracy, BinaryMCC, BinarySensitivity, BinarySpecificity, BinaryF1Score
from constant.index import DataIdx, MetricIdx
from data import loader
from callbacks.core import build_callbacks
from process.core import evaluate_on_threshold

import numpy as np
import os
from saved_model import SAVED_MODEL_PATH
from resource import RESOURCE_PATH
from constant.url import DataPath
import pandas as pd
from constant.shape import InputShape

from data.dictionary import EmbDict
from data.loader import read_emb_data, read_pssm_data

from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf(raw_data):
    vectorizer = TfidfVectorizer()


def get_metrics(threshold=0.1) -> list:

    return [
        BinaryF1Score(threshold),
        BinaryAccuracy(threshold),
        BinaryMCC(threshold),
        BinarySensitivity(threshold),
        BinarySpecificity(threshold)
    ]


def parse_data(emb_data, pssm_data, label):
    emb_size = (InputShape.EMB_LENGTH, InputShape.EMB_DIM)
    pssm_size = (InputShape.PSSM_LENGTH, InputShape.PSSM_DIM)

    emb_input = np.reshape(emb_data, newshape=(-1, InputShape.EMB_LENGTH, InputShape.EMB_DIM))

    print(pssm_data.shape)
    pssm_input = np.stack(
        [
            pssm_data[:, (i * InputShape.PSSM_DIM):((i + 1) * InputShape.PSSM_DIM)]
            for i in range(InputShape.PSSM_LENGTH)
        ],
        axis=1
    )

    label = label
    return emb_input, pssm_input, label


def build_model() -> keras.models.Model:
    keras.backend.clear_session()
    emb_input = layers.Input(shape=(InputShape.EMB_LENGTH, InputShape.EMB_DIM))
    pssm_input = layers.Input(shape=(InputShape.PSSM_LENGTH, InputShape.PSSM_DIM))

    pssm_imd = pssm_input

    pssm_imd = layers.Conv1D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='SAME',
        activation="relu"
    )(pssm_imd)

    pssm_imd = layers.GRU(
        units=512,
        return_sequences=True,
        dropout=0.1
    )(pssm_imd)

    pssm_imd = layers.GRU(
        units=512,
        return_sequences=False,
        dropout=0.1
    )(pssm_imd)

    emb_imd = emb_input

    emb_imd = layers.Conv1D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='SAME',
        activation="relu"
    )(emb_imd)

    emb_imd = layers.GRU(
        units=512,
        return_sequences=True,
        dropout=0.1
    )(emb_imd)

    emb_imd = layers.GRU(
        units=512,
        return_sequences=False,
        dropout=0.1
    )(emb_imd)

    imd = layers.Concatenate(axis=-1)([emb_imd, pssm_imd])
    imd = layers.Dropout(rate=0.2)(imd)
    imd = layers.Dense(units=512, activation="relu")(imd)

    output_tf = layers.Dense(
      units=1,
      activation=tf.keras.activations.sigmoid
    )(imd)

    model = tf.keras.models.Model(inputs=[emb_input, pssm_input], outputs=output_tf)

    model.compile(
      optimizer=keras.optimizers.Adam(
          learning_rate=0.00016280409164167792,
          # decay=1e-6
      ),
      loss=keras.losses.binary_crossentropy,
      metrics=get_metrics(threshold=0.5)
    )

    return model


def train(train_ds, test_ds):
    n_epoch = 100

    callbacks = build_callbacks(log_dir="my_exp")

    model = build_model()
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=n_epoch,
        verbose=2,
        shuffle=True,
        class_weight={
              0: 1,
              1: 20
        },
        callbacks=callbacks
    )

    train_result = evaluate_on_threshold(model, train_ds, threshold=0.1)
    test_result = evaluate_on_threshold(model, test_ds, threshold=0.1)
    return train_result, test_result


def build_train_ds(emb_input, pssm_input, label):
    ds = tf.data.Dataset.from_tensor_slices(((emb_input, pssm_input), label))
    ds = ds.shuffle(10000).batch(32)
    return ds


def build_test_ds(emb_input, pssm_input, label):
    ds = tf.data.Dataset.from_tensor_slices(((emb_input, pssm_input), label))
    ds = ds.batch(32)
    return ds


train_emb, test_emb, train_emb_label, test_emb_label = read_emb_data()
train_pssm, test_pssm, train_pssm_label, test_pssm_label = read_pssm_data()

train_emb_input, train_pssm_input, train_label = parse_data(train_emb, train_pssm, train_pssm_label)
test_emb_input, test_pssm_input, test_label = parse_data(test_emb, test_pssm, test_pssm_label)

print(train_emb_input.shape, train_pssm_input.shape, train_label.shape)
print(test_emb_input.shape, test_pssm_input.shape, test_label.shape)

train_ds = build_train_ds(train_emb_input, train_pssm_input, train_label)
test_ds = build_test_ds(test_emb_input, test_pssm_input, test_label)
# print(train_emb_input.shape)
# print(train_pssm_input.shape)
# print(train_label.shape)

train(train_ds, test_ds)