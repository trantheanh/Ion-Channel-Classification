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
import shutil

from data.dictionary import EmbDict
from data.loader import read_emb_data, read_pssm_data, read_from_emb

from sklearn.feature_extraction.text import TfidfVectorizer
import fasttext

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


def get_tfidf(raw_data):
    vectorizer = TfidfVectorizer()


def train_sup_emb():
    model = fasttext.train_supervised(
        os.path.join(RESOURCE_PATH, "29052020", "raw.train"),
        epoch=25,
        dim=InputShape.EMB_DIM,
        # autotuneValidationFile=os.path.join(RESOURCE_PATH, "29052020", "raw.ind.test")
    )
    model.save_model(os.path.join(RESOURCE_PATH, "29052020", "emb.bin"))
    InputShape.EMB_DIM = model.get_dimension()
    return model


def load_emb() -> fasttext.FastText:
    model = fasttext.load_model(os.path.join(RESOURCE_PATH, "29052020", "emb.bin"))
    return model


def get_metrics(threshold=0.1) -> list:

    return [
        BinaryF1Score(threshold),
        BinaryAccuracy(threshold),
        BinaryMCC(threshold),
        BinarySensitivity(threshold),
        BinarySpecificity(threshold)
    ]


def parse_data(emb_data, pssm_data, tfidf_data, label):
    emb_size = (InputShape.EMB_LENGTH, InputShape.EMB_DIM)
    pssm_size = (InputShape.PSSM_LENGTH, InputShape.PSSM_DIM)

    emb_input = np.reshape(emb_data, newshape=(-1, InputShape.EMB_LENGTH, InputShape.EMB_DIM))

    pssm_input = np.stack(
        [
            pssm_data[:, (i * InputShape.PSSM_DIM):((i + 1) * InputShape.PSSM_DIM)]
            for i in range(InputShape.PSSM_LENGTH)
        ],
        axis=1
    )

    tfidf_input = tfidf_data

    label = label
    return emb_input, pssm_input, tfidf_input, label


def build_model() -> keras.models.Model:
    keras.backend.clear_session()
    emb_input = layers.Input(shape=(InputShape.EMB_LENGTH, InputShape.EMB_DIM))
    pssm_input = layers.Input(shape=(InputShape.PSSM_LENGTH, InputShape.PSSM_DIM))
    tfidf_input = layers.Input(shape=(InputShape.TFIDF_DIM,))

    pssm_imd = pssm_input

    # pssm_imd = layers.Conv1D(
    #     filters=32,
    #     kernel_size=3,
    #     strides=1,
    #     padding='SAME',
    #     activation="relu"
    # )(pssm_imd)
    # pssm_imd = layers.MaxPool1D()(pssm_imd)

    # pssm_imd = layers.GRU(
    #     units=512,
    #     return_sequences=True,
    #     dropout=0.1
    # )(pssm_imd)

    pssm_imd = layers.GRU(
        units=512,
        return_sequences=False,
        dropout=0.1
    )(pssm_imd)

    emb_imd = emb_input

    # emb_imd = layers.Conv1D(
    #     filters=32,
    #     kernel_size=3,
    #     strides=1,
    #     padding='SAME',
    #     activation="relu"
    # )(emb_imd)
    # emb_imd = layers.MaxPool1D()(emb_imd)

    emb_imd = layers.GlobalAveragePooling1D()(emb_imd)

    # emb_imd = layers.GRU(
    #     units=32,
    #     return_sequences=False,
    #     dropout=0.1
    # )(emb_imd)

    # emb_imd = layers.GRU(
    #     units=512,
    #     return_sequences=False,
    #     dropout=0.1
    # )(emb_imd)

    tfidf_imd = tfidf_input
    tfidf_imd = layers.Dropout(rate=0.1)(tfidf_imd)
    tfidf_imd = layers.Dense(units=128, activation="relu")(tfidf_imd)

    # imd = tfidf_imd
    imd = layers.Concatenate(axis=-1)([emb_imd, pssm_imd, tfidf_imd])
    imd = layers.Dropout(rate=0.3)(imd)
    # imd = layers.Dense(units=512, activation="relu")(imd)

    output_tf = layers.Dense(
      units=1,
      activation=tf.keras.activations.sigmoid
    )(imd)

    model = tf.keras.models.Model(inputs=[emb_input, pssm_input, tfidf_input], outputs=output_tf)
    print(model.summary())

    model.compile(
      optimizer=keras.optimizers.Nadam(
          learning_rate=0.00016280409164167792,
          # decay=1e-6
      ),
      loss=keras.losses.binary_crossentropy,
      metrics=get_metrics(threshold=0.1)
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
              0: 9,
              1: 91
        },
        callbacks=callbacks
    )

    train_result = evaluate_on_threshold(model, train_ds, threshold=0.1)
    test_result = evaluate_on_threshold(model, test_ds, threshold=0.1)
    print(train_result)
    print(test_result)
    return train_result, test_result


def build_train_ds(emb_input, pssm_input, tfidf_input, label):
    ds = tf.data.Dataset.from_tensor_slices(((emb_input, pssm_input, tfidf_input), label))
    ds = ds.shuffle(10000).batch(1)
    return ds


def build_test_ds(emb_input, pssm_input, tfidf_input, label):
    ds = tf.data.Dataset.from_tensor_slices(((emb_input, pssm_input, tfidf_input), label))
    ds = ds.batch(32)
    return ds


model_emb = train_sup_emb()
InputShape.EMB_DIM = model_emb.get_dimension()
print(model_emb.get_dimension())
train_emb, test_emb, train_tfidf, test_tfidf = read_from_emb(model_emb)
train_pssm, test_pssm, train_pssm_label, test_pssm_label = read_pssm_data()

train_emb_input, train_pssm_input, train_tfidf_input, train_label = parse_data(
    emb_data=train_emb,
    pssm_data=train_pssm,
    tfidf_data=train_tfidf,
    label=train_pssm_label
)
test_emb_input, test_pssm_input, test_tfidf_input, test_label = parse_data(
    emb_data=test_emb,
    pssm_data=test_pssm,
    tfidf_data=test_tfidf,
    label=test_pssm_label
)

train_ds = build_train_ds(train_emb_input, train_pssm_input, train_tfidf_input, train_label)
test_ds = build_test_ds(test_emb_input, test_pssm_input, test_tfidf_input, test_label)

# os.system("rm -r my_exp/*")
# shutil.rmtree("D:\workspace\anhtt\Ion-Channel-Classification\my_exp")
train(train_ds, test_ds)

# train_sup_emb()