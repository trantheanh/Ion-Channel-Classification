import os
import numpy as np
from process.core import experiment
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

batch_size = [1, 2, 4, 8, 16, 32]
lr = (0.00001, 0.5)
n_epoch = (50, 150)
optimizer = ["adam", "rmsprop", "sgd", "adamax", "adadelta", "nadam"]

PSSM_GRU = [256, 512, 1024]
PSSM_Dropout = (0.01, 0.5)

EMB_Conv1D = [16, 32, 64]
EMB_LSTM = [256, 512, 1024]
EMB_Dropout = (0.01, 0.5)

TFIDF_Dropout = (0.01, 0.5)
TFIDF_units = [64, 128, 256, 512]

dropout = (0.01, 0.5)
units = [256, 512, 1024]

decay = 0

n_experiments = 1000


def random_enum(_number):
    return np.array(_number)[
       np.random.randint(low=0, high=len(_number), size=(n_experiments,))]


def random_dropout(_number):
    return np.random.random(size=(n_experiments,)) * (_number[1] - _number[0])


def random_number(_number):
    return np.random.randint(low=_number[0], high=_number[1] + 1, size=(n_experiments,))


def random_log10(_number):
    return np.logspace(
        np.log10(_number[0]),
        np.log10(_number[1]),
        base=10,
        num=n_experiments
    )


hparams = {
    "threshold": np.random.random(size=(n_experiments,)),
    "batch_size": random_enum(batch_size),
    "lr": random_log10(lr),
    "n_epoch": random_number(n_epoch),
    "optimizer": random_enum(optimizer),
    "PSSM_GRU": random_enum(PSSM_GRU),
    "PSSM_Dropout": random_dropout(PSSM_Dropout),
    "EMB_Conv1D": random_enum(EMB_Conv1D),
    "EMB_LSTM": random_enum(EMB_LSTM),
    "EMB_Dropout": random_dropout(EMB_Dropout),
    "TFIDF_Dropout": random_dropout(TFIDF_Dropout),
    "TFIDF_units": random_enum(TFIDF_units),
    "dropout": random_dropout(dropout),
    "units": random_enum(units),
}

configs = [
    {"hparams": {
        "threshold": hparams["threshold"][i],
        "batch_size": int(hparams["batch_size"][i]),
        "lr": hparams["lr"][i],
        "n_epoch": int(hparams["n_epoch"][i]),
        "optimizer": hparams["optimizer"][i],
        "PSSM_GRU": int(hparams["PSSM_GRU"][i]),
        "PSSM_Dropout": hparams["PSSM_Dropout"][i],
        "EMB_Conv1D": int(hparams["EMB_Conv1D"][i]),
        "EMB_LSTM": int(hparams["EMB_LSTM"][i]),
        "EMB_Dropout": hparams["EMB_Dropout"][i],
        "TFIDF_Dropout": hparams["TFIDF_Dropout"][i],
        "TFIDF_units": int(hparams["TFIDF_units"][i]),
        "dropout": hparams["dropout"][i],
        "units": int(hparams["PSSM_GRU"][i]),
        "decay": decay,
    }}
    for i in range(n_experiments)]

for config in configs:
    experiment(
        configs=[config],
        log_dir=os.path.join(os.getcwd(), "log", "hparam_tuning")
    )
