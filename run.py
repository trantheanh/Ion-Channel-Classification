# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

import os

from data.loader import download_from_shareable_link, parse_csv_data
from constant.index import DataIdx
from process.core import experiment
from absl import flags, app

"""# MAIN FUNCTION"""

FLAGS = flags.FLAGS
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

flags.DEFINE_enum("is_tuning", "Y", ["Y", "N"], "running for auto hyper param tuning or specific setting")

flags.DEFINE_enum("optimizer", "nadam", ["adam", "rmsprop", "sgd", "adamax", "adadelta", "nadam"], "Name of optimizer")
flags.DEFINE_integer("batch_size", 1, "Batch size of traning data")
# flags.DEFINE_integer("batch_size", 1024, "Batch size of traning data")
flags.DEFINE_float("learning_rate", 0.00016280409164167792, "learning rate of optimizer")
flags.DEFINE_integer("n_epoch", 100, "Number of training epoch")
# flags.DEFINE_integer("n_epoch", 1, "Number of training epoch")
flags.DEFINE_integer("maxout_head", 2, "Number of maxout activation head")
flags.DEFINE_integer("maxout_units", 128, "Number of maxout units")
# flags.DEFINE_integer("maxout_units", 1, "Number of maxout units")
flags.DEFINE_integer("rnn_layers", 1, "Number of LSTM layer")
flags.DEFINE_integer("rnn_units", 1024, "Number of RNN units")
# flags.DEFINE_integer("rnn_units", 1, "Number of RNN units")
flags.DEFINE_float("dropout", 0.1, "Dropout Rate")
flags.DEFINE_integer("conv1d_depth", 32, "Number of Conv1D filter")
flags.DEFINE_integer("conv1d_size", 3, "Window size of Conv1D")
flags.DEFINE_integer("conv1d_stride", 1, "Stride of Conv1D")


def main(argv):
    train_data_path = "https://drive.google.com/open?id=1EJ0VcOmKUUOpSQ4dDkhbhVzvbIjz-Wg5"
    test_data_path = "https://drive.google.com/open?id=14Xo1zxFKHrus1KPxmVJe-D0S8Px9OIlX"
    download_from_shareable_link(url=train_data_path, destination="train_data.csv")
    download_from_shareable_link(url=test_data_path, destination="test_data.csv")

    train_data = parse_csv_data(path="train_data.csv")
    test_data = parse_csv_data(path="test_data.csv")

    print("Train Data Info:\n  MLP Feature: {}\n  RNN Feature: {}\n  Label: {}".format(
      train_data[DataIdx.MLP_FEATURE].shape,
      train_data[DataIdx.RNN_FEATURE].shape,
      train_data[DataIdx.LABEL].shape
    ))

    print("Test Data Info:\n  MLP Feature: {}\n  RNN Feature: {}\n  Label: {}".format(
      test_data[DataIdx.MLP_FEATURE].shape,
      test_data[DataIdx.RNN_FEATURE].shape,
      test_data[DataIdx.LABEL].shape
    ))

    log_dir = os.path.join(os.getcwd(),"log", "hparam_tuning")

    n_experiments = 1000

    batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    #batch_size = [1, 2, 4, 8, 16, 32]
    learning_rate = (0.0001, 0.5)
    n_epoch = (20, 100)
    optimizer = ["adam", "rmsprop", "sgd", "adamax", "adadelta", "nadam"]
    maxout_head = [1, 2, 3, 4]
    maxout_units = [32, 64, 128, 256, 512, 1024]
    rnn_layers = [1, 2, 3]
    rnn_units = [32, 64, 128, 256, 512]
    # rnn_units = [32, 64, 128, 256, 512, 1024]
    lr_decay = 1e-6
    conv1d_depth = [8, 16, 32, 64]
    conv1d_size = [2, 3, 4]
    conv1d_stride = [1, 2, 3]
    dropout = (0, 0.3)

    hparams = {
        "threshold": np.random.random(size=(n_experiments,)),
        "batch_size": np.array(batch_size)[
        np.random.randint(low=0, high=len(batch_size), size=(n_experiments,))],
        "learning_rate": np.logspace(np.log10(learning_rate[0]), np.log10(learning_rate[1]), base=10,
                                    num=n_experiments),
        "n_epoch": np.random.randint(low=n_epoch[0], high=n_epoch[1] + 1, size=(n_experiments,)),
        "optimizer": np.array(optimizer)[np.random.randint(low=0, high=len(optimizer), size=(n_experiments,))],
        "maxout_head": np.array(maxout_head)[
           np.random.randint(low=0, high=len(maxout_head), size=(n_experiments,))],
        "maxout_units": np.array(maxout_units)[
           np.random.randint(low=0, high=len(maxout_units), size=(n_experiments,))],
        "rnn_layers": np.array(rnn_layers)[
           np.random.randint(low=0, high=len(rnn_layers), size=(n_experiments,))],
        "rnn_units": np.array(rnn_units)[np.random.randint(low=0, high=len(rnn_units), size=(n_experiments,))],
        "dropout": np.random.random(size=(n_experiments,)) * (dropout[1] - dropout[0]),
        "conv1d_depth": np.array(conv1d_depth)[np.random.randint(low=0, high=len(conv1d_depth), size=(n_experiments,))],
        "conv1d_size": np.array(conv1d_size)[np.random.randint(low=0, high=len(conv1d_size), size=(n_experiments,))],
        "conv1d_stride": np.array(conv1d_stride)[np.random.randint(low=0, high=len(conv1d_stride), size=(n_experiments,))]
    }

    configs = [{"hparams": {
        "threshold": 0.5,
        "batch_size": int(hparams["batch_size"][i]),
        "learning_rate": hparams["learning_rate"][i],
        "n_epoch": 1#int(hparams["n_epoch"][i]),
        "optimizer": hparams["optimizer"][i],
        "maxout_head": int(hparams["maxout_head"][i]),
        "maxout_units": int(hparams["maxout_units"][i]),
        "rnn_layers": int(hparams["rnn_layers"][i]),
        "rnn_units": int(hparams["rnn_units"][i]),
        "lr_decay": lr_decay,
        "conv1d_depth": int(hparams["conv1d_depth"][i]),
        "dropout": hparams["dropout"][i],
        "conv1d_size": int(hparams["conv1d_size"][i]),
        "conv1d_stride": int(hparams["conv1d_stride"][i]),
    }} for i in range(n_experiments)]

    # configs = [{"hparams": {
    #     "threshold": 0.5,
    #     "batch_size": FLAGS.batch_size,
    #     "learning_rate": FLAGS.learning_rate,
    #     "n_epoch": FLAGS.n_epoch,
    #     "optimizer": FLAGS.optimizer,
    #     "maxout_head": FLAGS.maxout_head,
    #     "maxout_units": FLAGS.maxout_units,
    #     "rnn_layers": FLAGS.rnn_layers,
    #     "rnn_units": FLAGS.rnn_units,
    #     "lr_decay": lr_decay,
    #     "conv1d_depth": FLAGS.conv1d_depth,
    #     "dropout": FLAGS.dropout,
    #     "conv1d_size": FLAGS.conv1d_size,
    #     "conv1d_stride": FLAGS.conv1d_stride
    # }} for i in range(n_experiments)]

    if FLAGS.is_tuning == "Y":
        experiment(
          configs=configs,
          log_dir=log_dir
        )
    else:
        config = {
            "hparams": {
                "threshold": 0.5,
                "batch_size": FLAGS.batch_size,
                "learning_rate": FLAGS.learning_rate,
                "n_epoch": FLAGS.n_epoch,
                "optimizer": FLAGS.optimizer,
                "maxout_head": FLAGS.maxout_head,
                "maxout_units": FLAGS.maxout_units,
                "rnn_layers": FLAGS.rnn_layers,
                "rnn_units": FLAGS.rnn_units,
                "lr_decay": lr_decay,
                "conv1d_depth": FLAGS.conv1d_depth,
                "dropout": FLAGS.dropout,
                "conv1d_size": FLAGS.conv1d_size,
                "conv1d_stride": FLAGS.conv1d_stride,
              }
        }

        experiment(
            configs=[config],
            log_dir=log_dir
        )


if __name__ == "__main__":
    app.run(main)

