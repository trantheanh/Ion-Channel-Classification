# -*- coding: utf-8 -*-
import numpy as np

import os
import shutil

from data.loader import download_from_shareable_link, parse_csv_data
from constant.index import DataIdx
from process.core import experiment

"""# MAIN FUNCTION"""


def main():
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

    if os.path.isdir(log_dir):
      shutil.rmtree(log_dir)

    n_experiments = 1000

    batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    learning_rate = (0.0001, 5)
    n_epoch = (20, 200)
    optimizer = ["adam", "rmsprop", "sgd", "adamax", "adadelta", "nadam"]
    maxout_head = [1,2,3,4]
    maxout_units = [32, 64, 128, 256, 512, 1024]
    rnn_layers = [1,2,3]
    rnn_units = [32, 64, 128, 256, 512, 1024]

    hparams = {"threshold": np.random.random(size=(n_experiments,)), "batch_size": np.array(batch_size)[
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
               "rnn_units": np.array(rnn_units)[np.random.randint(low=0, high=len(rnn_units), size=(n_experiments,))]}

    configs = [{"hparams": {
      "threshold": hparams["threshold"][i],
      "batch_size": int(hparams["batch_size"][i]),
      "learning_rate": hparams["learning_rate"][i],
      "n_epoch": 3,
      # "n_epoch": int(hparams["n_epoch"][i]),
      "optimizer": hparams["optimizer"][i],
      "maxout_head": int(hparams["maxout_head"][i]),
      "maxout_units": int(hparams["maxout_units"][i]),
      "rnn_layers": int(hparams["rnn_layers"][i]),
      "rnn_units": int(hparams["rnn_units"][i]),
    }} for i in range(n_experiments)]

    if True:
        experiment(
          configs=configs,
          train_data=train_data,
          test_data=test_data,
          log_dir=log_dir
        )
    else:
        config1 = {
            "hparams": {
                "threshold": 0.5,
                "batch_size": 32,
                "learning_rate": 0.001,
                "n_epoch": 1,
                "optimizer": "adam",
                "maxout_head": 1,
                "maxout_units": 32,
                "rnn_layers": 2,
                "rnn_units": 32,
              }
        }

        config2 = {
            "hparams": {
                "threshold": 0.5,
                "batch_size": 32,
                "learning_rate": 0.001,
                "n_epoch": 1,
                "optimizer": "adam",
                "maxout_head": 0,
                "maxout_units": 32,
                "rnn_layers": 1,
                "rnn_units": 62,
              }
        }

    experiment(
        configs=[config1, config2],
        train_data=train_data,
        test_data=test_data,
        log_dir=log_dir
    )


if __name__ == "__main__":
    main()

