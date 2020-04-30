import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tensorboard.plugins.hparams import api as hp
import os

from models.core import build_model
from data.dataset import build_train_ds, build_test_ds
from constant.index import DataIdx, MetricIdx
from util.log import log_result


"""# Build single training process"""


def train(config, train_ds, test_ds, need_summary=False):
    hparams = config["hparams"]
    n_epoch = hparams["n_epoch"]
    verbose = config["verbose"]

    callbacks = []
    validation_ds = None

    if need_summary:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=config["log_dir"]))
        validation_ds = test_ds

    model = build_model(hparams)
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=n_epoch,
        verbose=verbose,
        shuffle=True,
        callbacks=callbacks
    )

    train_result = model.evaluate(train_ds, verbose=verbose)
    test_result = model.evaluate(test_ds, verbose=verbose)
    return train_result, test_result


"""# Build k-fold training process"""


def k_fold_experiment(config, train_data, test_data):
    hparams = config["hparams"]
    n_fold = config["n_fold"]

    # Train on K-fold
    folds = StratifiedKFold(
        n_splits=n_fold,
        shuffle=True,
        random_state=0
    ).split(train_data[DataIdx.MLP_FEATURE], train_data[DataIdx.LABEL])

    train_results = []
    dev_results = []
    for fold_index, fold in enumerate(folds):
        print("FOLD: {}".format(fold_index + 1))
        train_indexes, dev_indexes = fold

        train_ds = build_train_ds(
            mlp_x=train_data[DataIdx.MLP_FEATURE][train_indexes],
            rnn_x=train_data[DataIdx.RNN_FEATURE][train_indexes],
            y=train_data[DataIdx.LABEL][train_indexes],
            hparams=hparams
        )

        dev_ds = build_test_ds(
            mlp_x=train_data[DataIdx.MLP_FEATURE][dev_indexes],
            rnn_x=train_data[DataIdx.RNN_FEATURE][dev_indexes],
            y=train_data[DataIdx.LABEL][dev_indexes]
        )

        train_result, dev_result = train(
            config=config,
            train_ds=train_ds,
            test_ds=dev_ds
        )

        train_results.append(train_result)
        dev_results.append(dev_result)

    train_result = np.mean(np.array(train_results), axis=0)
    dev_result = np.mean(np.array(dev_results), axis=0)

    # Train on all & evaluate on test set
    train_ds = build_train_ds(
        mlp_x=train_data[DataIdx.MLP_FEATURE],
        rnn_x=train_data[DataIdx.RNN_FEATURE],
        y=train_data[DataIdx.LABEL],
        hparams=hparams,
    )

    test_ds = build_test_ds(
        mlp_x=test_data[DataIdx.MLP_FEATURE],
        rnn_x=test_data[DataIdx.RNN_FEATURE],
        y=test_data[DataIdx.LABEL]
    )

    _, test_result = train(config=config, train_ds=train_ds, test_ds=test_ds, need_summary=True)

    return train_result, dev_result, test_result


"""# Build Hyperparameter experiment process"""


def experiment(configs, train_data, test_data, log_dir):
    for index, config in enumerate(configs):
        session_log_dir = os.path.join(log_dir, "session_{}".format(index+1))
        print(session_log_dir)
        with tf.summary.create_file_writer(session_log_dir).as_default():
            print("\nEXPERIMENT: {}/{}".format(index+1, len(configs)))
            config["log_dir"] = session_log_dir
            config["n_fold"] = 5
            config["verbose"] = 2
            hp.hparams(config["hparams"])
            train_result, dev_result, test_result = k_fold_experiment(
                config=config,
                train_data=train_data,
                test_data=test_data
            )

            log_result(train_result, dev_result, test_result)

            tf.summary.scalar(name="train_acc", data=train_result[MetricIdx.ACC], step=1)
            tf.summary.scalar(name="train_mcc", data=train_result[MetricIdx.MCC], step=1)
            tf.summary.scalar(name="train_sen", data=train_result[MetricIdx.SEN], step=1)
            tf.summary.scalar(name="train_spec", data=train_result[MetricIdx.SPEC], step=1)

            tf.summary.scalar(name="dev_acc", data=dev_result[MetricIdx.ACC], step=1)
            tf.summary.scalar(name="dev_mcc", data=dev_result[MetricIdx.MCC], step=1)
            tf.summary.scalar(name="dev_sen", data=dev_result[MetricIdx.SEN], step=1)
            tf.summary.scalar(name="dev_spec", data=dev_result[MetricIdx.SPEC], step=1)

            tf.summary.scalar(name="test_acc", data=test_result[MetricIdx.ACC], step=1)
            tf.summary.scalar(name="test_mcc", data=test_result[MetricIdx.MCC], step=1)
            tf.summary.scalar(name="test_sen", data=test_result[MetricIdx.SEN], step=1)
            tf.summary.scalar(name="test_spec", data=test_result[MetricIdx.SPEC], step=1)