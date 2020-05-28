import numpy as np
import os
import datetime

from models import build_model
from data.dataset import build_train_ds, build_test_ds
from constant.index import DataIdx, MetricIdx
from util.log import log_result, write
from callbacks.core import build_callbacks
from metrics.core import BinaryAccuracy, BinaryMCC, BinarySensitivity, BinarySpecificity, BinaryF1Score
from data.loader import get_fold, get_data, preprocess_data
from saved_model import SAVED_MODEL_PATH


"""# Build single training process"""


def train(config, train_ds, test_ds, need_summary=False, need_threshold=True, need_save=False, is_final=False):
    hparams = config["hparams"]
    n_epoch = hparams["n_epoch"]
    class_weight = config["class_weight"]
    verbose = config["verbose"]

    callbacks = []

    if need_summary:
        callbacks = build_callbacks(log_dir=config["log_dir"])

    model = build_model(hparams)
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=n_epoch,
        verbose=verbose,
        class_weight=class_weight,
        shuffle=True,
        callbacks=callbacks
    )

    if need_save:
        if is_final:
            model.save(os.path.join(SAVED_MODEL_PATH, "{}_{}_final_model.h5".format(
                config["log_dir"].split("/")[-1],
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )))
        else:
            model.save(os.path.join(SAVED_MODEL_PATH, "{}.h5".format(
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )))

    if need_threshold:
        train_result = evaluate(model, train_ds)
        test_result = evaluate(model, test_ds)
    else:
        train_result = evaluate_on_threshold(model, train_ds, threshold=hparams["threshold"])
        test_result = evaluate_on_threshold(model, test_ds, threshold=hparams["threshold"])
    return train_result, test_result


"""# Build evaluation process"""


def evaluate(model, test_ds):
    results = [[y, model.predict(x)] for x, y in test_ds]
    y_true = np.concatenate([results[i][0] for i in range(len(results))], axis=0)
    y_pred = np.concatenate([results[i][1] for i in range(len(results))], axis=0).flatten()
    thresholds = np.arange(start=0.00, stop=1, step=0.01)

    results = {}
    for i in range(len(thresholds)):
        acc = BinaryAccuracy(threshold=thresholds[i])(y_true=y_true, y_pred=y_pred)
        mcc = BinaryMCC(threshold=thresholds[i])(y_true=y_true, y_pred=y_pred)
        sen = BinarySensitivity(threshold=thresholds[i])(y_true=y_true, y_pred=y_pred)
        spec = BinarySpecificity(threshold=thresholds[i])(y_true=y_true, y_pred=y_pred)
        f1 = BinaryF1Score(threshold=thresholds[i])(y_true=y_true, y_pred=y_pred)
        results[thresholds[i]] = np.array([f1, acc, mcc, sen, spec])

    return results


def evaluate_on_threshold(model, test_ds, threshold=0.5):
    results = [[y, model.predict(x)] for x, y in test_ds]
    y_true = np.concatenate([results[i][0] for i in range(len(results))], axis=0)
    y_pred = np.concatenate([results[i][1] for i in range(len(results))], axis=0).flatten()

    acc = BinaryAccuracy(threshold=threshold)(y_true=y_true, y_pred=y_pred)
    mcc = BinaryMCC(threshold=threshold)(y_true=y_true, y_pred=y_pred)
    sen = BinarySensitivity(threshold=threshold)(y_true=y_true, y_pred=y_pred)
    spec = BinarySpecificity(threshold=threshold)(y_true=y_true, y_pred=y_pred)
    f1 = BinaryF1Score(threshold=threshold)(y_true=y_true, y_pred=y_pred)

    return {threshold: np.array([f1, acc, mcc, sen, spec])}


def get_best_threshold(results):
    best = 0
    best_threshold = 0
    for threshold, result in results.items():
        target = result[MetricIdx.F1]
        if target > best:
            best = target
            best_threshold = threshold

    return best_threshold


def avg_evaluate(all_results):
    n_fold = len(all_results)
    final_result = {}
    if len(all_results) > 0:
        for threshold, result in all_results[0].items():
            final_result[threshold] = np.zeros_like(result)

    for i, results in enumerate(all_results):
        for threshold, result in results.items():
            final_result[threshold] = final_result[threshold] + result/n_fold

    return final_result


"""# Build k-fold training process"""


def k_fold_experiment(config, folds_data, test_data):
    print(config)
    hparams = config["hparams"]
    n_fold = config["n_fold"]

    train_results = []
    dev_results = []
    train_data = get_fold(folds_data=folds_data)
    train_data, test_data = preprocess_data(train_data, test_data)
    for fold_index in range(len(folds_data)):
        print("FOLD: {}".format(fold_index + 1))
        fold_data, dev_data = get_fold(folds_data, fold_index)
        fold_data, dev_data = preprocess_data(fold_data, dev_data)

        train_ds = build_train_ds(
            mlp_x=fold_data[DataIdx.MLP_FEATURE],
            rnn_x=fold_data[DataIdx.RNN_FEATURE],
            y=fold_data[DataIdx.LABEL],
            hparams=hparams
        )

        dev_ds = build_test_ds(
            mlp_x=dev_data[DataIdx.MLP_FEATURE],
            rnn_x=dev_data[DataIdx.RNN_FEATURE],
            y=dev_data[DataIdx.LABEL]
        )

        train_result, dev_result = train(
            config=config,
            train_ds=train_ds,
            test_ds=dev_ds,
            need_threshold=False,
            need_save=True,
            is_final=False
        )

        train_results.append(train_result)
        dev_results.append(dev_result)

    train_result = avg_evaluate(train_results)
    dev_result = avg_evaluate(dev_results)

    hparams["threshold"] = get_best_threshold(dev_result)
    train_result = train_result[hparams["threshold"]]
    dev_result = dev_result[hparams["threshold"]]

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

    _, test_result = train(
        config=config,
        train_ds=train_ds,
        test_ds=test_ds,
        need_summary=True,
        need_save=True,
        is_final=True
    )
    test_result = test_result[hparams["threshold"]]
    return train_result, dev_result, test_result


"""# Build Hyperparameter experiment process"""


def experiment(configs, log_dir):
    n_fold = 5
    folds_data, test_data = get_data(n_fold)
    for index, config in enumerate(configs):
        session_log_dir = os.path.join(log_dir, "session_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        print("\nEXPERIMENT: {}/{}".format(index+1, len(configs)))
        print(config)
        config["log_dir"] = session_log_dir
        config["n_fold"] = n_fold
        config["class_weight"] = {
              0: 1,
              1: 1
        }
        config["verbose"] = 0
        train_result, dev_result, test_result = k_fold_experiment(
            config=config,
            folds_data=folds_data,
            test_data=test_data
        )

        write(session_log_dir, "train", train_result, config["hparams"])
        write(session_log_dir, "dev", dev_result, config["hparams"])
        write(session_log_dir, "test", test_result, config["hparams"])
