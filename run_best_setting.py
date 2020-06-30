# # -*- coding: utf-8 -*-
# import numpy as np
#
# import os
# import shutil
#
# from data.loader import download_from_shareable_link, parse_csv_data
# from constant.index import DataIdx
# from absl import flags, app
# import datetime
#
# from sklearn.model_selection import StratifiedKFold
# import numpy as np
# import os
# import datetime
#
# from data.dataset import build_train_ds, build_test_ds
# from constant.index import DataIdx, MetricIdx
# from process.core import train
#
# """# MAIN FUNCTION"""
#
# FLAGS = flags.FLAGS
#
# flags.DEFINE_enum("optimizer", "nadam", ["adam", "rmsprop", "sgd", "adamax", "adadelta", "nadam"], "Name of optimizer")
# flags.DEFINE_integer("batch_size", 1, "Batch size of traning data")
# # flags.DEFINE_integer("batch_size", 1024, "Batch size of traning data")
# flags.DEFINE_float("learning_rate", 0.00016280409164167792, "learning rate of optimizer")
# flags.DEFINE_integer("n_epoch", 100, "Number of training epoch")
# # flags.DEFINE_integer("n_epoch", 1, "Number of training epoch")
# flags.DEFINE_integer("maxout_head", 2, "Number of maxout activation head")
# flags.DEFINE_integer("maxout_units", 128, "Number of maxout units")
# # flags.DEFINE_integer("maxout_units", 1, "Number of maxout units")
# flags.DEFINE_integer("rnn_layers", 1, "Number of LSTM layer")
# flags.DEFINE_integer("rnn_units", 1024, "Number of RNN units")
# # flags.DEFINE_integer("rnn_units", 1, "Number of RNN units")
#
#
# def main(argv):
#     # GET DATA
#     train_data_path = "https://drive.google.com/open?id=1EJ0VcOmKUUOpSQ4dDkhbhVzvbIjz-Wg5"
#     test_data_path = "https://drive.google.com/open?id=14Xo1zxFKHrus1KPxmVJe-D0S8Px9OIlX"
#     download_from_shareable_link(url=train_data_path, destination="train_data.csv")
#     download_from_shareable_link(url=test_data_path, destination="test_data.csv")
#
#     train_data = parse_csv_data(path="train_data.csv")
#     test_data = parse_csv_data(path="test_data.csv")
#
#     print("Train Data Info:\n  MLP Feature: {}\n  RNN Feature: {}\n  Label: {}".format(
#       train_data[DataIdx.MLP_FEATURE].shape,
#       train_data[DataIdx.RNN_FEATURE].shape,
#       train_data[DataIdx.LABEL].shape
#     ))
#
#     print("Test Data Info:\n  MLP Feature: {}\n  RNN Feature: {}\n  Label: {}".format(
#       test_data[DataIdx.MLP_FEATURE].shape,
#       test_data[DataIdx.RNN_FEATURE].shape,
#       test_data[DataIdx.LABEL].shape
#     ))
#
#     # CONFIG DEFINE
#     log_dir = os.path.join(os.getcwd(), "log", "hparam_tuning")
#
#     config = {
#         "hparams": {
#             "threshold": 0.5,
#             "batch_size": FLAGS.batch_size,
#             "learning_rate": FLAGS.learning_rate,
#             "n_epoch": FLAGS.n_epoch,
#             "optimizer": FLAGS.optimizer,
#             "maxout_head": FLAGS.maxout_head,
#             "maxout_units": FLAGS.maxout_units,
#             "rnn_layers": FLAGS.rnn_layers,
#             "rnn_units": FLAGS.rnn_units,
#             "lr_decay": 0
#           }
#     }
#
#     session_log_dir = os.path.join(log_dir, "session_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
#     print(config)
#     config["log_dir"] = session_log_dir
#     config["n_fold"] = 5
#     config["class_weight"] = {
#           0: 1,
#           1: 1
#     }
#     config["verbose"] = 2
#
#     # RUN K FOLD
#     hparams = config["hparams"]
#     n_fold = config["n_fold"]
#
#     # Train on K-fold
#     folds = StratifiedKFold(
#         n_splits=n_fold,
#         shuffle=True,
#         random_state=0
#     ).split(train_data[DataIdx.MLP_FEATURE], train_data[DataIdx.LABEL])
#
#     for fold_index, fold in enumerate(folds):
#         print("FOLD: {}".format(fold_index + 1))
#         train_indexes, dev_indexes = fold
#
#         train_ds = build_train_ds(
#             mlp_x=train_data[DataIdx.MLP_FEATURE][train_indexes],
#             rnn_x=train_data[DataIdx.RNN_FEATURE][train_indexes],
#             y=train_data[DataIdx.LABEL][train_indexes],
#             hparams=hparams
#         )
#
#         dev_ds = build_test_ds(
#             mlp_x=train_data[DataIdx.MLP_FEATURE][dev_indexes],
#             rnn_x=train_data[DataIdx.RNN_FEATURE][dev_indexes],
#             y=train_data[DataIdx.LABEL][dev_indexes]
#         )
#
#         train_result, dev_result = train(
#             config=config,
#             train_ds=train_ds,
#             test_ds=dev_ds,
#             need_threshold=True
#         )
#
#         train_result = [
#             [
#                 threshold,
#                 result[MetricIdx.ACC],
#                 result[MetricIdx.SPEC],
#                 result[MetricIdx.SEN],
#                 result[MetricIdx.MCC],
#                 result[MetricIdx.SEN],
#                 1 - result[MetricIdx.SPEC]
#             ]
#             for threshold, result in train_result.items()]
#
#         np.savetxt(
#             "fold_{}_train.csv".format(fold_index+1),
#             np.array(train_result),
#             delimiter=",",
#             header="THRESHOLD, ACC, SPEC, SEN, MCC, TPR, FPR"
#         )
#
#         dev_result = [
#             [
#                 threshold,
#                 result[MetricIdx.ACC],
#                 result[MetricIdx.SPEC],
#                 result[MetricIdx.SEN],
#                 result[MetricIdx.MCC],
#                 result[MetricIdx.SEN],
#                 1 - result[MetricIdx.SPEC]
#             ]
#             for threshold, result in dev_result.items()]
#
#         np.savetxt(
#             "fold_{}_dev.csv".format(fold_index + 1),
#             np.array(dev_result),
#             delimiter=",",
#             header="THRESHOLD, ACC, SPEC, SEN, MCC, TPR, FPR"
#         )
#
#     # Train on all & evaluate on test set
#     train_ds = build_train_ds(
#         mlp_x=train_data[DataIdx.MLP_FEATURE],
#         rnn_x=train_data[DataIdx.RNN_FEATURE],
#         y=train_data[DataIdx.LABEL],
#         hparams=hparams,
#     )
#
#     test_ds = build_test_ds(
#         mlp_x=test_data[DataIdx.MLP_FEATURE],
#         rnn_x=test_data[DataIdx.RNN_FEATURE],
#         y=test_data[DataIdx.LABEL]
#     )
#
#     final_train_result, final_test_result = train(
#         config=config,
#         train_ds=train_ds,
#         test_ds=test_ds,
#         need_threshold=True
#     )
#
#     final_train_result = [
#         [
#             threshold,
#             result[MetricIdx.ACC],
#             result[MetricIdx.SPEC],
#             result[MetricIdx.SEN],
#             result[MetricIdx.MCC],
#             result[MetricIdx.SEN],
#             1 - result[MetricIdx.SPEC]
#         ]
#         for threshold, result in final_train_result.items()]
#
#     np.savetxt(
#         "train.csv",
#         np.array(final_train_result),
#         delimiter=",",
#         header="THRESHOLD, ACC, SPEC, SEN, MCC, TPR, FPR"
#     )
#
#     final_test_result = [
#         [
#             threshold,
#             result[MetricIdx.ACC],
#             result[MetricIdx.SPEC],
#             result[MetricIdx.SEN],
#             result[MetricIdx.MCC],
#             result[MetricIdx.SEN],
#             1 - result[MetricIdx.SPEC]
#         ]
#         for threshold, result in final_test_result.items()]
#
#     np.savetxt(
#         "test.csv",
#         np.array(final_test_result),
#         delimiter=",",
#         header="THRESHOLD, ACC, SPEC, SEN, MCC, TPR, FPR"
#     )
#
#     return
#
#
# if __name__ == "__main__":
#     app.run(main)
#
