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
# import os
# import datetime
#
# from data.dataset import build_train_ds, build_test_ds
# from constant.index import DataIdx, MetricIdx
# from process.core import train
# from data.loader import get_fold_idx, get_fold
#
# """# MAIN FUNCTION"""
#
# FLAGS = flags.FLAGS
#
# flags.DEFINE_enum("optimizer", "nadam", ["adam", "rmsprop", "sgd", "adamax", "adadelta", "nadam"], "Name of optimizer")
# # flags.DEFINE_integer("batch_size", 1, "Batch size of traning data")
# flags.DEFINE_integer("batch_size", 1024, "Batch size of traning data")
# flags.DEFINE_float("learning_rate", 0.00016280409164167792, "learning rate of optimizer")
# # flags.DEFINE_integer("n_epoch", 100, "Number of training epoch")
# flags.DEFINE_integer("n_epoch", 1, "Number of training epoch")
# flags.DEFINE_integer("maxout_head", 2, "Number of maxout activation head")
# # flags.DEFINE_integer("maxout_units", 128, "Number of maxout units")
# flags.DEFINE_integer("maxout_units", 1, "Number of maxout units")
# flags.DEFINE_integer("rnn_layers", 1, "Number of LSTM layer")
# # flags.DEFINE_integer("rnn_units", 1024, "Number of RNN units")
# flags.DEFINE_integer("rnn_units", 1, "Number of RNN units")
#
#
# def main(argv):
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
#
#     print(config)
#
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
#     folds_data, test_data = get_fold_idx(n_fold)
#
#     for fold_index in range(n_fold):
#         print("FOLD: {}".format(fold_index + 1))
#
#         train_data, dev_data = get_fold(folds_data, fold_index)
#
#         train_ds = build_train_ds(
#             mlp_x=train_data[DataIdx.MLP_FEATURE],
#             rnn_x=train_data[DataIdx.RNN_FEATURE],
#             y=train_data[DataIdx.LABEL],
#             hparams=hparams
#         )
#
#         dev_ds = build_test_ds(
#             mlp_x=dev_data[DataIdx.MLP_FEATURE],
#             rnn_x=dev_data[DataIdx.RNN_FEATURE],
#             y=dev_data[DataIdx.LABEL]
#         )
#
#         train_result, dev_result = train(
#             config=config,
#             train_ds=train_ds,
#             test_ds=dev_ds,
#             need_threshold=True
#         )
#
#     # Train on all & evaluate on test set
#     train_data = get_fold(folds_data)
#
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
#     return
#
#
# if __name__ == "__main__":
#     app.run(main)
#
