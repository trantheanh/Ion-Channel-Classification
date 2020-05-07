import tensorflow as tf


"""# Build Tensorflow Dataset"""


def build_train_ds(mlp_x, rnn_x, y, hparams):
    ds = tf.data.Dataset.from_tensor_slices(((mlp_x, rnn_x), y))#.take(100)
    ds = ds.shuffle(10000).batch(hparams["batch_size"])
    return ds


def build_test_ds(mlp_x, rnn_x, y):
    ds = tf.data.Dataset.from_tensor_slices(((mlp_x, rnn_x), y))
    ds = ds.batch(32)
    return ds
