import tensorflow as tf
import numpy as np


"""# Build Tensorflow Dataset"""


def build_train_ds(train_data, hparams):
    x_emb, x_pssm, x_tfidf, y = train_data
    y = y.astype(np.int)
    ds = tf.data.Dataset.from_tensor_slices(((x_emb, x_pssm, x_tfidf), y.astype(np.int)))
    ds = ds.shuffle(10000).batch(hparams["batch_size"])
    return ds


def build_test_ds(test_data):
    x_emb, x_pssm, x_tfidf, y = test_data
    y = y.astype(np.int)
    ds = tf.data.Dataset.from_tensor_slices(((x_emb, x_pssm, x_tfidf), y))
    ds = ds.batch(32)
    return ds
