import tensorflow.keras as keras
from metrics.core import BinaryAccuracy, BinaryMCC, BinarySensitivity, BinarySpecificity
from process.core import build_test_ds
from constant.index import DataIdx, MetricIdx
from data import loader
import numpy as np
import os
from saved_model import SAVED_MODEL_PATH
from resource import RESOURCE_PATH
from constant.url import DataPath
import pandas as pd
from constant.shape import InputShape

from data.dictionary import EmbDict
from data.loader import read_emb_data, read_pssm_data, read_raw_data


def parse_data(emb_data, pssm_data):
    emb_size = (InputShape.EMB_LENGTH, InputShape.EMB_DIM)
    pssm_size = (InputShape.PSSM_LENGTH, InputShape.PSSM_DIM)
    mlp_input = data[:, :cb_size]
    rnn_input = np.stack(
        [data[:, (i * pssm_size[1]):((i + 1) * pssm_size[1])] for i in range(InputShape.PSSM_LENGTH)],
        axis=1
    )
    label = data[:, cb_size + pssm_size[0] * pssm_size[1]]
    return mlp_input, rnn_input, label


train_emb, test_emb, train_emb_label, test_emb_label = read_emb_data()
train_pssm, test_pssm, train_pssm_label, test_pssm_label = read_pssm_data()








