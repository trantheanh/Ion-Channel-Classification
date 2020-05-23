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

train_path = os.path.join(RESOURCE_PATH, DataPath.train_file_name)
train_df = pd.read_csv(train_path)
# print(len(train_data.tolist()))
# print(len(set(train_data.tolist())))
# print(train_data.tolist()[:4])

print(train_df.shape)
print(train_df[:30].shape)
print(train_df[:30].drop_duplicates().shape)
print(train_df[30:-1].shape)
print(train_df[30:-1].drop_duplicates().shape)
