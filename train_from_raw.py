import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import sklearn
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from callbacks.core import build_callbacks

import os
import sys

import shutil
from constant.shape import InputShape

from resource import RESOURCE_PATH
from metrics.core import BinaryAccuracy, BinaryMCC, BinarySensitivity, BinarySpecificity, BinaryF1Score

# wd15
train_raw_file_name = "wd15/raw.train"
test_raw_file_name = "wd15/raw.ind.test"

# wd17
#train_raw_file_name = "wd17/input.train.added"
#test_raw_file_name = "wd17/input.ind.test.added"

# # wd19
# train_raw_file_name = "wd19/input.train.added"
# test_raw_file_name = "wd19/input.ind.test.added"

# # wd21
# train_raw_file_name = "wd21/input.train.added"
# test_raw_file_name = "wd21/input.ind.test.added"

train_data = pd.read_csv(os.path.join(RESOURCE_PATH, train_raw_file_name), header=None, delimiter=" ").values
test_data = pd.read_csv(os.path.join(RESOURCE_PATH, test_raw_file_name), header=None, delimiter=" ").values
print(train_data.shape)
print(test_data.shape)


def get_metrics(threshold=0.3) -> list:

    return [
        BinaryF1Score(threshold),
        BinaryAccuracy(threshold),
        BinaryMCC(threshold),
        BinarySensitivity(threshold),
        BinarySpecificity(threshold)
    ]


def get_origin_data(trigram_data):
    origin_data = []
    labels = []
    for example in trigram_data:
        tokens = example
        origin_example = []
        label = None
        for i in range(len(tokens)):
            if i == 0:
                label = int(tokens[i] != "__label__NonA")
            elif i < len(tokens) - 2:
                origin_example.append(tokens[i][0])

        origin_example += tokens[-2][:]

        origin_data.append(origin_example)
        labels.append(label)

    return origin_data, labels


train_data, train_label = get_origin_data(train_data)
test_data, test_label = get_origin_data(test_data)

train_dict = []
for data in train_data:
    train_dict += data[0]

train_dict = list(set(train_dict))

train_idx = np.array([[train_dict.index(token)+1 for token in data] for data in train_data]).reshape(-1, 15, 1)
test_idx = np.array([[train_dict.index(token)+1 for token in data] for data in test_data]).reshape(-1, 15, 1)
# print(len(train_dict))
# print(train_dict)

# frq = {}
# for data in train_data:
#     tokens = data[0]
#     for token in tokens:
#         if frq.get(token) is None:
#             frq[token] = 0
#         else:
#             frq[token] += 1


def tokenize(sentence: str):
    return sentence.split(sep=" ")


# train_seq = [" ".join(seq) for seq in train_data]
# test_seq = [" ".join(seq) for seq in test_data]
# print(test_seq[0])
# print(train_seq[0])
# vectorizer = TfidfVectorizer(max_features=InputShape.TFIDF_DIM, analyzer="char")
# vectorizer.fit(train_seq)
# train_tfidf = vectorizer.transform(train_seq).todense()
# test_tfidf = vectorizer.transform(test_seq).todense()
# print(train_tfidf.shape)
# print(test_tfidf.shape)

if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_exp")):
    shutil.rmtree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_exp"))


model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=len(train_dict)+1, output_dim=128, input_length=15))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=512, recurrent_dropout=0.1)))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1, activation="sigmoid"))
model.compile(optimizer="nadam", loss=keras.losses.binary_crossentropy,
              metrics=get_metrics(threshold=0.3)
              )
model.fit(
    train_idx, np.array(train_label),
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    callbacks=build_callbacks(log_dir="my_exp")
)

model.evaluate(test_idx, np.array(test_label))

# train_x = train_data[:, 1:-1]
# train_y = train_data[0, 0] == "__label__NonA"
# print(train_x, train_y)
# print(train_data[0])
