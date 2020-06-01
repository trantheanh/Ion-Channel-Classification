import requests
import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import StratifiedKFold
from resource import RESOURCE_PATH
from constant.url import DataPath
from constant.shape import InputShape
from data.sampling import random_from_minor
from data.dictionary import EmbDict
import fasttext


train_path = os.path.join(RESOURCE_PATH, DataPath.train_file_name)
test_path = os.path.join(RESOURCE_PATH, DataPath.test_file_name)
fold_path = os.path.join(RESOURCE_PATH, "fold_{}.csv")
emb_path = os.path.join(RESOURCE_PATH, "29052020", DataPath.emb_file_name)
train_raw_path = os.path.join(RESOURCE_PATH, "29052020", DataPath.train_raw_file_name)
test_raw_path = os.path.join(RESOURCE_PATH, "29052020", DataPath.test_raw_file_name)
train_pssm_path = os.path.join(RESOURCE_PATH, "29052020", DataPath.train_pssm_file_name)
test_pssm_path = os.path.join(RESOURCE_PATH, "29052020", DataPath.test_pssm_file_name)


"""# Get data from Google Drive"""


def download_file_from_drive_id(id, destination):
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    chunk_size = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def download_from_shareable_link(url, destination):
    file_id = url.split("=")[-1]
    download_file_from_drive_id(file_id, destination)


"""# Parsing data to features/label"""


def parse_csv_data(path):
    data = pd.read_csv(path, header=None).values

    return parse_data(data)


def parse_data(data):
    cb_size = InputShape.CB_SIZE
    pssm_size = (InputShape.PSSM_LENGTH, InputShape.PSSM_DIM)
    mlp_input = data[:, :cb_size]
    rnn_input = np.stack(
        [data[:, (cb_size + i * pssm_size[1]):(cb_size + (i + 1) * pssm_size[1])] for i in range(pssm_size[0])],
        axis=1
    )
    label = data[:, cb_size + pssm_size[0] * pssm_size[1]]
    return mlp_input, rnn_input, label


def normalize(data, mean=None, std=None):
    mlp_input, rnn_input, label = data
    if mean is None:
        mean = np.mean(rnn_input, axis=0)

    if std is None:
        std = np.std(rnn_input, axis=0)

    rnn_input = (rnn_input - mean) / std
    return (mlp_input, rnn_input, label), mean, std


def split_k_fold(n_fold=5):
    cb_size = InputShape.CB_SIZE
    pssm_size = (InputShape.PSSM_LENGTH, InputShape.PSSM_DIM)
    print("START SPLIT DATA TO {} FOLD".format(n_fold))
    train_data = pd.read_csv(train_path, header=None).values

    folds = StratifiedKFold(
        n_splits=n_fold,
        shuffle=True,
        random_state=0
    ).split(
        train_data[:, :(cb_size + pssm_size[0] * pssm_size[1])],
        train_data[:, cb_size + pssm_size[0] * pssm_size[1]]
    )

    folds_data = {}

    for fold_index, fold in enumerate(folds):
        train_indexes, dev_indexes = fold
        folds_data[fold_index] = train_data[dev_indexes]

    # Save folds to CSV file
    for i in range(n_fold):
        np.savetxt(
            fold_path.format(i),
            folds_data[i],
            delimiter=","
        )

    return


def get_data(n_fold):
    for i in range(n_fold):
        if not os.path.isfile(fold_path.format(i)):
            split_k_fold(n_fold)
            break

    folds_data = []
    for i in range(n_fold):
        data = pd.read_csv(fold_path.format(i), header=None).values
        folds_data.append(data)

    test_data = parse_csv_data(test_path)

    return folds_data, test_data


def get_fold(folds_data, fold_index=-1, need_oversampling=True):
    train_data = np.concatenate([folds_data[i] for i in range(len(folds_data)) if i != fold_index])
    train_data = parse_data(train_data)

    if need_oversampling:
        train_data = random_from_minor(train_data[:-1], train_data[-1])

    if fold_index < 0:
        return train_data

    dev_data = folds_data[fold_index]
    dev_data = parse_data(dev_data)

    return train_data, dev_data


def preprocess_data(train_data, test_data):
    train_data, mean, std = normalize(train_data)
    test_data, _, _ = normalize(test_data, mean, std)
    return train_data, test_data


def read_raw_data(file_path, need_int=True) -> np.ndarray:
    data = []
    with open(file_path, "r") as f:
        for index, line in enumerate(f):
            example = []
            seq = line.split(" ")
            example += seq[1:-1]
            if need_int:
                example.append(int(seq[0] == "__label__A"))
            else:
                example.append(seq[0])
            data.append(example)
    return np.array(data)


def read_emb_data():
    emb_lookup = EmbDict(emb_path)
    train_raw_data = read_raw_data(train_raw_path)
    test_raw_data = read_raw_data(test_raw_path)
    train_emb_data = emb_lookup(train_raw_data[:, :-1])
    test_emb_data = emb_lookup(test_raw_data[:, :-1])

    return train_emb_data, test_emb_data, train_raw_data[:, -1], test_raw_data[:, -1]


def read_from_emb(emb:fasttext.FastText):
    train_raw_data = read_raw_data(train_raw_path)
    test_raw_data = read_raw_data(test_raw_path)

    train_emb = []
    for _, tokens in enumerate(train_raw_data[:, :-1]):
        vec = []
        for i in range(len(tokens)):
            vec.append(emb.get_word_vector(tokens[i]))
        train_emb.append(vec)
    train_emb = np.array(train_emb)

    test_emb = []
    for _, tokens in enumerate(test_raw_data[:, :-1]):
        vec = []
        for i in range(len(tokens)):
            vec.append(emb.get_word_vector(tokens[i]))
        test_emb.append(vec)
    test_emb = np.array(test_emb)

    return train_emb, test_emb

    # for i in range(len(train_raw_data)):
        # train_emb.append(emb.)
    # print(train_raw_data[0])


def read_pssm_data():
    train_pssm = pd.read_csv(train_pssm_path, header=None).values
    test_pssm = pd.read_csv(test_pssm_path, header=None).values

    return train_pssm[:, :-1], test_pssm[:, :-1], train_pssm[:, -1], test_pssm[:, -1]







