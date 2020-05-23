import requests
import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import StratifiedKFold
from resource import RESOURCE_PATH
from constant.url import DataPath


train_path = os.path.join(RESOURCE_PATH, DataPath.train_file_name)
test_path = os.path.join(RESOURCE_PATH, DataPath.test_file_name)
fold_path = os.path.join(RESOURCE_PATH, "fold_{}.csv")


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
    data = pd.read_csv(path).values

    return parse_data(data)


def parse_data(data):
    mlp_input = data[:, :30]
    rnn_input = np.stack([data[:, (30 + i * 20):(30 + (i + 1) * 20)] for i in range(15)], axis=1)
    label = data[:, 330]
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
    print("START SPLIT DATA TO {} FOLD".format(n_fold))
    train_data = pd.read_csv(train_path).values

    folds = StratifiedKFold(
        n_splits=n_fold,
        shuffle=True,
        random_state=0
    ).split(train_data[:, :330], train_data[:, 330])

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
        data = pd.read_csv(fold_path.format(i)).values
        folds_data.append(data)

    test_data = parse_csv_data(test_path)

    return folds_data, test_data


def get_fold(folds_data, fold_index=-1):
    train_data = np.concatenate([folds_data[i] for i in range(len(folds_data)) if i != fold_index])
    train_data = parse_data(train_data)

    if fold_index < 0:
        return train_data

    dev_data = folds_data[fold_index]
    dev_data = parse_data(dev_data)

    return train_data, dev_data


def preprocess_data(train_data, test_data):
    train_data, mean, std = normalize(train_data)
    test_data, _, _ = normalize(test_data, mean, std)
    return train_data, test_data





