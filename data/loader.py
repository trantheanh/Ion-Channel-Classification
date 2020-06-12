import requests
import pandas as pd
import numpy as np
import os
import json
import shutil
from sklearn.model_selection import StratifiedKFold
from resource import RESOURCE_PATH
from constant.url import DataPath
from constant.shape import InputShape
from data.sampling import random_from_minor
from data.dictionary import EmbDict
import fasttext
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from models.core import train_tfidf

fold_idx_path = os.path.join(RESOURCE_PATH, "fold_idx.npy")
emb_path = os.path.join(RESOURCE_PATH, DataPath.emb_file_name)
train_raw_path = os.path.join(RESOURCE_PATH, DataPath.train_raw_file_name)
test_raw_path = os.path.join(RESOURCE_PATH, DataPath.test_raw_file_name)
train_pssm_path = os.path.join(RESOURCE_PATH, DataPath.train_pssm_file_name)
test_pssm_path = os.path.join(RESOURCE_PATH, DataPath.test_pssm_file_name)
train_cp_path = os.path.join(RESOURCE_PATH, DataPath.train_cp_file_name)
test_cp_path = os.path.join(RESOURCE_PATH, DataPath.test_cp_file_name)


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
    raw_data, pssm_data, cp_data, label = data

    # Load tfidf
    # with open(os.path.join(RESOURCE_PATH, DataPath.tfidf_file_name), "rb") as f:
    #     vectorizer = pickle.load(f)

    vectorizer = train_tfidf(raw_data)

    tfidf_data = vectorizer.transform([" ".join(tokens) for _, tokens in enumerate(raw_data)]).todense()

    # Load fasttext Emb
    model = fasttext.load_model(os.path.join(RESOURCE_PATH, DataPath.emb_file_name))

    emb_data = np.array([
        [model.get_word_vector(token) for _, token in enumerate(example[:])]
        for _, example in enumerate(raw_data)
    ])

    return emb_data, pssm_data, tfidf_data, cp_data, label


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
    train_pssm_data_x, train_pssm_data_y = read_pssm_data(train_pssm_path)
    train_raw_data_x, train_raw_data_y = read_raw_data(train_raw_path)

    folds = StratifiedKFold(
        n_splits=n_fold,
        shuffle=True,
        random_state=0
    ).split(
        train_pssm_data_x,
        train_pssm_data_y
    )

    fold_idx = []
    for fold_index, fold in enumerate(folds):
        fold_idx.append(fold)

    np.save(fold_idx_path, fold_idx)

    # Build TFIDF if not exist yet
    train_tfidf(train_raw_data_x)

    return


def get_fold_idx(n_fold):
    for i in range(n_fold):
        if not os.path.isfile(fold_idx_path):
            split_k_fold(n_fold)
            break

    fold_idx = np.load(fold_idx_path, allow_pickle=True)

    return fold_idx


def get_fold(fold_idx, fold_index=-1, need_oversampling=True):
    raw_data_x, raw_data_y = read_raw_data(train_raw_path)
    pssm_data_x, pssm_data_y = read_pssm_data(train_pssm_path)
    cp_data_x, cp_data_y = read_cp_data(train_cp_path)

    if fold_index == -1:
        train_data = (raw_data_x, pssm_data_x, cp_data_x, raw_data_y)

        test_raw_data_x, test_raw_data_y = read_raw_data(test_raw_path)
        test_pssm_data_x, test_pssm_data_y = read_pssm_data(test_pssm_path)
        test_cp_data_x, test_cp_data_y = read_cp_data(test_cp_path)
        dev_data = (test_raw_data_x, test_pssm_data_x, test_cp_data_x, test_raw_data_y)
    else:
        train_idx, dev_idx = fold_idx[fold_index]
        train_data = (raw_data_x[train_idx], pssm_data_x[train_idx], cp_data_x[train_idx], raw_data_y[train_idx])
        dev_data = (raw_data_x[dev_idx], pssm_data_x[dev_idx], cp_data_x[dev_idx], raw_data_y[dev_idx])

    train_data = parse_data(train_data)
    dev_data = parse_data(dev_data)

    return train_data, dev_data


# def preprocess_data(train_data, test_data):
#     train_data, mean, std = normalize(train_data)
#     test_data, _, _ = normalize(test_data, mean, std)
#     return train_data, test_data


# def read_from_emb(emb: fasttext.FastText):
#     train_raw_data = pd.read_csv(train_raw_path, header=None, delimiter=" ").values
#     test_raw_data = pd.read_csv(test_raw_path, header=None, delimiter=" ").values
#
#     # Get embedding
#     train_emb = []
#     for _, tokens in enumerate(train_raw_data[:, 0]):
#         vec = []
#         for i in range(len(tokens)):
#             vec.append(emb.get_word_vector(tokens[i]))
#
#         train_emb.append(vec)
#     train_emb = np.array(train_emb)
#
#     test_emb = []
#     for _, tokens in enumerate(test_raw_data[:, 0]):
#         vec = []
#         for i in range(len(tokens)):
#             vec.append(emb.get_word_vector(tokens[i]))
#
#         test_emb.append(vec)
#     test_emb = np.array(test_emb)
#
#     # Get TF-IDF
#     train_raw_data = [" ".join(tokens[:]) for _, tokens in enumerate(train_raw_data[:, 0])]
#     test_raw_data = [" ".join(tokens[:]) for _, tokens in enumerate(test_raw_data[:, 0])]
#     vectorizer = TfidfVectorizer(max_features=InputShape.TFIDF_DIM, analyzer="char")
#     vectorizer.fit(train_raw_data)
#     train_tfidf = vectorizer.transform(train_raw_data).todense()
#     test_tfidf = vectorizer.transform(test_raw_data).todense()
#
#     return train_emb, test_emb, train_tfidf, test_tfidf


def read_pssm_data(path):
    pssm_data = pd.read_csv(path, header=None).values
    pssm_x = pssm_data[:, :-1].astype(np.float)
    pssm_x = np.reshape(pssm_x, newshape=(-1, InputShape.PSSM_LENGTH, InputShape.PSSM_DIM))
    pssm_y = pssm_data[:, -1]

    return pssm_x, pssm_y


def read_raw_data(path):
    raw_data = pd.read_csv(path, header=None, delimiter=" ").values
    raw_x = raw_data[:, 0]
    raw_y = raw_data[:, 1]

    return raw_x, raw_y


def read_cp_data(path):
    cp_data = pd.read_csv(path, header=None).values
    cp_x = cp_data[:, :-1].astype(np.float)
    cp_y = cp_data[:, -1]

    return cp_x, cp_y







