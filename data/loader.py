import requests
import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import StratifiedKFold
from constant.url import DataPath
from data import DATA_PATH


working_data_folder = "working_data"
fold_name = "fold_{}.csv"
train_name = "train.csv"
test_name = "test.csv"


"""# Get data from Google Drive"""


def download_file_from_drive_id(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_from_shareable_link(url, destination):
    file_id = url.split("=")[-1]
    download_file_from_drive_id(file_id, destination)


"""# Parsing data to features/label"""


def parse_csv_data(path):
  data = pd.read_csv(path).values
  mlp_input = data[:, :30]
  rnn_input = np.stack([data[:, (30+i*20):(30+(i+1)*20)] for i in range(15)], axis=1)
  label = data[:, 330]
  return mlp_input, rnn_input, label


def split_k_fold(n_fold=5):
    tmp_folder = "tmp"
    # Remove Old working data folder
    if os.path.isdir(os.path.join(DATA_PATH)):
        shutil.rmtree(os.path.join(DATA_PATH))

    # Create tmp folder
    if not os.path.isdir(tmp_folder):
        os.mkdir(tmp_folder)

    target_train_path = os.path.join(tmp_folder, "train_data.csv")
    target_test_path = os.path.join(tmp_folder, "test_data.csv")

    download_from_shareable_link(url=DataPath.train_data_path, destination=target_train_path)
    download_from_shareable_link(url=DataPath.test_data_path, destination=target_test_path)

    train_data = pd.read_csv(target_train_path).values
    test_data = pd.read_csv(target_train_path).values

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
    if not os.path.isdir(os.path.join(DATA_PATH, working_data_folder)):
        os.mkdir(os.path.join(DATA_PATH, working_data_folder))

    for i in range(n_fold):
        np.savetxt(os.path.join(
            DATA_PATH,
            working_data_folder,
            fold_name.format(i+1)),
            folds_data[i], delimiter=",")

    np.savetxt(os.path.join(DATA_PATH, working_data_folder, train_name), train_data, delimiter=",")
    np.savetxt(os.path.join(DATA_PATH, working_data_folder, test_name), test_data, delimiter=",")

    # Remove tmp folder
    if os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)

    return


def get_data(fold_index, n_fold):
    if os.path.isdir(os.path.join(DATA_PATH, working_data_folder)) \
            and os.path.isfile(os.path.join(DATA_PATH, working_data_folder, fold_name.format(fold_index))):
        return os.path.join(DATA_PATH, working_data_folder, fold_name.format(fold_index))
    else:
        split_k_fold(n_fold=n_fold)



