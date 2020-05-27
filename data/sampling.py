import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def smote(data):
    x, y = data
    sm = SMOTE(random_state=72)
    x_res, y_res = sm.fit_resample(x, y)
    return x_res, y_res


def get_data_index(labels):
    negative_idx = []
    positive_idx = []
    for index in range(len(labels)):
        if labels[index] == 0:
            negative_idx.append(index)
        else:
            positive_idx.append(index)

    if len(negative_idx) > len(positive_idx):
        major_idx = negative_idx
        minor_idx = positive_idx
    else:
        major_idx = positive_idx
        minor_idx = negative_idx

    return np.array(major_idx), np.array(minor_idx)


def random_from_minor(features, labels):
    major_idx, minor_idx = get_data_index(labels)

    new_minor_idx = minor_idx[np.random.randint(low=0, high=len(minor_idx), size=(len(major_idx),))]

    new_data_idx = np.concatenate([major_idx, new_minor_idx], axis=0)

    new_data = [features[i][new_data_idx] for i in range(len(features))] + [labels[new_data_idx]]

    return new_data


def random_from_major(features, labels):
    major_idx, minor_idx = get_data_index(labels)

    new_major_idx = minor_idx[np.random.randint(low=0, high=len(major_idx), size=(len(minor_idx),))]

    new_data_idx = np.concatenate([minor_idx, new_major_idx], axis=0)

    new_data = ([features[i][new_data_idx] for i in range(len(features))], labels[new_data_idx])

    return new_data
