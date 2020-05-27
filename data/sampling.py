import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def smote(data):
    x, y = data
    sm = SMOTE(random_state=72)
    x_res, y_res = sm.fit_resample(x, y)
    return x_res, y_res


def get_data(data):
    negative_data = []
    positive_data = []
    for index in range(len(data)):
        if data[index][-1] == 0:
            negative_data.append(data[index])
        else:
            positive_data.append(data[index])

    major_data = None
    minor_data = None
    if len(negative_data) > len(positive_data):
        major_data = np.array(negative_data)
        minor_data = np.array(positive_data)
    else:
        major_data = np.array(positive_data)
        minor_data = np.array(negative_data)

    return major_data, minor_data


def random_from_minor(data):
    major_data, minor_data = get_data(data)

    new_minor_data = minor_data[np.random.randint(low=0, high=len(minor_data), size=(len(major_data),))]
    new_data = np.concatenate([major_data, new_minor_data], axis=0)

    return new_data


def random_from_major(data):
    major_data, minor_data = get_data(data)

    new_major_data = major_data[np.random.randint(low=0, high=len(major_data), size=(len(minor_data),))]
    new_data = np.concatenate([minor_data, new_major_data], axis=0)

    return new_data



