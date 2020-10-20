import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def load_data(filename, line):
    data = pd.read_csv(filename, nrows=line)
    data['Label'] = pd.Categorical(data['Label']).codes

    cols = list(data)
    for item in cols:
        data = data[~data[item].isin([np.nan, np.inf])]

    labelset = data.iloc[:, 79:80].values
    dataset = data.iloc[:, 3:79].values

    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labelset, test_size=0.20, random_state=50)

    return X_train, X_test, Y_train, Y_test

