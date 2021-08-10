import numpy as np
import pandas as pd
import

def get_train_data(rcs_data, columns, num_days=4, label_column=-1, sleep_stage=1):
    """Right now, this function assumes we are classifying deep sleep.
        Inputs: list of RCS epochs as pandas dataframes, and corresponding columns for training.
        Outputs: Concatenated power data across days and corresponding 0-1 label for deep sleep."""

    X = rcs_data[0].iloc[:,columns].values
    y = rcs_data[0].iloc[:,label_column].values

    indices = np.where(y > 0)

    X = X[indices,:][0]
    y = y[indices]

    y[np.where(y != sleep_stage)] = 0
    y[np.where(y == sleep_stage)] = 1

    for i in range(1,num_days):
        tmpX = rcs_data[i].iloc[:,columns].values
        tmpy = rcs_data[i].iloc[:,label_column].values
        indices = np.where(tmpy > 0)

        tmpX = tmpX[indices,:]
        tmpy = tmpy[indices]
        tmpy[np.where(tmpy != sleep_stage)] = 0
        tmpy[np.where(tmpy == sleep_stage)] = 1

        X = np.concatenate((X, tmpX[0]), axis=0)
        y = np.concatenate((y, tmpy), axis=0)

    return X, y
