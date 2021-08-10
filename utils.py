import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score
import scipy.stats as stats
from scipy import io
import scipy.signal as signal


def import_data(file_path):
    return pd.read_csv(file_path)


def load_dreem(file_path):
    return io.loadmat(file_path)


def truncate_time(rcs_data):
    """Creates a column with a truncated time stamp hh:mm:ss. This matches the timestamp format of the dreem data"""
    rcs_data['MatchTime'] = ''
    rcs_data['MatchTime'] = rcs_data['localTime'].str.slice(12, 20)


def match_index(rcs_data, dreem_data):
    """Creates a column in dreem_data that contains the index of the rcs_data with the corresponding timestamp.
        Only accepts a single recording epoch for rcs and dreem data."""

    dreem_data["RCS_Index"] = -1

    for i in range(dreem_data.shape[0]):
        index = np.min(rcs_data.index[rcs_data['MatchTime'] == dreem_data["Timehhmmss"].iloc[i]])
        if index >= 0:
            dreem_data["RCS_Index"].iloc[i] = int(index)


def label_rcs(rcs_data, dreem_data, interval=7500):
    """Labels the sleep stage of a time point in rcs_data using the corresponding timestamp in the dreem_data df.
        Assumes 250 Hz sample rate."""
    rcs_data["SleepStage"] = -1

    for i in range(dreem_data.shape[0]):
        index = dreem_data["RCS_Index"].iloc[i]
        if index >= 0:
            rcs_data['SleepStage'].iloc[index:index + interval] = dreem_data["SleepStage"].iloc[i]


def get_train_data(rcs_data, columns, num_days=4, label_column=-1, sleep_stage=1):
    """Right now, this function assumes we are classifying deep sleep.
        Inputs: list of RCS epochs as pandas dataframes, and corresponding columns for training.
        Outputs: Concatenated power data across days and corresponding 0-1 label for deep sleep."""

    X = rcs_data[0].iloc[:, columns].values
    y = rcs_data[0].iloc[:, label_column].values

    indices = np.where(y > 0)

    X = X[indices, :][0]
    y = y[indices]

    y[np.where(y != sleep_stage)] = 0
    y[np.where(y == sleep_stage)] = 1

    for i in range(1, num_days):
        tmpX = rcs_data[i].iloc[:, columns].values
        tmpy = rcs_data[i].iloc[:, label_column].values
        indices = np.where(tmpy > 0)

        tmpX = tmpX[indices, :]
        tmpy = tmpy[indices]
        tmpy[np.where(tmpy != sleep_stage)] = 0
        tmpy[np.where(tmpy == sleep_stage)] = 1

        X = np.concatenate((X, tmpX[0]), axis=0)
        y = np.concatenate((y, tmpy), axis=0)

    return X, y


def get_single_day_data(rcs_data, columns, day=4, label_column=-1, sleep_stage=1):
    """Returns data from single day, usually for using as a validation set
    Inputs: List of RCS pandas dataframes, columns for returning, and element in RCS list.
    Outputs: Corresponding data and 0-1 labels for deep sleep."""
    X_val = rcs_data[day].iloc[:, columns].values
    y_val = rcs_data[day].iloc[:, label_column].values
    y_val[np.where(y_val != sleep_stage)] = 0
    return X_val, y_val


def get_scores(predictions, y_val):
    print("Accuracy: " + str(accuracy_score(predictions, y_val)))
    print("F1: " + str(f1_score(y_val, predictions)))


def train_bagged_svm(X, y, c=10, n_estimators=10, kernel='linear', max_samples_denominator=10, bootstrap=False):
    clf = BaggingClassifier(SVC(C=c, kernel=kernel), n_estimators=n_estimators,
                            max_samples=1.0 / max_samples_denominator, bootstrap=bootstrap, verbose=True, n_jobs=-1)

    clf.fit(X, y)
    return clf


def average_block(data, block_size):
    """Takes in data array, and returns the average of blocks of the data array.
        Input: Data (numpy) array (2 dim), block size.
        Output: Data array of size floor(Data array.shape[0] / block size). Where the reduction in size arises
        from averaging of blocks of data entries."""

    trunc = data.shape[0] % block_size
    if trunc > 0:
        data_r = data[:-1 * trunc, :].reshape((-1, block_size, data.shape[-1]))
    else:
        data_r = data.reshape((-1, block_size, data.shape[-1]))
    return np.mean(data_r, axis=1).reshape((-1, data.shape[-1]))


def average_data_labels(data, labels, ave_size):
    ave_data = average_block(data, ave_size)
    ave_label = labels[(ave_size - 1)::ave_size]
    return ave_data, ave_label


def get_LD_parameters_for_config(clf_obj, scale_factor=100):
    """Returns the weights that go into the adaptive_config.json.
        i.e. These weights do not account for (are not scaled by) the FFPV,
        thus assuming that this scaling is included automatically downstream."""
    weights = clf_obj.coef_ * scale_factor
    intercept = clf_obj.intercept_ * scale_factor
    subtract_vector = np.ones_like(weights) * (intercept / np.size(weights)) * -1
    subtract_vector = subtract_vector / weights
    return weights, subtract_vector


def get_LD_parameters_for_API(clf_obj, fixedPointFactor=256, scale_factor=100):
    """This is the weights that would go directly into the RCS API.
        i.e. these weights account for (are scaled by) the FFPV"""
    weights = np.round(clf_obj.coef_ * scale_factor * fixedPointFactor)
    weights_prime = weights / fixedPointFactor
    intercept = clf_obj.intercept_ * scale_factor
    subtract_vector = np.ones_like(weights) * (intercept / np.size(weights)) * -1
    subtract_vector = subtract_vector / weights_prime
    return weights, np.round(subtract_vector)


def test_LD_parameters(val_data, val_labels, weights, sub_vec, FFPV=256, FFPV_scaled_weights=False):
    """This function computes the LD outputs of validation data. All predictions of 1 and 0 should be on opposite sides
    of your boundary value. Assumes second dimension of val_data matches size of weights and sub_vec.
        Inputs:
            val_data: Validation data set
            val_labels: Validation data labels
            weights: These are the linear classifier weights, usually as put into adaptive_config.json
            sub_vec: These are the subtraction vector values, usually as put into adaptive_config.json
            FFPV: fractional fixed point value. Only used is weights and sub_vec are the values directly put into API.
            FFPV_scaled_weights: Only set to true if your weights and sub_vec are scaled by FFPV"""
    if FFPV_scaled_weights:
        weights_ = weights / FFPV
    else:
        weights_ = weights

    predictions = np.sum(weights_ * (val_data - sub_vec), axis=1)
    print(np.count_nonzero(predictions == 0))
    print("1 :" + str(np.max(predictions[np.where(predictions > 0)])) + " ; " + str(
        np.min(predictions[np.where(predictions > 0)])))
    print("0 :" + str(np.max(predictions[np.where(predictions < 0)])) + " ; " + str(
        np.min(predictions[np.where(predictions < 0)])))

    predictions[np.where(predictions > 0)] = 1
    predictions[np.where(predictions < 0)] = 0
    get_scores(predictions, val_labels)
