import numpy as np
import pandas as pd
import utils


def numerize_time(time_string):
    return int(time_string[0:2]) * 10 ** 4 + int(time_string[3:5]) * 10 ** 2 + int(time_string[6:])


class RecordingSession:

    def __init__(self, session_file_path):
        self.rcs_df = utils.import_data(session_file_path)
        self.parsed_df = None
        self.dreem_df = None
        self.accel_df = None

    def load_dreem(self, dreem_file_path):
        self.dreem_df = utils.import_data(dreem_file_path)

    def drop_Nan(self, field, copy_selected_columns=False, cols_to_save=[]):

        print("Warning: This function replaces this object's rcs_df field with an rcs_df that has Nan rows permanently "
              "removed.")
        print("To create original rcs_df, recreate RecordingSession object with identical filepath. If flag is true,"
              " then original columns, as deemed in cols_to_save, will be copied to parsed_df field for saving.")
        if copy_selected_columns:
            self.parsed_df = self.rcs_df.iloc[:, cols_to_save].copy()
        tmp = self.rcs_df[pd.notnull(self.rcs_df[field])]
        self.rcs_df = tmp.reset_index().drop(columns=["index"]).copy()

    def label_rcs_data(self, interval=7500):
        """Labels RCS data with sleep stages from the dreem band df by matching time samples.
            Inputs:
                interval: The number of rows to label a continuous segment of the rcs_df with the same
                    sleep stage. Usually equates to 30 seconds worth of rows, as that is how the dreem data is labelled.
        """

        self.create_truncated_time()
        self.match_index()
        self.label_rcs_sleep_stage(interval)
        self.check_beginning()

    def create_truncated_time(self):
        """Creates a column with a truncated time stamp hh:mm:ss. This matches the timestamp format of the dreem data"""
        self.rcs_df['MatchTime'] = ''
        self.rcs_df['MatchTime'] = self.rcs_df['localTime'].str.slice(12, 20)

    def match_index(self):
        """Creates a column in dreem_df that contains the index of the rcs_data with the corresponding timestamp.
            Only accepts a single recording epoch for rcs and dreem data."""

        self.dreem_df["RCS_Index"] = -1

        for i in range(self.dreem_df.shape[0]):
            index = np.min(self.rcs_df.index[self.rcs_df['MatchTime'] == self.dreem_df["Timehhmmss"].iloc[i]])
            if index >= 0:
                self.dreem_df["RCS_Index"].iloc[i] = int(index)

    def label_rcs_sleep_stage(self, interval):
        """Labels the sleep stage of a time point in rcs_data using the corresponding timestamp in the dreem_df.
            Assumes 250 Hz sample rate.
            Inputs: interval is the number of rows in RCS data frame that occur in 30 seconds."""

        self.rcs_df["SleepStage"] = -1

        for i in range(self.dreem_df.shape[0]):
            index = self.dreem_df["RCS_Index"].iloc[i]
            if index >= 0:
                self.rcs_df['SleepStage'].iloc[index:index + interval] = self.dreem_df["SleepStage"].iloc[i]

    def check_beginning(self):
        """This checks which recording started first, the dreem rec or the rcs rec."""
        rcs_start = numerize_time(self.rcs_df["MatchTime"].iloc[0])

        dreem_start = numerize_time(self.dreem_df["Timehhmmss"].iloc[0])

        if (int(self.rcs_df["MatchTime"].iloc[0][0:2]) < 7) & (int(self.dreem_df["Timehhmmss"].iloc[0][0:2]) >= 20):
            self.label_beginning()

        elif dreem_start < rcs_start:
            self.label_beginning()

        else:
            return

    def label_beginning(self):
        """Only call if Dreem recording begins before rcs_data.
        This function will label the missing elements for the first unlabelled block of rcs data points."""

        dreem_tmp = np.min(np.where(self.dreem_df["RCS_Index"].values >= 0))
        rcs_tmp = np.min(np.where(self.rcs_df["SleepStage"] > 0))
        self.rcs_df["SleepStage"].iloc[0:rcs_tmp] = self.dreem_df["SleepStage"].iloc[dreem_tmp - 1]


class Patient:

    def __init__(self):
        self.sessions = {}
        self.sessions_info = pd.DataFrame({'name': [], 'side': [], 'session_object': []})

    def add_session(self, name, session, side):
        """Adds session epoch to patients recording sessions dictionary.
            Inputs:
                Name: Name of recording session. Type - string. Recording date is recommended.
                Session: Recording_session object.
                Side: Brain Hemisphere of device
        """
        self.sessions[name] = session
        self.sessions_info.append(pd.DataFrame({'name': [name], 'side': [side], 'session_object': [session]}))

    def get_sleep_train_data(self, train_sessions, columns, label_column=-1, sleep_stages=1, omit_label=None):
        """Returns a training data and binary training label set based upon session names.
            Inputs:
                train_sessions: list of names of sessions for training
                columns: data columns to be used as features in model
                label_column: column number with data labels.
                sleep_stages: sleep stage for which classification is positive
                omit_label: Any data label that should be omitted from training data
            Outputs: Concatenated power data across days and corresponding 0-1 label for deep sleep."""

        X = self.sessions[train_sessions[0]].rcs_df.iloc[:, columns].values
        y = self.sessions[train_sessions[0]].rcs_df.iloc[:, label_column].values

        if omit_label is not None:
            indices = np.where((y > 0) & (y != omit_label))
        else:
            indices = np.where(y > 0)

        X = X[indices, :][0]
        y = y[indices]

        y[np.where(~np.isin(y, sleep_stages))] = 0
        y[np.where(np.isin(y, sleep_stages))] = 1

        for i in range(1, len(train_sessions)):
            tmpX = self.sessions[train_sessions[i]].rcs_df.iloc[:, columns].values
            tmpy = self.sessions[train_sessions[i]].rcs_df.iloc[:, label_column].values
            if omit_label is not None:
                indices = np.where((tmpy > 0) & (tmpy != omit_label))
            else:
                indices = np.where(tmpy > 0)

            tmpX = tmpX[indices, :]
            tmpy = tmpy[indices]
            tmpy[np.where(~np.isin(tmpy, sleep_stages))] = 0
            tmpy[np.where(np.isin(tmpy, sleep_stages))] = 1

            X = np.concatenate((X, tmpX[0]), axis=0)
            y = np.concatenate((y, tmpy), axis=0)

        return X, y
