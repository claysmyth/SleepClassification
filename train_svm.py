import numpy as np
import pandas as pd
import utils

rcs_day_paths = [
    '/Users/claysmyth/Desktop/Starr/RCS_Hypno_Data/RCS02L/4_12_21/Session1618291552302/DeviceNPC700398H/combinedDataTable.csv',
    '/Users/claysmyth/Desktop/Starr/RCS_Hypno_Data/RCS02L/4_13_21/Session1618383275250/DeviceNPC700398H/combinedDataTable.csv',
    '/Users/claysmyth/Desktop/Starr/RCS_Hypno_Data/RCS02L/4_14_21/Session1618466025554/DeviceNPC700398H/combinedDataTable.csv',
    '/Users/claysmyth/Desktop/Starr/RCS_Hypno_Data/RCS02L/4_15_21/Session1618551933403/DeviceNPC700398H/combinedDataTable.csv',
    '/Users/claysmyth/Desktop/Starr/RCS_Hypno_Data/RCS02L/4_16_21/Session1618641928594/DeviceNPC700398H/combinedDataTable.csv'
]

dreem_paths = [
    '/Users/claysmyth/Desktop/Starr/RCS_Hypno_Data/DREEM/Patient_RCS02/4:12:21/hypnoCleaned.csv',
    '/Users/claysmyth/Desktop/Starr/RCS_Hypno_Data/DREEM/Patient_RCS02/4:13:21/hypnoCleaned.csv',
    '/Users/claysmyth/Desktop/Starr/RCS_Hypno_Data/DREEM/Patient_RCS02/4:14:21/hypnoCleaned.csv',
    '/Users/claysmyth/Desktop/Starr/RCS_Hypno_Data/DREEM/Patient_RCS02/4:15:21/hypnoCleaned.csv',
    '/Users/claysmyth/Desktop/Starr/RCS_Hypno_Data/DREEM/Patient_RCS02/4:16:21/hypnoCleaned.csv'
]

rcs_days = []

for i in rcs_day_paths:
    rcs_days.append(import_data(i))

dreem_days = []

for i in dreem_paths:
    dreem_days.append(import_data(i))

# Note: The below code OVERRIDES the dataframe, only storing non-null power data time points
for i in range(len(rcs_days)):
    tmp = rcs_days[i]
    tmp = tmp[pd.notnull(tmp["Power_Band1"])]
    rcs_days[i] = tmp.reset_index().drop(columns=["index"]).copy()

assert len(rcs_days) == len(dreem_days)

for i in range(len(rcs_days)):
    utils.truncate_time(rcs_days[i])
    utils.match_index(rcs_days[i], dreem_days[i])


for i in range(len(rcs_days)):
    utils.label_rcs(rcs_days[i], dreem_days[i], interval=60)


X,y = utils.get_train_data(rcs_days, np.arange(15,17))

X_val = rcs_days[4].iloc[:,15:17].values
y_val = rcs_days[4].iloc[:,-1].values
y_val[np.where(y_val != 1)] = 0

regs = [10000, 10000]

for reg in regs:
    print('\t' + str(reg))
    clf = utils.train_bagged_svm(X, y, c=reg, n_estimators=12, max_samples_denominator=100)
    predictions = clf.predict(X_val)
    utils.get_scores(predictions, y_val)

