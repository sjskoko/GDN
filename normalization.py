import pandas as pd
import numpy as np

train_ = pd.read_csv(f'data/HAI/train.csv') # 216001
train_num = 216001
test_ = pd.read_csv(f'data/HAI/test.csv') # 216001

sensor = list(train_.keys())
sensor.remove('time')
print(sensor)

train_sensor = train_[sensor]
test_sensor = test_[sensor]

time_cat = pd.concat([train_['time'], test_['time']])
train_test_cat = pd.concat([train_sensor, test_sensor])


# normalized_tt_cat=(train_test_cat[sensor] - train_test_cat[sensor].min())/(train_test_cat[sensor].max()-train_test_cat[sensor].min() + e)
normalized_tt_cat=(train_test_cat[sensor]-train_test_cat[sensor].mean())/(train_test_cat[sensor].std())
normalized_tt_cat = normalized_tt_cat.fillna(0.0)

normalized_tt_cat.insert(0, 'time', time_cat)

normalized_train = normalized_tt_cat[:train_num]
normalized_test = normalized_tt_cat[train_num:]

normalized_test.insert(len(normalized_test.columns), 'attack', test_['attack'])

dirt = 'HAI_norm'
normalized_train.to_csv(f'data/{dirt}/train.csv')
normalized_test.to_csv(f'data/{dirt}/test.csv')

