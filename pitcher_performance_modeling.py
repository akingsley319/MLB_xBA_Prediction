# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:48:16 2022

@author: cubs1
"""

import pandas as pd

from game_file_preparation import PitcherPrep
from pitcher_model import PitcherPerf

train_set = pd.read_csv('data/train/pitchers_condensed_train.csv')
test_set = pd.read_csv('data/test/pitchers_condensed_test.csv')

pitcher_prep = PitcherPrep()

train = pitcher_prep.data_prep(train_set)
test = pitcher_prep.data_prep(test_set)

x_train = train.loc[:,~train.columns.isin(['next_estimated_ba_using_speedangle','pitcher'])]
y_train = train.loc[:,train.columns.isin(['next_estimated_ba_using_speedangle'])]

x_test = test.loc[:,~test.columns.isin(['next_estimated_ba_using_speedangle','pitcher'])]
y_test = test.loc[:,test.columns.isin(['next_estimated_ba_using_speedangle'])]

pitcher_model = PitcherPerf()
#pitcher_model.fit_rf(x_train,y_train,replace=True)
pitcher_model.fit_rf_intense(x_train,y_train,replace=True)

from sklearn.metrics import mean_squared_error, mean_absolute_error

print('mse: ' + str(mean_squared_error(y_test,pitcher_model.model.predict(x_test))))
print('mae: ' + str(mean_absolute_error(y_test,pitcher_model.model.predict(x_test))))

print(pitcher_model.model.best_params_)


pitcher_model.save_model()

train.to_csv('data/train/pitchers_performance_train.csv')
test.to_csv('data/test/pitchers_performance_test.csv')