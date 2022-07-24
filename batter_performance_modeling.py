# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:38:23 2022

@author: cubs1
"""

import pandas as pd

from game_file_preparation import BatterPrep
from batter_model import Batter

train_set = pd.read_csv('data/train/batters_condensed_train.csv')
test_set = pd.read_csv('data/test/batters_condensed_test.csv')

batter_prep = BatterPrep()

train = batter_prep.data_prep(train_set)
train = batter_prep.data_clean(train)

test = batter_prep.data_prep(test_set)
test = batter_prep.data_clean(test)

x_train = train.loc[:,~train.columns.isin(['next_estimated_ba_using_speedangle','batter'])]
y_train = train.loc[:,train.columns.isin(['next_estimated_ba_using_speedangle'])]

x_test = test.loc[:,~test.columns.isin(['next_estimated_ba_using_speedangle','batter'])]
y_test = test.loc[:,test.columns.isin(['next_estimated_ba_using_speedangle'])]

batter_model = Batter()
batter_model.fit_rf(x_train,y_train,replace=True)
# batter_model.fit_rf_intense(x_train,y_train,replace=True)

batter_model.save_model()

train.to_csv('data/train/batters_performance_train.csv')
test.to_csv('data/test/batters_performance_test.csv')



# =============================================================================
# 
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# 
# print('mse: ' + str(mean_squared_error(y_test,batter_model.predict(x_test))))
# print('mae: ' + str(mean_absolute_error(y_test,batter_model.predict(x_test))))
# =============================================================================

