# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:50:07 2022

@author: cubs1
"""

import pandas as pd
import game_file_preparation as gfp

train = pd.read_csv('data/train/train_data_fix.csv')
test = pd.read_csv('data/test/test_data_fix.csv')

import game_file_preparation as gfp

# Train Batter Dataset
train_prep = gfp.GamePrep(train)
train_batters = train_prep.return_batters()
train_batters.to_csv('data/train/batters_condensed_train.csv')

# Test Batter Dataset
test_prep = gfp.GamePrep(test)
test_batters = test_prep.return_batters()
test_batters.to_csv('data/test/batters_condensed_test.csv')

train_set_batters = train[['game_date','batter','estimated_ba_using_speedangle']]
test_set_batters = test[['game_date','batter','estimated_ba_using_speedangle']]



