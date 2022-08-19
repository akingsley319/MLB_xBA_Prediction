# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 00:45:47 2022

@author: cubs1
"""

import pandas as pd

import sys
sys.path.insert(0, './libs')

import libs.matchup_model as mm

train = pd.read_csv('data/train/matchups_train.csv')
test = pd.read_csv('data/test/matchups_test.csv')

x_matchup_train, y_matchup_train = mm.matchup_prep(train)
x_matchup_test, y_matchup_test = mm.matchup_prep(test)

matchup_per_model = mm.matchup_perf(x_matchup_train,y_matchup_train,param_grid=None,intense=True,save=True)

import pickle as pkl

with open(r"models/matchup.pkl", "rb") as input_file:
    model = pkl.load(input_file)
    
train_pred = model.predict(x_matchup_train)
test_pred = model.predict(x_matchup_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Training")
print("   mae: " + str(mean_absolute_error(y_matchup_train,train_pred)))
print("   mse: " + str(mean_squared_error(y_matchup_train,train_pred)))
print("Testing")
print("   mae: " + str(mean_absolute_error(y_matchup_test,test_pred)))
print("   mse: " + str(mean_squared_error(y_matchup_test,test_pred)))