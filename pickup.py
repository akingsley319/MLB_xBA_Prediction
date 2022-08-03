# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:59:25 2022

@author: cubs1
"""

import sys
sys.path.insert(0, './libs')

import pandas as pd
import numpy as np

import requests
import bs4
import time

import os
from time import sleep
from urllib.error import HTTPError

from datetime import date

import get_data as ga
import cleaning_game_file as cgf
import game_file_preparation as gfp
import pitcher_model as pm
import batter_model as bm
import matchup_model as mm

train = pd.read_csv('data/train/train_set.csv')
test = pd.read_csv('data/test/test_set.csv')

# format files
train_prep = gfp.GamePrep(train)
test_prep = gfp.GamePrep(test)

# Train Batter Dataset
train_batters = train_prep.return_batters()
train_batters.to_csv('data/train/batters_train.csv')

# Test Batter Dataset
test_batters = test_prep.return_batters()
test_batters.to_csv('data/test/batters_test.csv')

print('batters formatted')

# Train Pitcher Dataset
train_pitchers = train_prep.return_pitchers()
train_pitchers.to_csv('data/train/pitchers_train.csv')

# Test Pitcher Dataset
test_pitchers = test_prep.return_pitchers()
test_pitchers.to_csv('data/test/pitchers_test.csv')

print('pitchers formatted')

# Train Combined Datset
train_matchups = train_prep.return_matchups()
train_matchups.to_csv('data/train/matchups_train.csv')

# Test Combined Datset
test_matchups = test_prep.return_matchups()
test_matchups.to_csv('data/test/matchups_test.csv')

print('matchups formatted')


# batter recent prediction
x_batter_train, y_batter_train, batter_train = bm.batter_prep(train_batters)
x_batter_test, y_batter_test, batter_test = bm.batter_prep(test_batters)

batter_perf_model = bm.batter_perf(x_batter_train,y_batter_train,param_grid=None,intense=False,save=True)

print('batters predicted')


# pitcher recent prediction
x_pitcher_train, y_pitcher_train, pitcher_train = pm.pitcher_prep(train_pitchers)
x_pitcher_test, y_pitcher_test, pitcher_test = pm.pitcher_prep(test_pitchers)

pitcher_perf_model = pm.pitcher_perf(x_pitcher_train,y_pitcher_train,param_grid=None,intense=False,save=True)

print('pitchers predicted')