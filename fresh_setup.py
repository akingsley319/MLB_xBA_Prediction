# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:17:26 2022

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
import split_data as sd


# create game files
# player_map will take 8 seconds between web scraping players, which will take
# a considerable amount of time.
# if desired, uncomment the "ga.player_map(df)" line

ga.game_file_creation(end_year = date.today().year)
df = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")

#ga.player_map(df)

print('data retrieved')


# clean files
data = cgf.clean_data(df)
data.to_csv('data/game_files_clean.csv')

print('data cleaned')


# train-test split
train, test, train_limit = gfp.train_test_split(data,pitch_limit=100,year_sens=1000)
print('split completed')


# clustering
pitch_mod = pm.Pitcher(train_limit)
_, score, n_clus = pitch_mod.full_package(covar_goal = 0.95)

print('clustering modeled')

train = pitch_mod.apply_cluster_modeling(train)
test = pitch_mod.apply_cluster_modeling(test)

train.to_csv('data/train/train_set.csv')
test.to_csv('data/test/test_set.csv')


print('clustering applied')


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


# =============================================================================
# train_pitchers = pd.read_csv('data/train/pitchers_train.csv')
# test_pitchers = pd.read_csv('data/test/pitchers_test.csv')
# 
# zzz = train_pitchers.loc[train_pitchers['pitcher'] == 453286]
# yyz = x_pitcher_train.loc[x_pitcher_train.index.get_level_values('pitcher') == 453286]
# =============================================================================

# pitcher recent prediction
x_pitcher_train, y_pitcher_train, pitcher_train = pm.pitcher_prep(train_pitchers)
x_pitcher_test, y_pitcher_test, pitcher_test = pm.pitcher_prep(test_pitchers)

pitcher_perf_model = pm.pitcher_perf(x_pitcher_train,y_pitcher_train,param_grid=None,intense=False,save=True)

print('pitchers predicted')


# matchup expectations
x_matchup_train, y_matchup_train = mm.matchup_prep(train_matchups)
x_matchup_test, y_matchup_test = mm.matchup_prep(test_matchups)

matchup_per_model = mm.matchup_perf(x_matchup_train, y_matchup_train,param_grid=None,intense=False,save=True)