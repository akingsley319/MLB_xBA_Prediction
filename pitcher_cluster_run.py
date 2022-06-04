# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:23:57 2022

@author: cubs1
"""

import pandas as pd
import cleaning_game_file as cgf

data = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")
test_data = cgf.Cleaner(data)
test_data.clean_new()
test_data.pitch_spin_euc()

test_df_copy = test_data.data.copy()

import game_file_preparation as gfp

prep = gfp.GamePrep(test_df_copy)
train_set, test_set = prep.train_test_pitchers('data/train/pitcher_train_set.csv','data/test/pitcher_test_set.csv',test_split=0.1)
train_setting = prep.pitch_limiter(train_set, pitch_limit=100)

import pitcher_model as pm

pitch_mod = pm.Pitcher(train_set)
final_data, score, n_clus = pitch_mod.full_package(covar_goal=0.95)