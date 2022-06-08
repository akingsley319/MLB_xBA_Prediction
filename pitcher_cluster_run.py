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
train_set_year, test_set_year = prep.train_test_by_year('data/train_data.csv', 'data/test_data.csv')
train_setting_year = prep.pitch_limiter(train_set_year, pitch_limit=100)

import pitcher_model as pm

pitch_mod = pm.Pitcher(train_setting_year)
final_data_year, score, n_clus = pitch_mod.full_package(covar_goal = 0.95)

train_year_fix = pitch_mod.apply_cluster_modeling(train_set_year)
test_year_fix = pitch_mod.apply_cluster_modeling(test_set_year)





print('CLuster by Only Year')
for col in train_year_fix.columns:
    if 'cluster_attribute' in col:
        print('max ' + str(col) + ': ' + str(train_year_fix[col].loc[train_year_fix[col].idxmax()]))
        print('min ' + str(col) + ': ' + str(train_year_fix[col].loc[train_year_fix[col].idxmin()]))