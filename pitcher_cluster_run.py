# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:23:57 2022

@author: cubs1
"""

import pandas as pd
import cleaning_game_file as cgf

data = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")
print('data retrieved')
test_data = cgf.Cleaner(data)
test_data.clean_data()
test_data.clean_data_more()

test_df_copy = test_data.data.copy()

temp_df = test_data.data
temp_df.to_csv('data/game_files_clean.csv')
del(temp_df)

import game_file_preparation as gfp

test_df_copy = pd.read_csv('data/game_files_clean.csv')
test_df_copy = test_df_copy[test_df_copy.spin_x.notnull()]

prep = gfp.TrainPrep(test_df_copy)
train_set_year, test_set_year = prep.train_test_by_year('data/train/train_data.csv', 'data/test/test_data.csv')
train_setting_year = prep.pitch_limiter(train_set_year.copy(), pitch_limit=100)

import pitcher_model as pm

pitch_mod = pm.Pitcher(train_setting_year)
final_data_year, score, n_clus = pitch_mod.full_package(covar_goal = 0.95)

train_year_fix = pitch_mod.apply_cluster_modeling(train_set_year)
test_year_fix = pitch_mod.apply_cluster_modeling(test_set_year)

train_year_fix.to_csv('data/train/train_data_fix.csv')
test_year_fix.to_csv('data/test/test_data_fix.csv')



print('Cluster by Only Year: Train Data')
for col in train_year_fix.columns:
    if 'cluster_attribute' in col:
        print('max ' + str(col) + ': ' + str(train_year_fix[col].loc[train_year_fix[col].idxmax()]))
        print('min ' + str(col) + ': ' + str(train_year_fix[col].loc[train_year_fix[col].idxmin()]))
        
print('Cluster by Only Year: Test Data')
for col in test_year_fix.columns:
    if 'cluster_attribute' in col:
        print('max ' + str(col) + ': ' + str(test_year_fix[col].loc[test_year_fix[col].idxmax()]))
        print('min ' + str(col) + ': ' + str(test_year_fix[col].loc[test_year_fix[col].idxmin()]))