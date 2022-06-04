# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:41:46 2022

@author: cubs1
"""

import pandas as pd
import cleaning_game_file as cgf

data = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")
test_d = cgf.Cleaner(data)
test_d.clean_data()

tester = test_d.data.head(10)

test_d.data.isna().sum()

df_e = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")
test_e = cgf.Cleaner(df_e)
test_e.clean_new()


df = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")
test_obj = cgf.Cleaner(df)
test_obj.clean_new()
test_obj.fill_players()

player_map = test_obj.retrieve_player_map()



# pitcher modeling testing



data = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")
test_data = cgf.Cleaner(data)
test_data.pitch_prep()
#test_data.pitch_spin_euc()

import pitcher_model as pm

test_df_copy = test_data.data.copy()
pitch_mod = pm.Pitcher(test_df_copy)
pitch_mod.full_package(test_df_copy)

data_orig = pitch_mod.remove_nulls(test_df_copy)

df = pitch_mod.standardize_data(data_orig)
df_dr = pitch_mod.dimensionality_reduction(df, data_orig, covar_goal=0.95)

df_clus = pitch_mod.pitcher_pitch_cluster(df_dr, pitch_limit=150, mini=-1, maxi=3)

score_max, n_clus = pm.fuzzy__clustering(df_clus)