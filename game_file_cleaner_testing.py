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
test_data.pitch_spin_euc()

import pitcher_model as pm
pitch_mod = pm.Pitcher(test_data.data)
test_df = pitch_mod.full_package()

