# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:41:46 2022

@author: cubs1
"""

import pandas as pd
import cleaning_game_file as cgf

df = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")

test_c = cgf.Cleaner(df)

test_c.clean_header()
test_c.fill_events()
test_c.des_info()
test_c.pitcher_name()
test_c.handle_game_date()
test_c.remove_cols()
test_c.simplify_events()

test_c.data.isna().sum()

data = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")
test_d = cgf.Cleaner(data)
test_d.clean_data()

tester = test_d.data.head(10)

test_d.data.isna().sum()

test_clean = data.head(10)

df_e = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")
test_e = cgf.Cleaner(df_e)
test_e.clean_new()
what = test_e.data[test_e.data.events == 'game_advisory']


player_list = list(df_e.batter.unique()) + list(df_e.pitcher.unique())

full_list = []
for player in player_list:
    if player not in full_list:
        full_list.append(player)
        
len(full_list)

import get_data as gd

gd.player_by_id(669211)