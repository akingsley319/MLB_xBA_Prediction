# -*- coding: utf-8 -*-
""" Get all of the Data """

import get_data as ga
import pandas as pd

# Grabs all game data from 2017 through 2021 season for all teams
ga.game_files(seasons=(2017,2021))

df = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")

player_list = list(df.batter.unique()) + list(df.pitcher.unique())

full_list = []
for player in player_list:
    if player not in full_list:
        full_list.append(player)
        
player_mapping = ga.player_map(full_list)

with open('data/player_map.csv', 'w') as f:
    for key in player_mapping.keys():
        f.write("%s; %s\n" % (key, player_mapping[key]))