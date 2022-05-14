# -*- coding: utf-8 -*-
""" Get all of the Data """

import get_data as ga
import pandas as pd

# Grabs all game data from 2017 through 2021 season for all teams
ga.game_files(seasons=(2017,2021))

df = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")

data_head = df.head(10)
df.dtypes()