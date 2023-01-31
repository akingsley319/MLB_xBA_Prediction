# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:40:08 2023

@author: cubs1
"""

import sys
sys.path.insert(0, './libs')

import pandas as pd
import numpy as np

import get_data as ga
import cleaning_game_file as cgf

# updates the data in the raw csv file
ga.game_file_update()
df = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")

print('data retrieved')

# clean files
data = cgf.clean_data(df)
data.to_csv('data/game_files_clean.csv')

print('data cleaned')
