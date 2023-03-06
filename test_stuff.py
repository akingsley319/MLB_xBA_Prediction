# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:39:40 2023

@author: cubs1
"""

import sys
sys.path.insert(0, './libs')

import plot_results as pr
import dataframe_image as dfi

test = pr.ResultsTable()

test.stacked_results()

test.train_block
test.test_block

test.save_table()
