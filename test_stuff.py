# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:39:40 2023

@author: cubs1
"""

import sys
sys.path.insert(0, './libs')

import plot_results as pr

test = pr.ResultsTable()

test.all_results()
test.performance_block
test.save_table()