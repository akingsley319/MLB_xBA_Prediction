# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:18:15 2023

@author: cubs1
"""

import sys
sys.path.insert(0, './libs')

import plot_results as pr
import dataframe_image as dfi

eval_table = pr.ResultsTable()
eval_table.all_results()
eval_table.save_table()

clust_plots = pr.ClusterPlots(n=100,alpha=0.25)
pr.cluster_plots(clust_plots)