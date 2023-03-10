# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:39:40 2023

@author: cubs1
"""

import sys
sys.path.insert(0, './libs')

import plot_results as pr
import dataframe_image as dfi

test = pr.ClusterPlots()
#test.hist_graph('pitch_type')
#test.scatter_3d("release_speed","spin_x","spin_z")
#test.scatter_2d("release_speed","spin_x")

pr.cluster_plots(test)

hist_cols = ['pitch_type','p_throws','effective_speed','release_speed']
scatter_3d = [('release_speed','spin_x','spin_z'),
              ('release_pos_x','release_pos_y','release_pos_z')]
scatter_2d = [('spin_x','spin_z'),('spin_x','release_speed'),
              ('release_speed','spin_z'),('release_pos_x','release_pos_z'),
              ('release_pos_x','release_pos_y'),('release_pos_y','release_pos_z')]

for item in hist_cols:
    test.hist_graph(item)
for item in scatter_3d:
    test.scatter_3d(item[0],item[1],item[2])
for item in scatter_2d:
    test.scatter_2d(item[0],item[1])