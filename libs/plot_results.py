# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:58:08 2023

@author: cubs1
"""

import sys
sys.path.insert(0, './libs')

import pitcher_model as pm
import batter_model as bm
import matchup_model as mm
import combined_model as cm
import stacked_model as sm

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dataframe_image as dfi
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.metrics import mean_squared_error, mean_absolute_error

def cluster_plots(cluster_plots_obj):
    hist_cols = ['pitch_type','p_throws','effective_speed','release_speed']
    scatter_3d = [('release_speed','spin_x','spin_z'),
                  ('release_pos_x','release_pos_y','release_pos_z')]
    scatter_2d = [('spin_x','spin_z'),('spin_x','release_speed'),
                  ('release_speed','spin_z'),('release_pos_x','release_pos_z'),
                  ('release_pos_x','release_pos_y'),('release_pos_y','release_pos_z')]
    for item in hist_cols:
        cluster_plots_obj.hist_graph(item)
    for item in scatter_3d:
        cluster_plots_obj.scatter_3d(item[0],item[1],item[2])
    for item in scatter_2d:
        cluster_plots_obj.scatter_2d(item[0],item[1])

class ClusterPlots():
    def __init__(self):
        self.data = None
        self.cluster_cols = None # the cluster defining columns
        self.pitch_features = ['release_speed','release_pos_x','release_pos_z',
                               'pfx_x','pfx_z','plate_x','plate_z','vx0','vy0',
                               'vz0','ax','ay','az', 'effective_speed',
                               'release_spin_rate','release_extension',
                               'release_pos_y','spin_x', 'spin_z']
        self.retrieve_data()
        self.define_cols()
        self.clusters_close = {} # n closest points for each cluster
        n = 500 # The number of pitches closest to cluster center used in plots
        for atr in self.cluster_cols:
            self.clusters_close[atr] = self.data.nlargest(n,atr)
        
    # saves a 2d-scatterplot, separating clusters by color
    def scatter_2d(self,x,y):
        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111)    
        for i in range(0,len(self.cluster_cols)):
            df_temp = self.clusters_close[self.cluster_cols[i]]
            ax.scatter(df_temp[x],df_temp[y],label=self.cluster_cols[i])
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title("Scatter of " + x + ", " + y)    
        plt.legend()
        image_name = 'images/cluster/scatter_2d/' + x + '_' + y + '_scatter_2d' + '.png'
        plt.savefig(image_name, bbox_inches='tight')
        plt.show()
        
    # saves a 3d-scatterplot, separating clusters by color
    def scatter_3d(self,x,y,z):
        fig = plt.figure(figsize=(27, 22))
        plt.title = ("Scatter of " + str(x) + ", " + str(y) + ", " + str(z))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(0,len(self.cluster_cols)):
            df_temp = self.clusters_close[self.cluster_cols[i]]
            ax.scatter(df_temp[x],df_temp[y],df_temp[z],label=self.cluster_cols[i])
            ax.set_xlabel('release_speed')
            ax.set_ylabel(x)
            ax.set_zlabel(y)
            ax.set_title(z)
        plt.legend()
        image_name = 'images/cluster/scatter_3d/' + x + '_' + y + '_' + z + '_scatter_3d' + '.png'
        plt.savefig(image_name, bbox_inches='tight')
        plt.show()
       
    # saves a histogram of each cluster by a defined columns; a multi-plot (3xn)
    def hist_graph(self,col):
        fig = plt.figure(figsize=(27, 22))
        for i in range(0,len(self.cluster_cols)):
            ax=fig.add_subplot(4,3,i+1)
            plt.subplot(4, 3, i+1)
            clust_name = self.cluster_cols[i]
            plt.title(str(clust_name))
            df_temp = self.clusters_close[clust_name]
            plt.hist(df_temp[col].dropna())
            plt.margins(0.05)
        fig.suptitle("Plot of " + str(col),fontsize=50)
        img_name = 'images/cluster/hist/' + col + '_hist' + '.png'
        plt.savefig(img_name, bbox_inches='tight')
        plt.show()
        
    # retrieves and transforms the training dataset for analysis
    def retrieve_data(self):
        train = pd.read_csv('data/train/train_set.csv')
        with open(r"models/standardize_pitching_data.pkl", "rb") as input_file:
            model = pkl.load(input_file)
        train[self.pitch_features] = model.inverse_transform(train[self.pitch_features])
        train.dropna(subset=self.pitch_features, inplace=True)
        self.data = train
    
    # returns the cluster attribution of each pitch
    def define_cols(self):
        self.cluster_cols = [col for col in self.data.columns if 'cluster' in col]
        

# type = "batters","pitchers","matchups"
class ResultsTable():
    def __init__(self,x=None,y=None):
        self.train_block = pd.DataFrame(columns=['model_type','mae','mse','wmae'])
        self.test_block = pd.DataFrame(columns=['model_type','mae','mse','wmae'])
    
    # saves performance_block as a table
    def save_table(self):
        dfi.export(self.train_block.set_index('model_type'),'images/train_evaluation.png')
        dfi.export(self.test_block.set_index('model_type'),'images/test_evaluation.png')
    
    # evaluation metrics
    def wmae(self,true,pred,weights=None):
        return np.average(abs(true-pred),weights=weights)
    
    def mae(self,true,pred):
        return mean_absolute_error(true,pred)
    
    def mse(self,true,pred):
        return mean_squared_error(true,pred)
    
    # Updates self.performance_block with all model results
    def all_results(self):
        self.batter_results()
        self.pitcher_results()
        self.matchup_results()
        self.stacked_results()
        self.combined_results()
    
    # Saves model results to self.performance block
    def batter_results(self):
        train_true, train_pred, train_weights, test_true, test_pred, test_weights = self.batter_predicted()

        train_entry = ['batter', self.mae(train_true,train_pred), 
                       self.mse(train_true,train_pred), self.wmae(train_true[:-1],train_pred[:-1],train_weights[1:])]
        test_entry = ['batter', self.mae(test_true,test_pred), 
                       self.mse(test_true,test_pred), self.wmae(test_true[:-1],test_pred[:-1],test_weights[1:])]
        self.train_block.loc[len(self.train_block)] = train_entry
        self.test_block.loc[len(self.test_block)] = test_entry
        print('Batter Model Evaluated')
        
    def pitcher_results(self):
        train_true, train_pred, train_weights, test_true, test_pred, test_weights = self.pitcher_predicted()
        train_entry = ['pitcher', self.mae(train_true,train_pred), 
                       self.mse(train_true,train_pred), self.wmae(train_true[:-1],train_pred[:-1],train_weights[1:])]
        test_entry = ['pitcher', self.mae(test_true,test_pred), 
                       self.mse(test_true,test_pred), self.wmae(test_true[:-1],test_pred[:-1],test_weights[1:])]
        self.train_block.loc[len(self.train_block)] = train_entry
        self.test_block.loc[len(self.test_block)] = test_entry
        print('Pitcher Model Evaluated')
    
    def matchup_results(self):
        train_true, train_pred, train_weights, test_true, test_pred, test_weights = self.matchup_predicted()
        train_entry = ['matchup', self.mae(train_true,train_pred), 
                       self.mse(train_true,train_pred), self.wmae(train_true[:-1],train_pred[:-1],train_weights[1:])]
        test_entry = ['matchup', self.mae(test_true,test_pred), 
                       self.mse(test_true,test_pred), self.wmae(test_true[:-1],test_pred[:-1],test_weights[1:])]
        self.train_block.loc[len(self.train_block)] = train_entry
        self.test_block.loc[len(self.test_block)] = test_entry
        print('Matchup Model Evaluated')
    
    def stacked_results(self):
        train_true, train_pred, train_weights, test_true, test_pred, test_weights = self.stacked_predicted()
        train_entry = ['stacked', self.mae(train_true,train_pred), 
                       self.mse(train_true,train_pred), self.wmae(train_true[:-1],train_pred[:-1],train_weights[1:])]
        test_entry = ['stacked', self.mae(test_true,test_pred), 
                       self.mse(test_true,test_pred), self.wmae(test_true[:-1],test_pred[:-1],test_weights[1:])]
        self.train_block.loc[len(self.train_block)] = train_entry
        self.test_block.loc[len(self.test_block)] = test_entry
        print('Stacked Model Evaluated')
    
    def combined_results(self):
        train_true, train_pred, train_weights, test_true, test_pred, test_weights = self.combined_predicted()
        train_entry = ['combined', self.mae(train_true,train_pred), 
                       self.mse(train_true,train_pred), self.wmae(train_true[:-1],train_pred[:-1],train_weights[1:])]
        test_entry = ['combined', self.mae(test_true,test_pred), 
                       self.mse(test_true,test_pred), self.wmae(test_true[:-1],test_pred[:-1],test_weights[1:])]
        self.train_block.loc[len(self.train_block)] = train_entry
        self.test_block.loc[len(self.test_block)] = test_entry
        print('Combined Model Evaluated')
    
    # Retrieves the necessary data, prepares the data, and returns the true
    # and predicted values for the model
    def batter_predicted(self):
        train, test = self.retrieve_data('batters')
        x_train, y_train, train = bm.batter_prep(train)
        x_test, y_test, test = bm.batter_prep(test)
        train_weights = x_train['pa']
        test_weights = x_test['pa']
        with open(r"models/batter_recent_performance.pkl", "rb") as input_file:
            model = pkl.load(input_file)
        train_pred = y_train
        return self.reshape_data(y_train, model.predict(x_train), train_weights, y_test, model.predict(x_test), test_weights)    
    
    def pitcher_predicted(self):
        train, test = self.retrieve_data('pitchers')
        x_train, y_train, train = pm.pitcher_prep(train)
        x_test, y_test, test = pm.pitcher_prep(test)
        train_weights = x_train['pa']
        test_weights = x_test['pa']
        with open(r"models/pitcher_recent_performance.pkl", "rb") as input_file:
            model = pkl.load(input_file)
        return self.reshape_data(y_train, model.predict(x_train), train_weights, y_test, model.predict(x_test), test_weights)    
    
    def matchup_predicted(self):
        train, test = self.retrieve_data('matchups')
        x_train, y_train, train_weights = mm.matchup_prep(train)
        x_test, y_test, test_weights = mm.matchup_prep(test)
        with open(r"models/matchup.pkl", "rb") as input_file:
            model = pkl.load(input_file)
        return self.reshape_data(y_train, model.predict(x_train), train_weights, y_test, model.predict(x_test), test_weights)    
    
    def stacked_predicted(self):
        train, test = self.retrieve_data('matchups')
        batter_train, batter_test = self.retrieve_data('batters')
        pitcher_train, pitcher_test = self.retrieve_data('pitchers')
        x_train, y_train, train_weights = sm.stacked_prep(train,batter_train,pitcher_train)
        x_test, y_test, test_weights = sm.stacked_prep(test,batter_test,pitcher_test)
        with open(r"models/stacked_model.pkl", "rb") as input_file:
            model = pkl.load(input_file)
        return self.reshape_data(y_train, model.predict(x_train), train_weights, y_test, model.predict(x_test), test_weights)    
    
    def combined_predicted(self):
        train, test = self.retrieve_data('matchups')
        batter_train, batter_test = self.retrieve_data('batters')
        pitcher_train, pitcher_test = self.retrieve_data('pitchers')
        x_train, y_train, train_weights = cm.combined_prep(train,batter_train,pitcher_train)
        x_test, y_test, test_weights = cm.combined_prep(test,batter_test,pitcher_test)
        with open(r"models/combined_model.pkl", "rb") as input_file:
            model = pkl.load(input_file)
        return self.reshape_data(y_train, model.predict(x_train), train_weights, y_test, model.predict(x_test), test_weights)
    
    # Retrieves the necessary data, already separated by train/test set
    def retrieve_data(self,plot_type):
        train_link = 'data/train/' + plot_type + '_train.csv'
        test_link = 'data/test/' + plot_type + '_test.csv'
        return pd.read_csv(train_link), pd.read_csv(test_link)
    
    # reshapes data into useable format for evaluation
    def reshape_data(self, train_true, train_pred, train_weights, test_true, test_pred, test_weights):
        train_true = np.array(train_true.values)
        train_pred = train_pred.reshape(-1,1)
        train_weights = np.array(train_weights.values).reshape((-1,1))
        test_true = np.array(test_true.values)
        test_pred = test_pred.reshape(-1,1)
        test_weights = np.array(test_weights.values).reshape((-1,1))
        return train_true, train_pred, train_weights, test_true, test_pred, test_weights