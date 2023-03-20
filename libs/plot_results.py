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

from scipy.spatial import ConvexHull

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
    def __init__(self,n=500,alpha=0.25):
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
        self.alpha = alpha # opacity of scatterplots
        #self.color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.color_list = []
        n = n # The number of pitches closest to cluster center used in plots
        for atr in self.cluster_cols:
            self.clusters_close[atr] = self.data.nlargest(n,atr)
        self.define_color_list()
        
    # saves a 2d-scatterplot, separating clusters by color
    def scatter_2d(self,x,y):
        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111)    
        for i in range(0,len(self.cluster_cols)):
            c = self.color_list[i]
            df_temp = self.clusters_close[self.cluster_cols[i]]
            ax.scatter(df_temp[x],df_temp[y],label=self.cluster_cols[i],alpha=self.alpha,color=c)
            ax.set_xlabel(self.rename_var(x))
            ax.set_ylabel(self.rename_var(y))
            ax.set_title("Scatter of " + x + ", " + y)
            self.encircle(df_temp[x],df_temp[y],ec=c,fc="none")
        plt.legend()
        image_name = 'images/cluster/scatter_2d/' + x + '_' + y + '_scatter_2d' + '.png'
        plt.savefig(image_name, bbox_inches='tight')
        plt.show()
        
    # saves a 3d-scatterplot, separating clusters by color
    def scatter_3d(self,x,y,z):
        fig = plt.figure(figsize=(27, 22))
        #plt.title = ("Scatter of " + str(x) + ", " + str(y) + ", " + str(z))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(0,len(self.cluster_cols)):
            c = self.color_list[i]
            df_temp = self.clusters_close[self.cluster_cols[i]]
            ax.scatter(df_temp[x],df_temp[y],df_temp[z],label=self.cluster_cols[i],alpha=self.alpha,color=c)
            ax.set_xlabel(self.rename_var(x))
            ax.set_ylabel(self.rename_var(y))
            ax.set_zlabel(self.rename_var(z))
            ax.set_title("Scatter of " + x + ", " + y + ", " + z)
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
            plt.title(self.rename_var(clust_name))
            df_temp = self.clusters_close[clust_name]
            plt.hist(df_temp[col].dropna())
            plt.margins(0.05)
        fig.suptitle("Plot of " + str(col),fontsize=50)
        img_name = 'images/cluster/hist/' + col + '_hist' + '.png'
        plt.savefig(img_name, bbox_inches='tight')
        plt.show()
        
    # encircles plot points for a cluster in a scatter plot
    def encircle(self,x,y, ax=None, **kw):
        if not ax: ax=plt.gca()
        p = np.c_[x,y]
        hull = ConvexHull(p)
        poly = plt.Polygon(p[hull.vertices,:], **kw)
        ax.add_patch(poly)
        
    # creates list of colors of length equal to the number of clusters for
    # consistent color coding in plots
    def define_color_list(self):
        hsv = plt.get_cmap('hsv')
        self.color_list = hsv(np.linspace(0, 1.0, len(self.clusters_close)))
    
    # Renames variables so it is easier to tell what it represents
    def rename_var(self, x):
        if x == "spin_x":
            return "Spin X: Horizontal Spin from First to Third Base"
        elif x == "spin_z":
            return "Spin Z: Vertical Spin from Pitcher to Catcher"
        elif x == "release_speed":
            return "Release Speed: Velocity of Pitch at Point of Release"
        elif x == "release_pos_x":
            return "Horizontal Release Point from Catcher's Perspective"
        elif x == "release_pos_y":
            return "Release Point in feet from the Catcher"
        elif x == "release_pos_z":
            return "Vertical Release Point from Catcher's Perspective"
        elif x == "p_throws":
            return "Pitcher Handedness"
        elif x == "effective_speed":
            return "Effective Speed of Pitch as Viewed by Batter"
        elif x == "pitch_type":
            return "Pitch Type"
        else:
            return x
        
        
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
        self.train_dates = pd.DataFrame(columns=['model_type','April','May','June','July','August','September'])
        self.test_dates = pd.DataFrame(columns=['model_type','April','May','June','July','August','September'])
        self.train_dates_w = pd.DataFrame(columns=['model_type','April','May','June','July','August','September'])
        self.test_dates_w = pd.DataFrame(columns=['model_type','April','May','June','July','August','September'])
        
    
    # saves performance_block as a table
    def save_table(self):
        dfi.export(self.train_block.set_index('model_type'),'images/train_evaluation.png')
        dfi.export(self.test_block.set_index('model_type'),'images/test_evaluation.png')
        dfi.export(self.train_dates.set_index('model_type'),'images/train_dates_evaluation.png')
        dfi.export(self.test_dates.set_index('model_type'),'images/test_dates_evaluation.png')
        dfi.export(self.train_dates_w.set_index('model_type'),'images/train_dates_w_evaluation.png')
        dfi.export(self.test_dates_w.set_index('model_type'),'images/test_dates_w_evaluation.png')
    
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
        train, test = self.batter_predicted()
        train_true = train['next_estimated_ba_using_speedangle']
        train_pred = train['train_pred']
        train_weights = train['weights']
        test_true = test['next_estimated_ba_using_speedangle']
        test_pred = test['test_pred']
        test_weights = test['weights']
        train_entry = ['batter', self.mae(train_true,train_pred), 
                       self.mse(train_true,train_pred), self.wmae(train_true[:-1],train_pred[:-1],train_weights[1:])]
        test_entry = ['batter', self.mae(test_true,test_pred), 
                       self.mse(test_true,test_pred), self.wmae(test_true[:-1],test_pred[:-1],test_weights[1:])]
        self.train_block.loc[len(self.train_block)] = train_entry
        self.test_block.loc[len(self.test_block)] = test_entry
        self.eval_by_month(train,test,'batter')
        print('Batter Model Evaluated')
        
    def pitcher_results(self):
        train, test = self.pitcher_predicted()
        train_true = train['estimated_ba_using_speedangle']
        train_pred = train['train_pred']
        train_weights = train['weights']
        test_true = test['estimated_ba_using_speedangle']
        test_pred = test['test_pred']
        test_weights = test['weights']
        train_entry = ['pitcher', self.mae(train_true,train_pred), 
                       self.mse(train_true,train_pred), self.wmae(train_true[:-1],train_pred[:-1],train_weights[1:])]
        test_entry = ['pitcher', self.mae(test_true,test_pred), 
                       self.mse(test_true,test_pred), self.wmae(test_true[:-1],test_pred[:-1],test_weights[1:])]
        self.train_block.loc[len(self.train_block)] = train_entry
        self.test_block.loc[len(self.test_block)] = test_entry
        self.eval_by_month(train,test,'pitcher','estimated_ba_using_speedangle')
        print('Pitcher Model Evaluated')
        
    def matchup_results(self):
        train, test = self.matchup_predicted()
        train_true = train['estimated_ba_using_speedangle']
        train_pred = train['train_pred']
        train_weights = train['weights']
        test_true = test['estimated_ba_using_speedangle']
        test_pred = test['test_pred']
        test_weights = test['weights']
        train_entry = ['matchup', self.mae(train_true,train_pred), 
                       self.mse(train_true,train_pred), self.wmae(train_true[:-1],train_pred[:-1],train_weights[1:])]
        test_entry = ['matchup', self.mae(test_true,test_pred), 
                       self.mse(test_true,test_pred), self.wmae(test_true[:-1],test_pred[:-1],test_weights[1:])]
        self.train_block.loc[len(self.train_block)] = train_entry
        self.test_block.loc[len(self.test_block)] = test_entry
        self.eval_by_month(train,test,'matchup','estimated_ba_using_speedangle')
        print('Matchup Model Evaluated')
        
    def stacked_results(self):
        train, test = self.stacked_predicted()
        train_true = train['estimated_ba_using_speedangle']
        train_pred = train['train_pred']
        train_weights = train['weights']
        test_true = test['estimated_ba_using_speedangle']
        test_pred = test['test_pred']
        test_weights = test['weights']
        train_entry = ['stacked', self.mae(train_true,train_pred), 
                       self.mse(train_true,train_pred), self.wmae(train_true[:-1],train_pred[:-1],train_weights[1:])]
        test_entry = ['stacked', self.mae(test_true,test_pred), 
                       self.mse(test_true,test_pred), self.wmae(test_true[:-1],test_pred[:-1],test_weights[1:])]
        self.train_block.loc[len(self.train_block)] = train_entry
        self.test_block.loc[len(self.test_block)] = test_entry
        self.eval_by_month(train,test,'stacked','estimated_ba_using_speedangle')
        print('Stacked Model Evaluated')
        
    def combined_results(self):
        train, test = self.combined_predicted()
        train_true = train['estimated_ba_using_speedangle']
        train_pred = train['train_pred']
        train_weights = train['weights']
        test_true = test['estimated_ba_using_speedangle']
        test_pred = test['test_pred']
        test_weights = test['weights']
        train_entry = ['combined', self.mae(train_true,train_pred), 
                       self.mse(train_true,train_pred), self.wmae(train_true[:-1],train_pred[:-1],train_weights[1:])]
        test_entry = ['combined', self.mae(test_true,test_pred), 
                       self.mse(test_true,test_pred), self.wmae(test_true[:-1],test_pred[:-1],test_weights[1:])]
        self.train_block.loc[len(self.train_block)] = train_entry
        self.test_block.loc[len(self.test_block)] = test_entry
        self.eval_by_month(train,test,'combined','estimated_ba_using_speedangle')
        print('Combined Model Evaluated')
        
    # evaluates prediction (wmae and mae) for desired model/dataset by month
    def eval_by_month(self,train,test,model_type,true_res='next_estimated_ba_using_speedangle'):
        train_months = [model_type]
        train_months_w = [model_type]
        test_months = [model_type]
        test_months_w = [model_type]
        if model_type == 'batter' or model_type == 'pitcher':
            model_params = [model_type]
        else:
            model_params = ['batter','pitcher']
        dfs_train,dfs_test = self.split_by_month(train, test, model_params,true_res)
        for i in range(len(dfs_train)): # April through September
            train_months.append(self.mae(dfs_train[i][0],dfs_train[i][1]))
            train_months_w.append(self.wmae(dfs_train[i][0].values,dfs_train[i][1].values,dfs_train[i][2].values))
            if i >= len(dfs_test):
                test_months.append(None)
                test_months_w.append(None)
            else:
                test_months.append(self.mae(dfs_test[i][0],dfs_test[i][1]))
                test_months_w.append(self.wmae(dfs_test[i][0].values,dfs_test[i][1].values,dfs_test[i][2].values))
        self.train_dates.loc[len(self.train_dates)] = train_months
        self.test_dates.loc[len(self.test_dates)] = test_months
        self.train_dates_w.loc[len(self.train_dates_w)] = train_months_w
        self.test_dates_w.loc[len(self.test_dates_w)] = test_months_w
        
    def split_by_month(self,train,test,model_params,true_res='next_estimated_ba_using_speedangle'):
        temp_train = train.droplevel(model_params)
        temp_train.index = pd.to_datetime(temp_train.index)
        temp_train = temp_train.loc[temp_train.index.month.isin([4,5,6,7,8,9])]
        temp_train = temp_train.groupby(temp_train.index.month)
        dfs_train = [(group[[true_res]],group[['train_pred']],group[['weights']]) for _,group in temp_train]
        temp_test = test.droplevel(model_params)
        temp_test.index = pd.to_datetime(temp_test.index)
        temp_test = temp_test[temp_test.index.month.isin([4,5,6,7,8,9])]
        temp_test = temp_test.groupby(temp_test.index.month)
        dfs_test = [(group[[true_res]],group[['test_pred']],group[['weights']]) for _,group in temp_test]
        return dfs_train, dfs_test
    
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
        y_train['train_pred'] = model.predict(x_train)
        y_train['weights'] = train_weights
        y_test['test_pred'] = model.predict(x_test)
        y_test['weights'] = test_weights
        return y_train,y_test    
    
    def pitcher_predicted(self):
        train, test = self.retrieve_data('pitchers')
        x_train, y_train, train = pm.pitcher_prep(train)
        x_test, y_test, test = pm.pitcher_prep(test)
        train_weights = x_train['pa']
        test_weights = x_test['pa']
        with open(r"models/pitcher_recent_performance.pkl", "rb") as input_file:
            model = pkl.load(input_file)
        y_train['train_pred'] = model.predict(x_train)
        y_train['weights'] = train_weights
        y_test['test_pred'] = model.predict(x_test)
        y_test['weights'] = test_weights
        return y_train, y_test    
    
    def matchup_predicted(self):
        train, test = self.retrieve_data('matchups')
        x_train, y_train, train_weights = mm.matchup_prep(train)
        x_test, y_test, test_weights = mm.matchup_prep(test)
        with open(r"models/matchup.pkl", "rb") as input_file:
            model = pkl.load(input_file)
        y_train['train_pred'] = model.predict(x_train)
        y_train['weights'] = train_weights
        y_test['test_pred'] = model.predict(x_test)
        y_test['weights'] = test_weights
        return y_train, y_test    
    
    def stacked_predicted(self):
        train, test = self.retrieve_data('matchups')
        batter_train, batter_test = self.retrieve_data('batters')
        pitcher_train, pitcher_test = self.retrieve_data('pitchers')
        x_train, y_train, train_weights = sm.stacked_prep(train,batter_train,pitcher_train)
        x_test, y_test, test_weights = sm.stacked_prep(test,batter_test,pitcher_test)
        with open(r"models/stacked_model.pkl", "rb") as input_file:
            model = pkl.load(input_file)
        y_train['train_pred'] = model.predict(x_train)
        y_train['weights'] = train_weights
        y_test['test_pred'] = model.predict(x_test)
        y_test['weights'] = test_weights
        return y_train, y_test    
    
    def combined_predicted(self):
        train, test = self.retrieve_data('matchups')
        batter_train, batter_test = self.retrieve_data('batters')
        pitcher_train, pitcher_test = self.retrieve_data('pitchers')
        x_train, y_train, train_weights = cm.combined_prep(train,batter_train,pitcher_train)
        x_test, y_test, test_weights = cm.combined_prep(test,batter_test,pitcher_test)
        with open(r"models/combined_model.pkl", "rb") as input_file:
            model = pkl.load(input_file)
        y_train['train_pred'] = model.predict(x_train)
        y_train['weights'] = train_weights
        y_test['test_pred'] = model.predict(x_test)
        y_test['weights'] = test_weights
        return y_train, y_test
    
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