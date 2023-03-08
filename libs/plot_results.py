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

import matplotlib.pyplot as plt
import dataframe_image as dfi
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.metrics import mean_squared_error, mean_absolute_error

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