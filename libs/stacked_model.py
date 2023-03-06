# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 01:24:18 2023

@author: cubs1
"""

import pandas as pd
import numpy as np
import pickle as pkl

import game_file_preparation as gfp
import split_data as sd

import pitcher_model as pm
import batter_model as bm
import matchup_model as mm

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def stacked_prep(df,df_batter,df_pitcher):
    matchup_prep = sd.MatchupPrep()
    batter_prep = sd.BatterPrep()
    pitcher_prep = sd.PitcherPrep()
    # prepares data from each source
    x_matchup, y_matchup, pa_data = mm.matchup_prep(df)
    x_batter, y_batter, _ = bm.batter_prep(df_batter)
    x_pitcher, y_pitcher, _ = pm.pitcher_prep(df_pitcher)
    # brings in batter and pitcher models
    with open(r"models/batter_recent_performance.pkl", "rb") as input_file:
        batters_model = pkl.load(input_file)
    with open(r"models/pitcher_recent_performance.pkl", "rb") as input_file:
        pitchers_model = pkl.load(input_file)
    with open(r"models/matchup.pkl", "rb") as input_file:
        matchups_model = pkl.load(input_file)
    # batter and pitcher predictions
    y_batter["batter_pred"] = batters_model.predict(x_batter)
    y_pitcher["pitcher_pred"] = pitchers_model.predict(x_pitcher)
    y_matchup["matchup_pred"] = matchups_model.predict(x_matchup)
    # joins predictions with data
    y_matchup['pa'] = pa_data
    y_matchup = y_matchup.join(y_batter['batter_pred'],on=['batter','game_date'])
    y_matchup = y_matchup.join(y_pitcher['pitcher_pred'],on=['pitcher','game_date'])
    y_matchup.dropna(inplace=True)
    # separates data
    X = y_matchup.loc[:,~y_matchup.columns.isin(['estimated_ba_using_speedangle','pa'])]
    y = y_matchup.loc[:,y_matchup.columns.isin(['estimated_ba_using_speedangle'])]
    pa_data = y_matchup.loc[:,y_matchup.columns.isin(['pa'])]
    return X, y, pa_data

def stacked_perf(x_train,y_train,param_grid=None,intense=False,save=False):
    stacked_model = Stacked()
    
    if intense == True:
        stacked_model.fit_rf_intense(x_train,y_train,param_grid=param_grid,replace=True)
    elif intense == False:
        stacked_model.fit_rf(x_train,y_train,replace=True)
        
    if save == True:
        stacked_model.save_model()
        
    return stacked_model.model

class Stacked:
    def __init__(self):
        self.model = None
       
    # predicts using saved model
    def predict(self,data):
        return self.model.predict(data)
    
    # saves model
    def save_model(self):
        with open(r"models/matchup.pkl", "wb") as output_file:
            pkl.dump(self.model, output_file)
            
    def retrieve_model(self):
        with open(r"models/matchup.pkl", "rb") as input_file:
            self.model = pkl.load(input_file)
       
    # fits to model with best parameters from trials
    def fit_rf(self,x_train,y_train,replace=True):
        model = RandomForestRegressor(n_estimators=1600,min_samples_split=2,
                                       min_samples_leaf=4,max_features='sqrt',
                                       max_depth=10,bootstrap=True)
#        model = RandomForestRegressor()
        
        model.fit(x_train,y_train)
        
        if replace == True:
            self.model = model
        else:
            return model
        
    # longer more intensive training process
    def fit_rf_intense(self,x_train,y_train,param_grid=None,replace=True):
        if param_grid == None:
            param_grid = self.default_param_grid()
            
        model = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator = model, param_distributions = param_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs=-1)
        rf_random.fit(x_train,y_train)
        
        if replace == True:
            self.model = rf_random
        else:
            return rf_random
           
    # default param grid for RandomSearchCV in fit_rf_intense
    def default_param_grid(self):
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        
        return {'n_estimators': n_estimators,
           'max_features': max_features,
           'max_depth': max_depth,
           'min_samples_split': min_samples_split,
           'min_samples_leaf': min_samples_leaf,
           'bootstrap': bootstrap}
        