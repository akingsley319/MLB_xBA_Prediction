# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 21:19:44 2022

@author: cubs1
"""

import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

class Batter:
    def __init__(self):
        with open(r"models/batter_recent_performance.pkl", "rb") as input_file:
            self.model = pkl.load(input_file)
        #self.model = None
       
    # predicts using saved model
    def predict(self,data):
        return self.model.predict(data)
    
    # saves model
    def save_model(self):
        with open(r"models/batter_recent_performance.pkl", "wb") as output_file:
            pkl.dump(self.model, output_file)
       
    # fits to model with best parameters from trials
    def fit_rf(self,x_train,y_train,replace=True):
        model = RandomForestRegressor(n_estimators=200,min_samples_split=5,
                                      min_samples_leaf=4,max_features='auto',
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
        rf_random = RandomizedSearchCV(estimator = rf_test, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_job=-1)
        rf_random.fit(x_train,y_train)
        
        if replace == True:
            self.model = model
        else:
            return model
           
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
        
        
    