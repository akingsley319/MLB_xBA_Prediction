# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:43:22 2022

@author: cubs1
"""

import pandas as pd
import numpy as np
import math

import csv

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pickle import dump, load

from fcmeans import FCM
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import game_file_preparation as gfp
import pickle as pkl

# Performs cleaning of dataset and returns the x and y components
def pitcher_prep(df):
    prep = PitcherPrep()
    
    temp_data = pitcher_prep.data_prep(train_set)
    temp_data = pitcher_prep.data_clean(train)
    
    X = temp_data.loc[:,~temp_data.columns.isin(['next_estimated_ba_using_speedangle','pitcher'])]
    y = temp_data.loc[:,temp_data.columns.isin(['next_estimated_ba_using_speedangle'])]
    pitcher = temp_data['pitcher']
    
    return X, y, pitcher

# Performs performance modeling and reutrns the model
def pitcher_perf(x_train,y_train,param_grid=None,intense=False,save=False):
    pitcher_model = gfp.PitcherPerf()
    
    if intense == True:
        pitcher_model.fit_rf_intense(x_train,y_train,replace=True)
    elif intense == False:
        pitcher_model.fit_rf(x_train,y_train,param_grid,replace=True)
        
    if save == True:
        pitcher_model.save_model()
        
    return pitcher_model.model

class PitcherPerf:
    def __init__(self):
        self.model = None
       
    # predicts using saved model
    def predict(self,data):
        return self.model.predict(data)
    
    # saves model
    def save_model(self):
        with open(r"models/pitcher_recent_performance.pkl", "wb") as output_file:
            pkl.dump(self.model, output_file)
            
    def retrieve_model(self):
        with open(r"models/pitcher_recent_performance.pkl", "rb") as input_file:
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

class Pitcher:
    def __init__(self, data):
        self.pitcher_stats = ['release_speed','release_pos_x','release_pos_z',
                 'pfx_x','pfx_z','plate_x','plate_z','vx0','vy0','vz0','ax',
                 'ay','az', 'effective_speed','release_spin_rate', 
                 'release_extension','release_pos_y','spin_x','spin_z',
                 'pitcher','game_year','pitch_type','pitch_name']
        self.pitch_features = ['release_speed','release_pos_x','release_pos_z',
                 'pfx_x','pfx_z','plate_x','plate_z','vx0','vy0','vz0','ax',
                 'ay','az', 'effective_speed','release_spin_rate', 
                 'release_extension','release_pos_y','spin_x', 'spin_z']
        self.pitcher_features = ['pitcher','game_year','pitch_type','pitch_name']
        self.df = data[self.pitcher_stats]
    
    # apply models; standardization, dimensionality reduction, fuzzy clustering
    # This should be applied prior to game file concentration due to need for 
    # all pitch metrics
    def apply_cluster_modeling(self, data):
        if data is not None:
            data = data.copy()
        else:
            data = self.df.copy()
        
        data = self.apply_standardization(data)
        data, cols = self.apply_dimensionality_reduction(data)
        data, clus_cols = self.apply_fuzzy_cluster(data, cols)
        
        return data
    
    # standardize data, pca dimensionality reduction, kmeans clustering of 
    # individual pitcher repertoires, fuzzy c clustering of ideal pitches
    def full_package(self, covar_goal=0.95, mini=-1, maxi=3, pitch_limiter=100):
        data_orig = self.remove_nulls(self.df.copy())
        
        data = self.standardize_data(data_orig)
        data = self.dimensionality_reduction(data, data_orig, covar_goal)
        
        if pitch_limiter > 0:
            game_prep = gfp.GamePrep(data)
            data = game_prep.pitch_limiter(pitch_limit=pitch_limiter)
        
        data = self.pitcher_pitch_cluster(data, mini, maxi)
        
        score_max, n_clus, _ = self.fuzzy_clustering(data)
        
        return data, score_max, n_clus
    
    # Fuzzy Clustering of ideal pitch data for pitchers
    # Uses best silhouette score for k-means to determine number of clusters
    def fuzzy_clustering(self, data, mini=10, maxi=60):
        if 'pitch_type' in data.columns:
            data.drop(columns=['pitch_type'], inplace=True)
        if 'pitch_name' in data.columns:
            data.drop(columns=['pitch_name'], inplace=True)
        if 'pitcher' in data.columns:
            data.drop(columns=['pitcher'], inplace=True)
        if 'game_year' in data.columns:
            data.drop(columns=['game_year'], inplace=True)
        
        score_max, n_clus, n_init = self.cluster_k_def(mini, maxi, data, 0)
        
        data = data.to_numpy()
        my_model = FCM(n_clusters=n_clus)
        my_model.fit(data) ## X, numpy array. rows:samples columns:features
        
        with open('models/fuzzy_clustering_pitching_data.pkl', 'wb') as output_file:
            dump(my_model, output_file)
        
        print('fuzzy clustering completed')
        
        return score_max, n_clus, n_init
    
    # Standardize the data; negative values are important to keep!
    def standardize_data(self, data):
        scaler = StandardScaler()
        data[self.pitch_features] = scaler.fit_transform(data[self.pitch_features])
        
        with open('models/standardize_pitching_data.pkl', 'wb') as output_file:
            dump(scaler, output_file)
            
        print('done with standardization')
        
        return data
    
    # Dimensionality Reduction; done together to keep consistent measurements;
    # keeps 95%+ variance by default
    def dimensionality_reduction(self, data, orig_data, covar_goal):
        temp_data = data[self.pitch_features]
        
        for col in temp_data.columns:
            temp_data[col].astype('float')
        
        pca = PCA(n_components=covar_goal, random_state=42)
        np_pca = pca.fit_transform(temp_data)
        
        with open('models/pca_pitches.pkl','wb') as output_file:
            dump(pca, output_file)
        
        df_pca = orig_data[self.pitcher_features].copy()
        for i in range(0,len(np_pca[0])):
            temp_column = []
            
            for row in np_pca:
                temp_column.append(row[i])
            
            column_name = 'attribute_' + str(i)
            df_pca[column_name] = temp_column
        
        print('done with dimensionality reduction')
        print('attributes: ' + str(len(np_pca[0])))
        
        return df_pca
    
    # clusters pitches by pitcher and year based on best sillhouette score
    def pitcher_pitch_cluster(self, data, mini, maxi):
        if 'pitch_type' in data.columns:
            data.drop(columns=['pitch_type'], inplace=True)
        if 'pitch_name' in data.columns:
            data.drop(columns=['pitch_name'], inplace=True)
        
        with open('data/pitcher_repertoire_clusters.csv', 'w', newline='') as fp:
            f = csv.writer(fp)
            f.writerow(list(data.columns))
        
        pitch_attr = [element for element in list(data.columns) if element not in ['pitcher','game_year']]
        pitch_df = pd.DataFrame(columns=data.columns)

        counter = 0
        max_counter = 0
        
        for play in data.pitcher.unique():
            max_counter += data[data.pitcher==play].game_year.nunique()
        
        print('Starting pitch_clustering')
        for pitcher in data.pitcher.unique():       
            for year in data[data.pitcher == pitcher].game_year.unique():
                temp_df = data[(data.pitcher == pitcher) & 
                               (data.game_year == year)][pitch_attr]
                
                num_pitch_type = int(self.df[(self.df.pitcher == pitcher) & 
                                         (self.df.game_year == year)].pitch_type.nunique())
                
                score_max, n_clus, n_init = self.cluster_k_def(mini, maxi, temp_df, num_pitch_type)
                
                km = KMeans(n_clusters=n_clus,n_init=n_init,random_state=42)
                km.fit_predict(temp_df)
                    
                with open('data/pitcher_repertoire_clusters.csv', 'a', newline='') as fp:
                    f = csv.writer(fp)
                    
                    for centroid in km.cluster_centers_:
                        to_append = [pitcher, year]
                        to_append.extend(centroid)
                        
                        f.writerow(to_append)
                        
                        pitch_df.loc[len(pitch_df)] = to_append
                    
                counter += 1
                print(str(counter) + '/' + str(max_counter)) 
                   
        print('pitch clustering completed')
        return pitch_df
    
    # apply standardization; transforms the input dataset
    def apply_standardization(self, data):
        pickled_model = load(open('models/standardize_pitching_data.pkl', 'rb'))
        data[self.pitch_features] = pickled_model.transform(data[self.pitch_features])
        return data
    
    # apply dimensionality reduction and create new columns in data set for the output columns
    def apply_dimensionality_reduction(self, data):
        pickled_model = load(open('models/pca_pitches.pkl', 'rb'))
        
        cols = []
        for i in range(0,pickled_model.n_components_):
            cols.append('attribute_' + str(i))
            
        data[cols] =  pickled_model.transform(data[self.pitch_features])
        
        return data, cols
        
    # apply fuzzy clustering with defined columns (designed with dimensionality reduction in mind)
    def apply_fuzzy_cluster(self, data, cols):
        pickled_model = load(open('models/fuzzy_clustering_pitching_data.pkl', 'rb'))
        
        cols_fc = []
        for i in range(0, len(pickled_model.centers)):
            cols_fc.append('cluster_attribute_' + str(i))
        
        data[cols_fc] = pickled_model.soft_predict(data[cols].to_numpy())
        
        return data, cols_fc
    
    # Remove null values
    def remove_nulls(self, data, columns=None):
        if columns == None:
            columns = data.columns
            
        for column in columns:
            data.drop(data[data[column].isna()].index, inplace=True)
        return data.reindex()
    
    # returns relevant columns for clustering
    def count_columns(self, data):
        cols = list(data.columns)
        
        for attr in cols:
            if attr in self.pitcher_features:
                cols.remove(attr)
            
        return cols
                    
    # kmeans clustering for pitches, finding best sillhouette score
    def cluster_k_def(self, mini, maxi, rep, pitch_count_num):
        score_max = 0
        n_clus = 0
        n_init = 0
        
        if mini <= 1:
            mini = 2
        
        maxi += pitch_count_num
    
        for i in range(mini,maxi+1):
            n_init_val, n_init_max = self.cluster_kmeans(rep, i)
        
            if n_init_max > score_max:
                score_max = n_init_max
                n_clus = i
                n_init = n_init_val
        
        return score_max, n_clus, n_init
    
    def cluster_kmeans(self, rep, i):
        n_init_val = 0
        n_init_max = 0
        
        for j in range(3,21):
            km = KMeans(i,n_init=j,random_state=42)
            km.fit_predict(rep)
            
            # Use euclidean due to mean-based application
            score = silhouette_score(rep, km.labels_, metric='euclidean')
            
            if score > n_init_max:
                n_init_val = i
                n_init_max = score
                
        return n_init_val, n_init_max
    
# =============================================================================
#     # returns pitcher and years where total pitches thrown for pitcher meets limit set
#     def pitch_limiter(self, data, pitch_limit=0):
#         pitcher_list = list(data[data.pitcher.notnull()].pitcher.unique())
#         game_years = list(data[data.game_year.notnull()].game_year.unique())
#         
#         pitch_count = {}
#         for pitcher in pitcher_list:
#             for year in game_years:
#                 temp_df = data[(data.pitcher == pitcher) & (data.game_year == year)]
#                 count = int(len(temp_df.index))
#             
#                 if count >= pitch_limit:
#                     if pitcher in pitch_count.keys():
#                         pitch_count[pitcher] += [year]
#                     elif pitcher not in pitch_count.keys():
#                         pitch_count[pitcher] = [year]
#                     else:
#                         print('what?')
#             
#         return pitch_count
# =============================================================================
    
    # returns number of reported pitch_types
    def num_pitch_types(self, df):
        pitchers, _ = self.pitcher_pitch_types(df)
        
        num_pitch_types = {}
        for pitcher in pitchers:
            num_pitch_types[pitcher] = len(pitchers[pitcher])
            
        return num_pitch_types
    
    # returns the pitch type each pitcher is reported to have thrown
    def pitcher_pitch_types(self, df):
        pitch_types = self.pitch_types(df)
        
        pitchers = {}
        for pitch in pitch_types.keys():
            for p in pitch_types[pitch]:
                if p in pitchers.keys():
                    pitchers[p] += pitch
                elif p not in pitchers.keys():
                    pitchers[p] = pitch
                else:
                    print('how?')
                    
        return pitchers, pitch_types
    
    # returns dictionary of who throws certain pitches
    def pitch_types(self, df):
        pitch_type = {}
        for pitch in list(df.pitch_type):
            pitch_type[pitch] = list(df[df.pitch_type == pitch]['pitcher'].unique())
            
        return pitch_type
        

