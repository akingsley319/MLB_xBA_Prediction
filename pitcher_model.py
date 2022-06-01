# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:43:22 2022

@author: cubs1
"""

import pandas as pd
import math

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pickle import dump

from fcmeans import FCM


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
    
    # standardize data, pca dimensionality reduction, kmeans clustering of 
    # individual pitcher repertoires, fuzzy c clustering of ideal pitches
    def full_package(self, covar_goal=0.95, pitch_limit = 100, mini=-1, maxi=3):
        data_orig = self.remove_nulls(self.df.copy())
        
        data = self.standardize_data(data_orig)
        data = self.dimensionality_reduction(data, data_orig, covar_goal)
        data = self.pitcher_pitch_cluster(data, pitch_limit, mini, maxi)
        
        score_max, n_clus = self.fuzzy__clustering(data)
        
        return data, score_max, n_clus
    
    # Remove null values
    def remove_nulls(self, data):
        for column in data.columns:
            data.drop(data[data[column].isna()].index, inplace=True)
                           
        return data
    
    # Fuzzy Clustering of ideal pitch data for pitchers
    def fuzzy_clustering(self, data, mini=10, maxi=60):
        score_max, n_clus, n_init = self.cluster_k_def(mini, maxi, data, 0)
        
        my_model = FCM(n_clusters=n_clus) # we use two cluster as an example
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
    
    # Dimensionality Reduction; done together to keep consistent measurements
    def dimensionality_reduction(self, data, orig_data, covar_goal):
        pca = PCA(covar_goal, random_state=42)
        np_pca = pca.fit_transform(data[self.pitch_features])
        
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
    def pitcher_pitch_cluster(self, data, pitch_limit, mini, maxi):
        pitch_attr = self.count_columns(data)
        pitch_df = self.empty_db(pitch_attr)
        
        pitch_limiter = self.pitch_limiter(data, pitch_limit)
        
        counter = 0
        max_counter = len(pitch_limiter.keys())
        progress = 0
        for pitcher in pitch_limiter.keys():
            counter += 1
            print(str(counter) + '/' + str(max_counter))
            
            for year in pitch_limiter[pitcher]:
                temp_df = data[(data == pitcher) & (data == year)][pitch_attr]
                pitch_type_counter = self.num_pitch_types(self.df[self.df.pitch_type.notnull()])
                num_pitch_type = pitch_type_counter[pitcher]
                
                score_max, n_clus, n_init = self.cluster_k_def(mini, maxi, temp_df, num_pitch_type)
                
                km = KMeans(n_clusters=n_clus,n_init=n_init,random_state=42)
                km.fit_predict(temp_df)
                
                for centroid in km.cluster_centers_:
                    to_append = [pitcher, year] + list(centroid)
                    
                    pitch_df.loc[len(pitch_df)] = to_append
                   
        print('pitch clustering completed')
        return pitch_df
    
    # returns relevant columns for clustering
    def count_columns(self, data):
        cols = list(data.columns)
        
        for attr in cols:
            if attr in self.pitcher_features:
                cols.remove(attr)
            
        return cols
    
    # returns empty database for pitcher_pitch_clusters
    def empty_db(self, pitch_attr):
        cols = ['pitcher','game_year']
        
        for attr in pitch_attr:
            cols.extend(attr)
        
        return pd.DataFrame(columns=cols)
                    
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
    
    # returns pitcher and years where total pitches thrown for pitcher meets limit set
    def pitch_limiter(self, data, pitch_limit=0):
        pitcher_list = list(data[data.pitcher.notnull()].pitcher.unique())
        game_years = list(data[data.game_year.notnull()].game_year.unique())
        
        pitch_count = {}
        for pitcher in pitcher_list:
            for year in game_years:
                temp_df = data[(data.pitcher == pitcher) & (data.game_year == year)]
                count = int(len(temp_df.index))
            
                if count >= pitch_limit:
                    if pitcher in pitch_count.keys():
                        pitch_count[pitcher] += [year]
                    elif pitcher not in pitch_count.keys():
                        pitch_count[pitcher] = [year]
                    else:
                        print('what?')
            
        return pitch_count
    
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
        

