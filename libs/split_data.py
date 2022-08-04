# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 09:22:54 2022

@author: cubs1
"""

import pandas as pd
import numpy as np

from random import sample


class GenericPrep:
    def data_clean(self,data):
        data = self.remove_rows(data)
        data = self.remove_cols(data)       
        return data
    
    def remove_cols(self,data):
        for col in data.columns:
            if ('days_off' in col) or ('play' in col):
                if col != 'next_play':
                    data.drop(col,axis=1,inplace=True)
                    
        return data
    
    def remove_rows(self,data):
        data.dropna(axis=0,inplace=True)
        
        mask = (data.next_play != 0)
        
        return data.loc[mask,:]
    
    def play(self,data):
        data['play'] = data['pa'].apply(lambda x: 1 if x>0 else 0)
        
        return data
    
    def game_date_to_index(self,data):
        data = data.sort_values('game_date')
        data['game_date'] = pd.to_datetime(data['game_date'])
        data.set_index('game_date',inplace=True,drop=True)
        return data
    
    def date_info(self,data):
        day_of_week_idx = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
        
        data['day_of_week'] = data.index.day_name()
        data['day_of_week'] = data.day_of_week.replace(day_of_week_idx)
        
        data['month'] = data.index.month
        data['year'] = data.index.year 
    
    def per_pa(self,data):
        mask = (data.next_play != 0)
        data.loc[mask,'k_per_pa'] = data.loc[mask,'k'] / data.loc[mask,'pa']
        data.loc[mask,'bb_per_pa'] = data.loc[mask,'bb'] / (data.loc[mask,'pa'] + data.loc[mask,'bb'])
        
        data[['k_per_pa','bb_per_pa']] = data[['k_per_pa','bb_per_pa']].fillna(0)


class MatchupPrep(GenericPrep):
    def __init__(self, target='next_estimated_ba_using_speedangle'):
        self.target = target
        
    def initial_clean(self,data):
        self.play(data)
        data = self.game_date_to_index(data)
        data = self.type_set(data)
        data = self.shift_target(data)    
        
        return data
    
    def type_set(self,data):
        int_cols = ['batter','pitcher','pitch_count','play','k','bb']
        float_cols = [col for col in data.columns if ('cluster' in col) & ('list' not in col)]
        float_cols = float_cols.append('estimated_ba_using_speedangle')
        
        data.loc[:,data.columns.isin(int_cols)] = data.loc[:,data.columns.isin(int_cols)].astype('int')
        data.loc[:,data.columns.isin(float_cols)] = data.loc[:,data.columns.isin(float_cols)].astype('float')
    
    
        
class PitcherPrep(GenericPrep):
    def __init__(self, target='next_estimated_ba_using_speedangle'):
        self.target = target
    
    def data_prep(self,data,depth=18,bin_size=6,roll_vars=['estimated_ba_using_speedangle','k_per_pa','bb_per_pa']):
        df = data[['estimated_ba_using_speedangle','bb','k','pa','pitcher','game_date']].copy()
        
        df = self.initial_clean(df)
        
        print('pitcher columns: ' + str(df.columns))
        df = self.depth_finish(df,depth,bin_size,roll_vars)
        return df.dropna()
    
    def fill_dates(self,data):
        pd_out = pd.DataFrame()
        
        for pitcher in data['pitcher'].unique():
            temp_df = data[data.pitcher == pitcher]
            pd_out = pd_out.append(temp_df.asfreq('D'))
            pd_out['pitcher'].fillna(pitcher, inplace=True)
            
        data = pd_out
        data.fillna(0, inplace=True)
        
        return data
    
    def shift_target(self,data):
        data[['next_estimated_ba_using_speedangle','next_play','day_of_week','month','year']] = data.groupby('pitcher')[['estimated_ba_using_speedangle','play','day_of_week','month','year']].shift(-1)
        
        return data
    
    def type_set(self,data):
        int_cols = ['k','bb','pa','pitch_count','pitcher','play']
        
        for col in data.columns:
            if ('cluster' in col) and ('attribute' not in col):
                int_cols.append(col)
                
        data.loc[:,data.columns.isin(int_cols)] = data.loc[:,data.columns.isin(int_cols)].astype('int')
        data.loc[:,~data.columns.isin(int_cols)] = data.loc[:,~data.columns.isin(int_cols)].astype('float')
        
        return data
    
    def initial_clean(self,data):
        self.play(data)
        data = self.game_date_to_index(data)
        data = self.fill_dates(data)
        self.date_info(data)
        data = self.type_set(data)
        data = self.shift_target(data)    
        self.per_pa(data)
        
        return data
    
    # rolling mean by date
    def rolling_data(self,data,roll_amount,target):
        mean_name = target + '_mean_' + str(roll_amount)
        
        temp_df = data.groupby('pitcher')[[target,'play']].rolling(roll_amount).sum().reset_index(level=0,drop=False)
        temp_df.columns = ['pitcher', target, 'play']
        
        mask = (temp_df.play != 0)
        temp_df.loc[mask,mean_name] = temp_df.loc[mask,target] / temp_df.loc[mask,'play']
        
        temp_df.drop([target,'play'],axis = 1,inplace=True)
        
        return data.merge(temp_df, on=['game_date','pitcher'])
            
    # rolling mean weighted by plate appearances
    def rolling_weighted_data(self,data,roll_amount,target):
        data['weighted_pa'] = data.pa
        data['weighted_target'] = data[target] * data.weighted_pa
        
        name = target + '_mean_weighted_' + str(roll_amount)
        
        temp_df = data.groupby('pitcher')[['weighted_target','weighted_pa','bb']].rolling(roll_amount).sum().reset_index(level=0,drop=False)
        
        mask = (temp_df.weighted_pa != 0)
        if 'bb' not in target:
            temp_df.loc[mask,name] = temp_df.loc[mask,'weighted_target'] / temp_df.loc[mask,'weighted_pa']
        elif 'bb' in target:
            temp_df.loc[mask,name] = temp_df.loc[mask,'weighted_target'] / (temp_df.loc[mask,'weighted_pa'] + temp_df.loc[mask,'bb'])
        
        temp_df.drop(['weighted_target','weighted_pa','bb'],axis = 1,inplace=True)
        data.drop(['weighted_target','weighted_pa'],axis=1,inplace=True)
        
        return data.merge(temp_df, on=['game_date','pitcher'], how='inner')
        
    # introduces log of previous xBA
    def lag_features(self,data,n):
        cols = ['bb_per_pa','k_per_pa','play','estimated_ba_using_speedangle']
        
        for col in cols:
            for i in range(n):
                name = col + '_' + str(i+1)
                    
                if i > 0:
                    prev_name = col + '_' + str(i)
                    data[name] = data.groupby('pitcher')[prev_name].shift(1)
                else:
                    data[name] = data.groupby('pitcher')[col].shift(1)
                  
        return data
        
    # combines rolling and lag methods
    def depth_features(self,data,depth,bin_size,roll_vars=['estimated_ba_using_speedangle','k_per_pa','bb_per_pa']):
        temp_df = data.copy()
        for item in roll_vars:
            for i in range(1,depth+1):
                if i % bin_size == 0:
                    temp_df = self.rolling_data(temp_df,i,item)
                    temp_df = self.rolling_weighted_data(temp_df,i,item)
        temp_df = self.lag_features(temp_df,depth)
        
        return temp_df
    
    def depth_finish(self,data,depth,bin_size,roll_vars=['estimated_ba_using_speedangle','k_per_pa','bb_per_pa']):
        data = self.depth_features(data,depth,bin_size,roll_vars)
        
        return self.data_clean(data)


class BatterPrep(GenericPrep):
    def __init__(self, target='next_estimated_ba_using_speedangle'):
        self.target = target
    
    def data_prep(self, data, depth=15):
        data = data[['estimated_ba_using_speedangle','bb','k','pa','batter','game_date']].copy()
        data.fillna(0, inplace=True)
        
        data = self.play(data)
        data = self.game_date_to_index(data)
        data = self.fill_dates(data)
        self.date_info(data)
        data = self.type_set(data)
        data = self.shift_target(data)
        self.per_pa(data)
        data = self.depth_features(data,depth)
        
        data = self.data_clean(data)
        print('batter columns: ' + str(data.columns))
        
        return data.dropna()
    
    # Correctly set dtype
    def type_set(self, data):
        data[self.target] = data['estimated_ba_using_speedangle'].astype('float')
        data.loc[:,~data.columns.isin(['estimated_ba_using_speedangle'])] = data.loc[:,~data.columns.isin(['estimated_ba_using_speedangle'])].astype('int')
        
        return data
        
    # Fills missing days between batter's first and last appearance
    def fill_dates(self, data):
        pd_out = pd.DataFrame()
        batters = list(data['batter'].unique())
        
        for batter in batters:
            temp_df = data[data.batter == batter]
            pd_out = pd_out.append(temp_df.asfreq('D'))
            pd_out['batter'].fillna(batter, inplace=True)
            
        data = pd_out
        data.fillna(0, inplace=True)
        return data
    
    # Shifts variables that will be available prior, and the target variable
    def shift_target(self,data):
        data[['next_estimated_ba_using_speedangle','next_play','day_of_week','month','year']] = data.groupby('batter')[['estimated_ba_using_speedangle','play','day_of_week','month','year']].shift(-1)
        data.dropna(inplace=True)
        
        return data
    
    # rolling mean by date
    def rolling_data(self,data,roll_amount,target):
        mean_name = target + '_mean_' + str(roll_amount)
        
        temp_df = data.groupby('batter')[[target,'play']].rolling(roll_amount).sum().reset_index(level=0,drop=False)
        temp_df.columns = ['batter', target, 'play']
        
        mask = (temp_df.play != 0)
        temp_df.loc[mask,mean_name] = temp_df.loc[mask,target] / temp_df.loc[mask,'play']
        
        temp_df.drop([target,'play'],axis = 1,inplace=True)
        
        return data.merge(temp_df, on=['game_date','batter'])
        
    # rolling mean weighted by plate appearances
    def rolling_weighted_data(self,data,roll_amount,target):
        data['weighted_pa'] = data.pa
        data['weighted_target'] = data[target] * data.weighted_pa
        
        name = target + '_mean_weighted_' + str(roll_amount)
        
        temp_df = data.groupby('batter')[['weighted_target','weighted_pa','bb']].rolling(roll_amount).sum().reset_index(level=0,drop=False)
        
        mask = (temp_df.weighted_pa != 0)
        if 'bb' not in target:
            temp_df.loc[mask,name] = temp_df.loc[mask,'weighted_target'] / temp_df.loc[mask,'weighted_pa']
        elif 'bb' in target:
            temp_df.loc[mask,name] = temp_df.loc[mask,'weighted_target'] / (temp_df.loc[mask,'weighted_pa'] + temp_df.loc[mask,'bb'])
        
        temp_df.drop(['weighted_target','weighted_pa','bb'],axis = 1,inplace=True)
        data.drop(['weighted_target','weighted_pa'],axis=1,inplace=True)
        
        return data.merge(temp_df, on=['game_date','batter'], how='inner')
    
    # introduces log of previous xBA
    def lag_features(self,data,n):
        cols = ['bb_per_pa','k_per_pa','play','estimated_ba_using_speedangle']
        
        for col in cols:
            for i in range(n):
                name = col + '_' + str(i+1)
                
                if i > 0:
                    prev_name = col + '_' + str(i)
                    data[name] = data.groupby('batter')[prev_name].shift(1)
                else:
                    data[name] = data.groupby('batter')[col].shift(1)
                    
        return data
    
    # combines rolling and lag methods
    def depth_features(self,data,depth,roll_vars=['estimated_ba_using_speedangle','k_per_pa','bb_per_pa']):
        temp_df = data.copy()
        for item in roll_vars:
            for i in range(1,depth+1):
                if i % 5 == 0:
                    temp_df = self.rolling_data(temp_df,i,item)
                    temp_df = self.rolling_weighted_data(temp_df,i,item)
        temp_df = self.lag_features(temp_df,depth)
        
        return temp_df