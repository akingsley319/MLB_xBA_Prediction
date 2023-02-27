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
        
        if 'next_play' in data.columns:
            mask = (data.next_play != 0)
        elif 'next_play' not in data.columns:
            mask = (data.pa != 0)
        
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
        if 'next_play' in data.columns:
            mask = (data.next_play != 0)
        elif 'next_play' not in data.columns:
            mask = (data.play != 0)
            
        data.loc[mask,'k_per_pa'] = data.loc[mask,'k'] / data.loc[mask,'pa']
        data.loc[mask,'bb_per_pa'] = data.loc[mask,'bb'] / (data.loc[mask,'pa'] + data.loc[mask,'bb'])
        
        data[['k_per_pa','bb_per_pa']] = data[['k_per_pa','bb_per_pa']].fillna(0)


class MatchupPrep(GenericPrep):
    def __init__(self, target='estimated_ba_using_speedangle'):
        self.target = target
        
    def data_prep(self,data,depth_num=30,depth_min_pitcher=3,depth_min_batter=10,depth_type='D'):
        data = self.game_date_to_index(data)
        data = self.data_focus(data)
        
        pitcher_hard_data = self.rolling_cluster_hard(data,depth_num,depth_min_pitcher,depth_type)
        pitcher_soft_data = self.rolling_cluster_soft(data,depth_num,depth_min_pitcher,depth_type)
        batter_data = self.rolling_batter(data,depth_num,depth_min_batter,depth_type)
        
        data = data.loc[:,['pitcher','batter','estimated_ba_using_speedangle','pa']]
        data.dropna(inplace=True)
        
        data = data.set_index(['pitcher',data.index])
        data = data.join(pitcher_hard_data,on=['pitcher','game_date'])
        data = data.join(pitcher_soft_data,on=['pitcher','game_date'])
        
        data = data.reset_index().set_index(['batter','game_date'])
        data = data.join(batter_data,on=['batter','game_date'])
        
        data = data.set_index(['pitcher',data.index])
        data.dropna(inplace=True)
        
        return data.loc[:,~data.columns.isin(['pa'])], data['pa']
    
    # COmbines batter steps
    def rolling_batter(self,data,depth_num,depth_min=10,depth_type='D'):
        data = data.copy()
        data = self.explode(data)
        data = self.cluster_day(data)
        data = self.rolling_xba(data,depth_num,depth_min=10,depth_type='D')
        
        return data
    
    # Expand list columns detailing each event of atbat
    def explode(self,data):
        keep_cols = [col for col in data.columns if 'list' in col]
        keep_cols.extend(['batter'])
        
        data = data[keep_cols]
        keep_cols.remove('batter')
            
        for col in keep_cols:
            data[col] = data[col].apply(lambda x: x.translate({ord(i): None for i in ' []'}).split(','))
            
        data = data.set_index(['batter',data.index]).apply(pd.Series.explode).reset_index(level=0)
        data = data.replace('nan',np.nan)
        data.dropna(inplace=True)
        data[keep_cols] = data[keep_cols].astype('float')
        return data
    
    # Takes expanded columns and condenses results into day focused results
    # Takes a weighted mean for handling soft clustering
    def cluster_day(self,data):
        data = data.apply(lambda x: x.abs())
        cluster_cols = [col for col in data.columns if 'cluster' in col]
        cluster_cols.append('estimated_ba_using_speedangle_list')
        col_names = [col + '_agg' for col in cluster_cols]
        
        def weighted_mean(x):
            list_out = []
            for i in range(len(cluster_cols)):
                try:
                    wm = np.average(x['estimated_ba_using_speedangle_list'],weights=x[cluster_cols[i]])
                except ZeroDivisionError:
                    wm = 0
                list_out.append(wm)
            return list_out
        
        temp_df = pd.DataFrame()
        temp_df[0] = data.groupby(['batter','game_date'])[cluster_cols].apply(lambda x: weighted_mean(x))
        temp_data = data.groupby(['batter','game_date'])[cluster_cols].sum()
        temp_df = pd.DataFrame(temp_df[0].values.tolist(), index=temp_df.index, columns=col_names)
        temp_df.columns = col_names
        
        temp_df = temp_df.join(temp_data)
        
        return temp_df.reset_index(level=0)
    
    # Creates a rolling tracker for batter performance against certain pitch cluster
    def rolling_xba(self,data,depth_num,depth_min=10,depth_type='D'):
        cluster_cols = [col for col in data.columns if 'cluster' in col and 'agg' in col]
        weight_cols = [col for col in data.columns if 'cluster' in col and 'list' in col and 'agg' not in col]
        
        col_names = [col.replace('_list','_estimated') for col in weight_cols]
        
        depth = str(depth_num) + depth_type
        
        for i in range(len(cluster_cols)):
            data[cluster_cols[i]] = data[cluster_cols[i]] * data[weight_cols[i]]
        
        data = data.apply(lambda x: x.abs())
        
        temp_df = pd.DataFrame()
        temp_data = pd.DataFrame()
        temp_df = data.groupby(['batter'])[cluster_cols].rolling(depth,min_periods=depth_min,closed='left').sum()
        temp_data = data.groupby(['batter'])[weight_cols].rolling(depth,min_periods=depth_min,closed='left').sum()
        
        temp_df = temp_df.join(temp_data)
        
        for i in range(len(cluster_cols)):
            temp_df[col_names[i]] = temp_df[cluster_cols[i]] / temp_df[weight_cols[i]]
        
        temp_df = temp_df[col_names].reset_index(level=0)
        temp_df.replace([np.inf,-np.inf],0,inplace=True)
        temp_df = temp_df.set_index(['batter',temp_df.index])
        
        temp_df.dropna(how='all',inplace=True)
        return temp_df.fillna(0)
    
    # Creates a rolling tracker of pitch clusters thrown in atbat defining situations using the soft clustering results
    # mean is taken and the max & min are recorded
    def rolling_cluster_soft(self,data,depth_num,depth_min=3,depth_type="D"):
        data = data.copy()
        
        cluster_cols = [col for col in data.columns if 'cluster_attribute' in col and 'list' not in col]
        cluster_cols.extend(['pa','bb'])
        depth = str(depth_num) + depth_type
        
        data.index = pd.to_datetime(data.index)
        
        data = data.groupby(['pitcher',data.index])[cluster_cols].sum().reset_index(level=0)
        temp_data = data.groupby(['pitcher'], as_index=False).rolling(depth,min_periods=depth_min,closed='left')[cluster_cols].mean()
        
        col_names = []
        for col in temp_data.columns:
            if col in cluster_cols:
                col_names.append(col + '_roll')
            else:
                col_names.append(col)
        temp_data.columns = col_names
        temp_data = temp_data.reset_index(level=0)
        
        soft_cols = [col for col in temp_data.columns if '_roll' in col and 'bb_' not in col and 'k_' not in col and 'pa_' not in col]
        
        temp_data = temp_data.set_index(['pitcher',temp_data.index])
        
        temp_data = temp_data[soft_cols].dropna(how='all',subset=soft_cols)
        return temp_data.fillna(0)
    
    # creates a rolling tracker of pitch clusters thrown in atbat defining situations
    def rolling_cluster_hard(self,data,depth_num,depth_min=3,depth_type='D'):
        data = data.copy()
        
        cluster_cols = [col for col in data.columns if 'cluster' in col and 'attribute' not in col and 'list' not in col]
        cluster_cols.extend(['pa','bb'])
        depth = str(depth_num) + depth_type
        
        data.index = pd.to_datetime(data.index)
        
        data = data.groupby(['pitcher',data.index])[cluster_cols].sum().reset_index(level=0)
        temp_data = data.groupby(['pitcher'], as_index=False)[cluster_cols].rolling(depth,closed='left',min_periods=depth_min).sum()
        temp_data.reset_index(level=0)
        
        col_names = []
        for col in temp_data.columns:
            if col in cluster_cols:
                col_names.append(col + '_roll')
            else:
                col_names.append(col)
        temp_data.columns = col_names
        
        temp_data = self.hard_per_pa(temp_data)
        
        hard_cols = [col for col in temp_data.columns if 'per_pa' in col]
        temp_data = temp_data.set_index(['pitcher',temp_data.index])
        
        temp_data = temp_data[hard_cols].dropna(how='all',subset=hard_cols)
        return temp_data.fillna(0)

    # transforms clusters into per-pa instances
    def hard_per_pa(self,data):
        cluster_cols = [col for col in data.columns if 'cluster' in col and 'attribute' not in col and 'list' not in col and 'roll' in col]
        col_names = [item + '_per_pa' for item in cluster_cols]
        
        for i in range(len(col_names)):
            data[col_names[i]] = data[cluster_cols[i]] / (data['pa_roll'] + data['bb_roll'])
            
        return data
    
    # restricts columns in data to only necessary columns
    def data_focus(self,data):
        columns_used = [col for col in data if ('cluster' in col)]
        columns_used.extend(['batter', 'pitcher', 'bb', 'events', 'pa', 'estimated_ba_using_speedangle', 'estimated_ba_using_speedangle_list'])
        
        return data[columns_used]
    
    def type_set(self,data):
        int_cols = ['batter','pitcher','pitch_count','play','k','bb']
        float_cols = [col for col in data.columns if ('cluster' in col) & ('list' not in col)]
        float_cols = float_cols.append('estimated_ba_using_speedangle')
        
        data.loc[:,data.columns.isin(int_cols)] = data.loc[:,data.columns.isin(int_cols)].astype('int')
        data.loc[:,data.columns.isin(float_cols)] = data.loc[:,data.columns.isin(float_cols)].astype('float')
    
    
        
class PitcherPrep(GenericPrep):
    def __init__(self, target='estimated_ba_using_speedangle'):
        self.target = target
    
    def data_prep(self,data,depth=22,bin_size=7,roll_vars=['estimated_ba_using_speedangle','k_per_pa','bb_per_pa']):
        df = data[['estimated_ba_using_speedangle','bb','k','pa','pitcher','game_date']].copy()
        
        df = self.initial_clean(df)
        
        #print('pitcher columns: ' + str(df.columns))
        df = self.depth_finish(df,depth,bin_size,roll_vars)
        return df
    
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
        #data = self.shift_target(data)    
        self.per_pa(data)
        
        return data
    
    # rolling mean by date
    def rolling_data(self,data,roll_amount,target,roll_min):
        mean_name = target + '_mean_' + str(roll_amount)
        roll_amount = str(roll_amount) + 'D'
        
        temp_df = data.groupby('pitcher')[[target,'play']].rolling(roll_amount,min_periods=roll_min,closed='left').sum().reset_index(level=0,drop=False)
        temp_df.columns = ['pitcher', target, 'play']
        
        mask = (temp_df.play != 0)
        temp_df.loc[mask,mean_name] = temp_df.loc[mask,target] / temp_df.loc[mask,'play']
        
        temp_df.drop([target,'play'],axis = 1,inplace=True)
        
        return data.merge(temp_df, on=['game_date','pitcher'])
            
    # rolling mean weighted by plate appearances
    def rolling_weighted_data(self,data,roll_amount,target,roll_min):
        data['weighted_pa'] = data.pa
        data['weighted_target'] = data[target] * data.weighted_pa
        
        name = target + '_mean_weighted_' + str(roll_amount)
        roll_amount = str(roll_amount) + 'D'
        
        temp_df = data.groupby('pitcher')[['weighted_target','weighted_pa','bb']].rolling(roll_amount,min_periods=roll_min,closed='left').sum().reset_index(level=0,drop=False)
        
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
                    roll_min = int(i / bin_size)
                    temp_df = self.rolling_data(temp_df,i,item,roll_min)
                    temp_df = self.rolling_weighted_data(temp_df,i,item,roll_min)
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
        #print('batter columns: ' + str(data.columns))
        
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
            pd_out = pd.concat([pd_out,temp_df.asfreq('D')])
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
    def rolling_data(self,data,roll_amount,target,roll_min):
        mean_name = target + '_mean_' + str(roll_amount)
        roll_amount = str(roll_amount) + 'D'
        
        temp_df = data.groupby('batter')[[target,'play']].rolling(roll_amount,min_periods=roll_min).sum().reset_index(level=0,drop=False)
        temp_df.columns = ['batter', target, 'play']
        
        mask = (temp_df.play != 0)
        temp_df.loc[mask,mean_name] = temp_df.loc[mask,target] / temp_df.loc[mask,'play']
        
        temp_df.drop([target,'play'],axis = 1,inplace=True)
        
        return data.merge(temp_df, on=['game_date','batter'])
        
    # rolling mean weighted by plate appearances
    def rolling_weighted_data(self,data,roll_amount,target,roll_min):
        data['weighted_pa'] = data.pa
        data['weighted_target'] = data[target] * data.weighted_pa
        
        name = target + '_mean_weighted_' + str(roll_amount)
        roll_amount = str(roll_amount) + 'D'
        
        temp_df = data.groupby('batter')[['weighted_target','weighted_pa','bb']].rolling(roll_amount,min_periods=roll_min).sum().reset_index(level=0,drop=False)
        
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
                    roll_min = int((i / 5)+1)
                    temp_df = self.rolling_data(temp_df,i,item,roll_min)
                    temp_df = self.rolling_weighted_data(temp_df,i,item,roll_min)
        temp_df = self.lag_features(temp_df,depth)
        
        return temp_df