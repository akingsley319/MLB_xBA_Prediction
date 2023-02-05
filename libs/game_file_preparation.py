# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:24:14 2022

@author: cubs1
"""

import pandas as pd
import numpy as np

from random import sample

def train_test_split(df,pitch_limit=100,year_sens=1000):
    prep = TrainPrep(df)
    train_set_year, test_set_year = prep.train_test_by_year('data/train/train_data.csv', 'data/test/test_data.csv', year_sens)
    train_setting_year = prep.pitch_limiter(train_set_year.copy(), pitch_limit)
    return train_set_year, test_set_year, train_setting_year

# Transforms game data into useable data for models
class GamePrep:
    def __init__(self, data):
        self.event_def = ['single','double','home_run','triple','field_out',
             'grounded_into_double_play','force_out','double_play',
             'sac_bunt','sac_fly','fielders_choice_out','fielders_choice',
             'sac_fly_double_play','triple_play','sac_bunt_double_play',
             'strikeout','strikeout_double_play','field_error','walk',
             'hit_by_pitch','catch_interf']
        self.data = data
        
    def atbat_cluster(self, data):
        attr_cols = []
        for col in data:
            if 'cluster_attribute' in col:
                attr_cols.append(col)
        cluster_col = ['cluster_' + str(i) for i in range(len(attr_cols))]
        data[cluster_col] = data[attr_cols].apply(lambda x: x == x.max(), axis=1).astype(int) 
        return data
        
    def return_matchups(self):
        data = self.consolidate_atbat(self.data)
        data = self.atbat_cluster(data)
        return self.consolidate_games(data,'both')
        
    def return_batters(self):
        data = self.consolidate_atbat(self.data)
        return self.consolidate_games(data,'batter')
        
    def return_pitchers(self):
        data = self.consolidate_atbat(self.data)
        data = self.atbat_cluster(data)
        return self.consolidate_games(data,'pitcher')
    
    # consolidates atbats into useable data
    def consolidate_atbat(self,data):
        mask = data['events'].isin(self.event_def)
        
        data['atbat_game_id'] = (mask).cumsum()
        data['pitch_count'] = data.groupby('atbat_game_id')['atbat_game_id'].count()
        
        atbats = data.groupby('atbat_game_id').first()
            
        return atbats
    
    # player_batter: 'batter', 'pitcher', 'both'
    # make sure to add hard cluster first
    # consolidates games into useable data
    def consolidate_games(self,data, pitcher_batter='batter'):
        data = self.data_handling(data,pitcher_batter)
        agg_dict = self.agg_dict_def(data.columns,pitcher_batter)
        if pitcher_batter == 'both':
            return data.groupby(['game_date', 'pitcher', 'batter']).agg(agg_dict)
        elif pitcher_batter=='pitcher' or pitcher_batter=='batter':
            return data.groupby(['game_date', pitcher_batter]).agg(agg_dict)
        else:
            raise ValueError
            
    # defines operations for consolidate_games()
    def agg_dict_def(self,list_vars,pitcher_batter):
        sum_var = ['bb','k','pitch_count','pa']
        count_var = []
        first_var = ['game_date']
        mean_var = ['estimated_ba_using_speedangle']
        if pitcher_batter=='both':
            first_var.append('pitcher')
            first_var.append('batter')
        else:
            first_var.append(pitcher_batter)
        for col in list_vars:
            if 'team' in col:
                first_var.append(col)
            elif 'list' in col:
                # data_handling() already formatted into list
                first_var = first_var.append(col)
            elif 'max' in col:
                # data_handling() already found max
                first_var = first_var.append(col)
            elif 'min' in col:
                # data_handling() already found min
                first_var = first_var.append(col)
            elif 'attribute' in col:
                mean_var = mean_var.append(col)
            elif 'cluster' in col:
                sum_var = sum_var.append(col)
        agg_dict = {}
        for item in sum_var:
            agg_dict[item] = 'sum'
        for item in count_var:
            agg_dict[item] = 'count'
        for item in first_var:
            agg_dict[item] = 'first'
        for item in mean_var:
            agg_dict[item] = 'mean'
        return agg_dict
    
    # formats atbat data into data to be used for day consolidation
    def data_handling(self,data, pitcher_batter):
        data['k'] = data['events'].apply(lambda x: 1 if x in ['strikeout','strikeout_double_play'] else 0)
        data['bb'] = data['events'].apply(lambda x: 1 if x in ['walk'] else 0)
        data['pa'] = 1
        if pitcher_batter == 'both':
            grouped = data.groupby(['game_date', 'pitcher', 'batter'])
        elif pitcher_batter=='pitcher' or pitcher_batter=='batter':
            grouped = data.groupby(['game_date', pitcher_batter])
        else:
            raise ValueError
        for col in data.columns:
            if 'attribute' in col and 'cluster' in col:
                data[col+'_max'] = grouped[col].max()
                data[col+'_min'] = grouped[col].min()
                data[col+'_list'] = grouped[col].apply(list)
            elif 'attribute' in col or 'cluster' in col:
                data[col+'_list'] = grouped[col].apply(list)
            #elif col == 'estimated_ba_using_speedangle' or col == 'events':
            #    data[col+'_list'] = grouped[col].apply(list)
        return data

# Used for prepping data prior to training it    
class TrainPrep:
    def __init__(self, data):
        self.data = data
        self.event_def = ['single','double','home_run','triple','field_out',
                      'grounded_into_double_play','force_out','double_play',
                     'sac_bunt','sac_fly','fielders_choice_out','fielders_choice',
                     'sac_fly_double_play','triple_play','sac_bunt_double_play',
                     'strikeout','strikeout_double_play','field_error','walk',
                     'hit_by_pitch','catch_interf']
        
    # Splits dataset into train and test set
    def train_test_by_year(self, file_name_train, file_name_test, year_sens=1000):
        temp_df = self.data[self.data.game_year.notna()]
        recent_year_list = list(temp_df['game_year'].unique())
        
        recent_year = recent_year_list[-1]
        
        found = False
        
        while not found:
            #print(recent_year)
            mask = self.data.game_year == recent_year
            #print(self.data.loc[mask,'game_pk'].nunique())
            if self.data.loc[mask,'game_pk'].nunique() >= year_sens:
                found = True
            else:
                recent_year -= 1
             
        self.data['game_date'] = pd.to_datetime(self.data['game_date'], format='%Y-%m-%d', errors='coerce')
        train_set = self.data[self.data.game_date.dt.year < int(recent_year)]
        test_set = self.data[self.data.game_date.dt.year == int(recent_year)]
        
        train_set.to_csv(file_name_train)
        test_set.to_csv(file_name_test)
        
        return train_set, test_set
    
    # Returns only pitchers in data set with a certain number of pitches thrown
    def pitch_limiter(self, data=None, pitch_limit=100):
        if data is not None:
            data = self.remove_nulls(data)
        else:
            data = self.remove_nulls(self.data)
        
        temp_df = pd.DataFrame()
        
        for pitcher in data.pitcher.unique():
            for game_year in data.game_year.unique():
                
                data_temp = data[(data.pitcher==pitcher) & (data.game_year==game_year)].copy()
                
                if len(data_temp.index) >= pitch_limit:
                    temp_df = temp_df.append(data_temp)
                    # data.drop(data[data_temp].index, inplace=True)
                    
        return temp_df
    
    # Returns selection of pitchers based on list of attributes [player_id, game_year]
    def return_df(self, array_players, pitcher=True):
        temp_df = pd.DataFrame()
        counter = 0
        max_counter = len(array_players)
        for entry in array_players:
            if pitcher == True:
                df_app = self.data[(self.data.pitcher == entry[0]) & (self.data.game_year == entry[1])]
            elif pitcher == False:
                df_app = self.data[(self.data.batter == entry[0]) & (self.data.game_year == entry[1])]
            temp_df = temp_df.append(df_app)
            counter += 1
            print(str(counter) + '/' + str(max_counter))
        return temp_df
        
    
    def remove_nulls(self, data):
        for column in data.columns:
            data.drop(data[data[column].isna()].index, inplace=True)
        return data.reindex()
       
"""
class GamePrep:
    def __init__(self, data, game_condensed_list=None, weight=1000.0):
        self.data = data.sort_values(['game_date','pitcher','batter'])
        self.data = self.data[self.data.game_date.notna()].copy()
        self.weight = weight
        self.event_def = ['single','double','home_run','triple','field_out',
                      'grounded_into_double_play','force_out','double_play',
                     'sac_bunt','sac_fly','fielders_choice_out','fielders_choice',
                     'sac_fly_double_play','triple_play','sac_bunt_double_play',
                     'strikeout','strikeout_double_play','field_error','walk',
                     'hit_by_pitch','catch_interf']
        
        if game_condensed_list is not None:
            self.game_condensed_list = game_condensed_list
        else:
            self.game_condensed_list = self.consolidated_game_files(self.data, self.weight)
            
        print(self.game_condensed_list[-1])
        self.game_condensed_list.pop()
    
    # add hard cluster component to pitches
    def atbat_cluster(self, data):
        attr_cols = []
        
        for col in data:
            if 'cluster_attribute' in col:
                attr_cols.append(col)
        cluster_col = ['cluster_' + str(i) for i in range(len(attr_cols))]
        
        data[cluster_col] = data[attr_cols].apply(lambda x: x == x.max(), axis=1).astype(int) 
        
        return data
    
    def return_matchups(self):
        df_out = pd.DataFrame()
        counter = 0
        max_counter = len(self.game_condensed_list)
        for game in self.game_condensed_list:
            game_df = self.atbat_cluster(game)
            game_df = self.consolidate_games(game_df,'both')
            df_out = df_out.append(game_df)
            
            counter += 1
            print(str(counter) + "/" + str(max_counter))
        df_out = df_out.loc[:, ~df_out.columns.str.contains('Unnamed')]
        df_out['pa'] = df_out['events'].apply(lambda x: len([event for event in x if event != 'walk']))
        return df_out
    
    def return_pitchers(self):
        df_out = pd.DataFrame()
        counter = 0
        max_counter = len(self.game_condensed_list)
        for game in self.game_condensed_list:
            game_df = self.atbat_cluster(game)
            game_df = self.consolidate_games(game_df,'pitcher')
            df_out = df_out.append(game_df)
            
            counter += 1
            print(str(counter) + "/" + str(max_counter))
        df_out = df_out.loc[:, ~df_out.columns.str.contains('Unnamed')]
        df_out['pa'] = df_out['events'].apply(lambda x: len([event for event in x if event != 'walk']))
        return df_out
    
    def return_batters(self):
        df_out = pd.DataFrame()
        counter = 0
        max_counter = len(self.game_condensed_list)
        for game in self.game_condensed_list:
            game_df = self.consolidate_games(game,'batter')
            df_out = df_out.append(game_df)
            
            counter += 1
            print(str(counter) + "/" + str(max_counter))
        df_out = df_out.loc[:, ~df_out.columns.str.contains('Unnamed')]
        df_out['pa'] = df_out['events'].apply(lambda x: len([event for event in x if event != 'walk']))
        return df_out
    
    # Consolidates all atbats for a single game
    def consolidate_games(self, game, player_type='batter'):
        game_db = pd.DataFrame()
        if player_type == 'batter':
            player_list = game.loc[:,'batter'].unique()
        elif player_type == 'pitcher':
            player_list = game.loc[:,'pitcher'].unique()
        elif player_type == 'both':
            batter_list = game.loc[:,'batter'].unique()
            pitcher_list = game.loc[:,'pitcher'].unique()
            
            player_list = [(pitcher,batter) for pitcher in pitcher_list for batter in batter_list]
        
        for player in player_list:
#            print(player)
            if player_type == 'batter':
                game_temp = game[game.batter == player]
            elif player_type == 'pitcher':
                game_temp = game[game.pitcher == player]
            elif player_type == 'both':
                game_temp = game[(game.pitcher==player[0]) & (game.batter==player[1])]
                
#            print(game_temp.head(20))
            if game_temp.empty:
                continue
            
            values = {}
            
            for col in game_temp.columns:
                if 'attribute' in col:
                    mean_val = game_temp[col].mean()
                    values[col] = mean_val
                    
                    if 'cluster' in col:
                        values[col + '_max'] = game_temp[col].max()
                        values[col + '_min'] = game_temp[col].min()
                        
                        values[col + '_list'] = list(game_temp[col])
                    
                elif col == 'estimated_ba_using_speedangle':
                    mean_val = game_temp[col].mean()
                    values[col] = mean_val
                    values[col + '_list'] = list(game_temp[col])
                    
                elif col == 'events':
                    unique_vals = game_temp.events.unique()
                    val_counts = game_temp.events.value_counts()
                    
                    if 'walk' in unique_vals:
                        values['bb'] = val_counts['walk']
                    else:
                        values['bb'] = 0
                     
                    if 'strikeout' in unique_vals:
                        values['k'] = val_counts['strikeout']
                    else:
                        values['k'] = 0  
                    if 'strikeout_double_play' in unique_vals:
                        values['k'] += val_counts['strikeout_double_play']
                    else:
                        values['k'] += 0
                    
                    values['events'] = game_temp.events.tolist()
                    
                elif col == 'pitch_count':
                    total = game_temp[col].sum()
                    values[col] = total
                    
                elif 'cluster' in col:
                    values[col] = game_temp[col].sum()
                    values[col + '_list'] = list(game_temp[col])
                    
                elif col == 'game_date':
                    values[col] = game_temp['game_date'].unique()[0]
                    
                elif col == 'batter':
                    values[col] = game_temp['batter'].unique()[0]
                           
                elif col == 'pitcher':
                    values[col] = game_temp['pitcher'].unique()[0]
                    
                elif 'team' in col:
                    values[col] = game_temp[col].unique()[0]
                    
                else:
                    val_temp = list(game_temp[col])
                    values[col] = val_temp
                
            game_db = game_db.append(values, ignore_index=True)

        return game_db
    
    """
    """
    The below code returns a list of games in which the atbats are consolidated
    into a single entry. For this done on original data, see self.df_condensed.
    For further use, the data must be designated as for batter or pitcher, and 
    condensed into singular game information.
    """
    """
    
    # returns dataset of consolidated atbats
    # For binning games, this might be where to do it
    def consolidated_game_files(self, data, weight):
        games = self.separate_games()
        
        games_out = []
        
        max_count = len(games)
        count = 0
        for game in games:
            atbats = self.separate_atbats(game)
            atbats_out = pd.DataFrame()
            
            for atbat in atbats:
                atbat = atbat.loc[:,atbat.columns.notna()]
                dict_atbats = self.consolidate_atbat(atbat,weight)
                atbats_out = atbats_out.append(dict_atbats, ignore_index=True)
                
            games_out.append(atbats_out)
            
            count += 1
            print('atbats')
            print(str(count) + "/" + str(max_count))
#            if count == 20:
#                print(atbats_out['estimated_ba_using_speedangle'])
        return games_out
        
    
    # takes separated atbat (separate_atbat) and consolidates atbat
    def consolidate_atbat(self, data, weight):
        values = {}
#        attr_col = []
#        attr_val = 0
        
        pitch_count = len(data['attribute_1'])
        
        for col in data.columns:
            
            if 'attribute' in col:
                values[col] = data[col].iat[-1]
#                attr_col.append(col)
#                
#                attr_sum = []
#                for i in range(0,pitch_count):
#                    if i == pitch_count - 1:
#                        attr_sum.append(weight)
#                    else:
#                        attr_sum.append(1)
#                        
#                weighted_sum =  (data.loc[:,col] * list(attr_sum)).sum()
#                
#                attr_val += weighted_sum
#                values[col] = weighted_sum
                
            elif col == 'estimated_ba_using_speedangle':
                val = data.estimated_ba_using_speedangle[data.events.first_valid_index()]
                values['estimated_ba_using_speedangle'] = val
                
            elif col == 'events':
                val = data.events.unique()[0]
                values[col] = data.events.unique()[0]
                
            elif col == 'batter':
                values['batter'] = data.batter.unique()[0]
                
            elif col == 'pitcher':
                values['pitcher'] = data.pitcher.unique()[0]
                
            elif col == 'game_date':
                values['game_date'] = data.game_date.unique()[0]
                
# =============================================================================
#             elif col == 'estimated_ba_using_speedangle':
#                 mask = data['estimated_ba_using_speedangle'].notna()
#                 val = data.loc[mask,data.columns.isin(['estimated_ba_using_speedangle'])][::-1]
#                 values[col] = val
# =============================================================================
            
            else:
                val = list(data[col].unique())[-1]
                
                values[col] = val
        
#        for col in attr_col:
#            values[col] = values[col] / attr_val
            
        values['pitch_count'] = pitch_count
        
        return values
    
    # takes game and separates atbats
    def separate_atbats(self, data):
        mask = data['events'].isin(self.event_def)
        
        data['atbat_game_id'] = (mask).cumsum()
        
        atbats = []
        for i in data.atbat_game_id.unique():
            atbats.append(data[data.atbat_game_id == i].copy())
            
        return atbats
    
    # returns list of all games in data
    def separate_games(self):
        games = [self.data[self.data.game_date == date].copy() for date in self.data.game_date.unique()]
        print("games separated")
        return games
"""    