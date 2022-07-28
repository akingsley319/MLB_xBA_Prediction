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
        data.loc[mask,'bb_per_pa'] = data.loc[mask,'bb'] / data.loc[mask,'pa']
        
        data[['k_per_pa','bb_per_pa']] = data[['k_per_pa','bb_per_pa']].fillna(0)


class MatchupPrep(GenericPrep):
    def __init__(self):
        pass
    
        
class PitcherPrep(GenericPrep):
    def __init__(self,metric_cols=['pitcher','next_estimated_ba_using_speedangle','pa', 'pitch_count','play','next_play','k','bb','estimated_ba_using_speedangle','k_per_pa','bb_per_pa','day_of_week','month','year']):
        self.metric_cols = metric_cols
        self.initial_drop = drop_columns = ['Unnamed: 0', 'pitch_type', 'release_speed', 'release_pos_x', 'release_pos_z', 'batter', 'events', 'zone', 'game_type', 'stand', 'p_throws', 'home_team', 'away_team', 'bb_type', 'balls', 
                'strikes', 'game_year', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'fielder_2', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'effective_speed', 
                'release_spin_rate', 'release_extension', 'game_pk', 'fielder_3', 'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9', 'release_pos_y', 'babip_value', 'at_bat_number',
                'pitch_number', 'pitch_name', 'home_score', 'away_score', 'bat_score', 'fld_score', 'spin_axis', 'batter_name', 'pitcher_name', 'game_month', 'game_day', 'bat_event', 'spin_x', 'spin_z', 
                'attribute_0', 'attribute_1', 'attribute_2', 'attribute_3', 'attribute_4', 'attribute_5', 'attribute_6', 'attribute_7', 'attribute_8']
    
    def data_prep(self,data,depth=18,bin_size=6,roll_vars=['estimated_ba_using_speedangle','k_per_pa','bb_per_pa']):
        df = data.copy()
        df.drop(self.initial_drop,axis=1,inplace=True)
        
        df = self.initial_clean(df)
        df = df[self.metric_cols]
        return self.depth_finish(df,depth,bin_size,roll_vars)
    
    def fill_dates(self,data):
        pd_out = pd.DataFrame()
        
        for pitcher in data.pitcher.unique():
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
        data['weighted_pa'] = data.pa - data.bb
        data['weighted_target'] = data[target] * data.weighted_pa
        
        name = target + '_mean_weighted_' + str(roll_amount)
        
        temp_df = data.groupby('pitcher')[['weighted_target','weighted_pa']].rolling(roll_amount).sum().reset_index(level=0,drop=False)
        
        mask = (temp_df.weighted_pa != 0)
        temp_df.loc[mask,name] = temp_df.loc[mask,'weighted_target'] / temp_df.loc[mask,'weighted_pa']
        
        temp_df.drop(['weighted_target','weighted_pa'],axis = 1,inplace=True)
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
        
        return data
    
    # Correctly set dtype
    def type_set(self, data):
        data[self.target] = data['estimated_ba_using_speedangle'].astype('float')
        data.loc[:,~data.columns.isin(['estimated_ba_using_speedangle'])] = data.loc[:,~data.columns.isin(['estimated_ba_using_speedangle'])].astype('int')
        
        return data
        
    # Fills missing days between batter's first and last appearance
    def fill_dates(self, data):
        pd_out = pd.DataFrame()
        
        for batter in data.batter.unique():
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
        data['weighted_pa'] = data.pa - data.bb
        data['weighted_target'] = data[target] * data.weighted_pa
        
        name = target + '_mean_weighted_' + str(roll_amount)
        
        temp_df = data.groupby('batter')[['weighted_target','weighted_pa']].rolling(roll_amount).sum().reset_index(level=0,drop=False)
        
        mask = (temp_df.weighted_pa != 0)
        temp_df.loc[mask,name] = temp_df.loc[mask,'weighted_target'] / temp_df.loc[mask,'weighted_pa']
        
        temp_df.drop(['weighted_target','weighted_pa'],axis = 1,inplace=True)
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
        

class GamePrep:
    def __init__(self, data, game_condensed_list=None, weight=1000.0):
        self.data = data.sort_values(['game_date','pitcher','batter'])
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
            pitcher_list = game.loc[:,'batter'].unique()
            batter_list = game.loc[:,'pitcher'].unique()
            
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
                    values[col] = game_temp['game_date'].mode()
                    
                else:
                    val_temp = list(game_temp[col])
                    values[col] = val_temp
                
            game_db = game_db.append(values, ignore_index=True)

        return game_db
    
    """
    The below code returns a list of games in which the atbats are consolidated
    into a single entry. For this done on original data, see self.df_condensed.
    For further use, the data must be designated as for batter or pitcher, and 
    condensed into singular game information.
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
        
        pitch_count = int(data['attribute_1'].count())
        
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
                
# =============================================================================
#             elif col == 'estimated_ba_using_speedangle':
#                 mask = data['estimated_ba_using_speedangle'].notna()
#                 val = data.loc[mask,data.columns.isin(['estimated_ba_using_speedangle'])][::-1]
#                 values[col] = val
# =============================================================================
            
            else:
                val = data[col].unique()[::-1]
                
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