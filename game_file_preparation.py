# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:24:14 2022

@author: cubs1
"""

import pandas as pd

from random import sample


class GamePrep:
    def __init__(self, data, game_condensed_list=None, weight=1000.0):
        self.data = data.sort_values(['game_pk','pitcher','batter'])
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
    
    def return_pitchers(self):
        pass
    
    def return_batters(self):
        df_out = pd.DataFrame()
        counter = 0
        max_counter = len(self.game_condensed_list)
        for game in self.game_condensed_list:
            game_df = self.consolidate_games(game,'batter')
            df_out = df_out.append(game_df)
            
            counter += 1
            print(str(counter) + "/" + str(max_counter))
        df_out = df_out.loc[:, ~df_out.columns.str.contains('^Unnamed')]
        return df_out
    
    # Consolidates all atbats for a single game
    def consolidate_games(self, game, player_type='batter'):
        game_db = pd.DataFrame()
        
        if player_type == 'batter':
            player_list = game.batter.unique()
        elif player_type == 'player':
            player_list = game.pitcher.unique()
        
        for player in player_list:
            if player_type == 'batter':
                game_temp = game[game.batter == player]
            elif player_type == 'player':
                game_temp = game[game.pitcher == player]
            
            #print(game_temp.head(20))
            
            walks = 0
            values = {}
            
            for col in game_temp.columns:
                if 'attribute' in col:
                    mean_val = game_temp[col].mean()
                    values[col] = mean_val
                    
                elif col == 'estimated_ba_using_speedangle':
                    mean_val = game_temp[col].mean()
                    values[col] = mean_val
                    
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
                    
                    values[col] = game_temp.events.unique()
                    
                elif col == 'pitch_count':
                    total = game_temp[col].sum()
                    values[col] = total
                    
                else:
                    val_temp = game_temp[col].unique()[0]
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
                dict_atbats = self.consolidate_atbat(atbat,weight)
                atbats_out = atbats_out.append(dict_atbats, ignore_index=True)
                
            games_out.append(atbats_out)
            
            count += 1
            print(str(count) + "/" + str(max_count))
            if count == 20:
                print(atbats_out['estimated_ba_using_speedangle'])
        return games_out
        
    
    # takes separated atbat (separate_atbat) and consolidates atbat
    def consolidate_atbat(self, data, weight):
        values = {}
        attr_col = []
        attr_val = 0
        
        pitch_count = int(data['attribute_1'].count())
        
        for col in data.columns:
            
            if 'attribute' in col:
                attr_col.append(col)
                
                attr_sum = []
                for i in range(0,pitch_count):
                    if i == pitch_count - 1:
                        attr_sum.append(weight)
                    else:
                        attr_sum.append(1)
                        
                weighted_sum =  (data.loc[:,col] * list(attr_sum)).sum()
                
                attr_val += weighted_sum
                values[col] = weighted_sum
                
            elif col == 'estimated_ba_using_speedangle':
                val = data.estimated_ba_using_speedangle[data.events.first_valid_index()]
                values[col] = val
                
            elif col == 'events':
                val = data.events.unique()[0]
                values[col] = data.events.unique()[0]
            
            else:
                val = data[col].unique()[0]
                
                values[col] = val
        
        for col in attr_col:
            values[col] = values[col] / attr_val
            
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
        games = [self.data[self.data.game_pk == id].copy() for id in self.data.game_pk.unique()]
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
    def train_test_by_year(self, file_name_train, file_name_test, year_sens=500):
        temp_df = self.data[self.data.game_year.notna()]
        recent_year_list = list(temp_df['game_year'].unique())
        
        recent_year = recent_year_list[-1]
        
        found = False
        
        while not found:
            print(recent_year)
            if self.data[self.data.game_year == recent_year]['game_pk'].count() >= year_sens:
                found = True
            else:
                recent_year -= 1
             
        train_set = self.data[self.data.game_year < recent_year]
        test_set = self.data[self.data.game_year == recent_year]
        
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