# -*- coding: utf-8 -*-
"""
Created on Wed May 25 21:03:55 2022

@author: cubs1
"""

import pandas as pd

from random import sample


class GamePrep:
    def __init__(self, data):
        self.data = data
        self.event_def = ['single','double','home_run','triple','field_out',
                      'grounded_into_double_play','force_out','double_play',
                     'sac_bunt','sac_fly','fielders_choice_out','fielders_choice',
                     'sac_fly_double_play','triple_play','sac_bunt_double_play',
                     'strikeout','strikeout_double_play','field_error','walk',
                     'hit_by_pitch','catch_interf']
    
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
    
    # Cleans game files with regard to all pitcher stats by game 
    def consolidate_game_pitcher(self, game_dict=None, weight=2.0):
        if game_dict == None:
            game_dict = self.return_atbats(weight)
            
        game_db = pd.DataFrame()
        
        for year in game_dict.keys():
            for game in game_dict[year]:
                game_temp = game.groupby('pitcher')
                values = {}
                
                for col in game_temp.columns():
                    if 'attribute' in col:
                        values[col] = game_temp[col].mean()
                        
                    elif col == 'expected_ba_using_speedangle':
                        values[col] = game_temp[col].mean()
                        
                    elif col == 'events':
                        values[col] = list(game_temp.events.unique())[0]
                    
                    elif col == 'pitch_count':
                        values[col] = game_temp[col].sum()
                        
                    else:
                        values[col] = game_temp[col].notnull().unique()[0]
                        
                game_db = game_db.append(values)
                
        return game_db
    
    # Returns batter prepared files
    # xba_calc: average="full expected batting average", hit="chance of getting at least one hit"
    # weight: the emphasis placed on the final pitch of the atbat
    def game_prepared_batter(self, batter, xba_calc='average', days=1, date=None, game_db=None, weight=2.0):
        if game_db == None:
            game_db = self.consolidate_game_batter(weight)
        if date == None:
            date = game_db.game_date.max()
          
        def xba_hit_calc(x):
            calc_val = 0
            
            for entry in x:
                calc_val = calc_val * (1 - float(entry))
                
            return 1-calc_val
          
        aggregate_col = {}
        for col in game_db:
            if 'attribute' in col:
                aggregate_col[col] = 'mean'
            if col == 'estimated_ba_using_speedangle':
                if xba_calc == 'average':
                    aggregate_col[col] = 'mean'
                elif xba_calc == 'hit':
                    aggregate_col[col] = lambda x: xba_hit_calc(pd.Series.tolist(x))
            if col == 'pitch_count':
                aggregate_col[col] = 'sum'
            else:
                aggregate_col[col] = lambda x: ','.join(sorted(pd.Series.unique(x)))
        
        bin_size = str(days) + 'D'
        game_out = game_db[game_db.batter == batter].copy().resample(bin_size, 'game_date').agg(aggregate_col)
        
        return game_out
    
    # Cleans game files with regard to all batter stats by game and returns dataset
    def consolidate_game_batter(self, game_dict=None, weight=2.0):
        if game_dict == None:
            game_dict = self.return_atbats(weight)
            
        game_db = pd.DataFrame()
        
        for year in game_dict.keys():
            for game in game_dict[year]:
                for batter in game.bater.unique():
                    game_temp = game[game.batter == batter]
                    walks = 0
                    values = {}
                    
                    for col in game_temp.columns():
                        if 'attribute' in col:
                            values[col] = game_temp[col].mean()
                            
                        elif col == 'expected_ba_using_speedangle':
                            values[col] = game_temp[col].mean()
                            
                        elif col == 'events':
                            values['bb'] = game_temp.events.value_counts()['walk']
                            values[col] = list(game_temp.events.unique())[0]
                            
                        elif col == 'pitch_count':
                            values[col] = game_temp[col].sum()
                            
                        else:
                            values[col] = game_temp[col].notnull().unique()[0]
                        
                        game_db = game_db.append(values)
                
        return game_db
    
    # Returns pitcher prepared files
    def game_prepared_pitcher(self, pitcher, days=5, date=None, game_db=None, weight=2.0):
        if game_db == None:
            game_db = self.consolidate_game_pitcher(weight)
        if date == None:
            date = game_db.game_date.max()
            
        aggregate_col = {}
        for col in game_db:
            if 'attribute' in col:
                aggregate_col[col] = 'mean'
            if col == 'expected_ba_using_speedangle':
                aggregate_col[col] = 'mean'
            if col == 'pitch_count':
                aggregate_col[col] = 'sum'
            else:
                aggregate_col[col] = lambda x: ','.join(sorted(pd.Series.unique(x)))
        
        bin_size = str(days) + 'D'
        game_out = game_db[game_db.pitcher == pitcher].copy().resample(bin_size, 'game_date').agg(aggregate_col)
        
        return game_out
    
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
        
    def return_atbats(self, weight=2.0):
        games_out = {}
        
        games_sep = self.separate_games()
        
        for year in games_sep.keys():
            games_out[year] = []
            for game in games_sep[year]:
                atbats_sep = self.separate_atbat(game)
                game_df_temp = pd.DataFrame()
                
                for atbat in atbats_sep:
                    atbat_cons = self.consolidate_atbat(atbat, weight)
                    game_df_temp = game_df_temp.append(atbat_cons, ignore_index=True)
                    
                games_out[year].append(game_df_temp.copy())
            
        return games_out
        # separate_games (returns games grouped in dictionary by year in list by game_pk)
        # separate_atbat (returns list of grouped atbats)
        # consolidate_atbat (returns dictionary of atbat)
        
    def separate_games(self):
        years = self.data[self.data.game_year.notna()]['game_year'].unique()
        
        games_grouped = {}
        for year in years:
            games_grouped[year] = [self.data[self.data.game_year == year][self.data.game_pk == x] for x in self.data[self.data.game_year == year].game_pk.unique()]
            
        return games_grouped
    
    def separate_atbat(self, data):
        mask = data['events'].isin(self.event_def)
        
        data['atbat_game_id'] = (mask).cumsum()
        
        atbats = []
        for i in data.atbat_game_id.unique():
            atbats.append(data[data.atbat_game_id == i])
            
        return atbats
    
    # consolidates at-bats, including pitch data and xBA
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
                
            elif col == 'expected_ba_using_speedangle':
                attr_sum = []
                for i in range(0,pitch_count):
                    if i == pitch_count - 1:
                        attr_sum.append(weight)
                    else:
                        attr_sum.append(1)
                        
                weighted_sum =  (data.loc[:,col] * list(attr_sum)).sum(axis=1)
                
                attr_val += weighted_sum
                values[col] = weighted_sum
                
            elif col == 'events':
                values[col] = list(data.events.unique())[0]
            
            else:
                values[col] = data[col].notnull().unique()[0]
        
        for col in attr_col:
            values[col] = values[col] / attr_val
            
        values['pitch_count'] = pitch_count
        
        return values
        # cumsum on 0-0 count for forward, events for backwards
        # combine pitch data (sum?)
    
    # consolidates 5 game stretch for pitcher prediction
    def consolidate_pitcher(self):
        # combine pitch_data (same as at-bats and game)
        # combine xBA
        # variable days?
        pass
    
    def sum_pitches(self, df):
        sum_pitches = {}
        
        for attr in df.columns:
            sum_pitches[attr] = df[attr].sum()
        
        return sum_pitches
    
    def combine_xba(self, df):
        pass