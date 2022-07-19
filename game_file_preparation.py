# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:24:14 2022

@author: cubs1
"""

import pandas as pd

from random import sample

class PitcherPrep:
    def __init__(self):
        pass

class BatterPrep:
    def __init__(self, target='estimated_ba_using_speedangle'):
        self.target = target
    
    def data_prep(self, data, depth=15):
        data = data[['estimated_ba_using_speedangle','bb','k','pa','batter','game_date']].copy()
        data.fillna(0, inplace=True)
        
        self.play(data)
        data = self.game_date_to_index(data)
        self.fill_dates(data)
        self.date_info(data)
        self.type_set(data)
        self.shift_target(data)
        self.per_pa(data)
        data = self.depth_features(data,depth)
        
        return data
    
    def data_clean(self,data):
        self.remove_cols(data)
        data = self.remove_rows(data)
        
        return data
    
    # Correctly set dtype
    def type_set(self, data):
        data[self.target] = data[self.target].astype('float')
        data.loc[:,~data.columns.isin([self.target])] = data.loc[:,~data.columns.isin([self.target])].astype('int')
    
    # Introduce play column to model
    # States whether or not a batter played that day
    def play(self, data):
        data['play'] = data['pa'].apply(lambda x: 1 if x>0 else 0)
        
    # Move game date to index
    def game_date_to_index(self, data):
        data = data.sort_values('game_date')
        data['game_date'] = pd.to_datetime(data['game_date'])
        data.set_index('game_date',inplace=True,drop=True)
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
    
    # Introduces seasonality components
    def date_info(self,data):
        day_of_week_idx = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
        
        data['day_of_week'] = data.index.day_name()
        data['day_of_week'] = data.day_of_week.replace(day_of_week_idx)
        
        data['month'] = data.index.month
        data['year'] = data.index.year    
    
    # Shifts variables that will be available prior, and the target variable
    def shift_target(self,data):
        data[['next_estimated_ba_using_speedangle','next_play','day_of_week','month','year']] = data.groupby('batter')[['estimated_ba_using_speedangle','play','day_of_week','month','year']].shift(-1)
        data.dropna(inplace=True)
        
    # Converts columns to per plate appearance
    def per_pa(self,data):
        mask = (data.pa != 0)
        data.loc[mask,'k_per_pa'] = data.loc[mask,'k'] / data.loc[mask,'pa']
        data.loc[mask,'bb_per_pa'] = data.loc[mask,'bb'] / data.loc[mask,'pa']
        
        data[['k_per_pa','bb_per_pa']] = data[['k_per_pa','bb_per_pa']].fillna(0)
        
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
    
    # removes columns determined not significant
    def remove_cols(self,data):
        for col in data.columns:
            if ('days_off' in col) or ('play' in col):
                if col != 'next_play':
                    data.drop(col,axis=1,inplace=True)
                    
    # Removes empty and null rows
    def remove_rows(self,data):
        data.dropna(axis=0,inplace=True)
        
        mask = (data.next_play != 0)
        
        return data.loc[mask,:]
    

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
        df_out['pa'] = df_out['events'].apply(lambda x: len(x))
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