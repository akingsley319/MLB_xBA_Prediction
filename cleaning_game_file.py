# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:43:22 2022

@author: cubs1
"""

import pandas as pd
import numpy as np
import get_data as ga
import csv
import pickle

class Cleaner:
    def __init__(self, data, orig_data=None, ):
        self.data = data
        self.del_col_list = ['des','hit_location','inning_topbot',
                'hc_x','hc_y','sz_top','sz_bot','post_home_score',
                'post_away_score','post_bat_score','if_fielding_alignment',
                'of_fielding_alignment','delta_home_win_exp','delta_run_exp',
                'post_fld_score','hit_distance_sc', 'sv_id','launch_angle',
                'launch_speed','launch_speed_angle','woba_value','woba_denom',
                'estimated_woba_using_speedangle','iso_value']
        self.del_maybe = ['batter','pitcher']
        if orig_data == None:
            self.orig_data = self.data.copy()
        else:
            self.orig_data = orig_data
    
    def clean_data(self):
        self.remove_empty()
        self.clean_header()
        self.fill_events()
        self.des_info()
        self.pitcher_name()
        self.handle_game_date()
        self.remove_cols()
        self.simplify_events()
        self.base_runner_class()
    
    def clean_data_more(self):
        self.game_type()
        self.remove_from_events()
        self.pitch_spin_euc()
        self.pitch_fill()
        self.xba_k()
        
    def clean_new(self):
        self.clean_data()
        self.clean_data_more()
        
    def pitch_prep(self):
        self.clean_new()
        self.pitch_spin_euc()
        #self.pitch_data_clean()
        
    def fill_data(self):
        self.fill_ebus()
        self.fill_median()
        self.fill_median_by_year()
        self.fill_players()
        
    # Batter name fix
    # Pitch (type, release_speed, release_pos_x, release_pos_z)
    # zone ?
    # away team
    # pfx_x, pfx_z, plate_x, plate_z
    # vx0, vy0, vy0, ax, ay, az
    # effective_speed, release_spin_rate, release_extension, release_pos_y
    # pitch_name
    # spin_axis
    # bat_event ?
    # events (single, double, double_play, triple, triple_play)
    
    def remove_empty(self):
        self.data = self.data[self.data.game_date.notnull()].copy()

    
    # fill pitch metric data
    # currently fills backward based on pitcher, which is how atbats are in data; 
    # hope to replace with better method later
    def pitch_fill(self):
        cols = ['release_speed','release_pos_x','release_pos_z',
                 'pfx_x','pfx_z','plate_x','plate_z','vx0','vy0','vz0','ax',
                 'ay','az', 'effective_speed','release_spin_rate', 
                 'release_extension','release_pos_y','spin_x', 'spin_z']
        
        df = self.data.copy()
        
        # takes rows with nan in 'cols' and replaces all pitching metrics with None
        self.zero_rows(df,cols)
        df.reindex()
         
        df[cols] = df.groupby(['pitcher'])[cols].fillna(method='bfill')
        df[cols] = df.groupby(['pitcher'])[cols].fillna(method='ffill')
        
        self.data = df
        
    def zero_rows(self, data, cols):
        mask = data[cols].isna().any(axis=1)
        print(len(data.loc[mask,cols].index))
        data.loc[mask, cols] = None
        
        return data
    
    # apply pitcher clustering to data
    def pitch_data_clean(self):
        pitch_features = ['release_speed','release_pos_x','release_pos_z',
                 'pfx_x','pfx_z','plate_x','plate_z','vx0','vy0','vz0','ax',
                 'ay','az', 'effective_speed','release_spin_rate', 
                 'release_extension','release_pos_y','spin_x', 'spin_z']
        
        standardize = pickle.load(open('model.pkl', 'rb'))
        dimension_reduction = pickle.load(open('model.pkl', 'rb'))
        fuzzy_cluster = pickle.load(open('model.pkl', 'rb'))
        
        standardize.transform(self.data[pitch_features])
        dimension_reduction.transform(standardize)
        
        dim_cols = []
        for i in range(0,len(dimension_reduction[0])):
            temp_column = []
            
            for row in dimension_reduction:
                temp_column.append(row[i])
            
            column_name = 'attribute_' + str(i)
            self.data[column_name] = temp_column
            dim_cols.append(column_name)
        
        fuzzy_cluster.predict(self.data[dim_cols])
    
    # Orients pitch spin in same plane as other pitch stats
    def pitch_spin_euc(self):
        # Represents the spin rate on axis from 1st base to 3rd base
        self.data['spin_x'] = self.data[['spin_axis','release_spin_rate']].apply(lambda x: 
                    float(x.release_spin_rate) * np.sin(np.deg2rad(float(x.spin_axis))), 
                    axis=1)
        
        # Represents the spin rate on axis from pitcher mound to home plate
        self.data['spin_z'] = self.data[['spin_axis','release_spin_rate']].apply(lambda x: 
                    float(x.release_spin_rate) * np.cos(np.deg2rad(float(x.spin_axis))), 
                    axis=1)
            
    # retrieves player name based on player_id from data/player_map.csv or mlb.com and updates dataframe and data/player_map.csv
    def fill_players(self):
        full_list = self.unique_id()
        player_map = self.update_player_map(full_list)
        
        for player_id in full_list:
            self.data[self.data.pitcher == player_id]['pitcher_name'] = self.data[self.data.pitcher == player_id]['pitcher'].apply(lambda x: player_map[x])
            self.data[self.data.batter == player_id]['batter_name'] = self.data[self.data.batter == player_id]['batter'].apply(lambda x: player_map[x])
    
    def update_player_map(self, full_list):
        player_map = self.retrieve_player_map()
        missing_players = []
        
        for player_id in full_list:
            if (player_id not in list(player_map.keys()) or (player_map[player_id] == None)):
                missing_players.append(player_id)
        
        for player_id in missing_players:
            player_map[player_id] = ga.player_by_id(player_id)
        
        if len(missing_players) > 0:
            print(missing_players)
            self.write_to_player_map(player_map)
        
        return player_map
    
    def write_to_player_map(self, player_map):
        with open('data/player_map.csv', 'w') as f:
            f.truncate(0)
            for key in player_map.keys():
                f.write("%s; %s\n" % (key, player_map[key]))
            f.close()
    
    def retrieve_player_map(self):
        reader = open('data/player_map.csv', 'r+')
        lines = reader.readlines()
        
        result = {}
        for row in lines:
            entry = row.split('; ')
            if len(entry) != 2:
                reader.write(row)
                continue
            key = entry[0]
            if key in result:
                pass
            result[key] = entry[1]
            
        return result
    
    def unique_id(self):
        player_list = list(self.data.batter.unique()) + list(self.data.pitcher.unique())

        full_list = []
        for player in player_list:
            if player not in full_list:
                full_list.append(player)
            
        return full_list

    # removes game_advsory from events
    def remove_from_events(self):
        self.data.drop(self.data[self.data.events == 'game_advisory'].index, inplace=True)
    
    # focus on regular season games
    def game_type(self, games=['R']):
        self.data.drop(self.data[~self.data.game_type.isin(games)].index, inplace=True)
       
    # fill missing estimated_ba_using_speedangle with event and multiple imputation
    def fill_ebus(self, rem_list=['single','double','double_play','triple','triple_play']):
        median_events = list(self.data.events.unique())
        for item in rem_list:
            median_events.remove(item)
        self.replace_median('estimated_ba_using_speedangle', median_events)
        self.replace_median_bb_type('estimated_ba_using_speedangle', rem_list)
    
    # These values were removed model after this was written
# =============================================================================
#     # fill with median and event:  woba_denom, babip_value, iso_value
#     def fill_median(self, cols=['woba_denom', 'babip_value', 'iso_value']):
#         for feature in cols:
#             self.replace_median(feature, list(self.data.events.unique()))
#     
#     # fill with median by year and event: estimated_woba_using_speedangle, woba_value
#     def fill_median_by_year(self):
#         year_list = list(self.data.game_year.unique())
# 
#         self.replace_median('estimated_woba_using_speedangle', list(self.data.events.unique()), years=year_list)
#         self.replace_median('woba_value', list(self.data.events.unique()), years=year_list)
# =============================================================================
    
    # Median Fill base
    def create_median_table(self, df_temp, category, events_list):
        median_table = {}
        temp_df = df_temp[df_temp[category].notnull()]
        
        for event in events_list:
            median_val = np.median(temp_df[temp_df.events == event][category].astype(float))
            median_table[event] = median_val
            
        return median_table

    def replace_median(self, category, events_list, orig_data_l=None, years=None):        
        if orig_data_l == None:
            orig_data_l = self.orig_data
        
        def handle_median(event, default):
            if event in events_list and pd.isnull(default):
                med_val = median_table[event]
                return med_val.astype(float)
            else:
                return float(default)
            
        if years == None:
            self.data[category].astype('float')
            median_table = self.create_median_table(orig_data_l, category, events_list)
            self.data[category] = self.data.apply(lambda x: handle_median(x.events, x[category]), axis=1)
        else:
            self.data[category].astype('float')
            for year in years:
                median_table = self.create_median_table(orig_data_l[orig_data_l.game_year == year], category, events_list)
                self.data[self.data.game_year == year][category] = self.data[self.data.game_year == year].apply(lambda x: handle_median(x.events, x[category]), axis=1)   
    
    def replace_median_bb_type(self, category, events_list, orig_data_l=None):
        if orig_data_l == None:
            orig_data_l = self.orig_data
        
        def handle_median(event, default):
            if event in events_list and pd.isnull(default):
                med_val = median_table[event]
                return med_val.astype(float)
            else:
                return float(default)
        
        bb_type_list = self.data[self.data.bb_type.notnull()].bb_type.unique()
        for item in bb_type_list:
            self.data[category].astype('float')
            median_table = self.create_median_table(orig_data_l[orig_data_l.bb_type == item], category, events_list)
            self.data[self.data.bb_type == item][category] = self.data[self.data.bb_type == item].apply(lambda x: handle_median(x.events, x[category]), axis=1)
    
    #Removes headers that found their way into the dataset
    def clean_header(self):
        self.data.drop(self.data[self.data.game_date == 'game_date'].index, inplace=True)
     
    # Combine event and description 
    def fill_events(self):
        self.data["events"].fillna(self.data.description, inplace=True)

    # handle 'des' field to get batter name
    def des_info(self):
        self.data['des'].fillna(value='blank', inplace=True)
        
        des_seps = ['grounds','singles','doubles','triples','pops','hit',
            'lines','flies','walks','called','reaches','strikes',
            'homers','out','ground','intentionally']

        des_seps_s = ['2nd base.', '3rd base.', 'home.']

        des_seps_c = ['upheld: ', 'overturned: ']

        default_sep = '>>'
        
        self.data['batter_name'] = self.data['des'].apply(lambda x: self.separate_batter(x, des_seps, default_sep))
        self.data['batter_name'] = self.data['batter_name'].apply(lambda x: self.handle_steals(x, des_seps_s, default_sep))
        self.data['batter_name'] = self.data['batter_name'].apply(lambda x: self.remove_challenge(x, des_seps_c, default_sep))
        self.data['batter_name'] = self.data['batter_name'].apply(lambda x: self.steal_end_inn(x))
        self.data['batter_name'] = self.data.apply(lambda x: None if x.events == 'wild_pitch' 
                                     else x.batter_name, axis=1)
        self.data['batter_name'] = self.data.apply(lambda x: None if x.events == 'passed_ball' 
                                     else x.batter_name, axis=1)
        
    def separate_batter(self, des, seps, default_sep):
        for sep in seps:
            des = des.replace(sep,default_sep)
        
        output_list = [des.split(default_sep)][0]
        output = output_list[0]
    
        return output
   
    def handle_steals(self, des, seps, default_sep):
        for sep in seps:
            des = des.replace(sep,default_sep)
            
        output_list = [des.split(default_sep)]
        output = output_list[-1][-1]
        return output
        
    def steal_end_inn(self, des):
        if len(des) > 40:
            return None
        else:
            return des

    def remove_challenge(self, des, seps, default_sep):    
        for sep in seps:
            des = des.replace(sep,default_sep)
            
        output_list = [des.split(default_sep)]
        output = output_list[-1][-1]
            
        return output
    
    #Correctly format player_name to pitcher_name
    def pitcher_name(self):
        self.data['player_name'].fillna('unknown', inplace=True)
        self.data['pitcher_name'] = self.data['player_name'].apply(lambda x: self.orient_name(x))
        self.data['pitcher_name'] = self.data['pitcher_name'].apply(lambda x: None if x=='unknown' 
                                                  else x)
        self.data.drop('player_name', axis=1, inplace=True)
        
    def orient_name(self, name):
        name_sep = name.split(',')
    
        try:
            true_name = name_sep[1] + ' ' + name_sep[0]
        except:
            true_name = 'unknwon'
            
        return true_name
        
    #Separate game_date to year, month, and day columns and convert column
    def handle_game_date(self):
        self.data['game_year'] = self.data['game_date'].apply(lambda x: self.separate_date(x, 'year'))
        self.data['game_month'] = self.data['game_date'].apply(lambda x: self.separate_date(x, 'month'))
        self.data['game_day'] = self.data['game_date'].apply(lambda x: self.separate_date(x, 'day'))
        
        self.data['game_date'] = pd.to_datetime(self.data.game_date)
    
    def separate_date(self, date, time_cat):
        date_sep = date.split('-')
        
        if time_cat.lower() == 'year':
            return date_sep[0]
        if time_cat.lower() == 'month':
            return date_sep[1]
        if time_cat.lower() == 'day':
            return date_sep[2]
        
    # Turn base runner columns into binary classification
    def base_runner_class(self):
        self.data['on_1b'].fillna(0, inplace=True)
        self.data['on_2b'].fillna(0, inplace=True)
        self.data['on_3b'].fillna(0, inplace=True)
        
        self.data['on_1b'] = self.data['on_1b'].apply(lambda x: 1 if x!=0 else x)
        self.data['on_2b'] = self.data['on_2b'].apply(lambda x: 1 if x!=0 else x)
        self.data['on_3b'] = self.data['on_3b'].apply(lambda x: 1 if x!=0 else x)
        print('base_runner_class')
    
    # Drop unneeded, uninterested, and deprecated columns
    def remove_cols(self):
        for item in self.del_col_list:
            self.data.drop(item, inplace=True, axis=1)
            
    # Drop entries with no play (i.e. game advisory)
    # 'game_advisory' tag signals a delay of game or status change, no actual play
    def no_play(self):
        self.data.drop(self.data[self.data.events == 'game_advisory'].index, inplace=True)
        
    # Simplify events
    def simplify_events(self):
        self.data['bat_event'] = self.data['events'].apply(lambda x: self.hit_simp(x))
        
        self.data.drop('description', axis=1, inplace=True)
        self.data.drop('type', axis=1, inplace=True)
        #self.data.drop('bb_type', axis=1, inplace=True)
    
    def hit_simp(self, event):
        hit = ['single','double','home_run','triple']

        field_out = ['field_out','grounded_into_double_play','force_out','double_play',
                     'sac_bunt','sac_fly','fielders_choice_out','fielders_choice',
                     'sac_fly_double_play','triple_play','sac_bunt_double_play']
        
        runner_out = ['caught_stealing_3b','caught_stealing_2b','pickoff_2b',
                      'pickoff_caught_stealing_3b','caught_stealing_home',
                      'pickoff_caught_stealing_home','pickoff_caught_stealing_2b',
                      'runner_double_play','pickoff_3b','other_out','pickoff_1b']
        
        stolen_base = ['stolen_base_2b','stolen_base_home','stolen_base_3b']
        
        free_base = ['hit_by_pitch','walk','catcher_interf']
        
        strikeout = ['strikeout','strikeout_double_play']
        
        strike = ['called_strike','swinging_strike','swinging_strike_blocked',
                  'foul_bunt','missed_bunt','bunt_foul_tip','swinging_pitchout']
        
        ball = ['ball','blocked_ball','pitchout','wild_pitch','passed_ball']
        
        foul = ['foul_tip','foul','foul_pitchout']
        
        error = ['field_error']
        
        if event in hit:
            return 'hit'
        if event in field_out:
            return 'field_out'
        elif event in runner_out:
            return 'runner_out'
        elif event in stolen_base:
            return 'stolen_base'
        elif event in free_base:
            return 'free_base'
        elif event in strikeout:
            return 'strikeout'
        elif event in strike:
            return 'strike'
        elif event in ball:
            return 'ball'
        elif event in foul:
            return 'foul'
        elif event in error:
            return 'error'
        else:
            return None
        
    def xba_k(self):
        mask = self.data['events'].isin(['strikeout','strikeout_double_play'])
        self.data.loc[mask,'estimated_ba_using_speedangle'] = 0