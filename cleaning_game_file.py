# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:43:22 2022

@author: cubs1
"""

import pandas as pd
import numpy as np

class Cleaner:
    def __init__(self, data, orig_data=None):
        self.data = data
        self.del_col_list = ['des','hit_location','inning_topbot',
                'hc_x','hc_y','sz_top','sz_bot','post_home_score',
                'post_away_score','post_bat_score','if_fielding_alignment',
                'of_fielding_alignment','delta_home_win_exp','delta_run_exp',
                'post_fld_score','hit_distance_sc', 'sv_id',
                'launch_angle','launch_speed','launch_speed_angle']
        self.del_maybe = ['estimated_woba_using_speedangle','woba_value',
                'woba_denom','iso_value','batter','pitcher']
        if orig_data == None:
            self.orig_data = self.data.copy()
        else:
            self.orig_data = orig_data
    
    def clean_data(self):
        self.clean_header()
        self.fill_events()
        self.des_info()
        self.pitcher_name()
        self.handle_game_date()
        self.remove_cols()
        self.simplify_events()
        self.base_runner_class()
    
    def clean_new(self):
        self.clean_data()
        self.game_type()
        self.fill_ebus()
        self.fill_median()
        self.fill_median_by_year()
    # Batter name (https://www.mlb.com/player/player_id from data source)
    # Pitch (type, release_speed, release_pos_x, release_pos_z)
    # zone ?
    # away team
    # pfx_x, pfx_z, plate_x, plate_z
    # vx0, vy0, vy0, ax, ay, az
    # effective_speed, release_spin_rate, release_extension, release_pos_y
    # pitch_name
    # spin_axis
    # bat_event ?
    # fix game_advisory
    
    # focus on regular season games
    def game_type(self, games=['R']):
        self.data.drop(self.data[~self.data.game_type.isin(games)].index, inplace=True)
       
    # fill missing estimated_ba_using_speedangle with event and multiple imputation
    def fill_ebus(self, rem_list=['single','double','double_play','triple','triple_play']):
        median_events = list(self.data.events.unique())
        for item in rem_list:
            median_events.remove(item)
        self.replace_median('estimated_ba_using_speedangle', median_events)
        # handle non-median missing data
    
    # fill with median and event:  woba_denom, babip_value, iso_value
    def fill_median(self, cols=['woba_denom', 'babip_value', 'iso_value']):
        for feature in cols:
            self.replace_median(feature, list(self.data.events.unique()))
    
    # fill with median by year and event: estimated_woba_using_speedangle, woba_value
    def fill_median_by_year(self):
        year_list = list(self.data.game_year.unique())

        self.replace_median('estimated_woba_using_speedangle', list(self.data.events.unique()), years=year_list)
        self.replace_median('woba_value', list(self.data.events.unique()), years=year_list)
    
    # Median Fill base
    def create_median_table(self, df_temp, category, events_list):
        median_table = {}
        temp_df = df_temp[df_temp[category].notnull()]
        
        for event in events_list:
            median_val = np.median(temp_df[temp_df.events == event][category].astype(float))
            median_table[event] = median_val
            
        return median_table

    def replace_median(self, category, events_list, years=None):        
        
        def handle_median(event, default):
            if event in events_list and pd.isnull(default):
                med_val = median_table[event]
                return med_val.astype(float)
            else:
                return float(default)
            
        if years == None:
            self.data[category].astype('float')
            median_table = self.create_median_table(self.orig_data, category, events_list)
            self.data[category] = self.data.apply(lambda x: handle_median(x.events, x.estimated_ba_using_speedangle), axis=1)
        else:
            self.data[category].astype('float')
            for year in years:
                median_table = self.create_median_table(self.orig_data[self.orig_data.game_year == year], category, events_list)
                self.data[self.data.game_year == year][category] = self.data[self.data.game_year == year].apply(lambda x: handle_median(x.events, x.estimated_ba_using_speedangle), axis=1)   
    
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