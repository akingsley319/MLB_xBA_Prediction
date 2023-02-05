# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 13:49:49 2023

@author: cubs1
"""

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
