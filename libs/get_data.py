# -*- coding: utf-8 -*-

"""Load detail level Baseball Savant data into an CSV file."""

"""https://github.com/alanrkessler/savantscraper/blob/master/savantscraper.py"""

import os
from time import sleep
from urllib.error import HTTPError
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
from pybaseball import statcast
import re

def savant_search(season, team, home_road, csv=False, sep=';'):
    """Return detail-level Baseball Savant search results.
    Breaks pieces by team, year, and home/road for reasonable file sizes.
    Args:
        season (int): the year of results.
        team (str): the modern three letter team abbreviation.
        home_road (str): whether the pitching team is "Home" or "Road".
        csv (bool): whether or not a csv
        sep (str): separat
    Returns:
        a pandas dataframe of results and optionally a csv.
    Raises:
        HTTPError: if connection is unsuccessful multiple times.
    """
    # Define the number of times to retry on a connection error
    num_tries = 6
    # Define the starting backoff time to grow exponentially
    pause_time = 30

    # Generate the URL to search based on team and year
    url = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true"
           "&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=&h"
           f"fC=&hfSea={season}%7C&hfSit=&player_type=pitcher&hfOuts=&opponent"
           "=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt="
           f"&hfInfield=&team={team}&position=&hfOutfield=&hfRO="
           f"&home_road={home_road}&hfFlag=&hfPull=&metric_1=&hfInn="
           "&min_pitches=0&min_results=0&group_by=name&sort_col=pitches"
           "&player_event_sort=pitch_number_thisgame&sort_order=desc"
           "&min_pas=0&type=details&")

    # Attempt to download the file
    # If unsuccessful retry with exponential backoff
    # If still unsuccessful raise HTTPError
    # Due to possible limit on access to this data
    for retry in range(0, num_tries):
        try:
            single_combination = pd.read_csv(url, low_memory=False)
        except HTTPError as connect_error:
            print("HTTP Error")
            if connect_error:
                if retry == num_tries - 1:
                    raise HTTPError
                else:
                    sleep(pause_time)
                    pause_time *= 2
                    continue
            else:
                break

    # Drop duplicate and deprecated fields
    single_combination.drop(['pitcher.1', 'fielder_2.1', 'umpire', 'spin_dir',
                             'spin_rate_deprecated', 'break_angle_deprecated',
                             'break_length_deprecated', 'tfs_deprecated',
                             'tfs_zulu_deprecated'], axis=1, inplace=True)

    # Optionally save as csv for loading to another file system
    if csv:
        single_combination.to_csv(f"{team}_{season}_{home_road}_detail.csv",
                                  index=False, sep=sep)

    return single_combination

def game_files(seasons, teams=None):
    """ Return a database of all desired information, designed to allow 
        flexibility while by default returning all information after the 2017
        season. This date was chosen due to differences in how pitch velocity
        was measured.
        Args:
            seasons (tuple): inclusive range of years to include
            teams (list): inclusive list of teams to include pitching from
        Returns:
            csv file titled "game_files.csv" that contains the data from the 
            specified parameters.
    """
    
    if teams is None:
        teams = ['LAA', 'HOU', 'OAK', 'TOR', 'ATL', 'MIL', 'STL',
                 'CHC', 'ARI', 'LAD', 'SF', 'CLE', 'SEA', 'MIA',
                 'NYM', 'WSH', 'BAL', 'SD', 'PHI', 'PIT', 'TEX',
                 'TB', 'BOS', 'CIN', 'COL', 'KC', 'DET', 'MIN',
                 'CWS', 'NYY']

    locations = ['home']
    
    if os.path.exists("data/game_files.csv"):
        os.remove("data/game_files.csv")
    
    # Creates and appends csv file titled "game_files.csv"
    for season in range(seasons[0],seasons[1]+1):
        print(str(season))
        for team in teams:
            for location in locations:
                single_combination = savant_search(str(season), team, location)
                
                if os.path.exists('data/game_files.csv'):
                    with open('data/game_files.csv', 'a') as f:
                        f.write(single_combination.to_csv(index=False, sep=';'))
                else:
                    single_combination.to_csv('./data/game_files.csv', index=False, 
                                              sep=';')
                    
def player_by_id(player_id):
    time.sleep(8)
    
    headers = {'User-Agent': 'This is my web scraping script, used for a school project; Contact me at akingsley@regis.edu'}
    url = "https://www.mlb.com/player/" + str(player_id)
    page = requests.get(url, headers = headers)
    
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="player-header")
    if results == None:
        return None
    else:
        job_elements = results.find_all("span", class_="player-header--vitals-name")
    
    if len(job_elements) >= 1:
        return job_elements[0].text
    else:
        return None

def player_map(player_ids):
    player_map = {}
    
    for player in player_ids:
        player_map[player] = player_by_id(player)
        
    return player_map


# Grabs all game data from 2017 through 2021 season for all teams
def game_file_creation(start_year=2017, end_year=2022):
    game_files(seasons=(start_year,end_year))
    
# Update existing game files; due to file size, not converted to pandas
def game_file_update(file_location="data/game_files.csv"):
    update_date = find_latest_data(file_location)
    start = update_date.strftime('%Y-%m-%d')
    now = datetime.now()
    end = now.strftime('%Y-%m-%d')
    new_data = statcast(start_dt=start, end_dt=end)
    new_data.drop(['pitcher.1', 'fielder_2.1', 'umpire', 'spin_dir',
                             'spin_rate_deprecated', 'break_angle_deprecated',
                             'break_length_deprecated', 'tfs_deprecated',
                             'tfs_zulu_deprecated'], axis=1, inplace=True)
    new_data.to_csv(file_location, mode='a', index=False, header=False,  sep=';')
    
    
def find_latest_data(file_location):
    with open(file_location,encoding="latin-1") as f:
        f.readline()
        first = f.readline().strip()
        start = first.find('2',2,15)
        end = start + 10
        print(first[int(start):int(end)])
        last_date = datetime.strptime(first[int(start):int(end)],'%Y-%m-%d')
        r = re.compile('.*-.*-.*')
        for line in f:
            start = line.find('2',2,15)
            end = start + 10
            date_str = line[int(start):int(end)]
            if r.match(date_str) is not None:
                date = datetime.strptime(date_str,'%Y-%m-%d')
                if date > last_date:
                    last_date = date
    print('last date before update: ' + last_date.strftime('%Y-%m-%d'))
    return last_date
 
# Maps player ids and player names
def player_map(df=None):
    if df == None:
        df = pd.read_csv('data/game_files.csv', sep=';', encoding="latin-1")
    
    player_list = list(df.batter.unique()) + list(df.pitcher.unique())
    
    full_list = []
    for player in player_list:
        if player not in full_list:
            full_list.append(player)
    
    player_mapping = player_map(full_list)
    
    with open('data/player_map.csv', 'w') as f:
        for key in player_mapping.keys():
            f.write("%s; %s\n" % (key, player_mapping[key]))