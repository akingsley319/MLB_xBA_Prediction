# MLB_xBA_Prediction

## Introduction

The purpose of this projet is to perform expected Batting Average (xBA) prediction of MLB batters against a pitcher on some given day. This project is still in its initial stage of feature engineering using various machine learning models. 

The final project will involve predictions based on the recent performance of batters and pitchers, as well as matchup potential. This matchup potential will involve the use of Statcast pitch data, which involves 23 different metrics detailing the path of the pitched baseball, from the release point to when it crosses home plate. This data was standardized across all MLB ballparks in 2017.

## Project Use

It will be later that I will condense all the different files needed to run this project. The current design is for segmenting the project into understandable steps.

For a fresh file run (which will take a long time with the unoptimized code), run the "fresh_setup.py". If a player map for ids is desired - this will increase run time due to delay between runs (web scraping practice) - make sure to uncomment the line in the code.

The file "create_game_files.py" will retrieve the designated game files from the BaseballSavant API and download it all as a csv file. It will also sift through player ids and create a map of player names and player ids by scraping from each player's official MLB page.

The file "pitcher_cluster_run.py" will cluster pitcher pitches, providing a framework for matchup potential. This process will also clean the game files. Analysis of this process can be found in "cluster_analysis.ipynb".

"forecasting.py" prepares files (currently only batter) for modeling use.

"batter_performance_modeling.py" models and properly formats data for recent performance prediction.

"pitcher_performance_modeling.py" models and properly formats data for recent performance prediction.

## To Do List

This is the to-do list which should be completed by August 21st, 2022 for submission for my Master's Program for Data Science through Regis University.

* Provide feature engineering to clusters for matchup purposes and respresentation of pitcher repertoire
* Perform modeling of matchup result modeling
* Combination of created models
* Condense files that need to be run for less file clutter and better optimized use

## Resources

The framework for retrieving the data (get_data.py) was based on the code found at the following github location: https://github.com/alanrkessler/savantscraper/blob/master/savantscraper.py

I do not own any of the game file data, which is all publicly available from the BaseballSavant API and belongs to the MLB.

