# MLB_xBA_Prediction

## Introduction

The purpose of this project is to perform expected Batting Average (xBA) prediction of MLB batters against a pitcher on some given day. This project is still in its initial stage of feature engineering using various machine learning models. 

The final project will involve predictions based on the recent performance of batters and pitchers, as well as matchup potential. This matchup potential will involve the use of Statcast pitch data, which involves 23 different metrics detailing the path of the pitched baseball, from the release point to when it crosses home plate. This data was standardized across all MLB ballparks in 2017.

## Project Use


For a fresh file run (which will take a long time with the unoptimized code), run the "fresh_setup.py". If a player map for ids is desired - this will increase run time due to delay between runs (web scraping practice) - make sure to uncomment the line in the code.

The ability for project updating will be included at a later date. The code is currently focusing on achieving results with the data currently held for it.

## To Do List

This is the to-do list which should be completed by August 21st, 2022 for submission for my Master's Program for Data Science through Regis University.

* Provide feature engineering to clusters for matchup purposes and respresentation of pitcher repertoire 
* Perform modeling of matchup result modeling
* Combination of created models
* Condense files that need to be run for less file clutter and better optimized use

## Resources

The framework for retrieving the data (get_data.py) was based on the code found at the following github location: https://github.com/alanrkessler/savantscraper/blob/master/savantscraper.py

I do not own any of the game file data, which is all publicly available from the BaseballSavant API and belongs to the MLB.

