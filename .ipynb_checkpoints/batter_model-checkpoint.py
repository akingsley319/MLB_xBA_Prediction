# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 21:19:44 2022

@author: cubs1
"""

import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic

class Batter:
    def __init__(self, data):
        self.data = data
       
    # Heirarchical Clustering
    # LSTM
    # VAR
    # VARMA
    # VARMAX
