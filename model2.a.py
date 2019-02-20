# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:06:29 2019

@author: akash
"""
########################################
# ENVIRONMENT PREP
import os

### Provide the path here
os.chdir('C:\\Users\\akash\\Desktop\\GWU\\6501_Capstone\\data')


### Basic Packages
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from pandas.plotting import scatter_matrix

from sklearn.neighbors import KNeighborsClassifier

########################################