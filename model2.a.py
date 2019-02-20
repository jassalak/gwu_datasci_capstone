# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:06:29 2019

@author: akash
"""
################################################################################
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

################################################################################

### Data Load
df1_flu16 = pd.read_excel('FluShots20162017v2.xlsx',sheet_name='2016')
df1_flu16.columns.values[7] = "Pt Ins"
df1_flu16 = df1_flu16.drop(['Diagnosis Codes','Diagnosis Codes_1'], axis =1)
df1_flu16.rename(columns={'Diagnosis Codes_2':'Diagnosis Codes','Account Number':'AcctNum'}, inplace=True)
df1_flu17 = pd.read_excel('FluShots20162017v2.xlsx',sheet_name='2017')
df1_flu17.rename(columns={'Pt Address 1':'Pt Address 2','CPT Description':'Pt Address 1','CPT.1':'CPT Description','Account Number':'AcctNum'}, inplace=True)

df1_flu1617 = pd.concat([df1_flu16,df1_flu17], sort = False)
df1_flu1617 = df1_flu1617[pd.to_numeric(df1_flu1617['AcctNum'], errors='coerce').notnull()]
df1_flu1617['AcctNum'] = pd.to_numeric(df1_flu1617['AcctNum'])


df1_fluDx = pd.read_excel('DiagnosisReport_20162017.xlsx',sheet_name='PT1005_pat_diagnosis_list.rpt')
df2_fluDx = df1_fluDx.copy(deep = True)
df2_fluDx.rename(columns={'Patient account':'AcctNum'}, inplace=True)
df2_fluDx = df2_fluDx[['AcctNum']]
df2_fluDx = df2_fluDx.dropna(how = 'all')
df2_fluDx['AcctNum'] = df2_fluDx['AcctNum'].str.split('-').str[0]
df2_fluDx = df2_fluDx[pd.to_numeric(df2_fluDx['AcctNum'], errors='coerce').notnull()]
df2_fluDx['AcctNum'] = pd.to_numeric(df2_fluDx['AcctNum'])
df2_fluDx['FluDx_YES'] = df2_fluDx.notnull().all(1).astype(int)
df2_fluDx =df2_fluDx.reset_index(drop = True)

df1 = pd.merge(df1_flu1617,df2_fluDx,left_on = 'AcctNum',right_on = 'AcctNum', how = 'left')
df1['FluDx_YES'].replace(np.nan,'0',inplace=True)
df1['FluDx_YES'] = pd.to_numeric(df1['FluDx_YES'])

################################################################################

### DV Density
plt.figure(2); plt.title('Normal')
sns.distplot(df1['FluDx_YES'], kde=False, fit=st.norm)

### Extra code
#df3_fluDx.replace('',np.nan,inplace=True)
#df3_fluDx.dropna(inplace=True)
