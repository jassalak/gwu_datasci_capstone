# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 10:12:51 2019

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
## DataLoad and Global Filtering

#dir_base = 'C:\\Users\\akash\\Desktop\\GWU\\capstone\\data'
#filename = 'flushots_2016.xls'
#sheetname = 'StatisticsICD10.rpt'
#def load_clean_file(filename,sheetname):                                       # creates a function that reads/opens a file
#    input_file_text = pd.read_excel(filename,sheetname)
#    input_file_text.copy(deep = False)
#    input_file_text.dropna(how = 'all')
#    input_file_text.drop(['Unnamed: 4','Unnamed: 5'], axis =1)
#    input_file_text.replace('',np.nan,inplace=True)
#    input_file_text.dropna(inplace=True)
#    return input_file_text
#df3_flushots16 = load_clean_file(filename,sheetname)

########################################
### Data Load
df1_flu16 = pd.read_excel('flushots_2016_edited1.xls',sheet_name='StatisticsICD10.rpt')
df1_flu17 = pd.read_excel('flushots_2017_edited1.xls',sheet_name='StatisticsICD10.rpt')
df1 = pd.concat([df1_flu16,df1_flu17])
df1.rename(columns={'Account Number':'AcctNum'}, inplace=True)

### Null Removal & Trimming
df2 = df1.copy(deep = False)
df2 = df2.dropna(how = 'all')
df2 = df2.drop(['Unnamed: 4','Unnamed: 5'], axis =1)
df2['AcctNum'] = df2['AcctNum'].str.split('-').str[0]

### Null Check (Quant Vars)
plt.figure(1)
df2_missing = df2.isnull().sum()
df2_missing = df2_missing[df2_missing >0]
df2_missing.sort_values(inplace=True)
plt.title("count of nulls")
if len(df2_missing) > 0:df2_missing.plot.bar()
# ^^ Need to remove HEADERS!!!!

### Null Management
df3 = df2.copy(deep = False)
#df3.replace('',np.nan,inplace=True)
#df3.dropna(inplace=True)

###Additional Datasources
df1_fluDx = pd.read_excel('DiagnosisReport_20162017.xls',sheet_name='PT1005_pat_diagnosis_list.rpt')
df1_fluDx.rename(columns={'Patient account':'AcctNum','DOB':'Bday'}, inplace=True)
df2_fluDx = df1_fluDx.copy(deep = False)
df2_fluDx = df2_fluDx.dropna(how = 'all')
df2_fluDx['AcctNum'] = df2_fluDx['AcctNum'].str.split('-').str[0]
df3_fluDx = df2_fluDx.copy(deep = False)
df3_fluDx.replace('',np.nan,inplace=True)
df3_fluDx.dropna(inplace=True)
df3_fluDx['FluShotReceived'] = df3_fluDx.notnull().all(1).astype(int)

### Merging Datsets
df4_left = pd.merge(df3,df3_fluDx,left_on = 'AcctNum',right_on = 'AcctNum', how = 'left')
df4_right = pd.merge(df3,df3_fluDx,left_on = 'AcctNum',right_on = 'AcctNum', how = 'right')
df4_inner = pd.merge(df3,df3_fluDx,left_on = 'AcctNum',right_on = 'AcctNum', how = 'inner')
df4_outer = pd.merge(df3,df3_fluDx,left_on = 'AcctNum',right_on = 'AcctNum', how = 'outer')

########################################
# DATA TIDYING

### Create More Managable DataFrame
df4 = df4_left.copy(deep = False)
df4['FluShotReceived'].replace(np.nan,'0',inplace=True)
df4_cols = list(df4)
df5 = df4.copy(deep = False)
df5 = df5.drop(['AcctNum','DOB','Chart','Bday','Zip','ICD Date'], axis = 1)

################################################
# ENCODING
### Encoding Qualitative (Dummy Variables)

#df5 = pd.get_dummies(df5)
# ^^ Need to fix.  Bin Ages!!

### Determines quantitative and qualitative columns
df5_qual = [f for f in df5.columns if df5.dtypes[f] == 'object']
df5_quant = [f for f in df5.columns if df5.dtypes[f] != 'object']


################################################

#EDA & FEATURE SELECTION

### DV Density
y = df5['FluShotReceived']
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)

### Correlation Matrix
#### Quantiative
plt.figure(3)
#corr = df2[df2_quant+['SalePrice']].corr()
corr = df2[df2_quant].corr()
sns.heatmap(corr)

plt.figure(4)
f = pd.melt(df2, value_vars=df2_quant)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")

################################################

