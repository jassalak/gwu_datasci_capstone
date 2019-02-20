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
df2 = df2[pd.to_numeric(df2['AcctNum'], errors='coerce').notnull()]

### Null Check (Quant Vars)
plt.figure(1)
df2_missing = df2.isnull().sum()
df2_missing = df2_missing[df2_missing >0]
df2_missing.sort_values(inplace=True)
plt.title("count of nulls")
if len(df2_missing) > 0:df2_missing.plot.bar()

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
df3_fluDx['FluDx_YES'] = df3_fluDx.notnull().all(1).astype(int)
df3_fluDx = df3_fluDx[pd.to_numeric(df3_fluDx['AcctNum'], errors='coerce').notnull()]


### Merging Datsets
df4_left = pd.merge(df3,df3_fluDx,left_on = 'AcctNum',right_on = 'AcctNum', how = 'left')
df4_right = pd.merge(df3,df3_fluDx,left_on = 'AcctNum',right_on = 'AcctNum', how = 'right')
df4_inner = pd.merge(df3,df3_fluDx,left_on = 'AcctNum',right_on = 'AcctNum', how = 'inner')
df4_outer = pd.merge(df3,df3_fluDx,left_on = 'AcctNum',right_on = 'AcctNum', how = 'outer')

# ^6 left shoudl be 51429 rows
########################################
# DATA TIDYING

### Create More Managable DataFrame
df4 = df4_left.copy(deep = False)
df4['FluDx_YES'].replace(np.nan,'0',inplace=True)
df4['FluDx_YES'] = pd.to_numeric(df4['FluDx_YES'])

df4['Gender_Female'] = df4['Gender'].map({'F':1,'M':0})
df4['Gender_Male'] = df4['Gender'].map({'M':1,'F':0})

## Convert DOB to Age! Remove over 100??

df4['Age_Bin']= pd.cut(df4['Age'],[0,18,35,55,80,105], labels = ['Child','YoungAdult','MiddleAge','Old','Oldest'] )

df4_cols = list(df4)
df5 = df4.copy(deep = False)
df5 = df5.drop(['AcctNum','DOB','Chart','Bday','Zip','ICD Date','Gender','Sex'], axis = 1)

################################################

################################################

#EDA & FEATURE SELECTION

### Determines quantitative and qualitative columns
print(df5.dtypes)
df5_qual = [f for f in df5.columns if df5.dtypes[f] == 'object']
df5_quant = [f for f in df5.columns if df5.dtypes[f] != 'object']

### DV Density
plt.figure(2); plt.title('Normal')
sns.distplot(df5['FluDx_YES'], kde=False, fit=st.norm)

### Correlation Matrix
#### Quantiative
plt.figure(3)
#corr = df5[df5_quant+['FluShotReceived']].corr()
sns.heatmap(df5[df5_quant].corr())

#plt.figure(4)
#f = pd.melt(df5, value_vars=df5_quant)
#g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
#g = g.map(sns.distplot, "value")


df5_corr = df5.corr()
df5_corr.sort_values(["FluShotReceived"], ascending = False, inplace = True)
print(df5_corr.FluShotReceived)

################################################

