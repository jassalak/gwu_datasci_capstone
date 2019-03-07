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
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from pandas.plotting import scatter_matrix

from sklearn.neighbors import KNeighborsClassifier
#from statsmodels.discrete.discrete_model import Logit

################################################################################

### Data Load
df1_flu16 = pd.read_excel('FluShots20162017v2.xlsx',sheet_name='2016')
df1_flu16.columns.values[7] = "Pt Ins"
df1_flu16 = df1_flu16.drop(['Diagnosis Codes','Diagnosis Codes_1'], axis =1)
df1_flu16.rename(columns={'Diagnosis Codes_2':'Diagnosis Codes','Account Number':'AcctNum'}, inplace=True)
df1_flu16['year'] = 2016
df1_flu17 = pd.read_excel('FluShots20162017v2.xlsx',sheet_name='2017')
df1_flu17.rename(columns={'Pt Address 1':'Pt Address 2','CPT Description':'Pt Address 1','CPT.1':'CPT Description','Account Number':'AcctNum'}, inplace=True)
df1_flu17['year'] = 2017
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
df1.columns = df1.columns.str.upper()
#df1.to_excel("df1.xlsx")
df1 = pd.read_excel('df1.xlsx',sheet_name='Sheet1')
################################################################################

df2 = df1.copy(deep = False)
df2 = df1.drop(['ACCTNUM','PCP','LMG PRACTICE','DOS','DIAGNOSIS CODES','PLACE OF SERVICE','CPT', 'CPT DESCRIPTION','PT ADDRESS 1','PT ADDRESS 2', 'PT ZIP', 'PT DOB'], axis=1)
print(df2.info())


df2_qual = df2.select_dtypes(include=['object']).copy()
df2_quant = df2.select_dtypes(include=['int64']).copy()
df2_feats = list(df2)
print(df2.groupby('FLUDX_YES').mean() )

print(df2.isnull().sum())                                                       #Null check again
### Dropping Rows
df2.dropna(inplace=True)

for f in df2.columns:
    print(df2[f].value_counts())


print(df2.dtypes)


### DV Density
plt.figure(2); plt.title('Normal')
sns.distplot(df2['FLUDX_YES'], kde=False, fit=st.norm)

################################################################################
### Logistic Regression
df3 = df2_quant.copy(deep = False)

# prepare X and y
x = df3.drop(['FLUDX_YES','UNITS'],axis=1,inplace=False)
y = df3[['FLUDX_YES']]

X_train, X_test, Y_train, Y_test =tts(x, y, test_size = 0.3, random_state=5026)

logit = LogisticRegression()
result = logit.fit(Y_train,X_train)
print(result.summary())

logit_yhat = logit.predict(X_test)
logit_prob = logit.predict_proba(X_test)
logit_ci90 = (np.percentile(logit_prob[:,1],90))
logit_threshold = logit_ci90
logit_yhat = np.where(logit_prob[:,1] >= logit_threshold,1,0)


glm0_score =  round(metrics.accuracy_score(glm_Y, glm0_Yhat)*100,4)
print('\n Score glm0:', metrics.accuracy_score(glm_Y, glm0_Yhat) )
#print(' \n Intercept glm0: ',glm0.intercept_)
glm0_coef = pd.DataFrame(glm0.coef_[0], glm_X.columns, columns=['glm1_Coefficients,14'])
glm0_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(glm_Y, glm0_Yhat), columns=['predicted 0','predicted 1'], index =['actual 0','actual 1'] )
print('\n Confusion Matrix glm0: \n',glm0_confusion_matrix)

### Evaluation Metrics
tn = glm0_confusion_matrix.iloc[0,0]
fp = glm0_confusion_matrix.iloc[0,1]
fn = glm0_confusion_matrix.iloc[1,0]
tp = glm0_confusion_matrix.iloc[1,1]
sensitivity = tp/(tp+fn)*100                                                    #print(sensitivity) # this percent... of all True values, the model was able to predict
specificity = tn / (tn + fp) *100                                               #print(specificity) # this percent... of all False values, the model was able to predict



### Extra code

def corry(x, y, **kwargs):
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 10, xycoords = ax.transAxes)
# Create a pair grid instance
grid = sns.PairGrid(data= df2,vars = ['PCP SPECIALTY','PT INS','UNITS','PT CITY','PT STATE','PT GENDER','PT RACE','YEAR','FLUDX_YES'])
# Map the plots to the locations
grid = grid.map_upper(plt.scatter, color = 'darkred')
grid = grid.map_upper(corry)
grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
grid = grid.map_diag(plt.hist, bins = 10, edgecolor =  'k');


#https://www.datacamp.com/community/tutorials/categorical-data
#df3_fluDx.replace('',np.nan,inplace=True)
#df3_fluDx.dropna(inplace=True)
#df2_qual = [f for f in df2.columns if df2.dtypes[f] == 'object']
#df2_quant = [f for f in df2.columns if df2.dtypes[f] != 'object']

print(df2['PCP SPECIALTY'].value_counts())
print(df2['PT INS'].value_counts())
print(df2['PT STATE'].value_counts())
print(df2['PT CITY'].value_counts())

### Null Check (Quant Vars)
plt.figure(1)
df2_missing = df2.isnull().sum()
df2_missing = df2_missing[df2_missing >0]
df2_missing.sort_values(inplace=True)
plt.title("count of nulls")
if len(df2_missing) > 0:df2_missing.plot.bar()




























