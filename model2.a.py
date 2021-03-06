# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:47:02 2019

@author: akash
"""

################################################################################
# ENVIRONMENT PREP

### Basic Packages

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from sklearn import metrics
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import datetime as DT
from imblearn.over_sampling import SMOTE
from collections import Counter

from pandas.plotting import scatter_matrix
from datetime import datetime
from datetime import date
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from statsmodels.discrete.discrete_model import Logit

### Provide the path here
os.chdir('C:\\Users\\akash\\Desktop\\GWU\\6501_Capstone\\data')
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

present_date = pd.Timestamp(DT.datetime.now())
relative_date = DT.datetime(2019,12,31)
df1_flu1617['Pt DOB'] = pd.to_datetime(df1_flu1617['Pt DOB'], format='%m%d%y')
df1_flu1617['Pt DOB'] = df1_flu1617['Pt DOB'].where(df1_flu1617['Pt DOB'] < relative_date, df1_flu1617['Pt DOB'] -  np.timedelta64(100, 'Y'))
df1_flu1617['Pt_Age'] = (relative_date - df1_flu1617['Pt DOB']).astype('<m8[Y]')

df1_fluDx = pd.read_excel('DiagnosisReport_20162017.xlsx',sheet_name='PT1005_pat_diagnosis_list.rpt')
df1_fluDx = df1_fluDx.copy(deep = True)
df1_fluDx.rename(columns={'Patient account':'AcctNum'}, inplace=True)
df1_fluDx = df1_fluDx[['AcctNum']]
df1_fluDx = df1_fluDx.dropna(how = 'all')
df1_fluDx['AcctNum'] = df1_fluDx['AcctNum'].str.split('-').str[0]
df1_fluDx = df1_fluDx[pd.to_numeric(df1_fluDx['AcctNum'], errors='coerce').notnull()]
df1_fluDx['AcctNum'] = pd.to_numeric(df1_fluDx['AcctNum'])
df1_fluDx['FluDx_YES'] = df1_fluDx.notnull().all(1).astype(int)
df1_fluDx =df1_fluDx.reset_index(drop = True)

df1 = pd.merge(df1_flu1617,df1_fluDx,left_on = 'AcctNum',right_on = 'AcctNum', how = 'left')
df1['FluDx_YES'].replace(np.nan,'0',inplace=True)
df1['FluDx_YES'] = pd.to_numeric(df1['FluDx_YES'])
df1.columns = df1.columns.str.upper()

#df1.to_excel("df1.xlsx")
df1 = pd.read_excel('df1.xlsx',sheet_name='Sheet1')

################################################################################

df1b = df1.copy(deep = False)
df1b = df1.drop(['ACCTNUM','PCP','LMG PRACTICE','DOS','DIAGNOSIS CODES',
        'PLACE OF SERVICE', 'CPT DESCRIPTION','PT ADDRESS 1',
        'PT ADDRESS 2','PT CITY', 'PT DOB', 'UNITS', 'YEAR'], axis=1) #'PT ZIP'
print(df1b.shape)


print(df1b.isnull().sum())                                                       #Null check again
### Dropping Rows
df1b.dropna(inplace=True)

### Value Counts
for f in df1b.columns:
    print(df1b[f].value_counts())

################################################################################

# Lower Limit Thresholds
lt_pt_state = 90
lt_pcp_speciality = 35
lt_pt_ins = 14
lt_pt_race = 30
lt_cpt = 15
lt_zip = 2.8

### Combining Variables into Other

df1b['AGE_BIN']= pd.cut(df1b['PT_AGE'],[0,7,17,54,106], right = True, labels = ['Baby','Child','Adult','Senior'] )
# (0,7]|(7,17]|(17,54]|(54,106]

series = pd.value_counts(df1b['PT STATE'])
mask = (series/series.sum() * 100)                                              # To replace df['column'] use np.where I.e
mask = (series/series.sum() * 100).lt(lt_pt_state)                                        # lt(%); where % is the cut off
df1b['PT STATE'] = np.where(df1b['PT STATE'].isin(series[mask].index),'Other',df1b['PT STATE'])

series = pd.value_counts(df1b['PCP SPECIALTY'])
mask = (series/series.sum() * 100)
mask = (series/series.sum() * 100).lt(lt_pcp_speciality)
df1b['PCP SPECIALTY'] = np.where(df1b['PCP SPECIALTY'].isin(series[mask].index),'Other',df1b['PCP SPECIALTY'])

series = pd.value_counts(df1b['PT INS'])
mask = (series/series.sum() * 100)
mask = (series/series.sum() * 100).lt(lt_pt_ins)
df1b['PT INS'] = np.where(df1b['PT INS'].isin(series[mask].index),'Other',df1b['PT INS'])

df1b['PT RACE'].replace('_R','Other',inplace=True)                               # DOES NOT WORK WELL, DUPE OF 210 and 202
series = pd.value_counts(df1b['PT RACE'])
mask = (series/series.sum() * 100)
mask = (series/series.sum() * 100).lt(lt_pt_race)
df1b['PT RACE'] = np.where(df1b['PT RACE'].isin(series[mask].index),'Other',df1b['PT RACE'])

series = pd.value_counts(df1b['CPT'])
mask = (series/series.sum() * 100)
mask = (series/series.sum() * 100).lt(lt_cpt)
df1b['CPT'] = np.where(df1b['CPT'].isin(series[mask].index),'Other',df1b['CPT'])

series = pd.value_counts(df1b['PT ZIP'])
mask = (series/series.sum() * 100)
mask = (series/series.sum() * 100).lt(lt_zip)
df1b['PT ZIP'] = np.where(df1b['PT ZIP'].isin(series[mask].index),'Other',df1b['PT ZIP'])

new = series[~mask]
new['Other'] = series[mask].sum()
series.index = np.where(series.index.isin(series[mask].index),'Other',series.index)

#df1b.to_excel("df1b.xlsx")

################################################################################
df2 = df1b.copy(deep = True)

df2 = df2.drop(['PT RACE','PT_AGE', 'CPT'],axis =1)
df2_feats = list(df2)

df2 = pd.get_dummies(df2,columns = ['PT GENDER','PT STATE','PCP SPECIALTY','PT INS','PT ZIP','AGE_BIN'], prefix = ['Gndr','State','Spclty','Ins','Zip','Age'])
print(df2.dtypes)

print( 'qualitive variables', df2.select_dtypes(include=['object']).copy() )
print( 'quantative variables',  df2.select_dtypes(include=['int64','uint8']).copy().head(10) )

print(df2.groupby('FLUDX_YES').mean() )

df2_feats = list(df2)

#df2.to_excel("df2.xlsx")
################################################################################

#df4= StandardScaler().fit_transform(df3)
#df4 = MinMaxScaler().fit_transform(df3)
#df4 = pd.DataFrame(df4,columns = df3.columns).copy()


### df2 -- regular and smote


df2_x = df2.drop(['FLUDX_YES'],axis = 1 ).astype(float)
df2_y = pd.DataFrame(df2['FLUDX_YES'].astype(float))

df2_x_train, df2_x_test, df2_y_train, df2_y_test = tts(df2_x, df2_y, test_size=0.3, random_state=5026)

print("Shape of x_train dataset: ", df2_x_train.shape)
print("Shape of y_train dataset: ", df2_y_train.shape)
print("Shape of x_test dataset: ", df2_x_test.shape)
print("Shape of y_test dataset: ", df2_y_test.shape)

sm = SMOTE(random_state=5026)
df2_x_train_smote, df2_y_train_smote = sm.fit_sample(df2_x_train, df2_y_train.values.ravel())
print('Resampled dataset shape %s' % Counter(df2_y_train_smote))
traincols_df2 = list(df2_x)
df2_x_train_smote = pd.DataFrame(data=df2_x_train_smote, columns = traincols_df2).reset_index(drop = True)

df2_y_train_smote = pd.DataFrame(data = df2_y_train_smote).reset_index(drop = True)
df2_y_train_smote.rename(columns={0:'FLUDX_YES'}, inplace=True)

df2smote_train = df2_x_train_smote.join(df2_y_train_smote)
df2_test = df2_x_test.join(df2_y_test)
df2smote = pd.concat([df2smote_train,df2_test])
df2smote.shape

#df2smote.to_excel("df2smote.xlsx")

################################################################################
# df3 (correlation) -- regular and smote

# Feature Selection using Correlation Rank
 #https://blog.datadive.net/selecting-good-features-part-iii-random-forests/

df2_corr = df2.corr()               # defaults to Pearson
df2_corr.sort_values(["FLUDX_YES"], ascending = False, inplace = True)
print("\n Find most important features relative to DV: \n", df2_corr.FLUDX_YES)

traincols_df3= ['FLUDX_YES','Spclty_Pediatrics','Age_Child','Age_Baby','CPT_90686',
                'Ins_CIGNA','State_VA','Ins_BCBS', 'Gndr_M']  # corr > 0

#df3b
#traincols_df3 =['FLUDX_YES','Spclty_Pediatrics','Age_Child','Age_Baby','CPT_90686',
#                'Age_Senior','Age_Adult','Spclty_Family Practice','Spclty_Other'] # cor > abs(.3)

df3 = df2[traincols_df3].astype(float)
print(df3.shape)                                                                # Inspect shape of the `reduced_data`

df3_x = df3.drop(['FLUDX_YES'],axis = 1 ).astype(float)
df3_y = pd.DataFrame(df3['FLUDX_YES'].astype(float))

###

df3_x_train, df3_x_test, df3_y_train, df3_y_test = tts(df3_x, df3_y, test_size=0.3, random_state=5026)

print("Shape of x_train dataset: ", df3_x_train.shape)
print("Shape of y_train dataset: ", df3_y_train.shape)
print("Shape of x_test dataset: ", df3_x_test.shape)
print("Shape of y_test dataset: ", df3_y_test.shape)

sm = SMOTE(random_state=5026)
df3_x_train_smote, df3_y_train_smote = sm.fit_sample(df3_x_train, df3_y_train.values.ravel())
print('Resampled dataset shape %s' % Counter(df3_y_train_smote))
traincols = list(df3_x)
df3_x_train_smote = pd.DataFrame(data=df3_x_train_smote, columns = traincols).reset_index(drop = True)

df3_y_train_smote = pd.DataFrame(data = df3_y_train_smote).reset_index(drop = True)
df3_y_train_smote.rename(columns={0:'FLUDX_YES'}, inplace=True)


#df3 = pd.DataFrame(data=df3, columns = traincols_df3).reset_index(drop = True)
################################################################################

################################################################################

# df4 (decision tree) + Mean decrease impurity -- regular and smote

## Partition Data (DecisionTree)
#dt_x = df2_quant.drop(['UNITS','PTAGE','Yr_2016','Yr_2017','FLUDX_YES'],axis=1,inplace=False)
#df4_feats = list(dt_x)
#dt_y = df2_quant[['FLUDX_YES']]
#x_train, x_test, y_train, y_test =tts(df2_x, df2_y, test_size = 0.3, random_state=5026)
#print(x_train.dtypes)

## Depth Determination (DecisionTree)
### Range of values to try, and where to store MSE output
max_depth_range = range(1, 12)
all_MSE_scores = []

### Calculate MSE for each value of max_depth
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=5026)
    MSE_scores = cross_val_score(treereg, df2_x_train, df2_y_train, cv=12, scoring='neg_mean_squared_error')
    all_MSE_scores.append(np.mean(np.sqrt(-MSE_scores)))

### Plot max_depth (x-axis) versus MSE (y-axis)
plt.figure(5)
plt.plot(max_depth_range, all_MSE_scores)
plt.title('Max Depth Range Plot (Decision Tree)')
plt.xlabel('max_depth')
plt.ylabel('MSE (lower is better)')
plt.show()

## Feature Importance
### Based on max_depth plot, depth = 8 is most ideal
dtreereg = DecisionTreeRegressor(max_depth=5, random_state=5026)
dtreereg.fit(df2_x_train, df2_y_train)

### "Gini importance" of each feature:
print(pd.DataFrame({'feature':list(df2_x_train), 'importance':sorted(dtreereg.feature_importances_ *1000, reverse = True)}))

### Mean decrease impurity
#print( "Features sorted by their score:" )
#print( sorted(zip(map(lambda df2_x_train: round(df2_x_train, 4), dtreereg.feature_importances_ *1000 ), list(df2_x_train)), reverse=True))

traincols_df4=['FLUDX_YES','Spclty_Pediatrics','Age_Child','Age_Adult','Ins_CIGNA','Age_Baby','Ins_AETNA','Spclty_Family Practice']

df4 = df2.copy(deep = True)
df4 = df4[traincols_df4].astype(float)
print(df4.shape)                                                                # Inspect shape of the `reduced_data`

df4_x = df4.drop(['FLUDX_YES'],axis = 1 ).astype(float)
df4_y = pd.DataFrame(df4['FLUDX_YES'].astype(float))

###

df4_x_train, df4_x_test, df4_y_train, df4_y_test = tts(df4_x, df4_y, test_size=0.3, random_state=5026)

print("Shape of x_train dataset: ", df4_x_train.shape)
print("Shape of y_train dataset: ", df4_y_train.shape)
print("Shape of x_test dataset: ", df4_x_test.shape)
print("Shape of y_test dataset: ", df4_y_test.shape)

sm = SMOTE(random_state=5026)
df4_x_train_smote, df4_y_train_smote = sm.fit_sample(df4_x_train, df4_y_train.values.ravel())
print('Resampled dataset shape %s' % Counter(df4_y_train_smote))
traincols = list(df4_x)
df4_x_train_smote = pd.DataFrame(data=df4_x_train_smote, columns = traincols).reset_index(drop = True)

df4_y_train_smote = pd.DataFrame(data = df4_y_train_smote).reset_index(drop = True)
df4_y_train_smote.rename(columns={0:'FLUDX_YES'}, inplace=True)

################################################################################

################################################################################
# df5 -- manual selection

traincols_df5=['FLUDX_YES','Spclty_Pediatrics','Age_Child','Age_Adult','Ins_CIGNA','Age_Baby','Ins_AETNA','Spclty_Family Practice']

df5 = df2.copy(deep = True)
df5 = df5[traincols_df4].astype(float)
print(df5.shape)                                                                # Inspect shape of the `reduced_data`

df5_x = df5.drop(['FLUDX_YES'],axis = 1 ).astype(float)
df5_y = pd.DataFrame(df5['FLUDX_YES'].astype(float))

###

df5_x_train, df5_x_test, df5_y_train, df5_y_test = tts(df5_x, df5_y, test_size=0.3, random_state=5026)

print("Shape of x_train dataset: ", df5_x_train.shape)
print("Shape of y_train dataset: ", df5_y_train.shape)
print("Shape of x_test dataset: ", df5_x_test.shape)
print("Shape of y_test dataset: ", df5_y_test.shape)

sm = SMOTE(random_state=5026)
df5_x_train_smote, df5_y_train_smote = sm.fit_sample(df5_x_train, df5_y_train.values.ravel())
print('Resampled dataset shape %s' % Counter(df5_y_train_smote))
traincols = list(df5_x)
df5_x_train_smote = pd.DataFrame(data=df5_x_train_smote, columns = traincols).reset_index(drop = True)

df5_y_train_smote = pd.DataFrame(data = df5_y_train_smote).reset_index(drop = True)
df5_y_train_smote.rename(columns={0:'FLUDX_YES'}, inplace=True)
################################################################################

### DV Density
plt.figure(2); plt.title('Normal')
sns.distplot(df2['FLUDX_YES'], kde=False, fit=st.norm)

plt.figure(3); plt.title('Normal')
sns.distplot(df2smote['FLUDX_YES'], kde=False, fit=st.norm)
################################################################################

## DataSets

# df2 regular
x_train = df2_x_train
y_train = df2_y_train
x_test = df2_x_test
y_test = df2_y_test

# df2 smote
x_train = df2_x_train_smote
y_train = df2_y_train_smote

# df3 regular
x_train = df3_x_train
y_train = df3_y_train
x_test = df3_x_test
y_test = df3_y_test

# df3 smote
x_train = df3_x_train_smote
y_train = df3_y_train_smote

# df4 regular
x_train = df4_x_train
y_train = df4_y_train
x_test = df4_x_test
y_test = df4_y_test

# df4 smote
x_train = df4_x_train_smote
y_train = df4_y_train_smote

# df5 regular
x_train = df5_x_train
y_train = df5_y_train
x_test = df5_x_test
y_test = df5_y_test

# df5 smote
x_train = df5_x_train_smote
y_train = df5_y_train_smote

################################################################################
#### Logistic Regression -- Regular / SMOTED

logit = LogisticRegression(random_state = 5026)
#logit = LogisticRegression(random_state = 5026, class_weight = 'balanced')      # Class Weights

result= logit.fit(x_train,y_train)

logit_yhat = logit.predict(x_test)
logit_prob = logit.predict_proba(x_test)
logit_ci90 = (np.percentile(logit_prob[:,1],90))
logit_threshold = logit_ci90
logit_yhat = np.where(logit_prob[:,1] >= logit_threshold,1,0)

### Evaluation Metrics
logit_accuracy =  round(metrics.accuracy_score(y_test, logit_yhat)*100,2)
print('\n Accuracy logit:\n', metrics.accuracy_score(y_test, logit_yhat) )
#print(' \n Intercept logit: ',logit.intercept_)
logit_coef = pd.DataFrame(logit.coef_[0], x_test.columns, columns=['logit_Coefficients'])
logit_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(y_test, logit_yhat), columns=['predicted 0','predicted 1'], index =['actual 0','actual 1'] )
print('\n Confusion Matrix logit: \n',logit_confusion_matrix)
logit_auc = metrics.roc_auc_score(y_test, logit_yhat)
print('\n AUC: \n', logit_auc)
logit_f1 = f1_score(y_test, logit_yhat)
print('\n F_Score: \n', (logit_f1))

tn = logit_confusion_matrix.iloc[0,0]
fp = logit_confusion_matrix.iloc[0,1]
fn = logit_confusion_matrix.iloc[1,0]
tp = logit_confusion_matrix.iloc[1,1]
sensitivity = tp/(tp+fn)*100                                                    #print(sensitivity) # this percent... of all True values, the model was able to predict
specificity = tn / (tn + fp) *100                                               #print(specificity) # this percent... of all False values, the model was able to predict

################################################################################

################################################################################
# Random Forest Model -- Regular / SMOTE

#tree = RandomForestClassifier(random_state=5026)
tree = RandomForestClassifier(class_weight = 'balanced', random_state=5026)

tree.fit(x_train, y_train)

tree_prob = tree.predict_proba(x_test)
tree_ci90 = (np.percentile(tree_prob[:,1],90))
tree_threshold = tree_ci90
tree_yhat = np.where(tree_prob[:,1] >= tree_threshold,1,0)


tree_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(y_test, tree_yhat), columns=['predicted 0','predicted 1'], index =['actual 0','actual 1'] )
print('\n Confusion Matrix Tree \n',tree_confusion_matrix)

tree_accuracy = metrics.accuracy_score(y_test, tree_yhat)
print( '\n Accuracy Tree: \n', tree_accuracy )

tree_prob = tree.predict_proba(x_test)
tree_prob = [p[1] for p in tree_prob]
tree_auc = roc_auc_score(y_test, tree_prob)
print( '\n AUC Tree: \n', tree_auc )

tree_f1 = f1_score(y_test, tree_yhat)
print('\n F_Score: \n', tree_f1)
################################################################################

#Plotting ROC Curve -- Not Correct AUC SCORES!!!!

logit_fpr, logit_tpr, _ = metrics.roc_curve(y_test, logit_yhat)
logit_auc = round(metrics.roc_auc_score(y_test, logit_yhat),3)
tree_fpr, tree_tpr, _ = metrics.roc_curve(y_test, tree_yhat)
tree_auc= round(metrics.roc_auc_score(y_test, tree_yhat),3)
plt.plot(logit_fpr,logit_tpr,label="Logistic: auc = "+str(logit_auc))
plt.plot(tree_fpr,tree_tpr,label="RandomForest: auc = "+str(tree_auc))
plt.plot([0, 1], [0, 1],'k--')
plt.legend(loc='lower right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Postiive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve \n Logistic-OverSampling vs RandomForest-ClassWeights')
plt.show


################################################################################
## Random Forest Model -- Regular / SMOTE
#
##class_wgts = {i: 1 for i in y_train}
##class_wgts[0] = 1
##class_wgts[1] = 23
##class_wgts.pop('FLUDX_YES')
#
#tree_wgt.fit(x_train, y_train)
#
#tree_wgt_prob = tree_wgt.predict_proba(x_test)
#tree_wgt_ci90 = (np.percentile(tree_wgt_prob[:,1],90))
#tree_wgt_threshold = tree_ci90
#tree_wgt_yhat = np.where(tree_wgt_prob[:,1] >= tree_wgt_threshold,1,0)
#
#
#tree_wgt_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(y_test, tree_wgt_yhat), columns=['predicted 0','predicted 1'], index =['actual 0','actual 1'] )
#print('\n Confusion Matrix Weighted Tree \n',tree_wgt_confusion_matrix)
#
#tree_wgt_accuracy = metrics.accuracy_score(y_test, tree_wgt_yhat)
#print( '\n Accuracy Weighted Tree: \n',tree_wgt_accuracy )
#
#tree_wgt_prob = tree_wgt.predict_proba(x_test)
#tree_wgt_prob = [p[1] for p in tree_wgt_prob]
#tree_wgt_auc = roc_auc_score(y_test, tree_wgt_prob)
#print( '\n AUC Weighted Tree: \n', tree_wgt_auc )
#
#tree_wgt_f1 = f1_score(y_test, tree_wgt_yhat)
#print('\n F_Score: \n', tree_f1)
#################################################################################
#
#df2_yes = df2[df2["FLUDX_YES"] == 1].drop(['FLUDX_YES'],axis = 1 ).astype(float)
#
#df2_yes_corr = df2_yes.corr()               # defaults to Pearson
#sns.heatmap(df2_yes.corr()) #, cmap='BuGn')
##df2_yes_corr.sort_values(["FLUDX_YES"], ascending = False, inplace = True)
##print("Find most important features relative to DV:", df2_yes_corr.FLUDX_YES)

################################################################################

#### Logistic Regression (statsmodel)
#
#
#### Value Counts
#for f in df2.columns:
#    print(df2[f].value_counts())
#
#
#traincols = ['FLUDX_YES','Gndr_F','Gndr_M','State_Other','State_VA',
#             'Spclty_Family Practice','Spclty_Other','Spclty_Pediatrics',
#             'Ins_AETNA','Ins_BCBS','Ins_CIGNA','Ins_Other','Ins_UNITED',
#             'Zip_20132','Zip_20141','Zip_20147','Zip_20148','Zip_20164',
#             'Zip_20165','Zip_20175','Zip_20176','Zip_20180','Zip_Other',
#             'Age_Baby','Age_Child','Age_Adult','Age_Senior']
#y = pd.DataFrame(df2['FLUDX_YES'].astype(float))
#x = df2[traincols].astype(float)
#
#logit = Logit(y,x)
#result = logit.fit()
#print(result.summary())

################################################################################











