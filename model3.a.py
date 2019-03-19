# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:47:02 2019

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

from datetime import datetime
from datetime import date
import datetime as DT

from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from statsmodels.discrete.discrete_model import Logit

from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

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

now = DT.datetime(2019,12,31) #now = pd.Timestamp(DT.datetime.now())
df1_flu1617['Pt DOB'] = pd.to_datetime(df1_flu1617['Pt DOB'], format='%m%d%y')    # 1
df1_flu1617['Pt DOB'] = df1_flu1617['Pt DOB'].where(df1_flu1617['Pt DOB'] < now, df1_flu1617['Pt DOB'] -  np.timedelta64(100, 'Y'))   # 2
df1_flu1617['Pt_Age'] = (now - df1_flu1617['Pt DOB']).astype('<m8[Y]')    # 3

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
df2 = df1.drop(['ACCTNUM','PCP','LMG PRACTICE','DOS','DIAGNOSIS CODES','PLACE OF SERVICE', 'CPT DESCRIPTION','PT ADDRESS 1','PT ADDRESS 2', 'PT ZIP', 'PT DOB'], axis=1)
print(df2.info())


print(df2.isnull().sum())                                                       #Null check again
### Dropping Rows
df2.dropna(inplace=True)

### Value Counts
for f in df2.columns:
    print(df2[f].value_counts())

# Lower Limit Thresholds
pt_race_lt = 5

### Combining Variables into Other
series = pd.value_counts(df2['PT STATE'])
mask = (series/series.sum() * 100)                                              # To replace df['column'] use np.where I.e
mask = (series/series.sum() * 100).lt(1)                                        # lt(%); where % is the cut off
df2['PT STATE'] = np.where(df2['PT STATE'].isin(series[mask].index),'Other',df2['PT STATE'])

series = pd.value_counts(df2['PCP SPECIALTY'])
mask = (series/series.sum() * 100)                                              # To replace df['column'] use np.where I.e
mask = (series/series.sum() * 100).lt(16)                                        # lt(%); where % is the cut off
df2['PCP SPECIALTY'] = np.where(df2['PCP SPECIALTY'].isin(series[mask].index),'Other',df2['PCP SPECIALTY'])

series = pd.value_counts(df2['PT INS'])
mask = (series/series.sum() * 100)
mask = (series/series.sum() * 100).lt(1)                                        # lt(%); where % is the cut off
df2['PT INS'] = np.where(df2['PT INS'].isin(series[mask].index),'Other',df2['PT INS'])

df2['PT RACE'].replace('_R','Other',inplace=True)                               # DOES NOT WORK WELL, DUPE OF 210 and 202
series = pd.value_counts(df2['PT RACE'])
mask = (series/series.sum() * 100)
mask = (series/series.sum() * 100).lt(pt_race_lt)                                        # lt(%); where % is the cut off
df2['PT RACE'] = np.where(df2['PT RACE'].isin(series[mask].index),'Other',df2['PT RACE'])

series = pd.value_counts(df2['CPT'])
mask = (series/series.sum() * 100)
mask = (series/series.sum() * 100).lt(7)                                        # lt(%); where % is the cut off
df2['CPT'] = np.where(df2['CPT'].isin(series[mask].index),'Other',df2['CPT'])

new = series[~mask]
new['Other'] = series[mask].sum()
series.index = np.where(series.index.isin(series[mask].index),'Other',series.index)

df2 = pd.get_dummies(df2,columns = ['YEAR','PT GENDER','PT STATE','PCP SPECIALTY','PT INS','CPT'], prefix = ['Yr','Gndr','State','Spclty','Ins','CPT'])

print(df2.dtypes)


df2_qual = df2.select_dtypes(include=['object']).copy()
df2_quant = df2.select_dtypes(include=['int64','uint8']).copy()

df2_feats = list(df2)
print(df2.groupby('FLUDX_YES').mean() )

### SMOTE

traincols =['Yr_2016','Yr_2017','Gndr_F','Gndr_M',
            'State_MD','State_Other','State_VA','State_WV',
            'Spclty_Family Practice','Spclty_Internal Medicine','Spclty_Other',
            'Ins_AETNA','Ins_BCBS','Ins_CIGNA','Ins_MCAID','Ins_MCARE','Ins_Other','Ins_TRICARE','Ins_UNITED',
            'CPT_90658', 'CPT_90662', 'CPT_90685','CPT_90686', 'CPT_90688', 'CPT_Other']
y = pd.DataFrame(df2['FLUDX_YES'].astype(float))
x = df2[traincols].astype(float)

X_train, X_test, Y_train, Y_test = tts(x, y, test_size=0.3, random_state=5026)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", Y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", Y_test.shape)


sm = SMOTE(random_state=5026)
X_train_smote, Y_train_smote = sm.fit_sample(X_train, Y_train.values.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_smote.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(Y_train_smote.shape))

print("After OverSampling, counts of label '1': {}".format(sum(Y_train_smote==1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_smote==0)))

X_cols = ['Yr_2016','Yr_2017','Gndr_F','Gndr_M',
            'State_MD','State_Other','State_VA','State_WV',
            'Spclty_Family Practice','Spclty_Internal Medicine','Spclty_Other',
            'Ins_AETNA','Ins_BCBS','Ins_CIGNA','Ins_MCAID','Ins_MCARE','Ins_Other','Ins_TRICARE','Ins_UNITED',
            'CPT_90658', 'CPT_90662', 'CPT_90685','CPT_90686', 'CPT_90688', 'CPT_Other']
X_train_smote = pd.DataFrame(data=X_train_smote, columns = X_cols).reset_index(drop = True)

Y_train_smote = pd.DataFrame(data = Y_train_smote).reset_index(drop = True)
Y_train_smote.rename(columns={0:'FLUDX_YES'}, inplace=True)


### DV Density
plt.figure(2); plt.title('Normal')
sns.distplot(df2['FLUDX_YES'], kde=False, fit=st.norm)

################################################################################
#### Logistic Regression (sklearn)
df3 = df2_quant.copy(deep = False)
# prepare X and y
x = df3.drop(['FLUDX_YES','UNITS','PTAGE'],axis=1,inplace=False) # PATIENT AGE!!!!!!!!!!!!!!!!!
y = df3[['FLUDX_YES']]

X_train, X_test, Y_train, Y_test =tts(x, y, test_size = 0.3, random_state=5026)

logit = LogisticRegression()
result = logit.fit(X_train,Y_train)
#print(result.summary())

logit_yhat = logit.predict(X_test)
logit_prob = logit.predict_proba(X_test)
logit_ci90 = (np.percentile(logit_prob[:,1],90))
logit_threshold = logit_ci90
logit_yhat = np.where(logit_prob[:,1] >= logit_threshold,1,0)


logit_score =  round(metrics.accuracy_score(Y_test, logit_yhat)*100,2)
print('\n Score logit:', metrics.accuracy_score(Y_test, logit_yhat) )
#print(' \n Intercept logit: ',logit.intercept_)
logit_coef = pd.DataFrame(logit.coef_[0], X_test.columns, columns=['logit_Coefficients'])
logit_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(Y_test, logit_yhat), columns=['predicted 0','predicted 1'], index =['actual 0','actual 1'] )
print('\n Confusion Matrix logit SMOTE: \n',logit_confusion_matrix)

### Evaluation Metrics
tn = logit_confusion_matrix.iloc[0,0]
fp = logit_confusion_matrix.iloc[0,1]
fn = logit_confusion_matrix.iloc[1,0]
tp = logit_confusion_matrix.iloc[1,1]
sensitivity = tp/(tp+fn)*100                                                    #print(sensitivity) # this percent... of all True values, the model was able to predict
specificity = tn / (tn + fp) *100                                               #print(specificity) # this percent... of all False values, the model was able to predict
################################################################################
#### Logistic Regression (sklearn) SMOTE
df3 = df2_quant.copy(deep = False)

# prepare X and y
#x = df3.drop(['FLUDX_YES','UNITS'],axis=1,inplace=False)
#y = df3[['FLUDX_YES']]

#X_train, X_test, Y_train, Y_test =tts(x, y, test_size = 0.3, random_state=5026)


logit = LogisticRegression()
result = logit.fit(X_train_smote,Y_train_smote)

logit_yhat = logit.predict(X_test)
logit_prob = logit.predict_proba(X_test)
logit_ci90 = (np.percentile(logit_prob[:,1],90))
logit_threshold = logit_ci90
logit_yhat = np.where(logit_prob[:,1] >= logit_threshold,1,0)


logit_score =  round(metrics.accuracy_score(Y_test, logit_yhat)*100,2)
print('\n Score logit:', metrics.accuracy_score(Y_test, logit_yhat) )
#print(' \n Intercept logit: ',logit.intercept_)
logit_coef = pd.DataFrame(logit.coef_[0], X_test.columns, columns=['logit_Coefficients'])
logit_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(Y_test, logit_yhat), columns=['predicted 0','predicted 1'], index =['actual 0','actual 1'] )
print('\n Confusion Matrix logit: \n',logit_confusion_matrix)

### Evaluation Metrics
tn = logit_confusion_matrix.iloc[0,0]
fp = logit_confusion_matrix.iloc[0,1]
fn = logit_confusion_matrix.iloc[1,0]
tp = logit_confusion_matrix.iloc[1,1]
sensitivity = tp/(tp+fn)*100                                                    #print(sensitivity) # this percent... of all True values, the model was able to predict
specificity = tn / (tn + fp) *100                                               #print(specificity) # this percent... of all False values, the model was able to predict
################################################################################

# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

# Train model
clf_3 = SVC(kernel='linear',
            class_weight='balanced', # penalize
            probability=True)

clf_3.fit(x, y)

# Predict on training set
pred_y_3 = clf_3.predict(x)

# Is our model still predicting just one class?
print( np.unique( pred_y_3 ) )


# How's our accuracy?
print( metrics.accuracy_score(y, pred_y_3) )


# What about AUROC?
prob_y_3 = clf_3.predict_proba(x)
prob_y_3 = [p[1] for p in prob_y_3]
print( roc_auc_score(y, prob_y_3) )
































