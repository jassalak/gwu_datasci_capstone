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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

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
df2 = df1.drop(['ACCTNUM','PCP','LMG PRACTICE','DOS','DIAGNOSIS CODES',
                'PLACE OF SERVICE', 'CPT DESCRIPTION','PT ADDRESS 1',
                'PT ADDRESS 2', 'PT ZIP', 'PT DOB'], axis=1)
print(df2.info())


print(df2.isnull().sum())                                                       #Null check again
### Dropping Rows
df2.dropna(inplace=True)

df2['AGE_BIN']= pd.cut(df2['PTAGE'],[0,18,35,55,80,110], labels = ['1','2','3','4','5'] )

### Value Counts
for f in df2.columns:
    print(df2[f].value_counts())

################################################################################

# Lower Limit Thresholds
pt_state_lt = 1
pcp_speciality_lt = 16
pt_ins_lt = 1
pt_race_lt = 5
cpt_lt = 7

### Combining Variables into Other
series = pd.value_counts(df2['PT STATE'])
mask = (series/series.sum() * 100)                                              # To replace df['column'] use np.where I.e
mask = (series/series.sum() * 100).lt(pt_state_lt)                                        # lt(%); where % is the cut off
df2['PT STATE'] = np.where(df2['PT STATE'].isin(series[mask].index),'Other',df2['PT STATE'])

series = pd.value_counts(df2['PCP SPECIALTY'])
mask = (series/series.sum() * 100)                                              # To replace df['column'] use np.where I.e
mask = (series/series.sum() * 100).lt(pcp_speciality_lt)                                        # lt(%); where % is the cut off
df2['PCP SPECIALTY'] = np.where(df2['PCP SPECIALTY'].isin(series[mask].index),'Other',df2['PCP SPECIALTY'])

series = pd.value_counts(df2['PT INS'])
mask = (series/series.sum() * 100)
mask = (series/series.sum() * 100).lt(pt_ins_lt)                                        # lt(%); where % is the cut off
df2['PT INS'] = np.where(df2['PT INS'].isin(series[mask].index),'Other',df2['PT INS'])

df2['PT RACE'].replace('_R','Other',inplace=True)                               # DOES NOT WORK WELL, DUPE OF 210 and 202
series = pd.value_counts(df2['PT RACE'])
mask = (series/series.sum() * 100)
mask = (series/series.sum() * 100).lt(pt_race_lt)                                        # lt(%); where % is the cut off
df2['PT RACE'] = np.where(df2['PT RACE'].isin(series[mask].index),'Other',df2['PT RACE'])

series = pd.value_counts(df2['CPT'])
mask = (series/series.sum() * 100)
mask = (series/series.sum() * 100).lt(cpt_lt)                                        # lt(%); where % is the cut off
df2['CPT'] = np.where(df2['CPT'].isin(series[mask].index),'Other',df2['CPT'])

new = series[~mask]
new['Other'] = series[mask].sum()
series.index = np.where(series.index.isin(series[mask].index),'Other',series.index)

df2 = pd.get_dummies(df2,columns = ['YEAR','PT GENDER','PT STATE','PCP SPECIALTY','PT INS','CPT','AGE_BIN'], prefix = ['Yr','Gndr','State','Spclty','Ins','CPT','Age'])
print(df2.dtypes)


df2_qual = df2.select_dtypes(include=['object']).copy()
df2_quant = df2.select_dtypes(include=['int64','uint8']).copy()

df2_feats = list(df2)
print(df2.groupby('FLUDX_YES').mean() )

################################################################################
# DF3
# Import `PCA` from `sklearn.decomposition`

traincols =['Yr_2016','Yr_2017','Gndr_F','Gndr_M',
            'State_MD','State_Other','State_VA','State_WV',
            'Spclty_Family Practice','Spclty_Internal Medicine','Spclty_Other',
            'Ins_AETNA','Ins_BCBS','Ins_CIGNA','Ins_MCAID','Ins_MCARE','Ins_Other','Ins_TRICARE','Ins_UNITED',
            'CPT_90658', 'CPT_90662', 'CPT_90685','CPT_90686', 'CPT_90688', 'CPT_Other',
            'Age_1','Age_2','Age_3','Age_4','Age_5']

df3 = df2_quant.copy(deep = True)
df3 = df3[traincols].astype(float)

# Build the model
pca = PCA(n_components=30)

# Reduce the data, output is ndarray
df3 = pca.fit_transform(df3)

# Inspect shape of the `reduced_data`
df3.shape

# print out the reduced data
print(df3)

df3 = pd.DataFrame(data=df3, columns = traincols).reset_index(drop = True)

################################################################################

# DF4
# Import `PCA` from `sklearn.decomposition`

traincols =['Gndr_F','Gndr_M',
            'State_MD','State_Other','State_VA','State_WV',
            'Spclty_Family Practice','Spclty_Internal Medicine','Spclty_Other',
            'Ins_AETNA','Ins_BCBS','Ins_CIGNA','Ins_MCAID','Ins_MCARE','Ins_Other','Ins_TRICARE','Ins_UNITED',
            'Age_1','Age_2','Age_3','Age_4','Age_5']

df4 = df2_quant.copy(deep = True)
df4 = df4[traincols].astype(float)

################################################################################
### IV Correlation to DV
print("Find most important features relative to DV")
df2_corr = df2.corr()
df2_corr.sort_values(["FLUDX_YES"], ascending = False, inplace = True)
print(df2_corr.FLUDX_YES)
df2_feats = list(df2)

################################################################################

### SMOTE (df2)

traincols =['Yr_2016','Yr_2017','Gndr_F','Gndr_M',
            'State_MD','State_Other','State_VA','State_WV',
            'Spclty_Family Practice','Spclty_Internal Medicine','Spclty_Other',
            'Ins_AETNA','Ins_BCBS','Ins_CIGNA','Ins_MCAID','Ins_MCARE','Ins_Other','Ins_TRICARE','Ins_UNITED',
            'CPT_90658', 'CPT_90662', 'CPT_90685','CPT_90686', 'CPT_90688', 'CPT_Other',
            'Age_1','Age_2','Age_3','Age_4','Age_5']
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


X_train_smote = pd.DataFrame(data=X_train_smote, columns = traincols).reset_index(drop = True)

Y_train_smote = pd.DataFrame(data = Y_train_smote).reset_index(drop = True)
Y_train_smote.rename(columns={0:'FLUDX_YES'}, inplace=True)

################################################################################

### SMOTE (df3)

#traincols =['Yr_2016','Yr_2017','Gndr_F','Gndr_M',
#            'State_MD','State_Other','State_VA','State_WV',
#            'Spclty_Family Practice','Spclty_Internal Medicine','Spclty_Other',
#            'Ins_AETNA','Ins_BCBS','Ins_CIGNA','Ins_MCAID','Ins_MCARE','Ins_Other','Ins_TRICARE','Ins_UNITED',
#            'CPT_90658', 'CPT_90662', 'CPT_90685','CPT_90686', 'CPT_90688', 'CPT_Other',
#            'Age_1','Age_2','Age_3','Age_4','Age_5']

y = pd.DataFrame(df2['FLUDX_YES'].astype(float))
x = df3[traincols].astype(float)

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


X_train_smote = pd.DataFrame(data=X_train_smote, columns = traincols).reset_index(drop = True)

Y_train_smote = pd.DataFrame(data = Y_train_smote).reset_index(drop = True)
Y_train_smote.rename(columns={0:'FLUDX_YES'}, inplace=True)

################################################################################

### SMOTE (df4)

traincols =['Gndr_F','Gndr_M',
            'State_MD','State_Other','State_VA','State_WV',
            'Spclty_Family Practice','Spclty_Internal Medicine','Spclty_Other',
            'Ins_AETNA','Ins_BCBS','Ins_CIGNA','Ins_MCAID','Ins_MCARE','Ins_Other','Ins_TRICARE','Ins_UNITED',
            'Age_1','Age_2','Age_3','Age_4','Age_5']

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


X_train_smote = pd.DataFrame(data=X_train_smote, columns = traincols).reset_index(drop = True)

Y_train_smote = pd.DataFrame(data = Y_train_smote).reset_index(drop = True)
Y_train_smote.rename(columns={0:'FLUDX_YES'}, inplace=True)
################################################################################

### DV Density
plt.figure(2); plt.title('Normal')
sns.distplot(df2['FLUDX_YES'], kde=False, fit=st.norm)

################################################################################
#### Logistic Regression (sklearn) DF2
#df3 = df2_quant.copy(deep = False)
# prepare X and y
#x = df3.drop(['FLUDX_YES','UNITS','PTAGE'],axis=1,inplace=False) # PATIENT AGE!!!!!!!!!!!!!!!!!
#y = df3[['FLUDX_YES']]

#X_train, X_test, Y_train, Y_test =tts(x, y, test_size = 0.3, random_state=5026)

logit = LogisticRegression()
result = logit.fit(X_train,Y_train)
#print(result.summary())

logit_yhat = logit.predict(X_test)
logit_prob = logit.predict_proba(X_test)
logit_ci90 = (np.percentile(logit_prob[:,1],90))
logit_threshold = logit_ci90
logit_yhat = np.where(logit_prob[:,1] >= logit_threshold,1,0)


logit_score =  round(metrics.accuracy_score(Y_test, logit_yhat)*100,2)
print('\n Score logit:', metrics.accuracy_score(Y_test, logit_yhat) ) #0.8584547181760608
#print(' \n Intercept logit: ',logit.intercept_)
logit_coef = pd.DataFrame(logit.coef_[0], X_test.columns, columns=['logit_Coefficients'])
logit_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(Y_test, logit_yhat), columns=['predicted 0','predicted 1'], index =['actual 0','actual 1'] )
print('\n Confusion Matrix logit SMOTE: \n',logit_confusion_matrix)
logit_auc = metrics.roc_auc_score(Y_test, logit_yhat)
print('\n AUC: \n', logit_auc)  #0.568841245785207

### Evaluation Metrics
tn = logit_confusion_matrix.iloc[0,0]
fp = logit_confusion_matrix.iloc[0,1]
fn = logit_confusion_matrix.iloc[1,0]
tp = logit_confusion_matrix.iloc[1,1]
sensitivity = tp/(tp+fn)*100                                                    #print(sensitivity) # this percent... of all True values, the model was able to predict
specificity = tn / (tn + fp) *100                                               #print(specificity) # this percent... of all False values, the model was able to predict
################################################################################
#### Logistic Regression (sklearn) DF3
#df3 = df2_quant.copy(deep = False)
# prepare X and y
#x = df3.drop(['FLUDX_YES','UNITS','PTAGE'],axis=1,inplace=False) # PATIENT AGE!!!!!!!!!!!!!!!!!
#y = df3[['FLUDX_YES']]

#X_train, X_test, Y_train, Y_test =tts(x, y, test_size = 0.3, random_state=5026)

logit = LogisticRegression()
result = logit.fit(X_train,Y_train)
#print(result.summary())

logit_yhat = logit.predict(X_test)
logit_prob = logit.predict_proba(X_test)
logit_ci90 = (np.percentile(logit_prob[:,1],90))
logit_threshold = logit_ci90
logit_yhat = np.where(logit_prob[:,1] >= logit_threshold,1,0)


logit_score =  round(metrics.accuracy_score(Y_test, logit_yhat)*100,2)
print('\n Score logit:', metrics.accuracy_score(Y_test, logit_yhat) ) #0.880050664977834
#print(' \n Intercept logit: ',logit.intercept_)
logit_coef = pd.DataFrame(logit.coef_[0], X_test.columns, columns=['logit_Coefficients'])
logit_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(Y_test, logit_yhat), columns=['predicted 0','predicted 1'], index =['actual 0','actual 1'] )
print('\n Confusion Matrix logit SMOTE: \n',logit_confusion_matrix)
logit_auc = metrics.roc_auc_score(Y_test, logit_yhat)
print('\n AUC: \n', logit_auc)  #  0.558901607998342

### Evaluation Metrics
tn = logit_confusion_matrix.iloc[0,0]
fp = logit_confusion_matrix.iloc[0,1]
fn = logit_confusion_matrix.iloc[1,0]
tp = logit_confusion_matrix.iloc[1,1]
sensitivity = tp/(tp+fn)*100                                                    #print(sensitivity) # this percent... of all True values, the model was able to predict
specificity = tn / (tn + fp) *100                                               #print(specificity) # this percent... of all False values, the model was able to predict
################################################################################

#### Logistic Regression (sklearn) DF4
#df3 = df2_quant.copy(deep = False)
# prepare X and y
#x = df3.drop(['FLUDX_YES','UNITS','PTAGE'],axis=1,inplace=False) # PATIENT AGE!!!!!!!!!!!!!!!!!
#y = df3[['FLUDX_YES']]

#X_train, X_test, Y_train, Y_test =tts(x, y, test_size = 0.3, random_state=5026)

logit = LogisticRegression()
result = logit.fit(X_train,Y_train)
#print(result.summary())

logit_yhat = logit.predict(X_test)
logit_prob = logit.predict_proba(X_test)
logit_ci90 = (np.percentile(logit_prob[:,1],90))
logit_threshold = logit_ci90
logit_yhat = np.where(logit_prob[:,1] >= logit_threshold,1,0)


logit_score =  round(metrics.accuracy_score(Y_test, logit_yhat)*100,2)
print('\n Score logit:', metrics.accuracy_score(Y_test, logit_yhat) ) #0.8235592146928435
#print(' \n Intercept logit: ',logit.intercept_)
logit_coef = pd.DataFrame(logit.coef_[0], X_test.columns, columns=['logit_Coefficients'])
logit_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(Y_test, logit_yhat), columns=['predicted 0','predicted 1'], index =['actual 0','actual 1'] )
print('\n Confusion Matrix logit SMOTE: \n',logit_confusion_matrix)
logit_auc = metrics.roc_auc_score(Y_test, logit_yhat)
print('\n AUC: \n', logit_auc)  #  0.584751342

### Evaluation Metrics
tn = logit_confusion_matrix.iloc[0,0]
fp = logit_confusion_matrix.iloc[0,1]
fn = logit_confusion_matrix.iloc[1,0]
tp = logit_confusion_matrix.iloc[1,1]
sensitivity = tp/(tp+fn)*100                                                    #print(sensitivity) # this percent... of all True values, the model was able to predict
specificity = tn / (tn + fp) *100                                               #print(specificity) # this percent... of all False values, the model was able to predict
################################################################################
#### Logistic Regression (sklearn) SMOTE df2
#df3 = df2_quant.copy(deep = False)

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
print('\n Score logit:', metrics.accuracy_score(Y_test, logit_yhat) ) #0.8560481317289423
#print(' \n Intercept logit: ',logit.intercept_)
logit_coef = pd.DataFrame(logit.coef_[0], X_test.columns, columns=['logit_Coefficients'])
logit_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(Y_test, logit_yhat), columns=['predicted 0','predicted 1'], index =['actual 0','actual 1'] )
print('\n Confusion Matrix logit: \n',logit_confusion_matrix)
logit_auc = metrics.roc_auc_score(Y_test, logit_yhat)
print('\n AUC: \n', logit_auc) #0.5575

### Evaluation Metrics
tn = logit_confusion_matrix.iloc[0,0]
fp = logit_confusion_matrix.iloc[0,1]
fn = logit_confusion_matrix.iloc[1,0]
tp = logit_confusion_matrix.iloc[1,1]
sensitivity = tp/(tp+fn)*100                                                    #print(sensitivity) # this percent... of all True values, the model was able to predict
specificity = tn / (tn + fp) *100                                               #print(specificity) # this percent... of all False values, the model was able to predict
################################################################################

#### Logistic Regression (sklearn) SMOTE df3
#df3 = df2_quant.copy(deep = False)

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
print('\n Score logit:', metrics.accuracy_score(Y_test, logit_yhat) ) #0.8797973400886637
#print(' \n Intercept logit: ',logit.intercept_)
logit_coef = pd.DataFrame(logit.coef_[0], X_test.columns, columns=['logit_Coefficients'])
logit_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(Y_test, logit_yhat), columns=['predicted 0','predicted 1'], index =['actual 0','actual 1'] )
print('\n Confusion Matrix logit: \n',logit_confusion_matrix)
logit_auc = metrics.roc_auc_score(Y_test, logit_yhat)
print('\n AUC: \n', logit_auc) #0.550510259

### Evaluation Metrics
tn = logit_confusion_matrix.iloc[0,0]
fp = logit_confusion_matrix.iloc[0,1]
fn = logit_confusion_matrix.iloc[1,0]
tp = logit_confusion_matrix.iloc[1,1]
sensitivity = tp/(tp+fn)*100                                                    #print(sensitivity) # this percent... of all True values, the model was able to predict
specificity = tn / (tn + fp) *100                                               #print(specificity) # this percent... of all False values, the model was able to predict
################################################################################

#### Logistic Regression (sklearn) SMOTE df4
#df3 = df2_quant.copy(deep = False)

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
print('\n Score logit:', metrics.accuracy_score(Y_test, logit_yhat) ) #0.8310956301456618
#print(' \n Intercept logit: ',logit.intercept_)
logit_coef = pd.DataFrame(logit.coef_[0], X_test.columns, columns=['logit_Coefficients'])
logit_confusion_matrix = pd.DataFrame(metrics.confusion_matrix(Y_test, logit_yhat), columns=['predicted 0','predicted 1'], index =['actual 0','actual 1'] )
print('\n Confusion Matrix logit: \n',logit_confusion_matrix)
logit_auc = metrics.roc_auc_score(Y_test, logit_yhat)
print('\n AUC: \n', logit_auc) #0.5785534541374721

### Evaluation Metrics
tn = logit_confusion_matrix.iloc[0,0]
fp = logit_confusion_matrix.iloc[0,1]
fn = logit_confusion_matrix.iloc[1,0]
tp = logit_confusion_matrix.iloc[1,1]
sensitivity = tp/(tp+fn)*100                                                    #print(sensitivity) # this percent... of all True values, the model was able to predict
specificity = tn / (tn + fp) *100                                               #print(specificity) # this percent... of all False values, the model was able to predict
################################################################################

#df2
# SVC -->  https://elitedatascience.com/imbalanced-classes
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

# Train model
svc = SVC(kernel='linear',
            class_weight='balanced', # penalize
            probability=True)

svc.fit(X_train, Y_train)

# Predict on training set
svc_yhat = svc.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( svc_yhat ) )

# How's our accuracy?
print( metrics.accuracy_score(Y_test, svc_yhat) ) # 0.5836605446485117

# What about AUROC?
svc_prob = svc.predict_proba(X_test)
svc_prob = [p[1] for p in svc_prob]
print( roc_auc_score(Y_test, svc_prob) ) # 0.6473181635658544

################################################################################
#df3
# SVC -->  https://elitedatascience.com/imbalanced-classes
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

# Train model
svc = SVC(kernel='linear',
            class_weight='balanced', # penalize
            probability=True)

svc.fit(X_train, Y_train)

# Predict on training set
svc_yhat = svc.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( svc_yhat ) )

# How's our accuracy?
print( metrics.accuracy_score(Y_test, svc_yhat) ) #0.5300823305889804

# What about AUROC?
svc_prob = svc.predict_proba(X_test)
svc_prob = [p[1] for p in svc_prob]
print( roc_auc_score(Y_test, svc_prob) ) #0.6780682796470335

##############################################################################
#df4
# SVC -->  https://elitedatascience.com/imbalanced-classes
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

# Train model
svc = SVC(kernel='linear',
            class_weight='balanced', # penalize
            probability=True)

svc.fit(X_train, Y_train)

# Predict on training set
svc_yhat = svc.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( svc_yhat ) )

# How's our accuracy?
print( metrics.accuracy_score(Y_test, svc_yhat) ) #0.5300823305889804

# What about AUROC?
svc_prob = svc.predict_proba(X_test)
svc_prob = [p[1] for p in svc_prob]
print( roc_auc_score(Y_test, svc_prob) ) #0.6764161779300285

##############################################################################
# SVC -->  SMOTE DF2
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

# Train model
svc = SVC(kernel='linear',
            class_weight='balanced', # penalize
            probability=True)

svc.fit(X_train_smote, Y_train_smote)

# Predict on training set
svc_yhat = svc.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( svc_yhat ) )

# How's our accuracy?
print( metrics.accuracy_score(Y_test, svc_yhat) ) # 0.6051931602279924

# What about AUROC?
svc_prob = svc.predict_proba(X_test)
svc_prob = [p[1] for p in svc_prob]
print( roc_auc_score(Y_test, svc_prob) ) # 0.6462403024886211

################################################################################

# SVC -->  SMOTE DF3
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

# Train model
svc = SVC(kernel='linear',
            class_weight='balanced', # penalize
            probability=True)

svc.fit(X_train_smote, Y_train_smote)

# Predict on training set
svc_yhat = svc.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( svc_yhat ) )

# How's our accuracy?
print( metrics.accuracy_score(Y_test, svc_yhat) ) #

# What about AUROC?
svc_prob = svc.predict_proba(X_test)
svc_prob = [p[1] for p in svc_prob]
print( roc_auc_score(Y_test, svc_prob) ) #
##############################################################################

# SVC -->  SMOTE DF4
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

# Train model
svc = SVC(kernel='linear',
            class_weight='balanced', # penalize
            probability=True)

svc.fit(X_train_smote, Y_train_smote)

# Predict on training set
svc_yhat = svc.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( svc_yhat ) )

# How's our accuracy?
print( metrics.accuracy_score(Y_test, svc_yhat) ) #0.5394553514882837

# What about AUROC?
svc_prob = svc.predict_proba(X_test)
svc_prob = [p[1] for p in svc_prob]
print( roc_auc_score(Y_test, svc_prob) ) #0.661868789906019
##############################################################################

# Tree Model --> df2
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

class_wgts = {i: 1 for i in Y_train}
class_wgts[1] = 10
class_wgts[0] = 1
class_wgts.pop('FLUDX_YES')

# Train model
tree = RandomForestClassifier( class_weight = class_wgts)
tree.fit(X_train, Y_train)

# Predict on training set
tree_yhat = tree.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( tree_yhat ) )


# How's our accuracy?
print( metrics.accuracy_score(Y_test, tree_yhat) ) # 0.966624445851805


# What about AUROC?
tree_prob = tree.predict_proba(X_test)
tree_prob = [p[1] for p in tree_prob]
print( roc_auc_score(Y_test, tree_prob) ) # 0.6793665953240706


################################################################################
# Tree Model --> SMOTE df2
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

# Train model
tree = RandomForestClassifier()
tree.fit(X_train_smote, Y_train_smote)

# Predict on training set
tree_yhat = tree.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( tree_yhat ) )


# How's our accuracy?
print( metrics.accuracy_score(Y_test, tree_yhat) ) # 0.5980367321089297


# What about AUROC?
tree_prob = tree.predict_proba(X_test)
tree_prob = [p[1] for p in tree_prob]
print( roc_auc_score(Y_test, tree_prob) ) # 0.673525358008306


################################################################################

# Tree Model --> df3
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

# Train model
tree = RandomForestClassifier()
tree.fit(X_train, Y_train)

# Predict on training set
tree_yhat = tree.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( tree_yhat ) )


# How's our accuracy?
print( metrics.accuracy_score(Y_test, tree_yhat) ) #0.966624446

# What about AUROC?
tree_prob = tree.predict_proba(X_test)
tree_prob = [p[1] for p in tree_prob]
print( roc_auc_score(Y_test, tree_prob) ) # 0.683327554

################################################################################

# Tree Model --> SMOTE df3
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

# Train model
tree = RandomForestClassifier()
tree.fit(X_train_smote, Y_train_smote)

# Predict on training set
tree_yhat = tree.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( tree_yhat ) )


# How's our accuracy?
print( metrics.accuracy_score(Y_test, tree_yhat) ) # 0.6377454084863838


# What about AUROC?
tree_prob = tree.predict_proba(X_test)
tree_prob = [p[1] for p in tree_prob]
print( roc_auc_score(Y_test, tree_prob) ) # 0.6811904523479286


################################################################################

################################################################################

# Tree Model --> df4
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

# Train model
tree = RandomForestClassifier()
tree.fit(X_train, Y_train)

# Predict on training set
tree_yhat = tree.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( tree_yhat ) )


# How's our accuracy?
print( metrics.accuracy_score(Y_test, tree_yhat) ) #0.9664344521849272

# What about AUROC?
tree_prob = tree.predict_proba(X_test)
tree_prob = [p[1] for p in tree_prob]
print( roc_auc_score(Y_test, tree_prob) ) # 0.6912616155909479

################################################################################

# Tree Model --> SMOTE df4
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)

# Train model
tree = RandomForestClassifier()
tree.fit(X_train_smote, Y_train_smote)

# Predict on training set
tree_yhat = tree.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( tree_yhat ) )


# How's our accuracy?
print( metrics.accuracy_score(Y_test, tree_yhat) ) # 0.5394553514882837


# What about AUROC?
tree_prob = tree.predict_proba(X_test)
tree_prob = [p[1] for p in tree_prob]
print( roc_auc_score(Y_test, tree_prob) ) # 0.661868789906019


################################################################################


















