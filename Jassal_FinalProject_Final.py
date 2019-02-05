# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 11:51:43 2018

@author: akash
"""

################################################
# ENVIRONMENT PREP
import os

### Provide the path here
os.chdir('C:\\Users\\akash\\Desktop\\GWU\\6202_ML_RXie\\FinalProject') 

### Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
################################################
### Data Load
df1  = pd.read_csv('train.csv')

################################################
# DATA TIDYING

### Null Check
plt.figure(1)
df1_missing = df1.isnull().sum()
df1_missing = df1_missing[df1_missing > 0]
df1_missing.sort_values(inplace=True)
plt.title( 'Count of Nulls' )
df1_missing.plot.bar()

### Dropping Columns
df2 = df1.drop(['Id','PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'], axis=1)

### Dropping Rows 
df2.dropna(inplace=True)

### Null Check
df2_missing = df2.isnull().sum()
df2_missing = df2_missing[df2_missing > 0]
df2_missing.sort_values(inplace=True)
#df2_missing.plot.bar()
del(df1_missing,df2_missing)

################################################
# ENCODING
### Encoding Qualitative (Dummy Variables)

#df3_qual = pd.get_dummies(df3_qual)
df3 = pd.get_dummies(df2)

### Determines quantitative and qualitative columns
df2_qual = [f for f in df2.columns if df2.dtypes[f] == 'object']
df2_quant = [f for f in df2.columns if df2.dtypes[f] != 'object']
df3_qual = [f for f in df3.columns if df3.dtypes[f] == 'object']
df3_quant = [f for f in df3.columns if df3.dtypes[f] != 'object']

################################################
#EDA & FEATURE SELECTION

### DV Density
y = df2['SalePrice']
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


####Qualitative
#plt.figure(4)
#corr = df3[df3_qual+['SalePrice']].corr()
#corr = df3[-df2_quant].corr()
#sns.heatmap(corr)

# multiple scatter plots in Seaborn
#feature_cols =  df2.drop('SalePrice',axis=1)
#plt.figure(5)
#sns.pairplot(df2, x_vars=feature_cols, y_vars='SalesPrice', kind='reg')




### IV Correlation to DV 
print("Find most important features relative to DV")
df3_corr = df3.corr()
df3_corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(df3_corr.SalePrice)
df3_feats = list(df3)

del(df2_qual,df2_quant,df3_qual,df3_quant,corr,df3_corr,df3_feats,y,f,g)
################################################
# SCALING
## Change MinMaxScaler to StandardScaler (viceversa)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#df4= StandardScaler().fit_transform(df3)
df4 = MinMaxScaler().fit_transform(df3)
df4 = pd.DataFrame(df4,columns = df3.columns).copy()

df4_2= StandardScaler().fit_transform(df3)
df4_2 = pd.DataFrame(df4,columns = df3.columns).copy()

################################################
# MODEL-1 (DECISION TREE)

## Partition Data (DecisionTree)
dt_x = df4.drop('SalePrice',axis=1,inplace=False)
df4_feats = list(dt_x)
dt_y = df4[['SalePrice']]
X_train, X_test, Y_train, Y_test =tts(dt_x, dt_y, test_size = 0.3, random_state=6202)
print(X_train.dtypes)

## Depth Determination (DecisionTree)
### Range of values to try, and where to store MSE output
max_depth_range = range(1, 12)
all_MSE_scores = []

### Calculate MSE for each value of max_depth
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=6103)
    MSE_scores = cross_val_score(treereg, X_train, Y_train, cv=14, scoring='neg_mean_squared_error')
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
treereg = DecisionTreeRegressor(max_depth=8, random_state=6202)
treereg.fit(X_train, Y_train)

### "Gini importance" of each feature: 
print(pd.DataFrame({'feature':df4_feats, 'importance':sorted(treereg.feature_importances_ *1000, reverse = True)}))

del(df4_feats,dt_x,dt_y,all_MSE_scores,MSE_scores,depth)

################################################
### Create More Manageable DataFrame (via Feature Selection)

#### All Features
#df5 = df4

#### Top 43 Features 
#df5 = df4[['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','1stFlrSF','TotalBsmtSF','FullBath','BsmtQual_Ex','TotRmsAbvGrd','KitchenQual_Ex','YearBuilt','YearRemodAdd','Foundation_PConc','GarageYrBlt','MasVnrArea','ExterQual_Ex','Fireplaces','ExterQual_Gd','HeatingQC_Ex','BsmtFinType1_GLQ','GarageFinish_Fin','Neighborhood_NridgHt','BsmtFinSF1','SaleType_New','SaleCondition_Partial','Neighborhood_NoRidge','MasVnrType_Stone','OpenPorchSF','2ndFlrSF','MSSubClass','LotArea','OverallCond','BsmtFinSF2','BsmtUnfSF','LowQualFinSF','BsmtFullBath','BsmtHalfBath','HalfBath','BedroomAbvGr','KitchenAbvGr','WoodDeckSF','EnclosedPorch','3SsnPorch']]

#### Top 15 Features
#df5 = df4[['SalePrice','MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','GrLivArea','GarageCars','GarageArea','1stFlrSF','TotalBsmtSF','FullBath','BsmtQual_Ex','TotRmsAbvGrd']]

#### Top 5 Features
df5 = df4[['SalePrice','MSSubClass' ,'LotArea','OverallQual','OverallCond','GrLivArea']]


df5_qual = [f for f in df5.columns if df5.dtypes[f] == 'object']
df5_quant = [f for f in df5.columns if df5.dtypes[f] != 'object']

del(df5_qual,df5_quant)
################################################
# SUMMARIES (Part II)
print(df5.shape)
print(df5.dtypes)
print(df5.describe())
list(df5)
################################################
## MODEL 2 -- LINEAR REGRESSION
from sklearn.linear_model import LinearRegression

### Define Variables
feature_colsXY = list(df5)
lm_X = df5.drop('SalePrice',axis=1)
#lm_X = df4[['AvgCoveredCharges','AvgTotPayments','AvgMedicarePayments','C2PRatio']]
lm_Y = df5.SalePrice

### Instantiate Fit
lm = LinearRegression()
lm.fit(lm_X,lm_Y)

### Print Results
print('Intercept:',lm.intercept_)
print('Number of Coefficients:',len(lm.coef_))
print('Coefficients:',lm.coef_)
pd.DataFrame(lm.coef_,lm_X.columns,columns=['lm_Coefficients,11'])

### Prediction (Yhat)
lm_Yhat = lm.predict(lm_X)
##### Compare Y to Yhat

### Scoring -- Rsquared
lm_Rsq = lm.score(lm_X, lm_Y)
print('Linear Model R-squared:',(lm_Rsq)*100,'%')

### Scoring -- MeanSquareError
from sklearn.metrics import mean_squared_error

lm_mse = mean_squared_error(lm_Y, lm_Yhat)
print("Linear Model MSE:", lm_mse)

# pair the feature names with the coefficients
print('\n')
print(sorted(list(zip(feature_colsXY, lm.coef_))))


del(feature_colsXY,lm_Rsq,lm_X,lm_Yhat,lm_mse,lm_Y)

################################################

## MODEL 3 -- RIDGE REGRESSION
from sklearn import linear_model
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)

### Define Variables
rdg_X = df5.drop('SalePrice',axis=1)
rdg_Y = df5.SalePrice

### Instantiate Fit and Predict Yhat
clf.fit(rdg_X,rdg_Y)
rdg_Yhat = clf.predict(rdg_X)

### Print Results
print('Intercept:',clf.intercept_)
print('Number of Coefficients:',len(clf.coef_))
print('Coefficients:',clf.coef_)
pd.DataFrame(clf.coef_,rdg_X.columns,columns=['RIDGE_Coefficients,11'])

### Scoring -- Rsquared
rdg_Rsq = clf.score(rdg_X, rdg_Y)
print('RIDGE Model R-squared:',(rdg_Rsq)*100,'%')

### Scoring -- MeanSquareError
from sklearn.metrics import mean_squared_error

rdg_mse = mean_squared_error(rdg_Y, rdg_Yhat)
print("RIDGE Model MSE:", rdg_mse)

del(rdg_Rsq,rdg_X,rdg_Y,rdg_Yhat,rdg_mse)

#http://scikit-learn.org/stable/modules/linear_model.html
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
#https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/


################################################

## MODEL 4 -- STOCHASTIC GRADIENT DESCENT
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
clf = linear_model.SGDRegressor()
SGDRegressor(alpha=0.01, average=False, epsilon=0.1, eta0=0.01,
             fit_intercept=True, l1_ratio=0.5, learning_rate='invscaling',
       loss='squared_loss', n_iter=100, penalty='l2',
       power_t=0.25, random_state=None, shuffle=True, 
       verbose=0, warm_start=False)

### Define Variables
sgd_X = df5.drop('SalePrice',axis=1)
sgd_Y = df5.SalePrice

### Instantiate Fit and Predict Yhat
clf.fit(sgd_X,sgd_Y)
sgd_Yhat = clf.predict(sgd_X)

### Print Results
print('Intercept:',clf.intercept_)
print('Number of Coefficients:',len(clf.coef_))
print('Coefficients:',clf.coef_)
pd.DataFrame(clf.coef_,sgd_X.columns,columns=['SGD_Coefficients,11'])

### Scoring -- Rsquared
sgd_Rsq = clf.score(sgd_X, sgd_Y)
print('SGD Model R-squared:',(sgd_Rsq)*100,'%')

### Scoring -- MeanSquareError
from sklearn.metrics import mean_squared_error

sgd_mse = mean_squared_error(sgd_Y, sgd_Yhat)
print("SGD Model MSE:", sgd_mse)

del(sgd_Rsq,sgd_X,sgd_Y,sgd_Yhat,sgd_mse)

################################################

# MODEL 5 -- KNN

### X-Variable Categorization

#### All Features
#df6 = df4_2

#### Top 44 Features 
#df6 = df4_2[['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','1stFlrSF','TotalBsmtSF','FullBath','BsmtQual_Ex','TotRmsAbvGrd','KitchenQual_Ex','YearBuilt','YearRemodAdd','Foundation_PConc','GarageYrBlt','MasVnrArea','ExterQual_Ex','Fireplaces','ExterQual_Gd','HeatingQC_Ex','BsmtFinType1_GLQ','GarageFinish_Fin','Neighborhood_NridgHt','BsmtFinSF1','SaleType_New','SaleCondition_Partial','Neighborhood_NoRidge','MasVnrType_Stone','OpenPorchSF','2ndFlrSF','MSSubClass','LotArea','OverallCond','BsmtFinSF2','BsmtUnfSF','LowQualFinSF','BsmtFullBath','BsmtHalfBath','HalfBath','BedroomAbvGr','KitchenAbvGr','WoodDeckSF','EnclosedPorch','3SsnPorch']]

#### Top 15 Features
#df6 = df4_2[['SalePrice','MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','GrLivArea','GarageCars','GarageArea','1stFlrSF','TotalBsmtSF','FullBath','BsmtQual_Ex','TotRmsAbvGrd']]

#### Top 5 Features
df6 = df4_2[['SalePrice','MSSubClass' ,'LotArea','OverallQual','OverallCond','GrLivArea']]


### Y-Variable Categorization 
#### Create values to group the C2P Ratio into bins

### DV Density
y = df6['SalePrice']
plt.figure(8); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)

conditions = [
     (df6['SalePrice'] >= 0) & (df6['SalePrice'] < 0.135),
     (df6['SalePrice'] >= .135) & (df6['SalePrice'] < .18),
     (df6['SalePrice'] >= .18) & (df6['SalePrice'] < .25),
     (df6['SalePrice'] >= .25)] 
choices=[0,1,2,3]

df6['SalePrice_bin'] = np.select(conditions, choices, default=3)
df6_y = df6[['SalePrice_bin']]

### Distribution of Bins
plt.subplot(1,1,1)
plt.hist(df6.SalePrice_bin, color = "xkcd:lavender")
plt.xlabel("SalePrice Bin")
plt.title("Frequency of SalePrice")
plt.xticks(np.arange(4),('[0,135k)','[135k,170k)','[170k,225k)','[225k,+)'))
plt.tight_layout()

## Partition Data 
knn_x = df6.drop(['SalePrice','SalePrice_bin'],axis=1,inplace=False)
#knn_y = df6[['SalePrice']]
knn_y = df6_y[['SalePrice_bin']]
X_train, X_test, Y_train, Y_test =tts(knn_x, knn_y, test_size = 0.3, random_state=6202)
print(X_train.dtypes)

## KNN-Tuning -->
### Determines what number should K should be

### Store results
train_accuracy = []
test_accuracy  = []
### Set KNN setting from 1 to 15
knn_range = range(1, 15)
for neighbors in knn_range:
### Start Nearest Neighbors Classifier with K of 1
    knn = KNeighborsClassifier(n_neighbors=neighbors,metric='minkowski', p=1)
### Train the data using Nearest Neighbors
    knn.fit(X_train, Y_train)
### Capture training accuracy
    train_accuracy.append(knn.score(X_train, Y_train))
### Predict using the test dataset  
    Y_pred = knn.predict(X_test)
### Capture test accuracy
    test_accuracy.append(knn.score(X_test, Y_test))
  
## Plot Results from KNN Tuning
plt.figure(6)
plt.plot(knn_range, train_accuracy, label='training accuracy')
plt.plot(knn_range, test_accuracy,  label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Neighbors')
plt.legend()
plt.title('KNN Tuning ( # of Neighbors vs Accuracy')
#plt.savefig('KNNTuning.png')
plt.show()

### Plot Results from KNN Tuning (Test Accuracy Only)
plt.figure(7)
plt.plot(knn_range, test_accuracy,  label='test accuracy', 
         color = "orange")
plt.ylabel('Accuracy')
plt.xlabel('Neighbors')
plt.legend()
plt.title('KNN Tuning ( # of Neighbors vs Accuracy')
#plt.savefig('KNNTuning2.png')
plt.show()

## KNN Using Output from KNN-Tuning, K = 5 is most ideal
knn = KNeighborsClassifier(n_neighbors=5, 
                            metric='minkowski', p=1)

### Re-Train the data using Nearest Neighbors
knn.fit(X_train, Y_train)

### Model Accuracy
knn_Ypred= knn.predict(X_test)
print('\nPrediction from X_test:')
print(knn_Ypred)

knn_score = knn.score(X_test, Y_test)
print('Score:', knn_score*100,'%')

### Prediction -- SalePrice_bin
X_new = np.array([[20, 8000, 6, 5 ,1800]])
knn_pred2 = knn.predict(X_new)
print('Prediction:', knn_pred2,'Saleprice')

del(choices, conditions,df6_y,Y_pred,knn_Ypred,knn_score,knn_x,knn_y, neighbors,test_accuracy,train_accuracy)

################################################

#ExtraCode
### DV Density
y = df2['LotArea']
plt.figure(10); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)













