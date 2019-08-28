import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt

import sys

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sklearn as sk
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm

import datetime
from dateutil.parser import *

X = pd.read_parquet("C:/Users/vvrhovec/Veronika/kodovi/radni_folder/parquet/X.parquet")
X.head()

std_scaler = StandardScaler()
lm = linear_model.LinearRegression()

list_of_items = [38,39,40,41,57] 

rmse_train = []
rmse_test = []
r2 = []


def scaling_columns_seperately(self_train,self_test,col):
    aux_df = self_train[col]
    std_scaler.fit(aux_df.values.reshape(-1,1))
    aux_df = std_scaler.transform(self_train[col].values.reshape(-1,1))
    self_train[col] = aux_df
    aux_df = self_test[col]
    aux_df = std_scaler.transform(self_test[col].values.reshape(-1,1))
    self_test[col] = aux_df

    
def removing_outliers(self):
    upper_lim = self['amount'].quantile(.95)
    lower_lim = self['amount'].quantile(.05)
    self=self[(self['amount'] < upper_lim) & (self['amount'] > lower_lim)]
    return self


def X_train_test_separation(self):
    aux = self.loc[self.index.get_level_values('date') <= '2018-01-01']
    train = aux.drop('amount',axis=1)
    aux = self.loc[self.index.get_level_values('date') > '2018-01-01']
    test = aux.drop('amount',axis=1)
    return train, test


def y_train_test_separation(self):
    aux = self.loc[self.index.get_level_values('date') <= '2018-01-01']
    train = aux['amount']
    aux = self.loc[self.index.get_level_values('date') > '2018-01-01']
    test = aux['amount']
    return train, test

#_______________________________________________________________________________________
## Item 38

X_38 = X.loc[38]
X_38 = removing_outliers(X_38)

### Separating data into training and test set
X_38_train, X_38_test =  X_train_test_separation(X_38)
y_38_train, y_38_test =  y_train_test_separation(X_38)

### Scaling the Dataframe
for column in X_38_train:                               
    scaling_columns_seperately(X_38_train,X_38_test,column)

#### Fitting the model on the data from training set
model_38 = lm.fit(X_38_train, y_38_train)
y_38_train_predicted = model_38.predict(X_38_train)

rmse_38_train = np.sqrt(mean_squared_error(y_38_train, y_38_train_predicted))
rmse_train.append(rmse_38_train)

#### Fitting the model on the data from test set
y_38_test_predicted = model_38.predict(X_38_test)
r2.append(r2_score(y_38_test,y_38_test_predicted))

rmse_38_test = np.sqrt(mean_squared_error(y_38_test, y_38_test_predicted))
rmse_test.append(rmse_38_test)


#_______________________________________________________________________________________
## Item 39
X_39 = X.loc[39]
X_39 = removing_outliers(X_39)

### Separating data into training and test set
X_39_train, X_39_test =  X_train_test_separation(X_39)
y_39_train, y_39_test =  y_train_test_separation(X_39)

### Scaling the Dataframe
for column in X_39_train:                               
    scaling_columns_seperately(X_39_train,X_39_test,column)

### Fitting the model on the data from training set
model_39 = lm.fit(X_39_train, y_39_train)
y_39_train_predicted = model_39.predict(X_39_train)

rmse_39_train = np.sqrt(mean_squared_error(y_39_train, y_39_train_predicted))
rmse_train.append(rmse_39_train)

### Fitting the model on the data from test set
y_39_test_predicted = model_39.predict(X_39_test)

r2.append(r2_score(y_39_test,y_39_test_predicted))

rmse_39_test = np.sqrt(mean_squared_error(y_39_test, y_39_test_predicted))
rmse_test.append(rmse_39_test)


#____________________________________________________________________________________
## Item 40
X_40 = X.loc[40]
X_40 = removing_outliers(X_40)

### Separating data into training and test set
X_40_train, X_40_test =  X_train_test_separation(X_40)
y_40_train, y_40_test =  y_train_test_separation(X_40)

### Scaling the Dataframe
for column in X_40_train:                               
    scaling_columns_seperately(X_40_train,X_40_test,column)

### Fitting the model on the data from training set
model_40 = lm.fit(X_40_train, y_40_train)
y_40_train_predicted = model_40.predict(X_40_train)

rmse_40_train = np.sqrt(mean_squared_error(y_40_train, y_40_train_predicted))
rmse_train.append(rmse_40_train)

### Fitting the model on the data from test set
y_40_test_predicted = model_40.predict(X_40_test)

r2.append(r2_score(y_40_test,y_40_test_predicted))

rmse_40_test = np.sqrt(mean_squared_error(y_40_test, y_40_test_predicted))
rmse_test.append(rmse_40_test)
#____________________________________________________________________________________


## Item 41
X_41 = X.loc[41]
X_41 = removing_outliers(X_41)

### Separating data into training and test set
X_41_train, X_41_test =  X_train_test_separation(X_41)
y_41_train, y_41_test =  y_train_test_separation(X_41)

### Scaling the Dataframe
for column in X_41_train:                               
    scaling_columns_seperately(X_41_train,X_41_test,column)

### Fitting the model on the data from training set
model_41 = lm.fit(X_41_train, y_41_train)
y_41_train_predicted = model_41.predict(X_41_train)

rmse_41_train = np.sqrt(mean_squared_error(y_41_train, y_41_train_predicted))
rmse_train.append(rmse_41_train)

### Fitting the model on the data from test set
y_41_test_predicted = model_41.predict(X_41_test)

r2.append(r2_score(y_41_test,y_41_test_predicted))

rmse_41_test = np.sqrt(mean_squared_error(y_41_test, y_41_test_predicted))
rmse_test.append(rmse_41_test)


#_______________________________________________________________________________________
## Item 57

X_57 = X.loc[57]
X_57 = removing_outliers(X_57)

### Separating data into training and test set
X_57_train, X_57_test =  X_train_test_separation(X_57)
y_57_train, y_57_test =  y_train_test_separation(X_57)

### Scaling the Dataframe
for column in X_57_train:                               
    scaling_columns_seperately(X_57_train,X_57_test,column)

### Fitting the model on the data from training set
model_57 = lm.fit(X_57_train, y_57_train)
y_57_train_predicted = model_57.predict(X_57_train)

rmse_57_train = np.sqrt(mean_squared_error(y_57_train, y_57_train_predicted))
rmse_train.append(rmse_57_train)

### Fitting the model on the data from test set
y_57_test_predicted = model_57.predict(X_57_test)

r2.append(r2_score(y_57_test,y_57_test_predicted))

rmse_57_test = np.sqrt(mean_squared_error(y_57_test, y_57_test_predicted))
rmse_test.append(rmse_57_test)

print()
print("******************************************************************************")
print()
print("ITEM\t rmse_train\t\t rmse_test\t\t r2")
for i in range(0,len(list_of_items)):
    print (list_of_items[i],"\t", rmse_train[i],"\t",rmse_test[i],"\t", r2[i])
print()
print("******************************************************************************")
print()