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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm

import datetime
from dateutil.parser import *

from sklearn.pipeline import Pipeline
from transformers import dayName, dummy_days, DFStandardScaler, ZeroFillTransformer, DFFeatureUnion

import pickle

def removing_outliers(self):
    upper_lim1 = self['amount'].quantile(.95)
    lower_lim1 = self['amount'].quantile(.05)
    self=self[(self['amount'] < upper_lim1) & (self['amount'] > lower_lim1)]
    
    upper_lim2 = self['amount'].quantile(.975)
    lower_lim2 = self['amount'].quantile(.025)
    self=self[(self['amount'] < upper_lim2) & (self['amount'] > lower_lim2)]
    
    upper_lim3 = self['amount'].quantile(.99)
    lower_lim3 = self['amount'].quantile(.01)
    self=self[(self['amount'] < upper_lim3) & (self['amount'] > lower_lim3)]    
    
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

std_scaler = StandardScaler()
lm = linear_model.LinearRegression()

list_of_items = [38,39,40,41,57] 

rmse_train = []
rmse_test = []
r2 = []

categorical_features = ['day_of_week']
numerical_features = ['regular_price', 'discounted_price','number_of_stores']

pipeline_pickle_path = 'pipeline_pickle.pkl'
pipeline_unpickle = open(pipeline_pickle_path, 'rb')
  
pipeline_from_pickle = pickle.load(pipeline_unpickle)


import workalendar.europe
from datetime import date
from workalendar.europe import Slovenia
cal = Slovenia()

lista =[]

for i in range (0,len(cal.holidays(2016))-1):
    lista.append(cal.holidays(2016)[i][0])

for i in range (0,len(cal.holidays(2017))-1):
    lista.append(cal.holidays(2017)[i][0])

for i in range (0,len(cal.holidays(2018))-1):
    lista.append(cal.holidays(2018)[i][0])


def holiDay(self):    
    self ['holidays']  = 0
    for i in range (0,self.shape[0]):
        for j in range (0, len(lista)):
            if self.index.get_level_values('date')[i].date() == lista[j] :
                self['holidays'][i] = 0
                self['holidays'][i-1] = 1
                self['holidays'][i-2] = 1
    return self
                      
def payDay (self):           
    self ['paycheque_days']  = 0
    for i in range (0,self.shape[0]):
        if ((self.index.get_level_values('date')[i].date().day>10) & (self.index.get_level_values('date')[i].date().day<20)):
            self['paycheque_days'][i] = 1
    return self

X = pd.read_parquet("C:/Users/vvrhovec/Veronika/kodovi/radni_folder/parquet/X_prices_stores_amount.parquet")

X_38 = X.loc[38]
X_39 = X.loc[39]
X_40 = X.loc[40]
X_41 = X.loc[41]
X_57 = X.loc[57]


X_38 = payDay(X_38)
X_39 = payDay(X_39)
X_40 = payDay(X_40)
X_41 = payDay(X_41)
X_57 = payDay(X_57)


X_38 = holiDay(X_38)
X_39 = holiDay(X_39)
X_40 = holiDay(X_40)
X_41 = holiDay(X_41)
X_57 = holiDay(X_57)



X_38 = removing_outliers(X_38)

X_38_train, X_38_test =  X_train_test_separation(X_38)
y_38_train, y_38_test =  y_train_test_separation(X_38)

X_38_train = pipeline_from_pickle.fit_transform(X_38_train)
X_38_test = pipeline_from_pickle.transform(X_38_test)

model_38 = lm.fit(X_38_train, y_38_train)

y_38_train_predicted = model_38.predict(X_38_train)

model_38.score(X_38_train,y_38_train)

mae_38_train = mean_absolute_error(y_38_train, y_38_train_predicted)
mae_38_train

mse_38_train = mean_squared_error(y_38_train, y_38_train_predicted)
mse_38_train

rmse_38_train = np.sqrt(mean_squared_error(y_38_train, y_38_train_predicted))
rmse_train.append(rmse_38_train)
rmse_38_train

y_38_test_predicted = model_38.predict(X_38_test)

mae_38_test = mean_absolute_error(y_38_test, y_38_test_predicted)
mae_38_test

mse_38_test = mean_squared_error(y_38_test, y_38_test_predicted)
mse_38_test

r2_score(y_38_test,y_38_test_predicted)

r2.append(r2_score(y_38_test,y_38_test_predicted))

rmse_38_test = np.sqrt(mean_squared_error(y_38_test, y_38_test_predicted))
rmse_test.append(rmse_38_test)
rmse_38_test

X_39 = removing_outliers(X_39)

X_39_train, X_39_test =  X_train_test_separation(X_39)
y_39_train, y_39_test =  y_train_test_separation(X_39)

X_39_train = pipeline_from_pickle.fit_transform(X_39_train)
X_39_test = pipeline_from_pickle.transform(X_39_test)

model_39 = lm.fit(X_39_train, y_39_train)

y_39_train_predicted = model_39.predict(X_39_train)

model_39.score(X_39_train, y_39_train)

mae_39_train = mean_absolute_error(y_39_train, y_39_train_predicted)
mae_39_train

mse_39_train = mean_squared_error(y_39_train, y_39_train_predicted)
mse_39_train

rmse_39_train = np.sqrt(mean_squared_error(y_39_train, y_39_train_predicted))
rmse_train.append(rmse_39_train)
rmse_39_train

y_39_test_predicted = model_39.predict(X_39_test)

mae_39_test = mean_absolute_error(y_39_test, y_39_test_predicted)
mae_39_test

mse_39_test = mean_squared_error(y_39_test, y_39_test_predicted)
mse_39_test

r2_score(y_39_test,y_39_test_predicted)

r2.append(r2_score(y_39_test,y_39_test_predicted))

rmse_39_test = np.sqrt(mean_squared_error(y_39_test, y_39_test_predicted))
rmse_test.append(rmse_39_test)
rmse_39_test

X_40 = removing_outliers(X_40)

X_40_train, X_40_test =  X_train_test_separation(X_40)
y_40_train, y_40_test =  y_train_test_separation(X_40)

X_40_train = pipeline_from_pickle.fit_transform(X_40_train)
X_40_test = pipeline_from_pickle.transform(X_40_test)

model_40 = lm.fit(X_40_train, y_40_train)

y_40_train_predicted = model_40.predict(X_40_train)

model_40.score(X_40_train, y_40_train)

mae_40_train = mean_absolute_error(y_40_train, y_40_train_predicted)
mae_40_train

mse_40_train = mean_squared_error(y_40_train, y_40_train_predicted)
mse_40_train

rmse_40_train = np.sqrt(mean_squared_error(y_40_train, y_40_train_predicted))
rmse_train.append(rmse_40_train)
rmse_40_train

y_40_test_predicted = model_40.predict(X_40_test)

mae_40_test = mean_absolute_error(y_40_test, y_40_test_predicted)
mae_40_test

mse_40_test = mean_squared_error(y_40_test, y_40_test_predicted)
mse_40_test

r2_score(y_40_test,y_40_test_predicted)

r2.append(r2_score(y_40_test,y_40_test_predicted))

rmse_40_test = np.sqrt(mean_squared_error(y_40_test, y_40_test_predicted))
rmse_test.append(rmse_40_test)
rmse_40_test

X_41 = removing_outliers(X_41)

X_41_train, X_41_test =  X_train_test_separation(X_41)
y_41_train, y_41_test =  y_train_test_separation(X_41)

X_41_train = pipeline_from_pickle.fit_transform(X_41_train)
X_41_test = pipeline_from_pickle.transform(X_41_test)

model_41 = lm.fit(X_41_train, y_41_train)

y_41_train_predicted = model_41.predict(X_41_train)

model_41.score(X_41_train, y_41_train)

mae_41_train = mean_absolute_error(y_41_train, y_41_train_predicted)
mae_41_train

mse_41_train = mean_squared_error(y_41_train, y_41_train_predicted)
mse_41_train

rmse_41_train = np.sqrt(mean_squared_error(y_41_train, y_41_train_predicted))
rmse_train.append(rmse_41_train)
rmse_41_train

y_41_test_predicted = model_41.predict(X_41_test)

mae_41_test = mean_absolute_error(y_41_test, y_41_test_predicted)
mae_41_test

mse_41_test = mean_squared_error(y_41_test, y_41_test_predicted)
mse_41_test

r2_score(y_41_test,y_41_test_predicted)

r2.append(r2_score(y_41_test,y_41_test_predicted))

rmse_41_test = np.sqrt(mean_squared_error(y_41_test, y_41_test_predicted))
rmse_test.append(rmse_41_test)
rmse_41_test

X_57 = removing_outliers(X_57)

X_57_train, X_57_test =  X_train_test_separation(X_57)
y_57_train, y_57_test =  y_train_test_separation(X_57)

X_57_train = pipeline_from_pickle.fit_transform(X_57_train)
X_57_test = pipeline_from_pickle.transform(X_57_test)

model_57 = lm.fit(X_57_train, y_57_train)

y_57_train_predicted = model_57.predict(X_57_train)

model_57.score(X_57_train, y_57_train)

mae_57_train = mean_absolute_error(y_57_train, y_57_train_predicted)
mae_57_train

mse_57_train = mean_squared_error(y_57_train, y_57_train_predicted)
mse_57_train

rmse_57_train = np.sqrt(mean_squared_error(y_57_train, y_57_train_predicted))
rmse_train.append(rmse_57_train)
rmse_57_train

y_57_test_predicted = model_57.predict(X_57_test)

mae_57_test = mean_absolute_error(y_57_test, y_57_test_predicted)
mae_57_test

mse_57_test = mean_squared_error(y_57_test, y_57_test_predicted)
mse_57_test

r2_score(y_57_test,y_57_test_predicted)

r2.append(r2_score(y_57_test,y_57_test_predicted))

rmse_57_test = np.sqrt(mean_squared_error(y_57_test, y_57_test_predicted))
rmse_test.append(rmse_57_test)
rmse_57_test


print()
print("******************************************************************************")
print()
print("ITEM\t rmse_train\t\t rmse_test\t\t r2")
for i in range(0,len(list_of_items)):
    print (list_of_items[i],"\t", rmse_train[i],"\t",rmse_test[i],"\t", r2[i])
print()
print("******************************************************************************")
print()