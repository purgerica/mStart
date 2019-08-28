import time
start = time.time()

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

def test_train_separation (self):
    
    aux = self.loc[self.index.get_level_values('date') <= '2018-01-01']
    train_X = aux.drop('amount',axis=1)
    train_y = aux['amount']
    
    aux = self.loc[self.index.get_level_values('date') > '2018-01-01']
    test_X = aux.drop('amount',axis=1)
    test_y = aux['amount']
    
    return train_X, test_X, train_y, test_y

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

X = pd.read_parquet("C:/Users/vvrhovec/Veronika/kodovi/radni_folder/parquet/X_prices_stores_amount.parquet")

X_38 = X.loc[38]
X_39 = X.loc[39]
X_40 = X.loc[40]
X_41 = X.loc[41]
X_57 = X.loc[57]


#___________________________________________________________________________________________________________
X_38 = removing_outliers(X_38)

X_38_train, X_38_test, y_38_train, y_38_test =  test_train_separation(X_38)

X_38_train = pipeline_from_pickle.fit_transform(X_38_train)
X_38_test = pipeline_from_pickle.transform(X_38_test)

model_38 = lm.fit(X_38_train, y_38_train)

y_38_train_predicted = model_38.predict(X_38_train)

model_38.score(X_38_train,y_38_train)

rmse_train.append(np.sqrt(mean_squared_error(y_38_train, y_38_train_predicted)))

y_38_test_predicted = model_38.predict(X_38_test)

r2_score(y_38_test,y_38_test_predicted)

r2.append(r2_score(y_38_test,y_38_test_predicted))

rmse_test.append(np.sqrt(mean_squared_error(y_38_test, y_38_test_predicted)))


#___________________________________________________________________________________________________________
X_39 = removing_outliers(X_39)

X_39_train, X_39_test, y_39_train, y_39_test =  test_train_separation(X_39)

X_39_train = pipeline_from_pickle.fit_transform(X_39_train)
X_39_test = pipeline_from_pickle.transform(X_39_test)

model_39 = lm.fit(X_39_train, y_39_train)

y_39_train_predicted = model_39.predict(X_39_train)

model_39.score(X_39_train, y_39_train)

rmse_train.append(np.sqrt(mean_squared_error(y_39_train, y_39_train_predicted)))

y_39_test_predicted = model_39.predict(X_39_test)

r2_score(y_39_test,y_39_test_predicted)

r2.append(r2_score(y_39_test,y_39_test_predicted))

rmse_test.append(np.sqrt(mean_squared_error(y_39_test, y_39_test_predicted)))


#___________________________________________________________________________________________________________
X_40 = removing_outliers(X_40)

X_40_train, X_40_test, y_40_train, y_40_test =  test_train_separation(X_40)

X_40_train = pipeline_from_pickle.fit_transform(X_40_train)
X_40_test = pipeline_from_pickle.transform(X_40_test)

model_40 = lm.fit(X_40_train, y_40_train)

y_40_train_predicted = model_40.predict(X_40_train)

model_40.score(X_40_train, y_40_train)

rmse_train.append(np.sqrt(mean_squared_error(y_40_train, y_40_train_predicted)))

y_40_test_predicted = model_40.predict(X_40_test)

r2_score(y_40_test,y_40_test_predicted)

r2.append(r2_score(y_40_test,y_40_test_predicted))

rmse_test.append(np.sqrt(mean_squared_error(y_40_test, y_40_test_predicted)))


#___________________________________________________________________________________________________________
X_41 = removing_outliers(X_41)

X_41_train, X_41_test, y_41_train, y_41_test =  test_train_separation(X_41)

X_41_train = pipeline_from_pickle.fit_transform(X_41_train)
X_41_test = pipeline_from_pickle.transform(X_41_test)

model_41 = lm.fit(X_41_train, y_41_train)

y_41_train_predicted = model_41.predict(X_41_train)

model_41.score(X_41_train, y_41_train)

rmse_train.append(np.sqrt(mean_squared_error(y_41_train, y_41_train_predicted)))

y_41_test_predicted = model_41.predict(X_41_test)

r2_score(y_41_test,y_41_test_predicted)

r2.append(r2_score(y_41_test,y_41_test_predicted))

rmse_test.append(np.sqrt(mean_squared_error(y_41_test, y_41_test_predicted)))


#___________________________________________________________________________________________________________
X_57 = removing_outliers(X_57)

X_57_train, X_57_test, y_57_train, y_57_test =  test_train_separation(X_57)

X_57_train = pipeline_from_pickle.fit_transform(X_57_train)
X_57_test = pipeline_from_pickle.transform(X_57_test)

model_57 = lm.fit(X_57_train, y_57_train)

y_57_train_predicted = model_57.predict(X_57_train)

model_57.score(X_57_train, y_57_train)

rmse_train.append(np.sqrt(mean_squared_error(y_57_train, y_57_train_predicted)))

y_57_test_predicted = model_57.predict(X_57_test)

r2_score(y_57_test,y_57_test_predicted)

r2.append(r2_score(y_57_test,y_57_test_predicted))

rmse_test.append(np.sqrt(mean_squared_error(y_57_test, y_57_test_predicted)))


#___________________________________________________________________________________________________________
print()
print("******************************************************************************")
print()
print("ITEM\t rmse_train\t\t rmse_test\t\t r2")
for i in range(0,len(list_of_items)):
    print (list_of_items[i],"\t", rmse_train[i],"\t",rmse_test[i],"\t", r2[i])
print()
print("******************************************************************************")
print()

#___________________________________________________________________________________________________________

end = time.time()
#print("end:\t", end, "\nstart:\t", start,  "\ntime: \t  \t ",end - start)
print("time: ",end - start)
print()