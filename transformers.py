import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from functools import reduce

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import Imputer, MultiLabelBinarizer

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm

import datetime
from dateutil.parser import *

from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion


#__________________________________________________________________________________________________________________________________________________________________________

class dayName(TransformerMixin):                                            # transformer that takes date (which is index) and creates new column named "day_of_week"
                                                                            # where it converts the date into a day name
    def __init__(self,day_of_week = True):
        self.dani = day_of_week

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        day_of_week = 0
        
        X['day_of_week'] = X.index.get_level_values('date').day_name()
        return X

#__________________________________________________________________________________________________________________________________________________________________________


class dummy_days(TransformerMixin):                                          # takes the day name from the column "day_of_week" and creates 7 new columns
                                                                             # Mon, Tue, ..., Sun where it inputs 0 or 1, depending on which day it is

    def __init__(self,day_of_week = True):
        self.dani = day_of_week

    def fit(self, X, y = None):
        return self    
    
    def transform(self, X, y = None):
        encoded_columns = pd.get_dummies(X['day_of_week'])
        
        X = X.join(encoded_columns)
        X = X.drop('day_of_week',axis=1)
        return X
             
             
#__________________________________________________________________________________________________________________________________________________________________________

                            # following transformers are taken (and some of them changed a bit) from
                            # https://github.com/jem1031/pandas-pipelines-custom-transformers/blob/master/code/custom_transformers.py#L256
                            # Julie Michelman, PyData, Seattle 2017

#__________________________________________________________________________________________________________________________________________________________________________

class DFStandardScaler(TransformerMixin):
    # StandardScaler but for pandas DataFrames

    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled

#__________________________________________________________________________________________________________________________________________________________________________

class ZeroFillTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xz = X.fillna(value=0)
        return Xz

#__________________________________________________________________________________________________________________________________________________________________________

class DFFeatureUnion(TransformerMixin):
    # FeatureUnion but for pandas DataFrames

    def __init__(self):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion
 