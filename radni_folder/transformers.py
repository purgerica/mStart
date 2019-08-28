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
#___________________________________________________________________________________________________________________________________________________________________________________________


class dayName(TransformerMixin):                                            # transformer that takes date (which is index) and creates new column named "day_of_week"
                                                                            # where it converts the date into a day name
    def __init__(self, day_of_week = True):
        self.dani = day_of_week

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        day_of_week = 0
        
        X['day_of_week'] = X.index.get_level_values('date').day_name()
        return X
#___________________________________________________________________________________________________________________________________________________________________________________________

    
# following transformers are taken (and some of them changed a bit)
# from https://github.com/jem1031/pandas-pipelines-custom-transformers/blob/master/code/custom_transformers.py#L256
# Julie Michelman, PyData, Seattle 2017


class DFFunctionTransformer(TransformerMixin):
    # FunctionTransformer but for pandas DataFrames

    def __init__(self, *args, **kwargs):
        self.ft = FunctionTransformer(*args, **kwargs)

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        Xt = self.ft.transform(X)
        Xt = pd.DataFrame(Xt, index=X.index, columns=X.columns)
        return Xt

#___________________________________________________________________________________________________________________________________________________________________________________________


class DFFeatureUnion(TransformerMixin):
    # FeatureUnion but for pandas DataFrames

    def __init__(self, transformer_list):
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

#___________________________________________________________________________________________________________________________________________________________________________________________

class DFImputer(TransformerMixin):
    # Imputer but for pandas DataFrames

    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imp = None
        self.statistics_ = None

    def fit(self, X, y=None):
        self.imp = Imputer(strategy=self.strategy)
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled

#___________________________________________________________________________________________________________________________________________________________________________________________

class dummy_days(TransformerMixin):

    def __init__(self,day_of_week = True):
        self.dani = day_of_week

    def fit(self, X, y = None):
        return self    
    
    def transform(self, X, y = None):
        encoded_columns = pd.get_dummies(X['day_of_week'])
        
        X = X.join(encoded_columns)
        X = X.drop('day_of_week',axis=1)
        return X
             
        
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

#___________________________________________________________________________________________________________________________________________________________________________________________


class ColumnExtractor(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols

#___________________________________________________________________________________________________________________________________________________________________________________________

class ZeroFillTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xz = X.fillna(value=0)
        return Xz

#___________________________________________________________________________________________________________________________________________________________________________________________

class DummyTransformer(TransformerMixin):

    def __init__(self):
        self.dv = None

    def fit(self, X, y = None):
        return self    
    
    def transform(self, X, y = None):
        encoded_columns = pd.get_dummies(self['day_of_week'])
        
        X = X.join(encoded_columns)
        X = X.drop('day_of_week',axis=1)
        return X
             
#___________________________________________________________________________________________________________________________________________________________________________________________ 

class MultiEncoder(TransformerMixin):
    # Multiple-column MultiLabelBinarizer for pandas DataFrames

    def __init__(self, sep=','):
        self.sep = sep
        self.mlbs = None

    def _col_transform(self, x, mlb):
        cols = [''.join([x.name, '=', c]) for c in mlb.classes_]
        xmlb = mlb.transform(x)
        xdf = pd.DataFrame(xmlb, index=x.index, columns=cols)
        return xdf

    def fit(self, X, y=None):
        Xsplit = X.applymap(lambda x: x.split(self.sep))
        self.mlbs = [MultiLabelBinarizer().fit(Xsplit[c]) for c in X.columns]
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xsplit = X.applymap(lambda x: x.split(self.sep))
        Xmlbs = [self._col_transform(Xsplit[c], self.mlbs[i])
                 for i, c in enumerate(X.columns)]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xmlbs)
        return Xunion

    
#____________________________________________________________________________________________________________________________________________________________________________________________________
class DFRobustScaler(TransformerMixin):
    # RobustScaler but for pandas DataFrames

    def __init__(self):
        self.rs = None
        self.center_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.rs = RobustScaler()
        self.rs.fit(X)
        self.center_ = pd.Series(self.rs.center_, index=X.columns)
        self.scale_ = pd.Series(self.rs.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xrs = self.rs.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=X.columns)
        return Xscaled

#_________________________________________________________    
class ZeroFillTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xz = X.fillna(value=0)
        return Xz

    
class Log1pTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xlog = np.log1p(X)
        return Xlog
