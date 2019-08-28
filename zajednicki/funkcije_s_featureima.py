from sklearn.base import TransformerMixin
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from scipy.stats import iqr
from functools import reduce
import pandas as pd
import holidays
SI_holidays = holidays.SI()


from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from sklearn.linear_model import LinearRegression

class log(TransformerMixin):
    def fit(self, X, y=None):
        #print (self)
        return self

    def transform(self, X):
        pas=np.log(X.loc[:,'amount'])
        X.loc[:,'amount'] = pas
        return X
    
class log2(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[:,'amount'] = np.log2(X.loc[:,'amount'])
        return X
    
class log10(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[:,'amount'] = np.log10(X.loc[:,'amount'])
        return X
    
class korijen(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[:,'amount'] = np.sqrt(X.loc[:,'amount'])
        return X
    
class kroz(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[:,'amount'] = 1/(X.loc[:,'amount'])
        return X

class nista(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
  
    
class dayName(TransformerMixin):                                            
                                                                           
    def __init__(self,day_of_week = True):
        self.dani = day_of_week
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        day_of_week = 0
        X['day_of_week'] = X.index.get_level_values('date').day_name()
        return X
    
class holiday(TransformerMixin):
    
    def __init__(self,holiday = True):
        self.dani = holiday
    
    def fit(self, X, y = None):
        return self
    
    def transform(self,X,y=None):
        X['holiday']=0
        for index, row in X.iterrows():
            klosar = row.name[1].date() in SI_holidays
            if(klosar == True):
            #print(row.name[0],row.name[1].date())
                X.loc[(row.name[0],row.name[1].date()),'holiday'] = 1
        return X

"""class dpp(TransformerMixin):
    def __init__(self,holidays = True):
        self.dani = holidays
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y=None):   
        import workalendar.europe
        from datetime import date
        from workalendar.europe import Slovenia
        cal = Slovenia()

        lista = []

        for i in range (0,len(cal.holidays(2016))-1):
            lista.append(cal.holidays(2016)[i][0])

        for i in range (0,len(cal.holidays(2017))-1):
            lista.append(cal.holidays(2017)[i][0])
    
        for i in range (0,len(cal.holidays(2018))-1):
            lista.append(cal.holidays(2018)[i][0])

        X ['holidays']  = 0
        X.head()
        for i in range (0, X.shape[0]):
            for j in range (0, len(lista)):
                if X.index.get_level_values('date')[i].date() == lista[j] :
                    X['holidays'][i] = 0
                    X['holidays'][i-1] = 1
                    X['holidays'][i-2] = 1
        return X   

class payDay(TransformerMixin):
    
    def __init__(self, paycheque_days = True):
        self.dani = paycheque_days
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y=None):
        X['paycheque_days'] = 0
        for i in range (0,X.shape[0]):
            if ((X.index.get_level_values('date')[i].date().day>10) & (X.index.get_level_values('date')[i].date().day<20)):
                X['paycheque_days'][i] = 1
        return X
    
class trecine(TransformerMixin):
    
    def __init__(self, prva = True, druga = True, treca = True):
        self.trecina1 = prva
        self.trecina2 = druga
        self.trecina3 = treca
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y=None):
        X['prva'] = 0
        X['druga'] = 0
        X['treca'] = 0
        
        for i in range (0,X.shape[0]):
            if (X.index.get_level_values('date')[i].date().day <= 10):
                X['prva'][i] = 1
            if ((X.index.get_level_values('date')[i].date().day > 10) & (X.index.get_level_values('date')[i].date().day <= 20)): 
                X['druga'][i] = 1
            if (X.index.get_level_values('date')[i].date().day > 20):
                X['treca'][i] = 1
        return X
"""    
    
class IQ(TransformerMixin): 
    
    def __init__(self):
        self.UL = 0
        self.LL = 0
   
    def fit(self,X):
        self.LL = np.percentile(X['amount'],25)-iqr(X['amount'])*(3/2)
        self.UL = np.percentile(X['amount'],75)+iqr(X['amount'])*(3/2)
        #print(self.LL)
        #print(self.UL)
        return X

    def transform(self,X):
        #print(self.LL)
        #print(self.UL)
        X=X[(X['amount'] > self.LL) & (X['amount'] < self.UL)]
        return X


    
class devetBani(TransformerMixin):
    
    def __init__(self):
        self.UL = 0
        self.LL = 0
    
    def fit(self, X, y=None):
        self.UL = np.percentile(X['amount'],95)
        self.LL = np.percentile(X['amount'],5)
        #print ('pas mater')
        return X

    def transform(self, X):
        X = X[(X['amount'] > self.LL) & (X['amount'] < self.UL)]
        return X
    
    
class STD(TransformerMixin):
    
    def __init__(self):
        self.UL = 0
        self.LL = 0
    
    def fit(self, X, y=None):
        self.UL=X['amount'].mean() + X['amount'].std()*1.5
        self.LL=X['amount'].mean() - X['amount'].std()*1.5
        return X

    def transform(self, X):
        X = X[(X['amount'] > self.LL) & (X['amount'] < self.UL)]
        return X
    
class scale(TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaler = StandardScaler()
        for col in X.columns:
                scaler.fit(X[col].values.reshape(-1,1)) # fit na trainu
                col_train_scaled = scaler.transform(X[col].values.reshape(-1,1)) # skaliram red
                X[col]=col_train_scaled
        return X

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

class DummyTransformer(TransformerMixin):

    def __init__(self):
        self.dv = None

    def fit(self, X, y=None):
        # assumes all columns of X are strings
        Xdict = X.to_dict('records')
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(Xdict)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdict = X.to_dict('records')
        Xt = self.dv.transform(Xdict)
        cols = self.dv.get_feature_names()
        cols = [s.replace('day_of_week=', '') for s in cols]
        Xdum = pd.DataFrame(Xt, index=X.index, columns=cols)
        return Xdum

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
    
class ZeroFillTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xz = X.fillna(value=0)
        return Xz
    
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
    
CAT_FEATS = ['day_of_week']
NUM_FEATS = ['regular_price','discounted_price','no_stores','holiday', 'dpp', 'payDay','trecine']



ALFA = Pipeline([('days',dayName()), ('praznici',holiday()), ('dani_prije_praznika',dpp()), ('dani_prije_place',payDay()), ('trecine',trecine()), 
                 ('features', DFFeatureUnion([('numerics', Pipeline([
            ('extract', ColumnExtractor(NUM_FEATS)),
            ('zero_fill', ZeroFillTransformer()),('opop',DFStandardScaler())
            ])),('categoricals', Pipeline([('extract', ColumnExtractor(CAT_FEATS)),
                                           ('dummy', DummyTransformer())]))])),('opop',DFStandardScaler()),('classifier', LinearRegression())])

BETA = Pipeline([('days',dayName()), ('praznici',holiday()), ('dani_prije_praznika',dpp()), ('dani_prije_place',payDay()), ('trecine',trecine()), 
                 ('features', DFFeatureUnion([('numerics', Pipeline([
            ('extract', ColumnExtractor(NUM_FEATS)),
            ('zero_fill', ZeroFillTransformer())
            ])),('categoricals', Pipeline([('extract', ColumnExtractor(CAT_FEATS)),
                                           ('dummy', DummyTransformer())]))])),('opop',DFStandardScaler()),('classifier', LinearRegression())])

GAMA = Pipeline([('days',dayName()), ('praznici',holiday()), ('dani_prije_praznika',dpp()), ('dani_prije_place',payDay()), ('trecine',trecine()), 
                 ('features', DFFeatureUnion([('numerics', Pipeline([
            ('extract', ColumnExtractor(NUM_FEATS)),
            ('zero_fill', ZeroFillTransformer())
            ])),('categoricals', Pipeline([('extract', ColumnExtractor(CAT_FEATS)),
                                           ('dummy', DummyTransformer())]))])),('classifier', LinearRegression())])

DELTA = Pipeline([('days',dayName()), ('praznici',holiday()), ('dani_prije_praznika',dpp()), ('dani_prije_place',payDay()), ('trecine',trecine()), 
                  ('features', DFFeatureUnion([('numerics', Pipeline([
            ('extract', ColumnExtractor(NUM_FEATS)),
            ('zero_fill', ZeroFillTransformer()),('opop',DFStandardScaler())
            ])),('categoricals', Pipeline([('extract', ColumnExtractor(CAT_FEATS)),
                                           ('dummy', DummyTransformer())]))])),('classifier', LinearRegression())])


filename = 'ALFA.sav'
joblib.dump(ALFA, filename)

filename = 'BETA.sav'
joblib.dump(BETA, filename)

filename = 'GAMA.sav'
joblib.dump(GAMA, filename)

filename = 'DELTA.sav'
joblib.dump(DELTA, filename)
