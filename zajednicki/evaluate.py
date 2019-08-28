import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from funkcije import scale,ColumnExtractor,DummyTransformer,DFFeatureUnion, ZeroFillTransformer,DFStandardScaler,dayName,holiday
from funkcije import log,log2,log10,kroz,korijen,IQ,devetBani,STD,nista

from sklearn.externals import joblib
from math import e, sqrt


# FILEOVI

X38 = pd.read_parquet("C:/Users/vvrhovec/Veronika/kodovi/zajednicki/parquet/item_X38.parquet")
X39 = pd.read_parquet("C:/Users/vvrhovec/Veronika/kodovi/zajednicki/parquet/item_X39.parquet")
X40 = pd.read_parquet("C:/Users/vvrhovec/Veronika/kodovi/zajednicki/parquet/item_X40.parquet")
X41 = pd.read_parquet("C:/Users/vvrhovec/Veronika/kodovi/zajednicki/parquet/item_X41.parquet")
X57 = pd.read_parquet("C:/Users/vvrhovec/Veronika/kodovi/zajednicki/parquet/item_X57.parquet")

ALFA = joblib.load('ALFA.sav')
BETA = joblib.load('BETA.sav')
GAMA = joblib.load('GAMA.sav')
DELTA = joblib.load('DELTA.sav')

# PRIPREMA

test_df = pd.DataFrame()
test_df = pd.DataFrame(index =[38,39,40,41,57], columns =['baseline']) 

skaliranje = [log, log2, log10, kroz, korijen, nista]
outlieri = [IQ, devetBani, STD, nista]
pipe = [ALFA, BETA, GAMA, DELTA]
nacin=[]

# FUNKCIJE

def to_eng(table):
    #print('tu sam')
    table = table.rename(columns={"redna_cena":"regular_price", "akcijska_cena":"discounted_price","n_trgovin":"no_stores","kol":"amount"})
    table = table.rename_axis(['item','date'])
    return table
    
def baseline (df):
    X_train, X_test = train_test_split(df, shuffle = False, test_size = 0.2)
    #print(X_train)
    X_train_mean = X_train['amount'].mean()
    
    predictions_train = [X_train_mean] * X_train.shape[0] 
    predictions_test = [X_train_mean] * X_test.shape[0] 
    
    #train_df.loc[(df.index[0][0]),'baseline'] = np.sqrt(mean_squared_error(X_train['amount'], predictions_train))
    test_df.loc[(df.index[0][0]),'baseline'] = np.sqrt(mean_squared_error(X_test['amount'], predictions_test)) 

def SO(table,postotak=0):                             # PROVJERI JEL TREBA ONDA UOPCE OVO!!!!!
    pd.options.mode.chained_assignment = None  # default='warn'
    #print(postotak)
    MIN_J = 5000      #    postavimo neki veliki min
    NACIN_J = ''      #    prazan string koji ce nam reci kako smo dosli do rezultata
    MIN_D = 5000
    NACIN_D = ''
    DULJINA = len(table.index)
    
    a = 0
    b = 0
    c = 0
    d = 0
    f = 0
    
    
    #print('----------------------------------------------------OVO JE POCETAK-----------------------------------------------------')
    #print(table.head)
    #tablica=table
    for i in range(len(skaliranje)):                
        for z in range (len(outlieri)):
            tablica = table.copy()    # radimo na tablici,table cuvamo
            if postotak!=0:
                tablica_train, tablica_test = train_test_split(tablica, test_size=postotak,shuffle=False)
            
                proces1 = skaliranje[i]()
                proces2 = outlieri[z]()
            
            
                tablica_train = proces1.fit_transform(tablica_train)
                tablica_test = proces1.transform(tablica_test) 
                proces2.fit(tablica_train)
                tablica_train = proces2.transform(tablica_train)
                tablica_test = proces2.transform(tablica_test)           
                X_test = tablica_test.drop('amount', axis=1)
                y_test = tablica_test['amount']
                X_train = tablica_train.drop('amount', axis=1)
                y_train = tablica_train['amount']
            
                for j in range(len(pipe)):
                    #print('tu sam--->',i,'-----',j)#----------------------------AKO ZELIS PROVJERAVAT ZA JEDNOG
                    string = (str(skaliranje[i]))[17:-2] + '+' + (str(outlieri[z]))[17:-2] + '+' + str(j)   
                    pipe[j].fit(X_train,y_train)
                    predictions = pipe[j].predict(X_test)
                
                #print(predictions[:10])
                
                    if i == 0:                                    # VRACANJE TESTA SUKLADNO SKALIRANJU     
                    #print("............................................. ",predictions[:5])
                        predictions = np.power(e,predictions)          # <-----------------------------e je jednak nula
                    #print(e)
                    #print("............................................. ",predictions[:5])
                    
                        if a == 0:  
                        #print("............................................. ",y_test[:5])
                            y_test = np.power(e,y_test)
                        #print("............................................. ",y_test[:5])
                            a += 1
                    
                    
                    if i == 1:
                        predictions = np.power(2,predictions)
                    #y_test = np.power(2,y_test)
                        if b == 0:  
                        #print("............................................. ",y_test[:5])
                            y_test = np.power(2,y_test)
                        #print("............................................. ",y_test[:5])
                            b += 1
                    if i == 2:
                        predictions = np.power(10,predictions)
                    #y_test = np.power(10,y_test)
                        if c == 0:  
                        #print("............................................. ",y_test[:5])
                            y_test = np.power(10,y_test)
                        #print("............................................. ",y_test[:5])
                            c += 1
                    if i == 3:
                        predictions = np.power(predictions,-1)
                    #y_test = np.power(y_test,-1)
                        if d == 0:  
                        #print("............................................. ",y_test[:5])
                            y_test = np.power(y_test,-1)
                        #print("............................................. ",y_test[:5])
                            d += 1
                    if i == 4:
                        predictions = np.power(predictions,2)
                    #y_test = np.power(y_test,2)
                        if f == 0:  
                        #print("............................................. ",y_test[:5])
                            y_test = np.power(y_test,2)
                        #print("............................................. ",y_test[:5])
                            f += 1
                #print('tu sam--->',i,'---',z,'--',j,'posto podataka:',len(tablica.index)/DULJINA)    #DA VIDIS KOLIKO IMA PODATAKA
                    rms = sqrt(mean_squared_error(y_test, predictions))
    
                #test_df[string] = 0
                #print(table.index[0][0],string)    ### PROMJENI STRING IMA 4 NACINA
                
                    test_df.loc[(table.index[0][0]),string] = rms
                #test_df.loc[(df.index[0][0]),string] = rms
    
                
    
                #print ('tu sam',i,z,j,'---------->',rms)#----------------------------AKO ZELIS PROVJERAVAT ZA KOLIKO JE RMS
                    if (DULJINA - len(tablica.index)) / DULJINA < 0.1:
                        if rms < MIN_J:
                            MIN_J = rms
                            NACIN_J = str(i) + '+' + str(z) + '+' + str(j)
                    if (DULJINA - len(tablica.index)) / DULJINA < 0.2:
                        if rms < MIN_D:
                            MIN_D = rms
                            NACIN_D = str(i) + '+' + str(z) + '+' + str(j)
            else:
                proces1 = skaliranje[i]()
                proces2 = outlieri[z]()
            
            
                tablica = proces1.fit_transform(tablica)
                proces2.fit(tablica)
                tablica = proces2.transform(tablica)
                       
                X_train = tablica.drop('amount', axis=1)
                y_train = tablica['amount']
            
                for j in range(len(pipe)):
                    #print('tu sam--->',i,'-----',j)#----------------------------AKO ZELIS PROVJERAVAT ZA JEDNOG
                    string = (str(skaliranje[i]))[17:-2] + '+' + (str(outlieri[z]))[17:-2] + '+' + str(j)   
                    pipe[j].fit(X_train,y_train)
                    predictions = pipe[j].predict(X_train)
                
                #print(predictions[:10])
                
                    if i == 0:                                    # VRACANJE TESTA SUKLADNO SKALIRANJU     
                    #print("............................................. ",predictions[:5])
                        predictions = np.power(e,predictions)          # <-----------------------------e je jednak nula
                    #print(e)
                    #print("............................................. ",predictions[:5])
                    
                        if a == 0:  
                        #print("............................................. ",y_test[:5])
                            y_test = np.power(e,y_train)
                        #print("............................................. ",y_test[:5])
                            a += 1
                    
                    
                    if i == 1:
                        predictions = np.power(2,predictions)
                    #y_test = np.power(2,y_test)
                        if b == 0:  
                        #print("............................................. ",y_test[:5])
                            y_test = np.power(2,y_train)
                        #print("............................................. ",y_test[:5])
                            b += 1
                    if i == 2:
                        predictions = np.power(10,predictions)
                    #y_test = np.power(10,y_test)
                        if c == 0:  
                        #print("............................................. ",y_test[:5])
                            y_test = np.power(10,y_train)
                        #print("............................................. ",y_test[:5])
                            c += 1
                    if i == 3:
                        predictions = np.power(predictions,-1)
                    #y_test = np.power(y_test,-1)
                        if d == 0:  
                        #print("............................................. ",y_test[:5])
                            y_test = np.power(y_train,-1)
                        #print("............................................. ",y_test[:5])
                            d += 1
                    if i == 4:
                        predictions = np.power(predictions,2)
                    #y_test = np.power(y_test,2)
                        if f == 0:  
                        #print("............................................. ",y_test[:5])
                            y_test = np.power(y_train,2)
                        #print("............................................. ",y_test[:5])
                            f += 1
                #print('tu sam--->',i,'---',z,'--',j,'posto podataka:',len(tablica.index)/DULJINA)    #DA VIDIS KOLIKO IMA PODATAKA
                    rms = sqrt(mean_squared_error(y_train, predictions))
    
                #test_df[string] = 0
                #print(table.index[0][0],string)    ### PROMJENI STRING IMA 4 NACINA
                
                    test_df.loc[(table.index[0][0]),string] = rms
                #test_df.loc[(df.index[0][0]),string] = rms
    
                
    
                #print ('tu sam',i,z,j,'---------->',rms)#----------------------------AKO ZELIS PROVJERAVAT ZA KOLIKO JE RMS
                    if (DULJINA - len(tablica.index)) / DULJINA < 0.1:
                        if rms < MIN_J:
                            MIN_J = rms
                            NACIN_J = str(i) + '+' + str(z) + '+' + str(j)
                    if (DULJINA - len(tablica.index)) / DULJINA < 0.2:
                        if rms < MIN_D:
                            MIN_D = rms
                            NACIN_D = str(i) + '+' + str(z) + '+' + str(j)
                
        #print('----------------------------------------------KRAJ LOOPA-----------------------------------------------')
    PATH = (str(skaliranje[int(NACIN_J[0])]))[17:-2] + '+' + (str(outlieri[int(NACIN_J[2])]))[17:-2] + '+' + str(NACIN_J[4])   
    nacin.append(str(table.index[0][0])+' '+PATH)
    test_df.loc[(table.index[0][0]),'put'] = NACIN_J
    print ('Najmanju greska koju dobijemo a da sacuvamo barem 90% podataka je :',MIN_J,',a dobijemo ju na ovaj nacin:',NACIN_J)
    print ('Najmanju greska koju dobijemo a da sacuvamo barem 80% podataka je :',MIN_D,',a dobijemo ju na ovaj nacin:',NACIN_D,'\n')    

def saveModela(X):
    pas=test_df.loc[X.index[0][0],'put']
    X_train = X.drop('amount', axis=1)
    y_train = X['amount']
    pipe[int(pas[4])].fit(X_train,y_train)
    filename = 'Model'+str(X.index[0][0])+'.sav'
    joblib.dump(pipe[int(pas[4])], filename)
    return

# POZIVANJE FJA  

X38=to_eng(X38)
X39=to_eng(X39)
X40=to_eng(X40)
X41=to_eng(X41)
X57=to_eng(X57)

"""X38= X38[:-1]
X39= X39[:-1]
X40= X40[:-1]
X41= X41[:-1]
X57= X57[:-1]"""


baseline(X38)
baseline(X39)
baseline(X40)
baseline(X41)
baseline(X57)

SO(X38)
SO(X39)
SO(X40)
SO(X41)
SO(X57)

saveModela(X38)
saveModela(X39)
saveModela(X40)
saveModela(X41)
saveModela(X57)

print(test_df)

print(test_df.min(axis=1))