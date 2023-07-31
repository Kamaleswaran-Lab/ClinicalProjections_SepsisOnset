# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 23:18:54 2021

@author: mehak
"""
import numpy as np
import random
import pandas as pd
import pickle

from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans


def norm_to_none(x,name,ranges):
    c1,c2,c3,c4 = ranges[name][:4]
    log_variables = ['Creatinine', 'Bilirubin_direct', 
                     'Bilirubin_total', 'Glucose', 
                     'Lactate','WBC', 'TroponinI']

    #reverse normalization of normal range
    y = x*(c4-c3) + c3
    if(name in log_variables):
        y = 10**y-1
    return y

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def train_test_split(data, pat_ids, seed):
    
    data['patient_id'] = pat_ids
    s_index = data.loc[data['SepsisLabel'] == 1.0].index
    sepsis_patients = np.unique(data.loc[s_index]['patient_id'].values)
    
    random.seed(seed)
    samples = random.sample(set(sepsis_patients), len(sepsis_patients))
    
    test_sepsis = samples[0:int(0.25*len(samples))]
    train_sepsis = samples[int(0.25*len(samples)):]
    
    df_sepsis_test = data.loc[data['patient_id'].isin(test_sepsis)]
    df_sepsis_train = data.loc[data['patient_id'].isin(train_sepsis)]
    
    print("Train Sepsis: ", len(df_sepsis_train))
    print("Test Sepsis: ", len(df_sepsis_test))
    
    ns_index = data.loc[data['SepsisLabel'] == 0.0].index
    nonsepsis_patients = np.unique(data.loc[ns_index]['patient_id'].values)
    
    random.seed(seed)
    samples = random.sample(set(nonsepsis_patients), len(nonsepsis_patients))
    
    test_nonsepsis = samples[0:int(0.25*len(samples))]
    train_nonsepsis = samples[int(0.25*len(samples)):]
    
    df_nonsepsis_test = data.loc[data['patient_id'].isin(test_nonsepsis)]
    df_nonsepsis_train = data.loc[data['patient_id'].isin(train_nonsepsis)]
    
    print("Train non Sepsis: ", len(df_nonsepsis_train))
    print("Test non Sepsis: ", len(df_nonsepsis_test))
    
    
    dfX = pd.concat([df_sepsis_train, df_nonsepsis_train])
    dfy = dfX['SepsisLabel']
    dfPatID = dfX['patient_id']
    dfX = dfX.drop('SepsisLabel', axis = 1)
    dfX = dfX.drop('patient_id', axis = 1)
    
    TestX = pd.concat([df_sepsis_test, df_nonsepsis_test])
    Testy = TestX['SepsisLabel']
    TestPatID = TestX['patient_id']
    TestX = TestX.drop('SepsisLabel', axis = 1)
    TestX = TestX.drop('patient_id', axis = 1)
    
    print ("dfX: ", dfX.shape)
    print ("dfy: ", dfy.shape)
    print("TestX: ", TestX.shape)
    print ("Testy: ", Testy.shape)
    
    return dfX, dfy, TestX, Testy, dfPatID, TestPatID

def smote_pipeline():
    over = SMOTE(sampling_strategy=0.25, random_state=14)
    under = RandomUnderSampler(sampling_strategy=0.35, random_state=7)
    pipeline = Pipeline(steps=[('o', over),('u', under)])
    return pipeline

def clustering(X, clusters):
    
    print("Clustering ... ")
    kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 500, "random_state":20}
    kmeans = KMeans(n_clusters=clusters, **kmeans_kwargs)
    kmeans.fit(X)
    
    return kmeans

def save_obj(object_var, folder, filename):
    
    if(filename[-3:] == 'pkl'):
        with open(folder + filename,'wb') as f:
            pickle.dump(object_var,f)
    else:
        with open(folder + filename + '.pkl','wb') as f:
            pickle.dump(object_var,f)
    
        
def load_obj(folder, filename):
    
    if(filename[-3:] == 'pkl'):
        with open(folder + filename,'rb') as f:
            object_var = pickle.load(f)
    else:
        with open(folder + filename + '.pkl','rb') as f:
            object_var = pickle.load(f)
        
    return object_var