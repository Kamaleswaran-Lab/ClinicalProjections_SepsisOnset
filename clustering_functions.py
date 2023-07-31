# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 18:38:05 2022

@author: mehak
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import date
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn import preprocessing


from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.metrics import silhouette_score
# Dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.express as px
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

import utils

import sys
import warnings
warnings.filterwarnings("ignore")
import time
import shutil
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import matplotlib

import holoviews as hv
from holoviews import opts, dim
from bokeh.plotting import show, output_file
from holoviews.plotting import Plot
import numpy as np

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def generate_colormap(N):
    arr = np.arange(N)/N
    N_up = int(math.ceil(N/7)*7)
    arr.resize(N_up)
    arr = arr.reshape(7,N_up//7).T.reshape(-1)
    ret = matplotlib.cm.hsv(arr)
    n = ret[:,3].size
    a = n//2
    b = n-a
    for i in range(3):
        ret[0:n//2,i] *= np.arange(0.2,1,0.8/a)
    ret[n//2:,3] *= np.arange(1,0.1,-0.9/b)
#     print(ret)
    return ret


def chord_diagram(matrix, name):
    hv.extension('bokeh')
    hv.output(size = 200)
    n = len(matrix)
    data = hv.Dataset((list(np.arange(n)), list(np.arange(n)), matrix),
                  ['source', 'target'], 'value').dframe()

    Plot.fig_rcparams={'axes.labelsize':40, 'axes.titlesize':40}

    color_map = ListedColormap(generate_colormap(n))
    chord = hv.Chord(data.astype('int32')).opts(fontsize = {'labels' : 20})
    chord.opts(  
        node_color='index', edge_color='source', labels ='index', 
        cmap= color_map, edge_cmap= color_map, width =500, height=500)
    chord.opts(label_text_font_size='15pt')
    output_file(name)
    show(hv.render(chord))
    
    

def norm_to_none(x,name,ranges):
    c1,c2,c3,c4 = ranges[name][:4]
    log_variables = ['Creatinine', 'Bilirubin_direct', 'Bilirubin_total', 'Glucose', 'Lactate', 'WBC', 'TroponinI']
    #reverse normalization of normal range
    y = x*(c4-c3) + c3
    if(name in log_variables):
        y = 10**(y) - 1
        
    return y

def get_ranges(constraint_file_name):
    #Set the constraints for each variable
    ranges = {}
    
    #The min and max values of these variables are log transformed 
    log_variables = ['Creatinine', 'Bilirubin_direct', 'Bilirubin_total', 'Glucose', 'Lactate', 'WBC', 'TroponinI']
    
    with open(constraint_file_name, 'r') as f:
        for x in f:
            line = x.replace('\n', '').split(', ')
            if line[0] in log_variables:
                ranges[line[0]] = [np.log10(float(i) +1) for i in line[1:]]
            else:
                ranges[line[0]] = [float(i) for i in line[1:]]

    return ranges

ranges = get_ranges('constraints_wo_calcium.txt')

def clustering_stats(clustering_labels_kmeans12, dfX):
    
    dfX['kmeans'] = clustering_labels_kmeans12
    df_mean = (dfX.loc[dfX.kmeans!=-1, :]
                        .groupby('kmeans').mean())

    df_median = (dfX.loc[dfX.kmeans!=-1, :]
                        .groupby('kmeans').median())

    df_std = (dfX.loc[dfX.kmeans!=-1, :]
                        .groupby('kmeans').std())

    columns = list(dfX.columns)
    variables = [s[:-2] for s in columns if '-0' in s]
    print(variables)
    dfX_stats =  pd.DataFrame()

    for var in variables:

        if var in ['Calcium']:
            continue
        elif var in ['SOFA', 'SIRS']:
            dfX_stats[var + '-median'] = df_median[[var + '-' + str(i) for i in range(6)]].median(axis = 1)
            dfX_stats[var + '-std'] = df_std[[var + '-' + str(i) for i in range(6)]].std(axis = 1)
            dfX_stats[var + '-mean'] = df_mean[[var + '-' + str(i) for i in range(6)]].mean(axis = 1)
        else:
            dfX_stats[var + '-median'] = norm_to_none( df_median[[var + '-' + str(i) for i in range(6)]].median(axis = 1), var, ranges)
            dfX_stats[var + '-std'] = norm_to_none(df_std[[var + '-' + str(i) for i in range(6)]].std(axis = 1), var, ranges)
            dfX_stats[var + '-mean'] = norm_to_none(df_mean[[var + '-' + str(i) for i in range(6)]].mean(axis = 1), var, ranges)
    
    return dfX_stats

def sepsis_concentration(clustering_labels_kmeans12, dfy, dfPatID):
    pats = pd.DataFrame(columns = ['SepsisLabel', 'patid'])
    pats['SepsisLabel'] = dfy 
    pats['patid'] = dfPatID.values
    p = pats.groupby('patid').max()
    zip_iterator = zip(p.index, p['SepsisLabel'].values)


    sepsisp = dict(zip_iterator)

    sepsis_conc12 = []

    for c in range(12):
        total = len(clustering_labels_kmeans12[clustering_labels_kmeans12 == c])
        sepsis_label = dfy[clustering_labels_kmeans12 == c]
        sepsis = len(sepsis_label[sepsis_label['SepsisLabel'] == 1])
        sepsis_conc12.append(np.round(sepsis*100/total, 2))

        print("Cluster {} : {} sepsis concentration, {}, {} ".format(c, (sepsis/total)*100, sepsis, total))


    sepsis_prev12 = []

    print( " \n")
    for c in range(12):
        pats_in_cluster = np.unique(dfPatID[clustering_labels_kmeans12 == c])
        sepsis_label = np.array([sepsisp[pat] for pat in pats_in_cluster])
        total = len(sepsis_label)
        sepsis = len(sepsis_label[sepsis_label == 1])
        sepsis_prev12.append(np.round(sepsis*100/total, 2))

        print("Cluster {} : {} sepsis prevalence".format(c, (sepsis/total)*100))
        
    sepsis = pd.DataFrame( columns = ['Sepsis Concentration', 'Sepsis Prevalence'])
    sepsis['Sepsis Concentration'] = sepsis_conc12
    sepsis['Sepsis Prevalence'] = sepsis_prev12
    
    return sepsis

def most_varying_feat( clustering_labels_kmeans12, dfX_feat, name):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(dfX_feat), columns = dfX_feat.columns)
    df_scaled['kmeans'] = clustering_labels_kmeans12
    df_mean = (df_scaled.loc[df_scaled.kmeans!=-1, :]
                        .groupby('kmeans').mean())

    results = pd.DataFrame(columns=['Variable', 'Var'])
    for column in df_mean.columns[1:]:
        results.loc[len(results), :] = [column, np.var(df_mean[column])]
    selected_columns = list(results.sort_values(
            'Var', ascending=False,
        ).head(10).Variable.values) + ['kmeans']
    tidy = df_scaled[selected_columns].melt(id_vars='kmeans')

    sns.set(rc = {'figure.figsize':(15,10)})

    sns.barplot(x='kmeans', y='value', hue='variable', data=tidy)
    plt.title('kmeans 12', fontsize = 20)
    plt.savefig('./' + name, dpi=300)

def randomForest_feat_imp(clustering_labels_kmeans12, dfX_feat, name):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(dfX_feat), columns = dfX_feat.columns)
    y = clustering_labels_kmeans12 
    X = dfX_feat
    df_scaled['kmeans'] = clustering_labels_kmeans12
    clf = RandomForestClassifier(n_estimators=100).fit(X, y)
    selected_columns = list(pd.DataFrame(np.array([clf.feature_importances_, X.columns]).T, columns=['Importance', 'Feature'])
               .sort_values("Importance", ascending=False)
               .head(7)
               .Feature
               .values)

    tidy = df_scaled[selected_columns+['kmeans']].melt(id_vars='kmeans')
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.barplot(x='kmeans', y='value', hue='variable', data=tidy, palette='Set3')
    plt.legend(loc='upper right')
    plt.title('Kmeans 12', fontsize = 20)
    plt.savefig('./' + name , dpi=300)
    
def get_transition_matrix(clustering_labels_kmeans12, dfy, dfPatID, name):
    clusters = 12

    pats = pd.DataFrame(columns = ['SepsisLabel', 'patid'])
    pats['SepsisLabel'] = dfy 
    pats['patid'] = dfPatID
    p = pats.groupby('patid').max()
    zip_iterator = zip(p.index, p['SepsisLabel'].values)

    sepsisp = dict(zip_iterator)

    matrix_kmeans12 = np.zeros((clusters, clusters))
    matrix_kmeans12_control = np.zeros((clusters, clusters))
    index = dfPatID.index
    for i in range(len(dfy)-1):
        
        if(dfPatID[index[i]] == dfPatID[index[i+1]]):
            if(sepsisp[dfPatID[index[i]]] == 1):
                matrix_kmeans12[clustering_labels_kmeans12[i], clustering_labels_kmeans12[i+1]] += 1
            else:
                matrix_kmeans12_control[clustering_labels_kmeans12[i], clustering_labels_kmeans12[i+1]] += 1

    np.save( 'sepsis_' + name, matrix_kmeans12)
    np.save('control_' +  name, matrix_kmeans12_control)
    
    return matrix_kmeans12, matrix_kmeans12_control

def get_transition_matrix_grady(clustering_labels_kmeans12, dfy, dfPatID, name):
    clusters = 12

    pats = pd.DataFrame(columns = ['SepsisLabel', 'patid'])
    pats['SepsisLabel'] = dfy 
    pats['patid'] = dfPatID.values
    p = pats.groupby('patid').max()
    zip_iterator = zip(p.index, p['SepsisLabel'].values)

    sepsisp = dict(zip_iterator)

    matrix_kmeans12 = np.zeros((clusters, clusters))
    matrix_kmeans12_control = np.zeros((clusters, clusters))
    index = dfPatID.index
    dfPatID = dfPatID.values[:,0].astype('int64')
    for i in range(len(dfy)-1):
        
        if(dfPatID[index[i]] == dfPatID[index[i+1]]):
            if(sepsisp[dfPatID[index[i]]] == 1):
                matrix_kmeans12[clustering_labels_kmeans12[i], clustering_labels_kmeans12[i+1]] += 1
            else:
                matrix_kmeans12_control[clustering_labels_kmeans12[i], clustering_labels_kmeans12[i+1]] += 1

    np.save( 'sepsis_' + name, matrix_kmeans12)
    np.save('control_' +  name, matrix_kmeans12_control)
    
    return matrix_kmeans12, matrix_kmeans12_control


def train_classifier(X_train, y_train, X_test, y_test, useGPU = True, final_params = None):


    wts = None
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    res = {}
    dtest = xgb.DMatrix(X_test, label = y_test)

    param_init = {
        "objective": "multi:softmax",
        "num_class": 12,
        "tree_method": "hist",
        "eval_metric": "auc",
        "sampling_method": "uniform",
        "learning_rate" : 0.3,
        "n_estimators": 1000,
        "max_depth":5,
        "min_child_weight":1,
        "gamma":0.1,
        "reg_alpha":0.1,
        "subsample": 1,
        "colsample_bytree":1,
        "nthread":4,
        "scale_pos_weight":1,
        "seed":27
    }
    
    if useGPU:
        param_init['gpu_id'] = 0
        param_init['tree_method'] = 'gpu_hist'
        param_init['sampling_method'] = 'gradient_based'
            
    if final_params is not None:
        params = final_params
    else:
        xgb1 = XGBClassifier(
                 **param_init)
        
        #Get n trees
        cvresult = xgb.cv(xgb1.get_xgb_params(), xgtrain, num_boost_round=param_init["n_estimators"], nfold=5, metrics='auc', \
                          early_stopping_rounds=10)
            
        xgb1.set_params(n_estimators=cvresult.shape[0])
        
        
        print(xgb1.get_params()['n_estimators'])
    
        #Tune tree parameters
    
        param_grid1 = { 
        'max_depth' : range(2,10,1),
        'min_child_weight': np.arange(0.01, 5, 0.5)
        }
        
        hyperparams = []
        accuracies = []
        for max_depth in param_grid1['max_depth']:
            for min_child_weight in param_grid1['min_child_weight']:
                
                xgb1.set_params(max_depth = max_depth, min_child_weight = min_child_weight)
                param = xgb1.get_xgb_params()
                
                model_pred = xgb.train(
                    param,
                    xgtrain,
                    evals=[(dtest, "test")],
                    evals_result=res,
                    early_stopping_rounds=10,
                    num_boost_round=200,
                    xgb_model= None,
                    verbose_eval = False
                )
    
                # Predict training set:
                dtrain_predictions = model_pred.predict(xgtrain)
    
                # Predict test set:
                dtest_predictions = model_pred.predict(dtest)
    
                    
                # Print model report:
                print("\nModel Report")
                auc_train = metrics.roc_auc_score(y_train, dtrain_predictions)
                auc_test = metrics.roc_auc_score(y_test, dtest_predictions)
                print(
                    "AUC Score (Train): %f"
                    % auc_train
                )
                print(
                    "AUC Score (Test): %f"
                    % auc_test
                )
            
                fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, dtrain_predictions)
                fpr, tpr, thresholds = metrics.roc_curve(y_test, dtest_predictions)
                
                gmean = np.sqrt(tpr_train * (1 - fpr_train))
                
                precision, recall, threshpr = metrics.precision_recall_curve(y_test, dtest_predictions)
                fscore = (2 * precision * recall) / (precision + recall)
                fscore[np.isnan(fscore)] = 0
                index = np.argmax(fscore)
                fscoreOpt = round(fscore[index], ndigits = 4)
                thresholdOpt = round(threshpr[index], ndigits = 4)
                    
                #print('Best Threshold: {} with F-Score: {}'.format(thresholdOpt, fscoreOpt))
            
                
                p_threshold = thresholdOpt
                cm = confusion_matrix(y_test.values, (dtest_predictions > p_threshold) * 1)
            
                TP = cm[1][1]
                TN = cm[0][0]
                FP = cm[0][1]
                FN = cm[1][0]
            
                # Sensitivity, hit rate, recall, or true positive rate
                TPR = TP / (TP + FN)
                # Specificity or true negative rate
                TNR = TN / (TN + FP)
                # Precision or positive predictive value
                PPV = TP / (TP + FP)
                # Negative predictive value
                NPV = TN / (TN + FN)
                # Fall out or false positive rate
                FPR = FP / (FP + TN)
                # False negative rate
                FNR = FN / (TP + FN)
                # False discovery rate
                FDR = FP / (TP + FP)
            
                #print("Val Sensitivity:", TPR)
                #print("Val Specificity:", TNR)
                #print("Val Precision:", PPV)
                #print("confusion matrix:\n", cm)
                
                fscore = f1_score(y_test.values, (dtest_predictions > p_threshold) * 1)
                accuracy = accuracy_score(y_test.values, (dtest_predictions > p_threshold) * 1)
                results = {
                    "sensitivity" : TPR,
                    "specificity" : TNR,
                    "precision" : PPV,
                    "cm" : cm,
                    "auc_score" : auc_test,
                    "fpr" : fpr,
                    "tpr" : tpr,
                    "threshold" : p_threshold,
                    "fscore" : fscore,
                    "thresholds_fscore" : threshpr,
                    "accuracy": accuracy
                }
                
                hyperparams.append(param)
                accuracies.append(accuracy)
                
        
        
        
        best_accuracy = np.argmax(accuracies)
        params = hyperparams[best_accuracy]
        
        
        xgb1.set_params(**params)
        param_grid3 = {
        'subsample':[0.01, 0.1, 0.5, 0.3, 1],
        'colsample_bytree':[0.01, 0.1, 0.5, 0.3, 1],
        }
        
        hyperparams = []
        accuracies = []
        
        for subsample in param_grid3['subsample']:
            for colsample_bytree in param_grid3['colsample_bytree']:
                
                xgb1.set_params(subsample = subsample, colsample_bytree = colsample_bytree)
                param = xgb1.get_xgb_params()
                
                model_pred = xgb.train(
                    param,
                    xgtrain,
                    evals=[(dtest, "test")],
                    evals_result=res,
                    early_stopping_rounds=10,
                    num_boost_round=200,
                    xgb_model=wts,
                    verbose_eval = False
                )
    
                # Predict training set:
                dtrain_predictions = model_pred.predict(xgtrain)
    
                # Predict test set:
                dtest_predictions = model_pred.predict(dtest)
    
                    
                # Print model report:
                print("\nModel Report")
                auc_train = metrics.roc_auc_score(y_train, dtrain_predictions)
                auc_test = metrics.roc_auc_score(y_test, dtest_predictions)
                print(
                    "AUC Score (Train): %f"
                    % auc_train
                )
                print(
                    "AUC Score (Test): %f"
                    % auc_test
                )
            
                fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, dtrain_predictions)
                fpr, tpr, thresholds = metrics.roc_curve(y_test, dtest_predictions)
                
                gmean = np.sqrt(tpr_train * (1 - fpr_train))
                
                precision, recall, threshpr = metrics.precision_recall_curve(y_test, dtest_predictions)
                fscore = (2 * precision * recall) / (precision + recall)
                fscore[np.isnan(fscore)] = 0
                index = np.argmax(fscore)
                fscoreOpt = round(fscore[index], ndigits = 4)
                thresholdOpt = round(threshpr[index], ndigits = 4)
                    
                #print('Best Threshold: {} with F-Score: {}'.format(thresholdOpt, fscoreOpt))
            
                
                p_threshold = thresholdOpt
                cm = confusion_matrix(y_test.values, (dtest_predictions > p_threshold) * 1)
            
                TP = cm[1][1]
                TN = cm[0][0]
                FP = cm[0][1]
                FN = cm[1][0]
            
                # Sensitivity, hit rate, recall, or true positive rate
                TPR = TP / (TP + FN)
                # Specificity or true negative rate
                TNR = TN / (TN + FP)
                # Precision or positive predictive value
                PPV = TP / (TP + FP)
                # Negative predictive value
                NPV = TN / (TN + FN)
                # Fall out or false positive rate
                FPR = FP / (FP + TN)
                # False negative rate
                FNR = FN / (TP + FN)
                # False discovery rate
                FDR = FP / (TP + FP)
            
                #print("Val Sensitivity:", TPR)
                #print("Val Specificity:", TNR)
                #print("Val Precision:", PPV)
                #print("confusion matrix:\n", cm)
                
                fscore = f1_score(y_test.values, (dtest_predictions > p_threshold) * 1)
                accuracy = accuracy_score(y_test.values, (dtest_predictions > p_threshold) * 1)
                results = {
                    "sensitivity" : TPR,
                    "specificity" : TNR,
                    "precision" : PPV,
                    "cm" : cm,
                    "auc_score" : auc_test,
                    "fpr" : fpr,
                    "tpr" : tpr,
                    "threshold" : p_threshold,
                    "fscore" : fscore,
                    "thresholds_fscore" : threshpr,
                    "accuracy": accuracy
                }
                
                hyperparams.append(param)
                accuracies.append(accuracy)
                
    
        best_accuracy = np.argmax(accuracies)
        params = hyperparams[best_accuracy]
        
        xgb1.set_params(**params)
        
        #Tune learning rate
        xgb1.set_params(learning_rate = 0.1)
        params = xgb1.get_xgb_params()
        
        cvresult = xgb.cv(param, xgtrain, num_boost_round=param_init["n_estimators"], nfold=5, \
                          metrics='auc', early_stopping_rounds=10)
        xgb1.set_params(n_estimators=cvresult.shape[0])   
        
        params = xgb1.get_xgb_params()
    
    
    #Final Model
    model_pred = xgb.train(
        params,
        xgtrain,
        evals=[(dtest, "test")],
        evals_result=res,
        early_stopping_rounds=10,
        num_boost_round=200,
        xgb_model= None,
        verbose_eval = False
    )

    # Predict training set:
    dtrain_predictions = model_pred.predict(xgtrain)

    # Predict test set:
    dtest_predictions = model_pred.predict(dtest)

        
    # Print model report:
    print("\nModel Report")
    auc_train = metrics.roc_auc_score(y_train, dtrain_predictions)
    auc_test = metrics.roc_auc_score(y_test, dtest_predictions)
    print(
        "AUC Score (Train): %f"
        % auc_train
    )
    print(
        "AUC Score (Test): %f"
        % auc_test
    )

    fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, dtrain_predictions)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, dtest_predictions)
    
    gmean = np.sqrt(tpr_train * (1 - fpr_train))
    
    precision, recall, threshpr = metrics.precision_recall_curve(y_test, dtest_predictions)
    fscore = (2 * precision * recall) / (precision + recall)
    fscore[np.isnan(fscore)] = 0
    index = np.argmax(fscore)
    fscoreOpt = round(fscore[index], ndigits = 4)
    thresholdOpt = round(threshpr[index], ndigits = 4)
        
    print('Best Threshold: {} with F-Score: {}'.format(thresholdOpt, fscoreOpt))

    
    p_threshold = thresholdOpt
    cm = confusion_matrix(y_test.values, (dtest_predictions > p_threshold) * 1)

    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    print("Val Sensitivity:", TPR)
    print("Val Specificity:", TNR)
    print("Val Precision:", PPV)
    print("confusion matrix:\n", cm)
    
    fscore = f1_score(y_test.values, (dtest_predictions > p_threshold) * 1)
    accuracy = accuracy_score(y_test.values, (dtest_predictions > p_threshold) * 1)
    results = {
        "sensitivity" : TPR,
        "specificity" : TNR,
        "precision" : PPV,
        "cm" : cm,
        "auc_score" : auc_test,
        "fpr" : fpr,
        "tpr" : tpr,
        "threshold" : p_threshold,
        "fscore" : fscore,
        "thresholds_fscore" : threshpr,
        "accuracy": accuracy
    }
    
    return model_pred, results, params

def how_I_trained_classifier(data, kmeans12):

    dfX, dfy, TestX, Testy, dfPatID, TestPatID = utils.train_test_split(data, data_physical['patient_id'], 23)
    
    X_train = data_feat.loc[dfX.index]
    y_train = kmeans12.predict(X_train)
    X_test = data_feat.loc[TestX.index]
    y_test = kmeans12.predict(X_test)
    
    classifier, results, params = train_classifier(dfX, dfy, TestX, Testy)
    
    utils.save_obj(classifier, './', 'xgb_classifier')
    utils.save_obj(results, './', 'xgb_results')
    utils.save_obj(params, './', 'xgb_params')
    
    xgbparams = utils.load_obj('./', 'xgb_params')
    
    wts = None
    xgtrain = xgb.DMatrix(dfX, label=y_train)
    res = {}
    dtest = xgb.DMatrix(TestX, label = y_test)
    
    
    params = xgbparams
        
    model_pred = xgb.train(
        params,
        xgtrain,
        evals=[(dtest, "test")],
        evals_result=res,
        early_stopping_rounds=10,
        num_boost_round=200,
        xgb_model= None,
        verbose_eval = False
    )
    
    
    
    utils.save_obj(model_pred, './', 'final_classifier')
    
    
    
    dtrain_predictions = model_pred.predict(xgtrain)
    
    # Predict test set:
    dtest_predictions = model_pred.predict(dtest)
    
    accuracy= metrics.accuracy_score(dtest_predictions, y_test)

def get_cluster_membership(patID, clustering_labels_kmeans12, sepsisLabel):
    import collections
    
    path_to_data = 'C:/KamalLab/ai_sepsis-master/input_data/training_set'
    pat_analyse = pd.DataFrame(columns = ['patid', 'clusterSeq', 'counter', 'time_to_sepsis', 'total_icu_time'])
    
    temp_df = pd.DataFrame(columns = ['patid', 'labels'])
    temp_df['patid'] = patID
    temp_df['labels'] = clustering_labels_kmeans12
    temp_df_pats = [x for _, x in temp_df.groupby(temp_df['patid'])]
    
    for df in temp_df_pats:
        label_list = df['labels'].values
        counter = collections.Counter(label_list)
        pat_id = df['patid'].values[0]
        print(pat_id)
        
        folder = pat_id[0]
        number = pat_id[1:]
        path_to_patient = path_to_data + folder + '/p' + number + '.psv'
        pat_data_psv = pd.read_csv(path_to_patient, sep='|')
        label_psv = pat_data_psv['SepsisLabel']
        
        total_icu_time = len(label_psv)
        try:
            time_to_sepsis = label_psv.index[label_psv.values == 1][0] + 6
        except IndexError:
            time_to_sepsis = total_icu_time
            
        pat_analyse.loc[len(pat_analyse)] = [pat_id, 
                                             label_list,
                                             counter,
                                             time_to_sepsis,
                                             total_icu_time]
    
    max_time_cluster = pat_analyse['counter'].apply(max)    
    pat_analyse['max_time_cluster'] = max_time_cluster
    
    greater_than_25_clusters = pat_analyse['counter'].apply(lambda c: np.array(list(c))[np.where(list(c.values()) > sum(c)*0)[0]])
    pat_analyse['greater_than_25_perc'] = greater_than_25_clusters
    
    for cluster_num in range(12):
        cluster_membership = pat_analyse['greater_than_25_perc'].apply(lambda x: cluster_num in x)
        pat_analyse['Cluster' + str(cluster_num)] = cluster_membership
    
    pats = pd.DataFrame(columns = ['SepsisLabel', 'patid'])
    pats['SepsisLabel'] = sepsisLabel 
    pats['patid'] = patID
    p = pats.groupby('patid').max()
    zip_iterator = zip(p.index, p['SepsisLabel'].values)

    sepsisp = dict(zip_iterator)
    pat_analyse['Sepsis'] = sepsisp.values()
    
    pat_analyse.to_csv('Grady_pat_analyse.csv')
    
    return pat_analyse

#%%

def trajectories(patID, clustering_labels_kmeans12, sepsisLabel):
    import collections
    pat_analyse = pd.DataFrame(columns = ['patid', 'clusterSeq', 'states_transition', 'counter', 
                                          'time_to_sepsis_subpat', 'total_icu_time_subpat',
                                          'last_cluster_before_sepsis', 'max_time_cluster'])
    
    temp_df = pd.DataFrame(columns = ['patid', 'labels', 'sepsislabels'])
    temp_df['patid'] = patID
    temp_df['labels'] = clustering_labels_kmeans12
    temp_df['sepsislabels'] = sepsisLabel
    temp_df_pats = [x for _, x in temp_df.groupby(temp_df['patid'])]
    
    for df in temp_df_pats:
        
        label_list = df['labels'].values
        counter = collections.Counter(label_list)
        pat_id = df['patid'].values[0]
        print(pat_id)
        
        total_icu_time = len(label_list)
        sepsis_time = np.where(df['sepsislabels'].values == 1)[0]
        if len(sepsis_time) != 0:
            time_to_sepsis = sepsis_time[0]
            last_cluster_before_sepsis = label_list[time_to_sepsis]
        else:
            time_to_sepsis = total_icu_time 
            last_cluster_before_sepsis = -1
            
        full_label_list = np.copy(label_list)
        l = len(label_list)
        i = 1
        while(i<l):
            if label_list[i] == label_list[i-1]:
                label_list = np.delete(label_list, i)
                l = l - 1
            else:
                i = i + 1
                
            
        pat_analyse.loc[len(pat_analyse)] = [pat_id, full_label_list, label_list, counter,
                                             time_to_sepsis, total_icu_time,
                                             last_cluster_before_sepsis, max(counter)]
    
    
    greater_than_25_clusters = pat_analyse['counter'].apply(lambda c: np.array(list(c))[np.where(list(c.values()) > sum(c)*0)[0]])
    pat_analyse['greater_than_25_perc'] = greater_than_25_clusters
    
    
    pats = pd.DataFrame(columns = ['SepsisLabel', 'patid'])
    pats['SepsisLabel'] = sepsisLabel 
    pats['patid'] = patID
    p = pats.groupby('patid').max()
    zip_iterator = zip(p.index, p['SepsisLabel'].values)

    sepsisp = dict(zip_iterator)
    pat_analyse['Sepsis'] = sepsisp.values()
    
    return pat_analyse