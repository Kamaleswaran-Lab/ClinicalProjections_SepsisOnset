# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 06:36:24 2021

@author: mehak
"""


import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import os

def model_predict_proba(
    X_test, y_test,
    model, xgboost = False
):
    
    if xgboost:
        X_test = xgb.DMatrix(X_test, label = y_test)
        test_predictions = model.predict(X_test)
    else:
        # Predict test set:
        test_predictions2 = model.predict_proba(X_test)
        test_predictions = np.array([t[1] for t in test_predictions2])
        
    fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predictions)
    gmean = np.sqrt(tpr * (1 - fpr))
    
    
    precision, recall, threshpr = metrics.precision_recall_curve(y_test, test_predictions)
    fscore = (2 * precision * recall) / (precision + recall)
    fscore[np.isnan(fscore)] = 0
    
    result = { 
        'predictions' : test_predictions,
        'gmean' : gmean,
        'fscore' : fscore,
        'thresholds_gmean': thresholds,
        'thresholds_fscore': threshpr
        }
    
    return result

def predict(y, threshold):
    
    pred = np.zeros(y.shape)
    
    pred[y > threshold] = 1
    
    return pred

def get_metrics(true, predictions):
    
    cm = confusion_matrix(true, predictions)
    
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
    
    auc = metrics.roc_auc_score(true, predictions)
    fscore = (2 * PPV * TPR)/(PPV + TPR)
    
    print("Val Sensitivity:", TPR)
    print("Val Specificity:", TNR)
    print("Val Precision:", PPV)
    print("confusion matrix:\n", cm)
    
    return [TPR, TNR, PPV, auc, fscore]

def final_results(train_predictions, test_predictions, 
                  train_predictions_bin, test_predictions_bin, 
                  dfy, Testy,
                  name, folder):
    
    fpr_test, tpr_test, thresholds_test = metrics.roc_curve(Testy, test_predictions)
    fpr_train, tpr_train, thresholds_train = metrics.roc_curve(dfy, train_predictions)
    
    precision_test, recall_test, threshpr_test = metrics.precision_recall_curve(Testy, test_predictions)
    precision_train, recall_train, threshpr_train = metrics.precision_recall_curve(dfy, train_predictions)
    auprc_train = metrics.auc(recall_train, precision_train)
    auprc_test = metrics.auc(recall_test, precision_test)
    
    f, a = plt.subplots(1,2, figsize = (15,10))
    a[0].plot(fpr_train, tpr_train, 'k', fpr_test, tpr_test, 'r')
    a[0].set_xlabel('False Positive Rate')
    a[0].set_ylabel('True Positive Rate')
    a[0].legend(['train', 'test'])
    a[0].set_title('ROC Curve: Imputed Dataset')
    
    a[1].plot(precision_train, recall_train, 'k', precision_test, recall_test, 'r')
    a[1].set_xlabel('Precision')
    a[1].set_ylabel('Recall')
    a[1].legend(['train', 'test'])
    a[1].set_title('Precision-Recall Curve: Imputed Dataset')
    
    f.savefig(os.path.join(folder, name + 'curves.png'), bbox_inches = 'tight', dpi = 1000)
    m = get_metrics(dfy, train_predictions_bin)
    mt = get_metrics(Testy, test_predictions_bin)
    
    df = pd.DataFrame(columns = ['Dataset', 'Sensitivity', 'Specificity', 'Precision', 'auc', 'fscore'])
    df = df.append({'Dataset': 'Train', 'Sensitivity' : m[0], 'Specificity': m[1], 
                    'Precision': m[2], 'auc_roc': m[3], 'auc_prc' : auprc_train, 
                    'fscore': m[4]}, ignore_index = True)
    df = df.append({'Dataset': 'Test', 'Sensitivity' : mt[0], 'Specificity': mt[1], 
                    'Precision': mt[2], 'auc_roc': mt[3], 'auc_prc' : auprc_test,
                    'fscore': mt[4]}, ignore_index = True)
    
    final = {
        'metrics' : df,
        'fpr_test' : fpr_test,
        'tpr_test': tpr_test,
        'thresholds_test': thresholds_test,
        'fpr_train' : fpr_train,
        'tpr_train' : tpr_train,
        'thresholds_train' : thresholds_train,
        'precision_test' : precision_test,
        'recall_test' : recall_test,
        'threshpr_test' : threshpr_test,
        'precision_train' : precision_train,
        'recall_train': recall_train,
        'threshpr_train': threshpr_train,
        'train_predictions': train_predictions,
        'train_predictions_bin': train_predictions_bin,
        'test_predictions' : test_predictions,
        'test_predictions_bin' : test_predictions_bin
        }
    
    utils.save_obj(final, folder, 'Result' + name)
    
    return final
    