# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:37:23 2021

@author: mehak
"""
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

def modelfit_logistic(
    X_train, y_train,
    X_test, y_test,
    param, param_tune = None, wts = None 
):
    
    print("Training ...")
    logisticRegr = LogisticRegression(**param)
    scaler = StandardScaler()
    pipe = Pipeline([('Scaler' , scaler), ('model' , logisticRegr)])
        
    if(param['penalty'] == 'l1'):
        print("GridSearch")

        grid = GridSearchCV(pipe, param_tune, n_jobs = 8, verbose = 3)
        grid.fit(X_train, y_train)

        print(grid.best_params_)

        pipe = grid.best_estimator_


    pipe.fit(X_train, y_train)

    # Predict training set:
    train_predictions = pipe.predict(X_train)

    # Predict test set:
    dtest_predictions = pipe.predict_proba(X_test)
    test_predictions = np.array([x[1] for x in dtest_predictions])

    # Print model report:
    if(all(y_test == 1) or all(y_test) == 0):
        results = {
            "sensitivity" : 1,
            "specificity" : 1,
            "precision" : 1,
            "cm" : np.eye(4),
            "auc_score" : 1,
            "fpr" : 0,
            "tpr" : 1,
            "threshold" : 0,
            "fscore" : 1,
            "thresholds_fscore" : 0
        }
    else:
        print("\nModel Report")
        auc_train = metrics.roc_auc_score(y_train, train_predictions)
        auc_test = metrics.roc_auc_score(y_test, test_predictions)
        print(
            "AUC Score (Train): %f"
            % auc_train
        )
        print(
            "AUC Score (Test): %f"
            % auc_test
        )
        
        fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, train_predictions)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predictions)
        
        gmean = np.sqrt(tpr * (1 - fpr))
        
        
        precision, recall, threshpr = metrics.precision_recall_curve(y_test, test_predictions)
        fscore = (2 * precision * recall) / (precision + recall)
        fscore[np.isnan(fscore)] = 0
        index = np.argmax(fscore)
        fscoreOpt = round(fscore[index], ndigits = 4)
        thresholdOpt = round(threshpr[index], ndigits = 4)
        
        
        # Find the optimal threshold
        """
        index = np.argmax(gmean)
        thresholdOpt = round(thresholds[index], ndigits = 4)
        gmeanOpt = round(gmean[index], ndigits = 4)
        fprOpt = round(fpr[index], ndigits = 4)
        tprOpt = round(tpr[index], ndigits = 4)
        """
        
        print('Best Threshold: {} with F-Score: {}'.format(thresholdOpt, fscoreOpt))
        
        p_threshold = thresholdOpt
        
        cm = confusion_matrix(y_test.values, (test_predictions > p_threshold) * 1)
        
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
        
        fscore_final = f1_score(y_test.values, (test_predictions > p_threshold) * 1)
            
        results = {
            "sensitivity" : TPR,
            "specificity" : TNR,
            "precision" : PPV,
            "cm" : cm,
            "auc_score" : auc_test,
            "fpr" : fpr,
            "tpr" : tpr,
            "threshold" : p_threshold,
            "fscore" : fscore_final,
            "thresholds_fscore" : threshpr
        }
    
    return pipe, results
