# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:30:08 2021

@author: mehak
"""
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import numpy as np

#%%

def modelfit_xgboost(
    X_train, y_train,
    X_test, y_test,
    useGPU = True, wts = None
):
    
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    res = {}
    dtest = xgb.DMatrix(X_test, label = y_test)
                
    param_init = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "auc",
        "sampling_method": "uniform",
        "learning_rate" : 0.3,
        "n_estimators": 1000,
        "max_depth":5,
        "min_child_weight":1,
        "gamma":0.1,
        "reg_alpha":0.1,
        "subsample":0.8,
        "colsample_bytree":0.8,
        "nthread":4,
        "scale_pos_weight":1,
        "seed":27
    }
    
    if useGPU:
        param_init['gpu_id'] = 0
        param_init['tree_method'] = 'gpu_hist'
        param_init['sampling_method'] = 'gradient_based'
    
    xgb1 = XGBClassifier(
             **param_init)
    
    #Get n trees
    cvresult = xgb.cv(xgb1.get_xgb_params(), xgtrain, num_boost_round=param_init["n_estimators"], nfold=5, metrics='auc', \
                      early_stopping_rounds=10)
        
    xgb1.set_params(n_estimators=cvresult.shape[0])
    
    
    print(xgb1.get_params()['n_estimators'])

    #Tune tree parameters

    param_grid1 = { 
    'max_depth' : range(5,20,1),
    'min_child_weight': range(2,10,1)
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
            
            hyperparams.append(param)
            accuracies.append(accuracy)
            
    
    
    
    best_accuracy = np.argmax(accuracies)
    params = hyperparams[best_accuracy]
    
    xgb1.set_params(**params)
    param_grid3 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
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
            
            hyperparams.append(param)
            accuracies.append(accuracy)
            

    best_accuracy = np.argmax(accuracies)
    params = hyperparams[best_accuracy]
    
    xgb1.set_params(**params)
    
    #Tune learning rate
    xgb1.set_params(learning_rate = 0.1)
    param = xgb1.get_xgb_params()
    
    cvresult = xgb.cv(param, xgtrain, num_boost_round=param_init["n_estimators"], nfold=5, \
                      metrics='auc', early_stopping_rounds=10)
    xgb1.set_params(n_estimators=cvresult.shape[0])
    print(xgb1.get_params()['n_estimators'])   
    
    #Final Model
    model_pred = xgb.train(
        params,
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
    

    return params, results