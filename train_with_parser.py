# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 06:31:15 2022

@author: mehak
"""
        
import json
import shutil
import os
from datetime import date
from sklearn import preprocessing
import utils
import pandas as pd
from sklearn.cluster import KMeans
from model_xgboost import trainXGBoostCrossVal, modelfit_xgboost
import time
from Test import *
import numpy as np
import sys
import argparse
from sklearn.model_selection import GroupShuffleSplit 

class Dict2Class(object):
    
    def __init__(self, args_dict):
        
        for key in args_dict:
            setattr(self, key, args_dict[key])
        
    def print_class(self):
        print(self.__dict__)
        
def read_json(json_filename):
    
    json_file_path = json_filename
    with open(json_file_path, 'r') as j:
     contents = json.loads(j.read())
    args = Dict2Class(contents)
    
    return args


def trainAllModels(dfX, dfy, TestX, Testy, patid,
                   train_clustering_labels, test_clustering_labels, 
                   num_clusters, folder, smote = None, bootstrap = -1, 
                   final_xgb = None, passed_clusters = None):
    final_model_xgboost = []
    avg_xgboost_time = 0
    
    pass_clusters = [] #no sepsis patient clusters - no learning on these
    subtract = 0
    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7)
    for c in range(num_clusters):
        if  passed_clusters is not None and c in passed_clusters:
            subtract = subtract + 1
            pass_clusters.append(c)
        else:
            try:
                i = c - subtract
                print("Cluster {}".format(i))
                X_cluster = dfX.loc[train_clustering_labels == c]
                y_cluster = dfy.loc[train_clustering_labels == c]
                
                split = splitter.split(X_cluster, groups=patid.loc[train_clustering_labels == c])
                train_inds, test_inds = next(split)
                
                X_train = X_cluster.loc[X_cluster.index[train_inds]]
                y_train = y_cluster.loc[y_cluster.index[train_inds]]
                
                X_val = X_cluster.loc[X_cluster.index[test_inds]]
                y_val = y_cluster.loc[y_cluster.index[test_inds]]
                
                X_test = TestX.loc[test_clustering_labels == c]
                y_test = Testy.loc[test_clustering_labels == c]
                
                if smote is not None:
                    X_sm, y_sm = smote.fit_resample(X_train, y_train)
                else:
                    X_sm = X_cluster
                    y_sm = y_cluster
            except ValueError:
                pass_clusters.append(c)
                continue
                
            #### XGBOOST
            start = time.time()
            if bootstrap != -1:
                model, results, params = modelfit_xgboost(
                    X_sm, y_sm, X_val, y_val, X_test, y_test, useGPU= True, final_params = final_xgb[i]['params']
                )
            else:
                model, results, params = modelfit_xgboost(
                    X_sm, y_sm, X_val, y_val, X_test, y_test, useGPU= True, final_params = None
                )
            xgboost_time = time.time() - start 
            results_test = model_predict_proba(X_test, y_test, model, xgboost = True)
            results_train = model_predict_proba(X_cluster, y_cluster, model, xgboost = True)
            model_dictionary = {'model_name': 'xgboost', 
                                'model': model, 
                                'result_test': results_test, 
                                'result_train':results_train, 
                                'results': results,
                                'params' : params}
            final_model_xgboost.append(model_dictionary)
        
    avg_xgboost_time = avg_xgboost_time/(num_clusters)
    
    if bootstrap == -1:
        utils.save_obj(final_model_xgboost, folder, 'final_xgb')
        np.save(os.path.join(folder, 'passedclusters.npy'), pass_clusters)
        with open(folder + 'time.txt', 'w') as f:
            f.write('XGBoost ')
            f.write(str(avg_xgboost_time))
    else:
        utils.save_obj(final_model_xgboost, folder, 'bts_model{}'.format(bootstrap))
        np.save(os.path.join(folder, 'bts_passedclusters_{}.npy'.format(bootstrap)), pass_clusters)
        
    return final_model_xgboost, pass_clusters


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='read config filepath')
    parser.add_argument('--config_path',type=str, 
                    help='path to data.json file')
    
    config = parser.parse_args()
    
    if os.path.isfile(config.config_path):
        args = read_json(config.config_path)
        args.print_class()
    else:
        print("config path is not a valid file!")
    
    today = date.today()
    d1 = today.strftime("%d_%B")
    seed = 23 #int(time.time()) %  100 #Was defined for another purpose, should be removed
    if args.use_projected_data:
        folder = os.path.join(args.data_path,'projected_' + d1 + '_' + str(args.seed) + '/')
        print("Using the projected dataset: ", args.projected_datap_filename, args.projected_datan_filename)
        data_physical = pd.read_pickle(os.path.join(args.data_path, args.projected_datap_filename))
        data_normal = pd.read_pickle(os.path.join(args.data_path, args.projected_datan_filename))
        
        dirty_labels = [i for i in data_physical.columns if (('dirty' in i) or ('dist' in i))]
        remove = dirty_labels + ['index'] + ['patient_id']
        remaining = [i for i in data_physical.columns if i not in remove]
        data = data_physical[remaining].copy()
        
        dist_labels = [i for i in data_normal.columns if (('dist' in i) and ('norm' in i))]
        data = pd.concat([data[remaining],data_normal[dist_labels]], axis = 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        data.loc[:,dist_labels] = min_max_scaler.fit_transform(data[dist_labels])
        data['Age'] = (data['Age']  - data['Age'].min()) / (data['Age'].max() - data['Age'] .min())
        dfX, dfy, TestX, Testy, dfPatID, TestPatID = utils.train_test_split(data, data_physical['patient_id'], args.seed)
    else:
        folder = os.path.join(args.data_path, './imputed_' + d1 + '_' + str(args.seed) + '/')
        print("Using the imputed dataset: ", args.imputed_data_filename)
        data = pd.read_pickle(os.path.join(args.data_path, args.imputed_data_filename))
        patient_id_col = data['patient_id']
        dirty_labels = [i for i in data.columns if (('dirty' in i))]
        remove = dirty_labels + ['index'] + ['patient_id'] + ['Unnamed: 0']
        remaining = [i for i in data.columns if i not in remove]
        data = data[remaining].copy()
        data['Age'] = (data['Age']  - data['Age'].min()) / (data['Age'].max() - data['Age'] .min())
        dfX, dfy, TestX, Testy, dfPatID, TestPatID = utils.train_test_split(data, patient_id_col, args.seed)

    if(not os.path.exists(folder)):
        os.mkdir(folder)
    
    shutil.copy('./train.py', folder)
    shutil.copy('./data.json', folder)
    dfX.to_pickle(os.path.join(folder, 'train_data.pkl'))
    dfy.to_pickle(os.path.join(folder, 'train_labels.pkl'))
    TestX.to_pickle(os.path.join(folder, 'test_data.pkl'))
    Testy.to_pickle(os.path.join(folder, 'test_labels.pkl'))
    TestPatID.to_pickle(os.path.join(folder, 'test_pats.pkl'))
    dfPatID.to_pickle(os.path.join(folder, 'train_pats.pkl'))
    
    if args.smote:
        smote = utils.smote_pipeline()
    else:
        smote = None
    
    if not args.perform_clustering:
        kmeans = utils.load_obj(args.data_path, args.kmeans_obj)
        utils.save_obj(kmeans, folder, 'kmeans')
    else:
        X_sm1, y_sm1 = smote.fit_resample(dfX, dfy)
        #Clustering on SMOTE Data
        kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 500, "random_state":10}
        clusters = args.num_clusters
        kmeans = KMeans(n_clusters=clusters, **kmeans_kwargs)
        kmeans.fit(X_sm1)
        utils.save_obj(kmeans, folder, 'kmeans')
        
    
    train_clustering_labels = kmeans.predict(dfX)
    test_clustering_labels = kmeans.predict(TestX)
    num_clusters = args.num_clusters
    
    if args.cross_val:
        xgbparams, xgbresults, xgbbest = trainXGBoostCrossVal(dfX, dfy, train_clustering_labels, 
                                                              folder, dfPatID, args.num_clusters, 
                                                              smote = smote, do_smote = args.smote)
    
    final_xgb, pass_clusters = trainAllModels(dfX, dfy, TestX, Testy, dfPatID,
                               train_clustering_labels, test_clustering_labels, 
                               args.num_clusters, folder, smote = smote, 
                               bootstrap = -1)
    
    
    #final_xgb = utils.load_obj('./', 'final_xgb')
    #bts = final_xgb
    #pass_clusters = np.load('./data/imputed_26_June_9/passedclusters.npy')
    #pass_clusters = None
    
    #dfX = utils.load_obj('./data/imputed_26_June_9/', 'train_data.pkl')
    #dfy = utils.load_obj('./data/imputed_26_June_9/', 'train_labels.pkl')
    #TestX = utils.load_obj('./data/imputed_26_June_9/', 'test_data.pkl')
    #Testy = utils.load_obj('./data/imputed_26_June_9/', 'test_labels.pkl')
    #TestPatID = utils.load_obj('./data/imputed_26_June_9/', 'test_pats.pkl')
    #dfPatID = utils.load_obj('./data/imputed_26_June_9/', 'train_pats.pkl')
    
    train_clustering_labels = kmeans.predict(dfX)
    test_clustering_labels = kmeans.predict(TestX)
    
    #dfy.to_pickle(os.path.join(folder, 'train_labels.pkl'))
    #TestX.to_pickle(os.path.join(folder, 'test_data.pkl'))
    #Testy.to_pickle(os.path.join(folder, 'test_labels.pkl'))
    #TestPatID.to_pickle(os.path.join(folder, 'test_pats.pkl'))
    #dfPatID.to_pickle(os.path.join(folder, 'train_pats.pkl'))
    
    
    test_predictions_xgb = np.zeros(Testy.shape)
    train_predictions_xgb = np.zeros(dfy.shape)

    bin_test_predictions_xgb = np.zeros(Testy.shape)
    bin_train_predictions_xgb = np.zeros(dfy.shape)

    thresholds_xgb = np.zeros((num_clusters,))
    subtract = 0
    for c in range(num_clusters):
        

        print("Cluster {}".format(c))
        
        X_cluster = dfX.loc[train_clustering_labels == c]
        y_cluster = dfy.loc[train_clustering_labels == c]
        
        X_test = TestX.loc[test_clustering_labels == c]
        y_test = Testy.loc[test_clustering_labels == c]
        
        if  pass_clusters is not None and c in pass_clusters:
            predictions_test_xgb = np.zeros(len(y_test))
            predictions_train_xgb = np.zeros(len(y_cluster))
            thresholds_xgb[c] = 0.5
            subtract = subtract + 1
        else:
            i = c - subtract
            predictions_test_xgb = final_xgb[i]['result_test']['predictions']
            predictions_train_xgb = final_xgb[i]['result_train']['predictions']
            index = np.argmax(final_xgb[i]['result_train']['fscore'])
            thresholds_xgb[c] = final_xgb[i]['result_train']['thresholds_fscore'][index]
        
        train_predictions_xgb[train_clustering_labels == c]  = predictions_train_xgb
        test_predictions_xgb[test_clustering_labels == c] = predictions_test_xgb
        bin_train_predictions_xgb[train_clustering_labels == c]  = predict(predictions_train_xgb, thresholds_xgb[c]) 
        bin_test_predictions_xgb[test_clustering_labels == c] = predict(predictions_test_xgb, thresholds_xgb[c])
            
            
    xgb_results = final_results(train_predictions_xgb, test_predictions_xgb,
                                bin_train_predictions_xgb, bin_test_predictions_xgb,
                                dfy, Testy,
                                'xgb_results', folder)
    
    
    if args.bootstrap:
        for i in range(50):
            if args.use_projected_data:
                dfX, dfy, TestX, Testy, dfPatID, TestPatID = utils.train_test_split(data, 
                                                                                    data_physical['patient_id'], 
                                                                                    args.seed*i)
            else:
                dfX, dfy, TestX, Testy, dfPatID, TestPatID = utils.train_test_split(data, 
                                                                                    patient_id_col, 
                                                                                    args.seed*i)
            train_clustering_labels = kmeans.predict(dfX)
            test_clustering_labels = kmeans.predict(TestX)
            bts, pass_clusters = trainAllModels(dfX, dfy, TestX, Testy, dfPatID,
                                 train_clustering_labels, test_clustering_labels, 
                                 args.num_clusters, folder, smote = smote,
                                 bootstrap = i, final_xgb = final_xgb, passed_clusters = pass_clusters)
            
            pass_clusters= np.unique(pass_clusters)
            test_predictions_xgb = np.zeros(Testy.shape)
            train_predictions_xgb = np.zeros(dfy.shape)

            bin_test_predictions_xgb = np.zeros(Testy.shape)
            bin_train_predictions_xgb = np.zeros(dfy.shape)

            thresholds_xgb = np.zeros((num_clusters,))
            subtract = 0
            for c in range(num_clusters):
                

                print("Cluster {}".format(c))
                
                X_cluster = dfX.loc[train_clustering_labels == c]
                y_cluster = dfy.loc[train_clustering_labels == c]
                
                X_test = TestX.loc[test_clustering_labels == c]
                y_test = Testy.loc[test_clustering_labels == c]
                
                if pass_clusters is not None and c in pass_clusters:
                    predictions_test_xgb = np.zeros(len(y_test))
                    predictions_train_xgb = np.zeros(len(y_cluster))
                    thresholds_xgb[c] = 0.5
                    subtract = subtract + 1
                else:
                    j = c - subtract
                    predictions_test_xgb = bts[j]['result_test']['predictions']
                    predictions_train_xgb = bts[j]['result_train']['predictions']
                    index = np.argmax(bts[j]['result_train']['fscore'])
                    thresholds_xgb[c] = bts[j]['result_train']['thresholds_fscore'][index]
                
                train_predictions_xgb[train_clustering_labels == c]  = predictions_train_xgb
                test_predictions_xgb[test_clustering_labels == c] = predictions_test_xgb
                bin_train_predictions_xgb[train_clustering_labels == c]  = predict(predictions_train_xgb, thresholds_xgb[c]) 
                bin_test_predictions_xgb[test_clustering_labels == c] = predict(predictions_test_xgb, thresholds_xgb[c])
                    
                    
            bts_results = final_results(train_predictions_xgb, test_predictions_xgb,
                                        bin_train_predictions_xgb, bin_test_predictions_xgb,
                                        dfy, Testy,
                                        'bts_results{}'.format(i), folder)
        
    