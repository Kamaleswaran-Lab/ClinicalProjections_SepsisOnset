# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 20:39:38 2022

@author: mehak
"""

from pipe.ClinicalProjection import ClinicalProjection
import pandas as pd
import os
import numpy as np
import time
import warnings
import sys
import multiprocessing as mp
from functools import partial
import pickle

def get_ranges(constraint_file_name):
    #Set the constraints for each variable
    ranges = {}
    
    #The min and max values of these variables are log transformed 
    log_variables = ['Creatinine', 'Bilirubin_direct', 'Bilirubin_total', 
                     'Glucose', 'Lactate', 'WBC', 'TroponinI']
    
    with open(constraint_file_name, 'r') as f:
        for x in f:
            line = x.replace('\n', '').split(', ')
            if line[0] in log_variables:
                ranges[line[0]] = [np.log10(float(i) +1) for i in line[1:]]
            else:
                ranges[line[0]] = [float(i) for i in line[1:]]

    return ranges, log_variables


def project(clinProj, data):
    
    warnings.filterwarnings("ignore")
    final_phy = pd.DataFrame()
    final_normal = pd.DataFrame()
    
    #Splitting into block to manage larger dataset
    data_split = np.array_split(data, 10)
    
    for i, data_ in enumerate(data_split[:2]):
        
        data_ = data_.reset_index()
        
        final_phy_, val_phy = clinProj.projection_model_physical(data_.copy())        
        final_normal_,val_normal = clinProj.projection_model_normal(final_phy_.copy())

        final_phy_ = final_phy_.set_index('index', drop = True)
        final_normal_ = final_normal_.set_index('index', drop = True)
        
        final_phy = final_phy.append(final_phy_)
        final_normal = final_normal.append(final_normal_)
        
    return [final_phy, final_normal] 


if __name__ == '__main__':
        
    #imputed_data_path = './imputed_6_3.pkl'    
    #output_dir = './projection_output/'
    
    num_args = len(sys.argv)
    

    if num_args <= 2:
        print("Enter name file path to imputed data, and destination folder")
        sys.exit(0)
    else:
        name = sys.argv[1]
        output_dir = sys.argv[2]
         
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)
        
    ranges, log_variables = get_ranges('./constraints_wo_calcium.txt')  
    interval = 6
    
    clinProj = ClinicalProjection(interval = interval, 
                                  ranges = ranges,
                                  log_variables = log_variables)
    """ Get Projected Data """
    if name[-3:] == 'csv':
        data = pd.read_csv(name)
    elif name[-3:] == 'pkl':
        data = pd.read_pickle(name)
    else:
        print("Check Filename")
        sys.exit(0)
    
    filtered = [col for col in data if not col.startswith('Unnamed:')]
    data = data[filtered]
    
    start = time.time()
    physical, normal = project(clinProj, data)
    
    physical_sofa_corrected = clinProj.compute_SOFA(physical)
    normal_sofa_corrected = clinProj.compute_SOFA(normal)
    
    physical_sirs_corrected = clinProj.compute_SIRS(physical_sofa_corrected)
    normal_sirs_corrected = clinProj.compute_SIRS(normal_sofa_corrected)

    physical_sirs_corrected.to_pickle(os.path.join(output_dir, 'physical_.pkl'))
    normal_sirs_corrected.to_pickle(os.path.join(output_dir, 'normal_.pkl'))
    
    end = time.time()
    
    print("Time taken without mp: {} seconds".format(round(end-start, 2)))
    
    
    start = time.time()
    f = partial(project, clinProj)
    processes = mp.cpu_count()
    data_split = np.array_split(data, processes)
    data_array = [data_ for data_ in data_split]
    
    with mp.Pool(processes) as pool:
        lst_mp = list(pool.map(f, data_array))
    end = time.time()
    
    physical = pd.DataFrame()
    normal = pd.DataFrame()
    
    for i in range(len(lst_mp)):
        
        physical_sofa_corrected = clinProj.compute_SOFA(lst_mp[0][0])
        normal_sofa_corrected = clinProj.compute_SOFA(lst_mp[0][1])
        
        physical_sirs_corrected = clinProj.compute_SIRS(physical_sofa_corrected)
        normal_sirs_corrected = clinProj.compute_SIRS(normal_sofa_corrected)
        
        #physical_ = physical_sirs_corrected.set_index('index', drop = True)
        #normal_ = normal_sirs_corrected.set_index('index', drop = True)
        
        physical = physical.append(physical_sirs_corrected)
        normal = normal.append(normal_sirs_corrected)

    physical.to_pickle(os.path.join(output_dir, 'physical_mp.pkl'))
    normal.to_pickle(os.path.join(output_dir, 'normal_mp.pkl'))
        
    print("Time taken with mp: {} seconds".format(round(end-start, 2)))


