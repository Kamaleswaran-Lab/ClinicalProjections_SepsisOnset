# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 20:39:38 2022

@author: mehak
"""

from pipe.ClinicalProjection import ClinicalProjection
import pandas as pd
import os
import numpy as np


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


def project(clinProj, filename):
    """ Get Projected Data """
    if filename[-3:] == 'csv':
        data = pd.read_csv(filename).iloc[:,1:]
    elif filename[-3:] == 'pkl':
        data = pd.read_pickle(filename).iloc[:,1:]
    else:
        print("Check Filename")
        return
        
    final_phy = pd.DataFrame()
    final_normal = pd.DataFrame()
    
    #Splitting into block to manage larger dataset
    data_split = np.array_split(data, 4000)
    
    for i, data_ in enumerate(data_split[:2]):
        
        data_ = data_.reset_index()
        
        final_phy_, val_phy = clinProj.projection_model_physical(data_.copy())        
        final_normal_,val_normal = clinProj.projection_model_normal(final_phy_.copy())

        final_phy_ = final_phy_.set_index('index', drop = True)
        final_normal_ = final_normal_.set_index('index', drop = True)
        
        final_phy = final_phy.append(final_phy_)
        final_normal = final_normal.append(final_normal_)
        
    return final_phy, final_normal 


if __name__ == '__main__':
        
    imputed_data_path = './imputed_6_3.pkl'    
    output_dir = './projection_output/'
    
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)
        
    ranges, log_variables = get_ranges('./constraints_wo_calcium.txt')  
    interval = 6
    
    clinProj = ClinicalProjection(interval = interval, 
                                  ranges = ranges,
                                  log_variables = log_variables)
    
    physical, normal = project(clinProj, imputed_data_path)


    physical_sofa_corrected = clinProj.compute_SOFA(physical)
    normal_sofa_corrected = clinProj.compute_SOFA(normal)
    
    physical_sirs_corrected = clinProj.compute_SIRS(physical_sofa_corrected)
    normal_sirs_corrected = clinProj.compute_SIRS(normal_sofa_corrected)

    physical_sirs_corrected.to_pickle(os.path.join(output_dir, 'physical_.pkl'))
    normal_sirs_corrected.to_pickle(os.path.join(output_dir, 'normal_.pkl'))

