# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 17:48:39 2022

@author: mehak
"""

import pandas as pd
import os
from gurobipy import *
import numpy as np
import sys

#%%

""" Converting scaled back to normal range """

def norm_to_none(x,name,ranges):
    c1,c2,c3,c4 = ranges[name][:4]
    log_variables = ['Creatinine', 'Bilirubin_direct', 'Bilirubin_total', 'Glucose', 'Lactate', 'WBC', 'TroponinI']
    #reverse normalization of normal range
    y = x*(c4-c3) + c3
    if(name in log_variables):
        y = 10**(y) - 1
        
    return y

def norm_to_phys(x,name,ranges):
    c1,c2,c3,c4 = ranges[name][:4]
    
    #reverse normalization of normal range
    y = x*(c4-c3) + c3
    
    #normalize again for physcial
    return (y-c1)/(c2 - c1)

def phys_to_norm(x,name,ranges):
    c1,c2,c3,c4 = ranges[name][:4]
    
    #reverse normalization of physical range
    y = x*(c2-c1) + c1
    
    #normalize again for physcial
    return (y-c3)/(c4 - c3)

def projection_model_physical3(data2,ranges, interval ):
    
    # Create model
    m = Model("projection")
    k = len(data2)
    proj_variables = list(ranges.keys())
    
    #define projection variables
    x = {}
    for var in proj_variables:
        x[var] = {}
        for i in range(k):
            x[var][i] = {}
            for t in range(interval):
                x[var][i][t] = m.addVar(lb=0, ub = 1, name='x_{}_{}_{}'.format(var,i,t))
                
    #define binary variables
    binary_variables = proj_variables.copy()
    binary_variables.extend(['Lactate_1', 'HCO3_1', 'HCO3_2', 'SaO2'])
    z = {}
    for var in binary_variables:
        z[var] = {}
        for i in range(k):
            z[var][i] = {}
            for t in range(interval):
                z[var][i][t] = m.addVar(vtype= GRB.BINARY, name='z_{}_{}_{}'.format(var,i,t))
    m.update()
    
    #define Euclidean projection objective
    objExp = quicksum(np.square(np.array([x[var][i][t] -norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges)\
                    for t in range(interval) for i in range(k) for var in  proj_variables])))
    m.setObjective(objExp, GRB.MINIMIZE)
    m.update()
    #Add constraints
    #hourly change constraints:
    for i in range(k):
        for t in range(1,interval):
            for var in proj_variables:
                if len(ranges[var]) == 5:
                    c1,c2,c3,c4,c5 = ranges[var]
                    m.addConstr(x[var][i][t] - x[var][i][t-1], '<=', c5/(c2-c1))
                    m.addConstr(x[var][i][t] - x[var][i][t-1], '>=', -c5/(c2-c1))
    #MAP[t] >= 0.95(⅔ DBP[t] + ⅓ SBP[t])
    #MAP[t] <= 1.05(⅔ DBP[t] + ⅓ SBP[t])
    for i in range(k):
        for t in range(interval):
            c1 = ranges['MAP'][1] - ranges['MAP'][0]
            c2 = ranges['DBP'][1] - ranges['DBP'][0]
            c3 = ranges['SBP'][1] - ranges['SBP'][0]
            m.addConstr(x['MAP'][i][t]*c1, '>=', 0.95*(2/3*(c2*x['DBP'][i][t] +ranges['DBP'][0])\
                                    + 1/3*(x['SBP'][i][t]*c3 + ranges['SBP'][0])) -  ranges['MAP'][0])
            m.addConstr(x['MAP'][i][t]*c1, '<=', 1.05*(2/3*(c2*x['DBP'][i][t] +ranges['DBP'][0])\
                                    + 1/3*(x['SBP'][i][t]*c3 + ranges['SBP'][0])) -  ranges['MAP'][0])
    #Bilirubin_direct[t] <= Bilirubin_total[t]
    for i in range(k):
        for t in range(interval):
            c1 = ranges['Bilirubin_direct'][1] - ranges['Bilirubin_direct'][0]
            c2 = ranges['Bilirubin_total'][1] - ranges['Bilirubin_total'][0]
            m.addConstr(c1*x['Bilirubin_direct'][i][t] + ranges['Bilirubin_direct'][0], '<=',\
                        c2*x['Bilirubin_total'][i][t]+ranges['Bilirubin_total'][0])
    #Hct[t] >= 1.5 Hgb[t]
    for i in range(k):
        for t in range(interval):
            c1 = ranges['Hgb'][1] - ranges['Hgb'][0]
            c2 = ranges['Hct'][1] - ranges['Hct'][0]
            m.addConstr(1.5*(c1*x['Hgb'][i][t] +  ranges['Hgb'][0]), '<=',c2*x['Hct'][i][t] + ranges['Hct'][0])
    #If HCO3[t] <= 10 (value), then BaseExcess[t] <= 0
    for i in range(k):
        for t in range(interval):
            var = 'HCO3'
            c1 = ranges['HCO3'][1] - ranges['HCO3'][0]
            c2 = ranges['BaseExcess'][1] - ranges['BaseExcess'][0]
            m.addConstr((10-ranges['HCO3'][0])/c1*z['HCO3'][i][t] + norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges)*(1 - z['HCO3'][i][t])\
                        ,'>=', norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges))
            m.addConstr(z['HCO3'][i][t]*norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges) + (10-ranges['HCO3'][0])/c1*(1-z['HCO3'][i][t]), '<=', norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges))
            m.addConstr(x['BaseExcess'][i][t], '<=', (0 -ranges['BaseExcess'][0])/c2*z['HCO3'][i][t]\
                        +(ranges['BaseExcess'][1]-ranges['BaseExcess'][0])/c2*(1- z['HCO3'][i][t]))
    #If Lactate >= 6, then BaseExcess <= 0. (values)
    for i in range(k):
        for t in range(interval):
            var = 'Lactate'
            c1 = ranges['Lactate'][1] - ranges['Lactate'][0]
            c2 = ranges['BaseExcess'][1] - ranges['BaseExcess'][0]
            m.addConstr((np.log10(6+1)-ranges['Lactate'][0])/c1*z['Lactate'][i][t] + \
                        norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges)*(1 - z['Lactate'][i][t]), '<=', norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges))
            m.addConstr((np.log10(6+1)-ranges['Lactate'][0])/c1*(1-z['Lactate'][i][t]) + \
                        norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges)*z['Lactate'][i][t], '>=', norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges))
            m.addConstr(x['BaseExcess'][i][t], '<=', (0 -ranges['BaseExcess'][0])/c2*z['Lactate'][i][t]\
                        +(ranges['BaseExcess'][1]-ranges['BaseExcess'][0])/c2*(1- z['Lactate'][i][t]))
    #If BaseExcess <= 0 then either HC03<=10 or Lactate >=6
    for i in range(k):
        for t in range(interval):
            var = 'BaseExcess'
            c1 = ranges['Lactate'][1] - ranges['Lactate'][0]
            c2 = ranges['BaseExcess'][1] - ranges['BaseExcess'][0]
            c3 = ranges['HCO3'][1] - ranges['HCO3'][0]
            #x[t] = 1 if BaseExcess <=0, x[t]=0 then BaseExcess <=20
            m.addConstr((0 -ranges['BaseExcess'][0])/c2*z['BaseExcess'][i][t] + norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],\
                var,ranges)* (1 - z['BaseExcess'][i][t]), '>=', norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges))
            #x[t] = 0 then baseExcess >= 0, x[t]=1, then BaseExcess >= -40
            m.addConstr(norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges)*z['BaseExcess'][i][t] + \
                        (1 - z['BaseExcess'][i][t])*(0 -ranges['BaseExcess'][0])/c2, '<=', norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges))
            #y[t] = 1 when HCO3 <=10, y[t]=0 when HCO3 <= 45
            m.addConstr((10 -ranges['HCO3'][0])/c3*z['HCO3_1'][i][t] + (ranges['HCO3'][1]-ranges['HCO3'][0])/c3* (1 - \
                                        z['HCO3_1'][i][t]), '>=', x['HCO3'][i][t])
            #y[t]=0 when HCO3 >=10, y[t]=1 when HCO3 >=0
            m.addConstr((10-ranges['HCO3'][0])/c3*(1-z['HCO3_1'][i][t])+ z['HCO3_1'][i][t]*(0 -ranges['HCO3'][0])/c3, \
                        '<=', x['HCO3'][i][t])
            #z[t] = 1 when Lactate >=6, z[t] =0 then Lactate >=0
            m.addConstr((np.log10(6+1) -ranges['Lactate'][0])/c1*z['Lactate_1'][i][t] + (0-ranges['Lactate'][0])/c1* (1 - z['Lactate_1'][i][t]),\
                        '<=', x['Lactate'][i][t])
            #z[t] =0 if Lactate <= 6, z[t]=1 then Lactate<=30
            m.addConstr((np.log10(6+1)-ranges['Lactate'][0])/c1*(1-z['Lactate_1'][i][t])+ \
                        z['Lactate_1'][i][t]*(ranges['Lactate'][1] -ranges['Lactate'][0])/c1, '>=', x['Lactate'][i][t])
            #x[t] <= z[t] + y[t]
            m.addConstr(z['BaseExcess'][i][t], '<=', z['Lactate_1'][i][t] + z['HCO3_1'][i][t])
    #If pH < 7, then either PaCO2 <= 35 or HCO3 <=10
    for i in range(k):
        for t in range(interval):
            var = 'pH'
            c1 = ranges['pH'][1] - ranges['pH'][0]
            c2 = ranges['PaCO2'][1] - ranges['PaCO2'][0]
            c3 = ranges['HCO3'][1] - ranges['HCO3'][0]
            #x[t] = 1 if ph <=7, x[t]=0 then Ph <= 7.7
            m.addConstr((7 -ranges['pH'][0])/c1*z['pH'][i][t] + norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges)\
                        * (1 - z['pH'][i][t]), '>=', norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges))
            #x[t] = 0 then pH >= 7, x[t]=1, then pH >= 6.5
            m.addConstr(norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges)*z['pH'][i][t] + \
                        (1 - z['pH'][i][t])*(7 -ranges['pH'][0])/c1, '<=', norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges))
            #y[t] = 1 when PaCO2 <=35, y[t]=0 when PaCO2 <= 120
            m.addConstr((35 -ranges['PaCO2'][0])/c2*z['PaCO2'][i][t] + (ranges['PaCO2'][1]-ranges['PaCO2'][0])/c2*\
                        (1 - z['PaCO2'][i][t]), '>=', x['PaCO2'][i][t])
            #y[t]=0 when PaCO2 >=35, y[t]=1 when PaCO2 >=16
            m.addConstr((35-ranges['PaCO2'][0])/c2*(1-z['PaCO2'][i][t])+ z['PaCO2'][i][t]*(16 -ranges['PaCO2'][0])/c2, \
                        '<=', x['PaCO2'][i][t])
            #z[t] = 1 when HCO3 <=10, z[t] =0 then HCO3 <= 45
            m.addConstr((10 -ranges['HCO3'][0])/c3*z['HCO3_2'][i][t] + (ranges['HCO3'][1]-ranges['HCO3'][0])/c3* \
                        (1 - z['HCO3_2'][i][t]), '>=', x['HCO3'][i][t])
            #x[t] =0 if HCO3 >=10, z[t]=1 then HCO3>=0
            m.addConstr((10-ranges['HCO3'][0])/c3*(1-z['HCO3_2'][i][t])+ z['HCO3_2'][i][t]*(0 -\
                        ranges['HCO3'][0])/c3, '<=', x['HCO3'][i][t])
            #x[t] <= z[t] + y[t]
            m.addConstr(z['pH'][i][t], '<=', z['PaCO2'][i][t] + z['HCO3_2'][i][t])
    
    #If 95 <= PaO2[t-1] and 95 <= PaO2[t+1] then SpO2[t] >= 90 -- CONSTRAINT REMOVED
    
    m.update()
    m.setParam( 'OutputFlag', False )
    m.write('QP_phys.lp')
    m.optimize()
    data_copy = data2.copy()
    for var in proj_variables:
        for i in range(k):
            for t in range(interval):
                data_copy.loc[i,'{}-{}'.format(var,t)] = phys_to_norm(x[var][i][t].x,var,ranges)
    for var in proj_variables:
        data_copy['{}-phys-dist'.format(var)] = [np.round(sum(np.square(np.array([x[var][i][t].x - \
                    norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)], var,ranges) for t in range(6)]))),4) for i in range(k)]
    return data_copy, m.ObjVal




def projection_model_normal(data, ranges, interval):

    # Create model
    m = Model("projection")
    k = len(data)
    proj_variables = list(ranges.keys())

    #define projection variables
    x = {}
    for var in proj_variables:
        x[var] = {}
        for i in range(k):
            x[var][i] = {}
            for t in range(interval):
                x[var][i][t] = m.addVar(lb=0, ub = 1, name='x_{}_{}_{}'.format(var,i,t))
                
    m.update()


    #define Euclidean projection objective
    objExp = quicksum(np.square(np.array([x[var][i][t] - data.loc[i,'{}-{}'.format(var,t)]\
                    for t in range(interval) for i in range(k) for var in  proj_variables])))
    m.setObjective(objExp, GRB.MINIMIZE)
    m.update()
    m.setParam( 'OutputFlag', False )
    m.write('QP_norm.lp')

    m.optimize()

    data_copy = data.copy()
    for var in proj_variables:
        for i in range(k):
            for t in range(interval):
                data_copy.loc[i,'{}-{}'.format(var,t)] = x[var][i][t].x
                
    for var in proj_variables:             
        data_copy['{}-norm-dist'.format(var)] = [np.round(sum(np.square(np.array([x[var][i][t].x - \
                    data.loc[i,'{}-{}'.format(var,t)] for t in range(interval)]))),4) for i in range(k)]
        
    return data_copy, m.ObjVal

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

def normalize(x, var_name, ranges, norm_to):
    
    c1,c2,c3,c4 = ranges[var_name][:4]
    
    if norm_to == 'physical':
        #reverse normalization of normal range
        y = x*(c4-c3) + c3
        
        #normalize again for physical
        return (y-c1)/(c2 - c1)
    
    elif norm_to == 'normal':
        #reverse normalization of physical range
        y = x*(c2-c1) + c1
        
        #normalize again for normal
        return (y-c3)/(c4 - c3)
    else:
        print("Invalid option ", norm_to)
        return x


def compute_SOFA_coag(platelets: float) -> float:
    if platelets >= 150:
        return 0
    if platelets >= 100:
        return 1
    if platelets >= 50:
        return 2
    if platelets >= 20:
        return 3
    return 4


def compute_SOFA_liver(bilirubin_total: float) -> float:
    if bilirubin_total <= 1.2:
        return 0
    if bilirubin_total <= 2.0:
        return 1
    if bilirubin_total <= 6.0:
        return 2
    if bilirubin_total <= 12:
        return 3
    return 4


def compute_SOFA_renal(creatinine: float) -> float:
    if creatinine <= 1.2:
        return 0
    if creatinine <= 2.0:
        return 1
    if creatinine <= 3.5:
        return 2
    if creatinine <= 5.0:
        return 3
    return 4


def compute_SOFA(df, interval, ranges):

    var = 'SOFA'
    k = len(df)
    for i in range(k):
        for t in range(interval):
            df.loc[i,'{}-{}'.format(var,t)] = max(
                # compute_SOFA_resp(df["FiO2"], df["O2Sat"]),
                compute_SOFA_coag(norm_to_none(df.loc[i,'{}-{}'.format('Platelets',t)], 'Platelets', ranges)),
                compute_SOFA_liver(norm_to_none(df.loc[i,'{}-{}'.format('Bilirubin_total',t)], 'Bilirubin_total', ranges)),
                compute_SOFA_renal(norm_to_none(df.loc[i,'{}-{}'.format('Creatinine',t)], 'Creatinine', ranges))
            )
    
    return df

def compute_SIRS(df, interval, ranges):
    
    var = 'SIRS'
    k = len(df)
    for i in range(k):
        for t in range(interval):
            temp = norm_to_none(df.loc[i,'{}-{}'.format('Temp',t)], 'Temp', ranges)
            hr = norm_to_none(df.loc[i,'{}-{}'.format('HR',t)], 'HR', ranges)
            wbc = norm_to_none(df.loc[i,'{}-{}'.format('WBC',t)], 'WBC', ranges)
            resp = norm_to_none(df.loc[i,'{}-{}'.format('Resp',t)], 'Resp', ranges)
            
            df.loc[i,'{}-{}'.format(var,t)] = ((temp < 36) | (temp > 38)) + (hr > 90) + ((wbc < 4) | ( wbc > 12)) + (resp > 20)

    return df

#%%
def project(interval, name, ranges):
    
    #Read Data
    
    #imputed_path = os.path.join(data_path_grady, 'Imputed' + str(year))
    data = pd.read_csv(name).iloc[:,1:]
    data_split = np.array_split(data, 4)
    
    
    final_phy3 = pd.DataFrame()
    final_normal3 = pd.DataFrame()
    
    for i, data_ in enumerate(data_split):
        
        data_ = data_.reset_index()
        
        final_phy_3,val_phy3 = projection_model_physical3(data_.copy(),ranges, interval)
        print("Done to physical" , i)
        
        final_normal_3,val_normal3 = projection_model_normal(final_phy_3.copy(),ranges, interval)
        print("Done to Normal", i)

        #Physionet
        final_phy_3 = final_phy_3.set_index('index', drop = True)
        final_normal_3 = final_normal_3.set_index('index', drop = True)
        
        final_phy3 = final_phy3.append(final_phy_3)
        final_normal3 = final_normal3.append(final_normal_3)
        
    return final_phy3, final_normal3 


if __name__ == '__main__':

    #num_args = len(sys.argv)
    
    #if num_args <= 2:
    #    print("Enter name file path to imputed data, and destination folder")
    #    exit(0)
    #else:
    #    name = sys.argv[1]
    #    output_dir = sys.argv[2]
        
    name = 'Imputed2014/summary_1654815096.csv'    
    ranges = get_ranges('constraints_wo_calcium.txt')
    output_dir = './'
    interval = 12
    physical, normal = project(interval, name, ranges)

    physical.to_csv(os.path.join(output_dir, 'physical.csv'))
    normal.to_csv(os.path.join(output_dir, 'normal.csv'))

    physical_sofa_corrected = compute_SOFA(physical, interval, ranges)
    normal_sofa_corrected = compute_SOFA(normal, interval, ranges)
    
    physical_sirs_corrected = compute_SIRS(physical_sofa_corrected, interval, ranges)
    normal_sirs_corrected = compute_SIRS(normal_sofa_corrected, interval, ranges)

    physical_sirs_corrected.to_csv(os.path.join(output_dir, 'physical_6_3.csv'))
    normal_sirs_corrected.to_csv(os.path.join(output_dir, 'normal_6_3.csv'))

#%%

