# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 07:53:20 2021

@author: mehak
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
New Projection Functions to:
    
    1. Physical 
    2. Normal

Calculates the distance. Overwrites the values for physical if needed by setting overwrite = True
"""
import pandas as pd
import os
from gurobipy import *
import numpy as np


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

#%%

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


#%%


#CORRECTED
def projection_model_physical2(data2,ranges):
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
            for t in range(6):
                x[var][i][t] = m.addVar(lb=0, ub = 1, name='x_{}_{}_{}'.format(var,i,t))
    #define binary variables
    binary_variables = proj_variables.copy()
    binary_variables.extend(['Lactate_1', 'HCO3_1', 'HCO3_2', 'SaO2'])
    z = {}
    for var in binary_variables:
        z[var] = {}
        for i in range(k):
            z[var][i] = {}
            for t in range(6):
                z[var][i][t] = m.addVar(vtype= GRB.BINARY, name='z_{}_{}_{}'.format(var,i,t))
    m.update()
    #define Euclidean projection objective
    objExp = quicksum(np.square(np.array([x[var][i][t] -norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges)\
                    for t in range(6) for i in range(k) for var in  proj_variables])))
    m.setObjective(objExp, GRB.MINIMIZE)
    m.update()
    #Add constraints
    #hourly change constraints:
    for i in range(k):
        for t in range(1,6):
            for var in proj_variables:
                if len(ranges[var]) == 5:
                    c1,c2,c3,c4,c5 = ranges[var]
                    m.addConstr(x[var][i][t] - x[var][i][t-1], '<=', c5/(c2-c1))
                    m.addConstr(x[var][i][t] - x[var][i][t-1], '>=', -c5/(c2-c1))
    #MAP[t] >= 0.95(⅔ DBP[t] + ⅓ SBP[t])
    #MAP[t] <= 1.05(⅔ DBP[t] + ⅓ SBP[t])
    for i in range(k):
        for t in range(6):
            c1 = ranges['MAP'][1] - ranges['MAP'][0]
            c2 = ranges['DBP'][1] - ranges['DBP'][0]
            c3 = ranges['SBP'][1] - ranges['SBP'][0]
            m.addConstr(x['MAP'][i][t]*c1, '>=', 0.95*(2/3*(c2*x['DBP'][i][t] +ranges['DBP'][0])\
                                    + 1/3*(x['SBP'][i][t]*c3 + ranges['SBP'][0])) -  ranges['MAP'][0])
            m.addConstr(x['MAP'][i][t]*c1, '<=', 1.05*(2/3*(c2*x['DBP'][i][t] +ranges['DBP'][0])\
                                    + 1/3*(x['SBP'][i][t]*c3 + ranges['SBP'][0])) -  ranges['MAP'][0])
    #If HCO3[t] <= 10 (value), then BaseExcess[t] <= 0
    for i in range(k):
        for t in range(6):
            c1 = ranges['HCO3'][1] - ranges['HCO3'][0]
            c2 = ranges['BaseExcess'][1] - ranges['BaseExcess'][0]
            m.addConstr((10-ranges['HCO3'][0])/c1*z['HCO3'][i][t] + (ranges['HCO3'][1]-ranges['HCO3'][0])/c1*(1 - z['HCO3'][i][t])\
                        ,'>=', x['HCO3'][i][t])
            m.addConstr((10-ranges['HCO3'][0])/c1*(1-z['HCO3'][i][t]), '<=', x['HCO3'][i][t])
            m.addConstr(x['BaseExcess'][i][t], '<=', (0 -ranges['BaseExcess'][0])/c2*z['HCO3'][i][t]\
                        +(ranges['BaseExcess'][1]-ranges['BaseExcess'][0])/c2*(1- z['HCO3'][i][t]))
    #If Lactate >= 6, then BaseExcess <= 0. (values)
    for i in range(k):
        for t in range(6):
            c1 = ranges['Lactate'][1] - ranges['Lactate'][0]
            c2 = ranges['BaseExcess'][1] - ranges['BaseExcess'][0]
            m.addConstr((np.log10(6+1)-ranges['Lactate'][0])/c1*z['Lactate'][i][t] + \
                        (ranges['Lactate'][0]-ranges['Lactate'][0])/c1*(1 - z['Lactate'][i][t]), '<=', x['Lactate'][i][t])
            m.addConstr((np.log10(6+1)-ranges['Lactate'][0])/c1*(1-z['Lactate'][i][t]) + \
                        (ranges['Lactate'][1] - ranges['Lactate'][0])/c1*z['Lactate'][i][t], '>=', x['Lactate'][i][t])
            m.addConstr(x['BaseExcess'][i][t], '<=', (0 -ranges['BaseExcess'][0])/c2*z['Lactate'][i][t]\
                        +(ranges['BaseExcess'][1]-ranges['BaseExcess'][0])/c2*(1- z['Lactate'][i][t]))
    #If BaseExcess <= 0 then either HC03<=10 or Lactate >=6
    for i in range(k):
        for t in range(6):
            c1 = ranges['Lactate'][1] - ranges['Lactate'][0]
            c2 = ranges['BaseExcess'][1] - ranges['BaseExcess'][0]
            c3 = ranges['HCO3'][1] - ranges['HCO3'][0]
            #x[t] = 1 if BaseExcess <=0, x[t]=0 then BaseExcess <=20
            m.addConstr((0 -ranges['BaseExcess'][0])/c2*z['BaseExcess'][i][t] + (ranges['BaseExcess'][1]-\
                    ranges['BaseExcess'][0])/c2* (1 - z['BaseExcess'][i][t]), '>=', x['BaseExcess'][i][t])
            #x[t] = 0 then baseExcess >= 0, x[t]=1, then BaseExcess >= -40
            m.addConstr((ranges['BaseExcess'][0]-ranges['BaseExcess'][0])/c2*z['BaseExcess'][i][t] + \
                        (1 - z['BaseExcess'][i][t])*(0 -ranges['BaseExcess'][0])/c2, '<=', x['BaseExcess'][i][t])
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
        for t in range(6):
            c1 = ranges['pH'][1] - ranges['pH'][0]
            c2 = ranges['PaCO2'][1] - ranges['PaCO2'][0]
            c3 = ranges['HCO3'][1] - ranges['HCO3'][0]
            #x[t] = 1 if ph <=7, x[t]=0 then Ph <= 7.7
            m.addConstr((7 -ranges['pH'][0])/c1*z['pH'][i][t] + (ranges['pH'][1]-\
                        ranges['pH'][0])/c1* (1 - z['pH'][i][t]), '>=', x['pH'][i][t])
            #x[t] = 0 then pH >= 7, x[t]=1, then pH >= 6.5
            m.addConstr((6.5-ranges['pH'][0])/c1*z['pH'][i][t] + \
                        (1 - z['pH'][i][t])*(7 -ranges['pH'][0])/c1, '<=', x['pH'][i][t])
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
    #Bilirubin_direct[t] <= Bilirubin_total[t]
    for i in range(k):
        for t in range(6):
            c1 = ranges['Bilirubin_direct'][1] - ranges['Bilirubin_direct'][0]
            c2 = ranges['Bilirubin_total'][1] - ranges['Bilirubin_total'][0]
            m.addConstr(c1*x['Bilirubin_direct'][i][t] + ranges['Bilirubin_direct'][0], '<=',\
                        c2*x['Bilirubin_total'][i][t]+ranges['Bilirubin_total'][0])
    #Hct[t] >= 1.5 Hgb[t]
    for i in range(k):
        for t in range(6):
            c1 = ranges['Hgb'][1] - ranges['Hgb'][0]
            c2 = ranges['Hct'][1] - ranges['Hct'][0]
            m.addConstr(1.5*(c1*x['Hgb'][i][t] +  ranges['Hgb'][0]), '<=',c2*x['Hct'][i][t] + ranges['Hct'][0])
    #If 95 <= PaO2[t-1] and 95 <= PaO2[t+1] then SpO2[t] >= 90
    for i in range(k):
        for t in range(1,6):
            c1 = ranges['O2Sat'][1] - ranges['O2Sat'][0]
            c2 = ranges['SaO2'][1] - ranges['SaO2'][0]
            #x[t] = 1 then PaO2[t-1] >=95, x[t]=0 then PaO2[t-1] >=0
            m.addConstr((95 - ranges['SaO2'][0])/c2*z['SaO2'][i][t], '<=', x['SaO2'][i][t-1])
            #x[t] = 1 then PaO2[t-1] <= 100, x[t]=0 then PaO2[t-1] <=95
            m.addConstr((95 - ranges['SaO2'][0])/c2 + (55 - ranges['SaO2'][0])/c2*z['SaO2'][i][t], '>=', x['SaO2'][i][t-1])
        for t in range(0,5):
            c1 = ranges['O2Sat'][1] - ranges['O2Sat'][0]
            c2 = ranges['SaO2'][1] - ranges['SaO2'][0]
            #y[t] = 1 then PaO2[t+1] >=95, y[t]=0 then PaO2[t+1] >=0
            m.addConstr((95 - ranges['SaO2'][0])/c2*z['O2Sat'][i][t], '<=', x['SaO2'][i][t+1])
            #y[t] = 1 then PaCO2[t+1] <= 100, y[t]=0 then PaCO2[t+1] <=95
            m.addConstr((95 - ranges['SaO2'][0])/c2 + (55 - ranges['SaO2'][0])/c2*z['O2Sat'][i][t], '>=', x['SaO2'][i][t+1])
        for t in range(1,5):
            #SpO2[t] >= 0.90 (x[t] + y[t] -1)
            m.addConstr((90 - ranges['O2Sat'][0])/c1*(z['O2Sat'][i][t] + z['SaO2'][i][t]-1), '<=', x['O2Sat'][i][t+1])
            
    m.update()
    m.setParam( 'OutputFlag', False )
    m.write('QP_phys.lp')
    m.optimize()
    data_copy = data2.copy()
    for var in proj_variables:
        for i in range(k):
            for t in range(6):
                data_copy.loc[i,'{}-{}'.format(var,t)] = phys_to_norm(x[var][i][t].x,var,ranges)
    for var in proj_variables:
        data_copy['{}-phys-dist'.format(var)] = [np.round(sum(np.square(np.array([x[var][i][t].x - \
                    norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)], var,ranges) for t in range(6)]))),4) for i in range(k)]
    return data_copy, m.ObjVal

#%%
def projection_model_physical1(data2,ranges):
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
            for t in range(6):
                x[var][i][t] = m.addVar(lb=0, ub = 1, name='x_{}_{}_{}'.format(var,i,t))
    #define binary variables
    binary_variables = proj_variables.copy()
    binary_variables.extend(['Lactate_1', 'HCO3_1', 'HCO3_2', 'SaO2'])
    z = {}
    for var in binary_variables:
        z[var] = {}
        for i in range(k):
            z[var][i] = {}
            for t in range(6):
                z[var][i][t] = m.addVar(vtype= GRB.BINARY, name='z_{}_{}_{}'.format(var,i,t))
    m.update()
    #define Euclidean projection objective
    objExp = quicksum(np.square(np.array([x[var][i][t] -norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)],var,ranges)\
                    for t in range(6) for i in range(k) for var in  proj_variables])))
    m.setObjective(objExp, GRB.MINIMIZE)
    m.update()
    #Add constraints
    #hourly change constraints:
    for i in range(k):
        for t in range(1,6):
            for var in proj_variables:
                if len(ranges[var]) == 5:
                    c1,c2,c3,c4,c5 = ranges[var]
                    m.addConstr(x[var][i][t] - x[var][i][t-1], '<=', c5/(c2-c1))
                    m.addConstr(x[var][i][t] - x[var][i][t-1], '>=', -c5/(c2-c1))
    #MAP[t] >= 0.95(⅔ DBP[t] + ⅓ SBP[t])
    #MAP[t] <= 1.05(⅔ DBP[t] + ⅓ SBP[t])
    for i in range(k):
        for t in range(6):
            c1 = ranges['MAP'][1] - ranges['MAP'][0]
            c2 = ranges['DBP'][1] - ranges['DBP'][0]
            c3 = ranges['SBP'][1] - ranges['SBP'][0]
            m.addConstr(x['MAP'][i][t]*c1, '>=', 0.95*(2/3*(c2*x['DBP'][i][t] +ranges['DBP'][0])\
                                    + 1/3*(x['SBP'][i][t]*c3 + ranges['SBP'][0])) -  ranges['MAP'][0])
            m.addConstr(x['MAP'][i][t]*c1, '<=', 1.05*(2/3*(c2*x['DBP'][i][t] +ranges['DBP'][0])\
                                    + 1/3*(x['SBP'][i][t]*c3 + ranges['SBP'][0])) -  ranges['MAP'][0])
    #Bilirubin_direct[t] <= Bilirubin_total[t]
    for i in range(k):
        for t in range(6):
            c1 = ranges['Bilirubin_direct'][1] - ranges['Bilirubin_direct'][0]
            c2 = ranges['Bilirubin_total'][1] - ranges['Bilirubin_total'][0]
            m.addConstr(c1*x['Bilirubin_direct'][i][t] + ranges['Bilirubin_direct'][0], '<=',\
                        c2*x['Bilirubin_total'][i][t]+ranges['Bilirubin_total'][0])
    #Hct[t] >= 1.5 Hgb[t]
    for i in range(k):
        for t in range(6):
            c1 = ranges['Hgb'][1] - ranges['Hgb'][0]
            c2 = ranges['Hct'][1] - ranges['Hct'][0]
            m.addConstr(1.5*(c1*x['Hgb'][i][t] +  ranges['Hgb'][0]), '<=',c2*x['Hct'][i][t] + ranges['Hct'][0])
    #If HCO3[t] <= 10 (value), then BaseExcess[t] <= 0
    for i in range(k):
        for t in range(6):
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
        for t in range(6):
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
        for t in range(6):
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
        for t in range(6):
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
    #If 95 <= PaO2[t-1] and 95 <= PaO2[t+1] then SpO2[t] >= 90
    for i in range(k):
        for t in range(1,5):
            var = 'SaO2'
            c1 = ranges['O2Sat'][1] - ranges['O2Sat'][0]
            c2 = ranges['SaO2'][1] - ranges['SaO2'][0]
            #x[t] = 1 then PaO2[t-1] >=95, x[t]=0 then PaO2[t-1] >=0
            m.addConstr((95 - ranges['SaO2'][0])/c2*z['SaO2'][i][t] +(1- z['SaO2'][i][t])*norm_to_phys(data2.loc[i,'{}-{}'.format(var,t-1)],var,ranges)\
                        , '<=', norm_to_phys(data2.loc[i,'{}-{}'.format(var,t-1)],var,ranges))
            #x[t] = 1 then PaO2[t-1] <= 100, x[t]=0 then PaO2[t-1] <=95
            m.addConstr((95 - ranges['SaO2'][0])/c2*(1-z['SaO2'][i][t]) + norm_to_phys(data2.loc[i,'{}-{}'.format(var,t-1)],var,ranges)*z['SaO2'][i][t], '>=', norm_to_phys(data2.loc[i,'{}-{}'.format(var,t-1)],var,ranges))
        for t in range(1,5):
            c1 = ranges['O2Sat'][1] - ranges['O2Sat'][0]
            c2 = ranges['SaO2'][1] - ranges['SaO2'][0]
            #y[t] = 1 then PaO2[t+1] >=95, y[t]=0 then PaO2[t+1] >=0
            m.addConstr((95 - ranges['SaO2'][0])/c2*z['O2Sat'][i][t]+ (1- z['O2Sat'][i][t])*norm_to_phys(data2.loc[i,'{}-{}'.format(var,t+1)],var,ranges)\
                        , '<=', norm_to_phys(data2.loc[i,'{}-{}'.format(var,t+1)],var,ranges))
            #y[t] = 1 then PaCO2[t+1] <= 100, y[t]=0 then PaCO2[t+1] <=95
            m.addConstr((95 - ranges['SaO2'][0])/c2*(1-z['O2Sat'][i][t]) + norm_to_phys(data2.loc[i,'{}-{}'.format(var,t+1)],var,ranges)*z['O2Sat'][i][t], '>=',norm_to_phys(data2.loc[i,'{}-{}'.format(var,t+1)],var,ranges))
        for t in range(1,5):
            #SpO2[t] >= 0.90 (x[t] + y[t] -1)
            m.addConstr((90 - ranges['O2Sat'][0])/c1*(z['O2Sat'][i][t] + z['SaO2'][i][t]-1), '<=', x['O2Sat'][i][t])
    m.update()
    m.setParam( 'OutputFlag', False )
    m.write('QP_phys.lp')
    m.optimize()
    data_copy = data2.copy()
    for var in proj_variables:
        for i in range(k):
            for t in range(6):
                data_copy.loc[i,'{}-{}'.format(var,t)] = phys_to_norm(x[var][i][t].x,var,ranges)
    for var in proj_variables:
        data_copy['{}-phys-dist'.format(var)] = [np.round(sum(np.square(np.array([x[var][i][t].x - \
                    norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)], var,ranges) for t in range(6)]))),4) for i in range(k)]
    return data_copy, m.ObjVal


#%%
def projection_model_normal(data,ranges):

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
            for t in range(6):
                x[var][i][t] = m.addVar(lb=0, ub = 1, name='x_{}_{}_{}'.format(var,i,t))
                
    m.update()


    #define Euclidean projection objective
    objExp = quicksum(np.square(np.array([x[var][i][t] - data.loc[i,'{}-{}'.format(var,t)]\
                    for t in range(6) for i in range(k) for var in  proj_variables])))
    m.setObjective(objExp, GRB.MINIMIZE)
    m.update()
    m.setParam( 'OutputFlag', False )
    m.write('QP_norm.lp')

    m.optimize()

    data_copy = data.copy()
    for var in proj_variables:
        for i in range(k):
            for t in range(6):
                data_copy.loc[i,'{}-{}'.format(var,t)] = x[var][i][t].x
                
    for var in proj_variables:             
        data_copy['{}-norm-dist'.format(var)] = [np.round(sum(np.square(np.array([x[var][i][t].x - \
                    data.loc[i,'{}-{}'.format(var,t)] for t in range(6)]))),4) for i in range(k)]
        
    return data_copy, m.ObjVal

#%%


def project():
    
    #Read Data
    data = pd.read_csv('Imputed/summary_1633396048.csv').iloc[:,1:]
    ranges = get_ranges('constraints.txt')
    
    final_phy_2,val_phy2 = projection_model_physical2(data.copy(),ranges )
    print("Done to physical - earlier")
    
    final_phy_1,val_phy1 = projection_model_physical1(data.copy(),ranges )
    print("Done to physical - new")
    
    final_normal_1,val_normal1 = projection_model_normal(final_phy_1.copy(),ranges)
    print("Done to Normal new - {}".format(i))
    
    final_normal_2,val_normal2 = projection_model_normal(final_phy_2.copy(),ranges)
    print("Done to Normal earlier - {}".format(i))
        
    return final_phy1, final_normal1, final_phy2, final_normal2

#%%

if __name__ == '__main__':

    physical_new, normal_new, physical_old, normal_old = project()
    physical_new.to_csv('physical_new_5oct.csv')
    physical_old.to_csv('physical_old_5oct.csv')
    
    normal_new.to_csv('normal_new_5oct.csv')
    normal_old.to_csv('normal_old_5oct.csv')


