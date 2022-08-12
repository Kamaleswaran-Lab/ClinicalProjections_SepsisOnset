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

def projection_model_physical(data, ranges, interval):
    
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

    print("Variables Added")
    #define Euclidean projection objective
    objExp = quicksum(np.square(np.array([x[var][i][t] - normalize(data.loc[i,'{}-{}'.format(var,t)],var,ranges, norm_to = 'physical')\
                    for t in range(interval) for i in range(k) for var in  proj_variables])))
    m.setObjective(objExp, GRB.MINIMIZE)
    m.update()

    print("Objective defined")
    #Add constraints

    #hourly change constraints:
    for i in range(k):
        for t in range(1,interval):
            for var in proj_variables:
                if len(ranges[var]) == interval-1:
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
                                    + 1/3*(x['SBP'][i][t]*c3 + ranges['SBP'][0]) -  ranges['MAP'][0]))
            m.addConstr(x['MAP'][i][t]*c1, '<=', 1.05*(2/3*(c2*x['DBP'][i][t] +ranges['DBP'][0])\
                                    + 1/3*(x['SBP'][i][t]*c3 + ranges['SBP'][0]) -  ranges['MAP'][0]))

    #If HCO3[t] <= 10 (value), then BaseExcess[t] <= 0
    for i in range(k):
        for t in range(interval):
            c1 = ranges['HCO3'][1] - ranges['HCO3'][0]
            c2 = ranges['BaseExcess'][1] - ranges['BaseExcess'][0]
            m.addConstr((10-ranges['HCO3'][0])/c1*z['HCO3'][i][t] + (ranges['HCO3'][1]-ranges['HCO3'][0])/c1*(1 - z['HCO3'][i][t])\
                        ,'>=', x['HCO3'][i][t])
            m.addConstr((10-ranges['HCO3'][0])/c1*(1-z['HCO3'][i][t]), '<=', x['HCO3'][i][t])
            m.addConstr(x['BaseExcess'][i][t], '<=', (ranges['BaseExcess'][1]-ranges['BaseExcess'][0])/c2*(1- z['HCO3'][i][t]))


    #If Lactate >= 6, then BaseExcess <= 0. (values)
    for i in range(k):
        for t in range(interval):
            c1 = ranges['Lactate'][1] - ranges['Lactate'][0]
            c2 = ranges['BaseExcess'][1] - ranges['BaseExcess'][0]
            m.addConstr((np.log10(6+1)-ranges['Lactate'][0])/c1*z['Lactate'][i][t] + \
                        (ranges['Lactate'][0]-ranges['Lactate'][0])/c1*(1 - z['Lactate'][i][t]), '<=', x['Lactate'][i][t])
            m.addConstr((np.log10(6+1)-ranges['Lactate'][0])/c1*(1-z['Lactate'][i][t]) + \
                        (ranges['Lactate'][1] - ranges['Lactate'][0])/c1*z['Lactate'][i][t], '>=', x['Lactate'][i][t])
            m.addConstr(x['BaseExcess'][i][t], '<=', (ranges['BaseExcess'][1]-ranges['BaseExcess'][0])/c2*(1- z['Lactate'][i][t]))

            m.addConstr(x['BaseExcess'][i][t], '<=', (0 -ranges['BaseExcess'][0])/c2*z['Lactate'][i][t]\
            +(ranges['BaseExcess'][1]-ranges['BaseExcess'][0])/c2*(1- z['Lactate'][i][t]))
#If BaseExcess <= 0 then either HC03<=10 or Lactate >=6
    #If BaseExcess <= 0 then either HC03<=10 or Lactate >=6

    for i in range(k):
        for t in range(interval):
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
        for t in range(interval):
            c1 = ranges['pH'][1] - ranges['pH'][0]
            c2 = ranges['PaCO2'][1] - ranges['PaCO2'][0]
            c3 = ranges['HCO3'][1] - ranges['HCO3'][0]

            #x[t] = 1 if ph <=7, x[t]=0 then Ph <= 7.7
            m.addConstr((7 -ranges['pH'][0])/c1*z['pH'][i][t] + (ranges['pH'][1]-\
                        ranges['pH'][0])/c1* (1 - z['pH'][i][t]), '>=', x['pH'][i][t])

            #x[t] = 0 then pH >= 7, x[t]=1, then pH >= 6.5
            m.addConstr((ranges['pH'][0]-ranges['pH'][0])/c1*z['pH'][i][t] + \
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
            m.addConstr((10-ranges['HCO3'][0])/c3*(1-z['HCO3_2'][i][t])+ z['HCO3_2'][i][t]*(ranges['HCO3'][1] -\
                        ranges['HCO3'][0])/c3, '<=', x['HCO3'][i][t])

            #x[t] <= z[t] + y[t]
            m.addConstr(z['pH'][i][t], '<=', z['PaCO2'][i][t] + z['HCO3_2'][i][t])


    #Bilirubin_direct[t] <= Bilirubin_total[t]
    for i in range(k):
        for t in range(interval):
            m.addConstr(x['Bilirubin_direct'][i][t], '<=',x['Bilirubin_total'][i][t])

    #Hct[t] >= 1.5 Hgb[t]
    for i in range(k):
        for t in range(interval):
            m.addConstr(1.5*x['Hgb'][i][t], '<=',x['Hct'][i][t])


    #If 95 <= PaO2[t-1] and 95 <= PaO2[t+1] then SpO2[t] >= 90
    for i in range(k):
        for t in range(1,interval):
            c1 = ranges['O2Sat'][1] - ranges['O2Sat'][0]
            c2 = ranges['SaO2'][1] - ranges['SaO2'][0]

            #x[t] = 1 then PaO2[t-1] >=95, x[t]=0 then PaO2[t-1] >=0
            m.addConstr((95 - ranges['SaO2'][0])/c2*z['SaO2'][i][t], '<=', x['SaO2'][i][t-1])

            #x[t] = 1 then PaO2[t-1] <= 100, x[t]=0 then PaO2[t-1] <=95
            m.addConstr((95 - ranges['SaO2'][0])/c2 + (5 - ranges['SaO2'][0])/c2*z['SaO2'][i][t], '>=', x['SaO2'][i][t-1])
            
        for t in range(0,interval-1):
            c1 = ranges['O2Sat'][1] - ranges['O2Sat'][0]
            c2 = ranges['SaO2'][1] - ranges['SaO2'][0]

            #y[t] = 1 then PaO2[t+1] >=95, y[t]=0 then PaO2[t+1] >=0
            m.addConstr((95 - ranges['SaO2'][0])/c2*z['O2Sat'][i][t], '<=', x['SaO2'][i][t+1])

            #y[t] = 1 then PaCO2[t+1] <= 100, y[t]=0 then PaCO2[t+1] <=95
            m.addConstr((95 - ranges['SaO2'][0])/c2 + (5 - ranges['SaO2'][0])/c2*z['O2Sat'][i][t], '>=', x['SaO2'][i][t+1])
            
        for t in range(1,interval-1):
            #SpO2[t] >= 0.90 (x[t] + y[t] -1)
            m.addConstr((90 - ranges['SaO2'][0])/c2*(z['O2Sat'][i][t] + z['SaO2'][i][t]-1), '<=', x['O2Sat'][i][t+1])

    m.update()
    m.setParam( 'OutputFlag', False )
    m.write('QP_phys.lp')
    
    print("Constrains written")

    m.optimize()

    print("Optimised")
    data_copy = data.copy()
    for var in proj_variables:
        for i in range(k):
            for t in range(interval):
                data_copy.loc[i,'{}-{}'.format(var,t)] = normalize(x[var][i][t].x,var,ranges, norm_to = 'normal')
                
    print("Data copied")
    for var in proj_variables:             
        data_copy['{}-phys-dist'.format(var)] = [np.round(sum(np.square(np.array([x[var][i][t].x - \
                    normalize(data.loc[i,'{}-{}'.format(var,t)], var,ranges, norm_to = 'physical') for t in range(interval)]))),4) for i in range(k)]
    print("Data diff")
    return data_copy, m.ObjVal

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
def projection_model_physical3(data2,ranges, interval ):
    
    # Create model
    m = Model("projection")
    k = len(data2)
    proj_variables = list(ranges.keys())
    
    #Preprocessing step
    """
    c1,c2,c3,c4 = ranges['O2Sat'][:4]
    temp1 = (90 - c3)/(c4 - c3)
    
    for i in range(k):
        for t in range(1,5):
            
            check1 = norm_to_none(data2.loc[i,'{}-{}'.format('SaO2',t-1)], 'SaO2', ranges) >= 95 and norm_to_none(data2.loc[i,'{}-{}'.format('SaO2',t+1)],'SaO2', ranges) >= 95
            check2 = abs(norm_to_none(data2.loc[i,'{}-{}'.format('SaO2',t-1)],'SaO2', ranges) - norm_to_none(data2.loc[i,'{}-{}'.format('O2Sat',t-1)], 'O2Sat', ranges)) >= 10\
                                  and  abs(norm_to_none(data2.loc[i,'{}-{}'.format('SaO2',t+1)], 'SaO2', ranges) - norm_to_none(data2.loc[i,'{}-{}'.format('O2Sat',t+1)], 'O2Sat', ranges)) >= 10
            check3 =  abs(norm_to_none(data2.loc[i,'{}-{}'.format('FiO2',t-1)], 'FiO2', ranges) - norm_to_none(data2.loc[i,'{}-{}'.format('FiO2',t+1)], 'FiO2', ranges)) <= 1e-4
                                   
                                                        
            if(check1 and check2 and check3 and norm_to_none(data2.loc[i,'{}-{}'.format('O2Sat',t)], 'O2Sat', ranges) < 90 ):
                data2.loc[i,'{}-{}'.format('O2Sat',t)] = temp1
                
      """          
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
    #If 95 <= PaO2[t-1] and 95 <= PaO2[t+1] then SpO2[t] >= 90
    
    """
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
    """
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
                    norm_to_phys(data2.loc[i,'{}-{}'.format(var,t)], var,ranges) for t in range(interval)]))),4) for i in range(k)]
    return data_copy, m.ObjVal

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

#%%

for i in range(k):
    for t in range(1,5):
        
        check1 = norm_to_none(data2.loc[i,'{}-{}'.format('SaO2',t-1)], 'SaO2', ranges) >= 95 and norm_to_none(data2.loc[i,'{}-{}'.format('SaO2',t+1)],'SaO2', ranges) >= 95
        check2 = abs(norm_to_none(data2.loc[i,'{}-{}'.format('SaO2',t-1)],'SaO2', ranges) - norm_to_none(data2.loc[i,'{}-{}'.format('O2Sat',t-1)], 'O2Sat', ranges)) >= 10\
                              and  abs(norm_to_none(data2.loc[i,'{}-{}'.format('SaO2',t+1)], 'SaO2', ranges) - norm_to_none(data2.loc[i,'{}-{}'.format('O2Sat',t+1)], 'O2Sat', ranges)) >= 10
        check3 =  abs(norm_to_none(data2.loc[i,'{}-{}'.format('FiO2',t-1)], 'FiO2', ranges) - norm_to_none(data2.loc[i,'{}-{}'.format('FiO2',t+1)], 'FiO2', ranges)) <= 1e-4
                               
                                                    
        if(check1 and check2 and check3 and norm_to_none(data2.loc[i,'{}-{}'.format('O2Sat',t)], 'O2Sat', ranges) < 90 ):
            data2.loc[i,'{}-{}'.format('O2Sat',t)] = temp1


#%%
def projection_model_normal(data,ranges, interval):

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

#%%

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


def compute_SOFA(df):
   
    #return df.apply(
    #    lambda df: max(
    #        # compute_SOFA_resp(df["FiO2"], df["O2Sat"]),
    #        compute_SOFA_coag(df["Platelets"]),
    #        compute_SOFA_liver(df["Bilirubin_total"]),
    #        compute_SOFA_renal(df["Creatinine"]),
    #    ),
    #    axis=1,
    #)

    var = 'SOFA'
    k = len(df)
    for i in range(k):
        for t in range(6):
            df.loc[i,'{}-{}'.format(var,t)] = max(
                # compute_SOFA_resp(df["FiO2"], df["O2Sat"]),
                compute_SOFA_coag(norm_to_none(df.loc[i,'{}-{}'.format('Platelets',t)], 'Platelets', ranges)),
                compute_SOFA_liver(norm_to_none(df.loc[i,'{}-{}'.format('Bilirubin_total',t)], 'Bilirubin_total', ranges)),
                compute_SOFA_renal(norm_to_none(df.loc[i,'{}-{}'.format('Creatinine',t)], 'Creatinine', ranges))
            )
    
    return df

def compute_SIRS(df):
    #return df.apply(
    #    lambda df: ((df["Temp"] < 36) | (df["Temp"] > 38))
    #    + (df["HR"] > 90)
    #    + ((df["WBC"] < 4) | (df["WBC"] > 12))
    #    + (df["Resp"] > 20),
    #    axis=1,
    #)
    
    var = 'SIRS'
    k = len(df)
    for i in range(k):
        for t in range(6):
            temp = norm_to_none(df.loc[i,'{}-{}'.format('Temp',t)], 'Temp', ranges)
            hr = norm_to_none(df.loc[i,'{}-{}'.format('HR',t)], 'HR', ranges)
            wbc = norm_to_none(df.loc[i,'{}-{}'.format('WBC',t)], 'WBC', ranges)
            resp = norm_to_none(df.loc[i,'{}-{}'.format('Resp',t)], 'Resp', ranges)
            
            df.loc[i,'{}-{}'.format(var,t)] = ((temp < 36) | (temp > 38)) + (hr > 90) + ((wbc < 4) | ( wbc > 12)) + (resp > 20)

    return df
    

#%%

def project(interval):
    
    #Read Data
    data = pd.read_csv('Imputed_3_1/summary_1652462935.csv').iloc[:,1:]
    data_split = np.array_split(data, 500)
    ranges = get_ranges('constraints_wo_calcium.txt')
    #final_phy1 = pd.DataFrame()
    #final_normal1 = pd.DataFrame()
    #final_phy2 = pd.DataFrame()
    #final_normal2 = pd.DataFrame()
    final_phy3 = pd.DataFrame()
    final_normal3 = pd.DataFrame()
    
    for i, data_ in enumerate(data_split):
        
        data_ = data_.reset_index()
        
        final_phy_3,val_phy3 = projection_model_physical3(data_.copy(),ranges, interval)
        print(str(i) + "Done to physical 3")
        #final_phy_2,val_phy2 = projection_model_physical2(data_.copy(),ranges )
        #print("Done to physical - earlier")
        
        #final_phy_1,val_phy1 = projection_model_physical1(data_.copy(),ranges )
        #print("Done to physical - new")
        
        final_normal_3,val_normal3 = projection_model_normal(final_phy_3.copy(),ranges, interval)
        print(str(i) + "Done to Normal 3")
        
        #final_normal_1,val_normal1 = projection_model_normal(final_phy_1.copy(),ranges)
        #print("Done to Normal new - {}".format(i))
        
        #final_normal_2,val_normal2 = projection_model_normal(final_phy_2.copy(),ranges)
        #print("Done to Normal earlier - {}".format(i))
        
        final_phy_3 = final_phy_3.set_index('index', drop = True)
        final_normal_3 = final_normal_3.set_index('index', drop = True)
        final_phy3 = final_phy3.append(final_phy_3)
        final_normal3 = final_normal3.append(final_normal_3)
        
        #final_phy_1 = final_phy_1.set_index('index', drop = True)
        #final_normal_1 = final_normal_1.set_index('index', drop = True)
        #final_phy1 = final_phy1.append(final_phy_1)
        #final_normal1 = final_normal1.append(final_normal_1)
        
        #final_phy_2 = final_phy_2.set_index('index', drop = True)
        #final_normal_2 = final_normal_2.set_index('index', drop = True)
        #final_phy2 = final_phy2.append(final_phy_2)
        #final_normal2 = final_normal2.append(final_normal_2)
        
        
    return final_phy3, final_normal3 #final_phy1, final_normal1, final_phy2, final_normal2

#%%

if __name__ == '__main__':

    #physical_new, normal_new, physical_old, normal_old = project()
    #physical_new.to_csv('physical_new_5oct.csv')
    #physical_old.to_csv('physical_old_5oct.csv')
    
    #normal_new.to_csv('normal_new_5oct.csv')
    #normal_old.to_csv('normal_old_5oct.csv')
    interval = 3
    physical3, normal3 = project(interval)
    
    physical3.to_csv('physical_3_1.csv')
    normal3.to_csv('normal_3_1.csv')

#%%

data = pd.read_csv('Imputed/summary_1633396048.csv').iloc[:,1:]
ranges = get_ranges('constraints_wo_calcium.txt')

#%%

data2 = data.copy()
k = len(data2)
pats = []
ts = []

c1,c2,c3,c4 = ranges['O2Sat'][:4]
temp1 = (95 - c3)/(c4 - c3)

for i in range(k):
    for t in range(1,5):
        
        check1 = norm_to_none(data2.loc[i,'{}-{}'.format('SaO2',t-1)], 'SaO2', ranges) >= 95 and norm_to_none(data2.loc[i,'{}-{}'.format('SaO2',t+1)],'SaO2', ranges) >= 95
        check2 = abs(norm_to_none(data2.loc[i,'{}-{}'.format('SaO2',t)],'SaO2', ranges) - norm_to_none(data2.loc[i,'{}-{}'.format('O2Sat',t)], 'O2Sat', ranges)) >= 10\
                              and  abs(norm_to_none(data2.loc[i,'{}-{}'.format('SaO2',t+1)], 'SaO2', ranges) - norm_to_none(data2.loc[i,'{}-{}'.format('O2Sat',t+1)], 'O2Sat', ranges)) >= 10
        check3 =  abs(norm_to_none(data2.loc[i,'{}-{}'.format('FiO2',t-1)], 'FiO2', ranges) - norm_to_none(data2.loc[i,'{}-{}'.format('FiO2',t+1)], 'FiO2', ranges)) <= 5                              
                                                    
        if(check1 and check2 and check3 and norm_to_none(data2.loc[i,'{}-{}'.format('O2Sat',t)], 'O2Sat', ranges) < 90 ):
            pats.append(i)
            ts.append(t)
            data2.loc[i,'{}-{}'.format('O2Sat',t)] = temp1


#%%
corrected_patients = np.unique(data.loc[pats]['patient_id'].values)

#%%

orig_sao2 = data.loc[data['patient_id'].isin(corrected_patients)][['SaO2-{}'.format(i) for i in range(6)]].apply(lambda x: norm_to_none(x,'SaO2', ranges))
corrected_sao2 = data2.loc[data2['patient_id'].isin(corrected_patients)][['SaO2-{}'.format(i) for i in range(6)]].apply(lambda x: norm_to_none(x,'SaO2', ranges))

orig_o2sat = data.loc[data['patient_id'].isin(corrected_patients)][['O2Sat-{}'.format(i) for i in range(6)]].apply(lambda x: norm_to_none(x,'O2Sat', ranges))
corrected_o2sat = data2.loc[data2['patient_id'].isin(corrected_patients)][['O2Sat-{}'.format(i) for i in range(6)]].apply(lambda x: norm_to_none(x,'O2Sat', ranges))


orig_fio2 = data.loc[data['patient_id'].isin(corrected_patients)][['FiO2-{}'.format(i) for i in range(6)]].apply(lambda x: norm_to_none(x,'O2Sat', ranges))
corrected_fio2 = data2.loc[data2['patient_id'].isin(corrected_patients)][['FiO2-{}'.format(i) for i in range(6)]].apply(lambda x: norm_to_none(x,'O2Sat', ranges))

#%%
orig_pats = np.copy(np.array(pats))

pats = np.unique(np.array(pats))

#%%


orig_sao2 = data.loc[pats][['SaO2-{}'.format(i) for i in range(6)]].apply(lambda x: norm_to_none(x,'SaO2', ranges))
corrected_sao2 = data2.loc[pats][['SaO2-{}'.format(i) for i in range(6)]].apply(lambda x: norm_to_none(x,'SaO2', ranges))

orig_o2sat = data.loc[pats][['O2Sat-{}'.format(i) for i in range(6)]].apply(lambda x: norm_to_none(x,'O2Sat', ranges))
corrected_o2sat = data2.loc[pats][['O2Sat-{}'.format(i) for i in range(6)]].apply(lambda x: norm_to_none(x,'O2Sat', ranges))

orig_fio2 = data.loc[pats][['FiO2-{}'.format(i) for i in range(6)]].apply(lambda x: norm_to_none(x,'O2Sat', ranges))
corrected_fio2 = data2.loc[pats][['FiO2-{}'.format(i) for i in range(6)]].apply(lambda x: norm_to_none(x,'O2Sat', ranges))

orig = pd.concat((orig_o2sat, orig_sao2, orig_fio2), axis = 1)
orig['patient_id'] = data.loc[data['patient_id'].isin(corrected_patients)]['patient_id']

changed = pd.concat((corrected_o2sat, orig_o2sat, corrected_sao2, corrected_fio2), axis = 1)
changed['patient_id'] = data.loc[data['patient_id'].isin(corrected_patients)]['patient_id']

#%%

for p in corrected_patients:
    
    index = data.loc[data['patient_id'] == p].index
    


#%%

for p in corrected_patients:
    
    index = data.loc[data['patient_id'] == p].index
    
    orig_sao2_ext = []
    corrected_sao2_ext = []
    
    orig_o2sat_ext = []
    corrected_o2sat_ext = []
    
    orig_fio2_ext = []
    corrected_fio2_ext = []
    
    for i in index:
        for j in range(5):
            orig_sao2_ext.append(orig_sao2.loc[i][j])
            corrected_sao2_ext.append(corrected_sao2.loc[i][j])
            
            orig_o2sat_ext.append(orig_o2sat.loc[i][j])
            corrected_o2sat_ext.append(corrected_o2sat.loc[i][j])
            
            orig_fio2_ext.append(orig_fio2.loc[i][j])
            corrected_fio2_ext.append(corrected_fio2.loc[i][j])

    orig_sao2_ext = np.array(orig_sao2_ext)
    corrected_sao2_ext = np.array(corrected_sao2_ext)
    
    orig_o2sat_ext = np.array(orig_o2sat_ext)
    corrected_o2sat_ext = np.array(corrected_o2sat_ext)
    
    orig_fio2_ext = np.array(orig_fio2_ext)
    corrected_fio2_ext = np.array(corrected_fio2_ext)
    
    fig, axes = plt.subplots(nrows=7, ncols=5, figsize = (25,25))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('PAT_ID :' + PAT_ID , fontsize = 20)
    for i in range(len(proj_variables)):     
        tot_phys_dist_i = df_pat['{}-phys-dist-total'.format(proj_variables[i])].loc[df_pat['patient_id'] == PAT_ID].values
        avg_phys_dist_i = df_pat['{}-phys-dist-avg'.format(proj_variables[i])].loc[df_pat['patient_id'] == PAT_ID].values
        l1 = axes[int(i/5)][int(i % 5)].plot(y, reconstructed_imputed[proj_variables[i]],'r', y, reconstructed_physical[proj_variables[i]], 'y')
        start, end = axes[int(i/5)][int(i % 5)].get_ylim()
        axes[int(i/5)][int(i % 5)].yaxis.set_ticks(np.arange(start, end, (end-start)/5))
        axes[int(i/5)][int(i % 5)].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
        if('Bilirubin' in proj_variables[i]):
            axes[int(i/5)][int(i % 5)].set_title('Blrb_' + proj_variables[i][-5:] + str(tot_phys_dist_i) + str(avg_phys_dist_i), fontsize = 18)
        else:
            axes[int(i/5)][int(i % 5)].set_title(proj_variables[i] + str(tot_phys_dist_i) + str(avg_phys_dist_i), fontsize = 18)
    
    fig.legend([l1[0], l1[1]], ['Imputed', 'Projected onto Physical'], fontsize = 20)


    fig.savefig(folder + '/check_' + PAT_ID)
    
    
    
    
    
#%%
ranges = get_ranges('constraints_wo_calcium.txt')
#pats = ['A003242', 'A016800', 'B101215' ,'B105116', 'B105122', 'B107096' ,'B108849', 'B108983' ,'B110701' ,'B111876' ,'B111922' ,'B112696', 'B119585']
pats = ['B101215']
final_phy = pd.DataFrame()


data_ = data.loc[data['patient_id'].isin( pats)]
data_ = data_.reset_index()
final_phy_,val_phy = projection_model_physical3(data_.copy(),ranges )

final_phy_ = final_phy_.set_index('index', drop = True)
final_phy = final_phy.append(final_phy_)

#final_phy.to_csv('HighPhys.csv')


#%%

data_.to_csv('HighPhysi.csv')


#%%

df1 = final_phy
df2 = phys

ne = (df1 != df2).any(1)

ne_stacked = (df1 != df2).stack()
changed = ne_stacked[ne_stacked]

changed.index.names = ['id', 'col']
difference_locations = np.where(df1 != df2)

changed_from = df1.values[difference_locations]

changed_to = df2.values[difference_locations]

out = pd.DataFrame({'from': changed_from, 'to': changed_to}, index=changed.index)



#%%

data_.head()
