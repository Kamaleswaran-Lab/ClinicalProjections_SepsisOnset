# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:00:58 2022

@author: mehak
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import date
import clustering_functions as clust_func
import utils
import seaborn as sns


from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler

kmeans = utils.load_obj('./', 'feat_kmeans12_median.pkl')

#%% Load Data 

def load_physionet(data_path = '../data/', sets = 'Both'):
    data_physical = pd.read_pickle(os.path.join(data_path, 'projected_physical_6_3.pkl'))
    data_normal = pd.read_pickle(os.path.join(data_path, 'projected_normal_6_3.pkl'))
    
    dirty_labels = [i for i in data_physical.columns if (('dirty' in i) or ('dist' in i))]
    remove = dirty_labels + ['index'] + ['patient_id']
    remaining = [i for i in data_physical.columns if i not in remove]
    data = data_physical[remaining].copy()
    
    dist_labels = [i for i in data_normal.columns if (('dist' in i) and ('norm' in i))]
    data = pd.concat([data[remaining],data_normal[dist_labels]], axis = 1)
    min_max_scaler = MinMaxScaler()
    data.loc[:,dist_labels] = min_max_scaler.fit_transform(data[dist_labels])
    data['Age'] = (data['Age']  - data['Age'].min()) / (data['Age'].max() - data['Age'] .min())
    patID = data_physical['patient_id']
    sepsisLabel = pd.DataFrame( columns = ['SepsisLabel'])
    sepsisLabel['SepsisLabel'] = data['SepsisLabel']
    
    #Dividing into site A and B
    if sets == 'A':
        data = data.loc[['A' in patID.values[i] for i in range(len(patID))]]
        sepsisLabel = sepsisLabel.loc[['A' in patID.values[i] for i in range(len(patID))]]
        patID = patID.loc[['A' in patID.values[i] for i in range(len(patID))]]
    elif sets == 'B':
        data = data.loc[['B' in patID.values[i] for i in range(len(patID))]]
        sepsisLabel = sepsisLabel.loc[['B' in patID.values[i] for i in range(len(patID))]]
        patID = patID.loc[['B' in patID.values[i] for i in range(len(patID))]]
    elif sets != 'Both':
        print(sets + " not recognized")
        
    columns = list(data.columns)
    variables = [s[:-2] for s in columns if '-0' in s]
    print(variables)
    data_feat = pd.DataFrame()
    
    for var in variables:
        data_feat[var + '-median'] = data[[var + '-' + str(i) for i in range(6)]].median(axis = 1)
        data_feat[var + '-std'] = data[[var + '-' + str(i) for i in range(6)]].std(axis = 1)
        
        if var in ['Calcium', 'SOFA', 'SIRS']:
            continue
        else:
    
            data_feat[var + '-norm-dist'] = data[var + '-norm-dist']

    return data, data_feat, sepsisLabel, patID 

data, data_feat, sepsisLabel, patID = load_physionet()
clustering_labels = kmeans.predict(data_feat)

#%%


data_imputed = pd.read_pickle(os.path.join('../data/', 'imputed_6_3.pkl'))
patID_imputed = data_imputed['patient_id']
sepsislabel_imputed = data_imputed['SepsisLabel']
data_imputed.drop(columns = ['patient_id', 'SepsisLabel'], inplace = True)
#%%

i = 1
splitter = GroupShuffleSplit(test_size=.20, n_splits=5, random_state = 7)
for split in splitter.split(data_imputed, groups=patID):
    train_idx = split[0]
    test_idx = split[0]
    model1, res, params = modelfit(data_imputed.loc[train_idx], clustering_labels[train_idx])    
    utils.save_obj('./', 'model_val_{}'.format(i) + '.pkl')
    i+=1
    test_predictions = model1.predict(data_imputed.loc[test_idx])
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train.values, dtrain_predictions))
    






#%% Stats 
#Calculates concentration and prevalence
df_stats = clust_func.clustering_stats(clustering_labels, data)
sepsis = clust_func.sepsis_concentration(clustering_labels, sepsisLabel, patID)
#clust_func.most_varying_feat(clustering_labels, data_feat, 'physionet_SetA_data_mvf12.png')
#clust_func.randomForest_feat_imp(clustering_labels, data_feat, 'physionet_SetB_data_rf12.png')

sepsis.to_csv('conc_setB_data.csv')
df_stats.to_csv('stats_setB_data.csv')

#%% Transition Matrix and chord diagram

matrix_kmeans12, matrix_kmeans12_control = clust_func.get_transition_matrix(clustering_labels, 
                                                           sepsisLabel, patID, 'kmeans12_setB_data.npy')
clust_func.chord_diagram(matrix_kmeans12, 'Physionet_sepsis_kmeans12.html')


#%% Load results 

transition_matrix_sepsis = np.load('Grady/sepsis_Grady_icu.npy')
transition_matrix_control = np.load('Grady/control_Grady_icu.npy')

#%%

clust_func.chord_diagram(transition_matrix_sepsis, 'Grady/Grady_sepsis_icu.html')
clust_func.chord_diagram(transition_matrix_control, 'Grady/Grady_control_icu.html')



#%%
transition_matrix_sepsis_diag = np.eye(12)*transition_matrix_sepsis
transition_matrix_control_diag = np.eye(12)*transition_matrix_control

diff_state_transitions_sepsis = transition_matrix_sepsis - transition_matrix_sepsis_diag
diff_state_transitions_control = transition_matrix_control - transition_matrix_control_diag

total_diff_state_transitions = diff_state_transitions_sepsis.sum() + diff_state_transitions_control.sum()
print(total_diff_state_transitions)

plt.figure(figsize = (15, 10))
sns.heatmap(diff_state_transitions_sepsis/diff_state_transitions_sepsis.sum())
plt.xlabel('Cluster to', fontsize = 20)
plt.ylabel('Cluster from', fontsize = 20)
plt.title('Cluster Transition Matrix for sepsis patients', fontsize = 20)

plt.figure(figsize = (15,10))
sns.heatmap(diff_state_transitions_control/diff_state_transitions_control.sum())
plt.xlabel('Cluster to', fontsize = 20)
plt.ylabel('Cluster from', fontsize = 20)
plt.title('Cluster Transition Matrix for control patients', fontsize = 20)


#%%

data_path = './Grady'
patID = pd.read_pickle(os.path.join(data_path, 'patid_icu_grady.pkl'))
sepsisLabel = pd.read_pickle(os.path.join(data_path, 'sepsisLabel_icu_grady.pkl'))
clustering_labels = np.load(os.path.join(data_path,'clustering_labels_grady_icu.npy'))
#trajectory = clust_func.trajectories(patID, clustering_labels, sepsisLabel)
#trajectory.to_pickle('trajectories_Grady_icu.pkl')
trajectory = pd.read_pickle('Physionet_all_data/trajectories.pkl')

#%%

x = trajectory.clusterSeq.values
length = max(map(len, x))
y = np.empty((len(x), length ))
yend = np.empty((len(x), length ))

## y and yend are of size length with None values in the end and in the beginning respectively
for idx, xi in enumerate(x):
    y[idx, :] = np.concatenate((xi, np.array([None]*(length - len(xi)))))
    yend[idx, :] = np.concatenate((np.array([None]*(length - len(xi))), xi))
    
#%%

# Get 48 hours before sepsis/discharge
length = 7
y48 = np.empty((len(x), length ))
yend48 = np.empty((len(x), length ))
yrep48 = np.empty((len(x),length))

## y and yend are of size length with None values in the end and in the beginning respectively
for idx, xi in enumerate(x):
    time_sepsis = trajectory.loc[trajectory.index[idx]]['time_to_sepsis_subpat']
    if time_sepsis == -1:
        xi = xi[-length:]
    elif time_sepsis < length:
        xi = xi[:time_sepsis+1]
    else:
        xi = xi[time_sepsis-length+1:time_sepsis+1]
    print(len(xi), time_sepsis)
    y48[idx, :] = np.concatenate((xi, np.array([None]*(length - len(xi)))))
    yend48[idx, :] = np.concatenate((np.array([None]*(length - len(xi))), xi))
        
    yrep48[idx, :] = np.concatenate((xi, np.array([xi[-1]]*(length - len(xi)))))
    
    
#%%

clusterDf = pd.DataFrame(columns = ['pat_id', 
                                    'cluster_transitions-0', 
                                    'cluster_transitions-1', 
                                    'cluster_transitions-2', 
                                    'cluster_transitions-3', 
                                    'cluster_transitions-4', 
                                    'cluster_transitions-5', 
                                    'cluster_transitions-6', 
                                    'time_to_sepsis', 'sepsis_label',
                                    'total_icu_time'])
    
clusterDf['pat_id'] = trajectory['patid']
clusterDf['sepsis_label'] = trajectory['Sepsis']
clusterDf['time_to_sepsis'] = trajectory['time_to_sepsis_subpat']
clusterDf['total_icu_time'] = trajectory['total_icu_time_subpat']
clusterDf['cluster_transitions-0'] = yrep48[:,0]
clusterDf['cluster_transitions-1'] = yrep48[:,1]
clusterDf['cluster_transitions-2'] = yrep48[:,2]
clusterDf['cluster_transitions-3'] = yrep48[:,3]
clusterDf['cluster_transitions-4'] = yrep48[:,4]
clusterDf['cluster_transitions-5'] = yrep48[:,5]
clusterDf['cluster_transitions-6'] = yrep48[:,6]

clusterDf.to_pickle('clusterDf_Grady_icu.pkl')

#%%

""" Sort """

plt.figure(figsize = (25,15))

parameters = {'axes.labelsize': 10,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)

ykeys = tuple([yrep48[:,i] for i in range(length)])
idx = np.lexsort(ykeys)
sepsislabel_sorted = trajectory.Sepsis.values[idx]
y_sorted = yrep48[idx]
yend_sorted = yrep48[idx]
sns.heatmap(yend_sorted) 
plt.title("All supatients - cluster transitions sorted")

#%%

plt.figure(figsize = (25,15))

parameters = {'axes.labelsize': 10,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)
sepsis = np.where(sepsislabel_sorted == 1)[0]
y_sepsis = yend_sorted[sepsis]
#y_sepsis.sort(axis = 0)
sns.heatmap(y_sepsis)
plt.title("Sepsis Subpatients - Sorted Cluster Transitions")

#%%
plt.figure(figsize = (25,15))

parameters = {'axes.labelsize': 10,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)
nonsepsis = np.where(sepsislabel_sorted == 0)[0]
y_nonsepsis = yend_sorted[nonsepsis]
#y_sepsis.sort(axis = 0)
sns.heatmap(y_nonsepsis)
plt.title("Non-sepsis Subpatients - Sorted Cluster Transitions")

#%%

plt.figure(figsize = (25,15))

parameters = {'axes.labelsize': 10,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)
sepsis = np.where(sepsislabel_sorted == 1)[0]
yend_sepsis = yend_sorted[sepsis]
#y_sepsis.sort(axis = 0)
sns.heatmap(yend_sepsis[:, -16:])
plt.title("Sepsis Subpatients - Sorted Cluster Transitions")

#%%

plt.figure(figsize = (25,15))

parameters = {'axes.labelsize': 10,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)
nonsepsis = np.where(sepsislabel_sorted == 0)[0]
yend_nonsepsis = yend_sorted[nonsepsis]
#y_sepsis.sort(axis = 0)
sns.heatmap(yend_nonsepsis[:, -16:])
plt.title("Non-sepsis Subpatients - Sorted Cluster Transitions")

#%%

isTransition = np.array([len(c) for c in trajectory.counter])
isTransition_sorted = isTransition[idx]

#%%
plt.figure(figsize = (25,15))

parameters = {'axes.labelsize': 10,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)
sepsis = np.where(np.bitwise_and(sepsislabel_sorted == 1, isTransition_sorted > 2))[0]

yend_sepsis = yend_sorted[sepsis]
#y_sepsis.sort(axis = 0)
sns.heatmap(yend_sepsis[:, -16:])
plt.title("Sepsis Subpatients - Sorted Cluster Transitions")

#%%

plt.figure(figsize = (25,15))

parameters = {'axes.labelsize': 10,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)
nonsepsis = np.where(np.bitwise_and(sepsislabel_sorted == 0, isTransition_sorted == 1))[0]

yend_nonsepsis = yend_sorted[nonsepsis]
#y_sepsis.sort(axis = 0)
sns.heatmap(yend_nonsepsis[:, -16:])
plt.title("Non Sepsis Subpatients - Sorted Cluster Transitions")

#%%

import lifelines
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

km_data = trajectory 
#kmf.fit(km_data['time_to_sepsis'], km_data['Sepsis'], label='control')
#ax = kmf.plot_survival_function()
plt.figure(figsize = (25,15))

parameters = {'axes.labelsize': 25,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)

for cluster_num in range(12):
    index = (trajectory.last_cluster_before_sepsis == cluster_num).values
    kmf.fit(km_data[index]['time_to_sepsis_subpat'], km_data[index]['Sepsis'], label='Cluster_' + str(cluster_num))
    ax = kmf.plot_survival_function(ci_show = False)
ax.set_title('Cluster association: Last Cluster Before Sepsis')
plt.xlabel('time_from_icu_start_to_sepsis')
plt.ylabel('probability of developing sepsis')
#plt.savefig('Kaplan_meier_25%.pdf', bbox_inches = 'tight', dpi = 1000)
#kmf.survival_function_
#kmf.cumulative_density_
#kmf.plot_survival_function()

#%%

km_data = trajectory 
#kmf.fit(km_data['time_to_sepsis'], km_data['Sepsis'], label='control')
#ax = kmf.plot_survival_function()
plt.figure(figsize = (25,15))

parameters = {'axes.labelsize': 25,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)

for transition in range[]:
    index = (trajectory.last_cluster_before_sepsis == cluster_num).values
    kmf.fit(km_data[index]['time_to_sepsis_subpat'], km_data[index]['Sepsis'], label='Cluster_' + str(cluster_num))
    ax = kmf.plot_survival_function(ci_show = False)
ax.set_title('Cluster association: Last cluster before Sepsis')


#%%


discharge_status = pd.read_pickle('Grady/discharge_status.pkl')
discharge_status['patid'] = discharge_status['patid'].astype(np.float64)
death = discharge_status['discharge_status'].isin(['Expired', 'Expired in Medical Facility'])

#%%

trajectory_new = pd.merge(trajectory, discharge_status, left_on = 'patid', right_on = 'patid', 
                          how = 'inner') 

trajectory_new['time_to_death'] = trajectory_new.total_icu_time_subpat - trajectory_new.time_to_sepsis_subpat
trajectory_new['death'] = np.zeros(len(trajectory_new))

dead_ind = trajectory_new.loc[trajectory_new['discharge_status'].isin(['Expired', 'Expired in Medical Facility'])].loc[trajectory_new['Sepsis'] == 1].index
trajectory_new['death'].loc[dead_ind] = 1
trajectory_new['time_to_death'].loc[trajectory_new['death'] != 1] = -1
trajectory_new.to_pickle('trajectories_Grady_icu_discharge.pkl')

#%%


km_data = trajectory_new.loc[trajectory_new['Sepsis'] == 1]
#kmf.fit(km_data['time_to_sepsis'], km_data['Sepsis'], label='control')
#ax = kmf.plot_survival_function()
plt.figure(figsize = (25,15))

parameters = {'axes.labelsize': 25,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)

for cluster_num in range(12):
    index = (km_data.max_time_cluster == cluster_num).values
    kmf.fit(km_data[index]['time_to_death'], km_data[index]['death'], label='Cluster_' + str(cluster_num))
    ax = kmf.plot_survival_function(ci_show = False)
ax.set_title('Cluster association: Last Cluster Before Sepsis')
plt.xlabel('time_from_sepsis')
plt.ylabel('probability of death')

#%%%

""" Analysis of Transitions """

transition_matrix_sepsis = np.load('Physionet_all_data/sepsis_kmeans12_all_data.npy')
transition_matrix_control = np.load('Physionet_all_data/control_kmeans12_all_data.npy')
trajectory = pd.read_pickle('Physionet_all_data/trajectories.pkl')
clusterdf = pd.read_pickle('Physionet_all_data/clusterDf.pkl')

#%%

num_transitions = trajectory.states_transition.apply(len)
no_transition_idx = trajectory.loc[num_transitions == 1].index

transition_trajectories = trajectory.loc[num_transitions > 1]
stationary_trajectories = trajectory.loc[num_transitions == 1]

#%%

stationary_sepsis_probabilities = pd.DataFrame(columns = ['cluster', 'num_sepsis', 'total', 'probability'])

for c in range(12):
    dfc = stationary_trajectories.loc[stationary_trajectories.max_time_cluster == c]
    total = len(dfc)
    num_sepsis = sum(dfc['Sepsis'])
    stationary_sepsis_probabilities.loc[len(stationary_sepsis_probabilities)] = [c, num_sepsis, total, num_sepsis/total]
    
print("Probability of sepsis|stationary_trajectory = {}".format(sum(stationary_trajectories['Sepsis'])/len(stationary_trajectories)))

stationary_sepsis_probabilities.to_csv('Physionet_all_data/Stationary_sepsis_probabilities.csv')

#%%

transitions_24hrs = clusterdf[['cluster_transitions-{}'.format(i) for i in range(6)]].loc[num_transitions > 1]



#%%


def get_transition_matrices(t):
    matrix_kmeans12 = np.zeros((clusters, clusters))
    matrix_kmeans12_control = np.zeros((clusters, clusters))
    index = dfPatID.index
    for i in range(len(dfy)-1):
        
        if(dfPatID[index[i]] == dfPatID[index[i+1]]):
            if(sepsisp[dfPatID[index[i]]] == 1):
                matrix_kmeans12[clustering_labels_kmeans12[i], clustering_labels_kmeans12[i+1]] += 1
            else:
                matrix_kmeans12_control[clustering_labels_kmeans12[i], clustering_labels_kmeans12[i+1]] += 1
                
                
clusters = 12

x = trajectory.clusterSeq.values
length = max(map(len, x))   
num_pats = len(trajectory)
hours_before = 24
prior = int(hours_before/3) - 1
labels_truncated = np.zeros((prior*num_pats, ))
corr_idx = np.zeros((prior*num_pats, ))
corr_pats = np.zeros((prior*num_pats, ))
corr_sepsis = np.zeros((prior*num_pats, ))

matrix_truncated_sepsis = np.zeros((clusters, clusters))
matrix_truncated_control = np.zeros((clusters, clusters))
num_sepsis = 0
num_control = 0
## y and yend are of size length with None values in the end and in the beginning respectively
for idx, xi in enumerate(x):
    time_sepsis = trajectory.loc[trajectory.index[idx]]['time_to_sepsis_subpat']
    last_cluster_before_sepsis = trajectory.loc[trajectory.index[idx]]['last_cluster_before_sepsis']
    patid = trajectory.loc[trajectory.index[idx]]['patid']
    if last_cluster_before_sepsis == -1 and len(xi) >= prior:
        if len(xi) == prior:
            rind = prior
        else:
            rind = np.random.randint(prior, len(xi))
        seq = xi[rind-prior:rind]
        labels_truncated[idx*prior: (idx+1)*prior] = seq 
        corr_idx[idx*prior: (idx+1)*prior] = np.arange(rind-prior, rind)
        corr_pats[idx*prior: (idx+1)*prior] = [int(patid[1:])]*prior
        corr_sepsis[idx*prior: (idx+1)*prior] = [0]*prior
        num_control += 1
        for j in range(len(seq)-1):
            matrix_truncated_control[seq[j], seq[j+1]] += 1
    elif time_sepsis < prior:
        seq = [-1]*prior
        labels_truncated[idx*prior: (idx+1)*prior] = [-1]*prior
        corr_idx[idx*prior: (idx+1)*prior] = [-1]*prior
        corr_pats[idx*prior: (idx+1)*prior] = [int(patid[1:])]*prior
        corr_sepsis[idx*prior: (idx+1)*prior] = [1]*prior
    elif last_cluster_before_sepsis == -1 and len(xi) < prior:
        seq = [-1]*prior
        labels_truncated[idx*prior: (idx+1)*prior] = [-1]*prior
        corr_idx[idx*prior: (idx+1)*prior] = [-1]*prior
        corr_pats[idx*prior: (idx+1)*prior] = [int(patid[1:])]*prior
        corr_sepsis[idx*prior: (idx+1)*prior] = [0]*prior
    else:
        seq = xi[time_sepsis-prior+1:time_sepsis+1]
        labels_truncated[idx*prior: (idx+1)*prior] = seq 
        corr_idx[idx*prior: (idx+1)*prior] = np.arange(time_sepsis-prior+1, time_sepsis+1)
        corr_pats[idx*prior: (idx+1)*prior] = [int(patid[1:])]*prior
        corr_sepsis[idx*prior: (idx+1)*prior] = [1]*prior
        num_sepsis += 1
        for j in range(len(seq)-1):
            matrix_truncated_sepsis[seq[j], seq[j+1]] += 1
    print(len(xi), time_sepsis)
    trajectory.loc[trajectory.index[idx]]['24_hrs_before_labels'] = [labels_truncated]
    
#%%

clust_func.chord_diagram(matrix_truncated_sepsis, 'Physionet_all_data/physionet_sepsis24.html')
clust_func.chord_diagram(matrix_truncated_control, 'Physionet_all_data/physionet_control24.html')

#%%

np.save('labels_truncated24.npy', labels_truncated)
np.save('num_sepsis24.npy', num_sepsis)
np.save('num_control24.npy', num_control)
np.save('transition_matrix_control24.npy', matrix_truncated_control)
np.save('transition_matrix_sepsis24.npy', matrix_truncated_sepsis)
np.save('corr_sepsis24.npy', corr_sepsis)
#%%

labels_truncated = np.load('labels_truncated24.npy')



#%%

clusters = 12

x = trajectory.clusterSeq.values
length = max(map(len, x))   
num_pats = len(trajectory)
hours_before = 18
prior = int(hours_before/3) - 1
labels_truncated18 = np.zeros((prior*num_pats, ))
corr_idx18 = np.zeros((prior*num_pats, ))
corr_pats18 = np.zeros((prior*num_pats, ))
corr_sepsis18 = np.zeros((prior*num_pats, ))

matrix_truncated_sepsis18 = np.zeros((clusters, clusters))
matrix_truncated_control18 = np.zeros((clusters, clusters))
num_sepsis18 = 0
num_control18 = 0

for idx, xi in enumerate(x):
    seq = np.int32(labels_truncated[idx*7: ((idx)*7) + prior])
    labels_truncated18[idx*prior: (idx+1)*prior] = seq
    
    if seq[0] != -1:
        if corr_sepsis[idx*7] == 1:
            num_sepsis18 += 1
            for j in range(len(seq)-1):
                matrix_truncated_sepsis18[seq[j], seq[j+1]] += 1
        elif corr_sepsis[idx*7] == 0:
            num_control18 += 1
            for j in range(len(seq)-1):
                matrix_truncated_control18[seq[j], seq[j+1]] += 1
        
np.save('labels_truncated18.npy', labels_truncated18)
np.save('num_sepsis18.npy', num_sepsis18)
np.save('num_control18.npy', num_control18)
np.save('transition_matrix_control18.npy', matrix_truncated_control18)
np.save('transition_matrix_sepsis18.npy', matrix_truncated_sepsis18)

clust_func.chord_diagram(matrix_truncated_sepsis18, 'Physionet_all_data/physionet_sepsis18.html')
clust_func.chord_diagram(matrix_truncated_control18, 'Physionet_all_data/physionet_control18.html')

#%%

clusters = 12

x = trajectory.clusterSeq.values
length = max(map(len, x))   
num_pats = len(trajectory)
hours_before = 12
prior = int(hours_before/3) - 1
labels_truncated12 = np.zeros((prior*num_pats, ))
corr_idx12 = np.zeros((prior*num_pats, ))
corr_pats12 = np.zeros((prior*num_pats, ))
corr_sepsis12 = np.zeros((prior*num_pats, ))

matrix_truncated_sepsis12 = np.zeros((clusters, clusters))
matrix_truncated_control12 = np.zeros((clusters, clusters))
num_sepsis12 = 0
num_control12 = 0

for idx, xi in enumerate(x):
    seq = np.int32(labels_truncated[idx*7: ((idx)*7) + prior])
    labels_truncated12[idx*prior: (idx+1)*prior] = seq
    
    if seq[0] != -1:
        if corr_sepsis[idx*7] == 1:
            num_sepsis12 += 1
            for j in range(len(seq)-1):
                matrix_truncated_sepsis12[seq[j], seq[j+1]] += 1
        elif corr_sepsis[idx*7] == 0:
            num_control12 += 1
            for j in range(len(seq)-1):
                matrix_truncated_control12[seq[j], seq[j+1]] += 1
        
np.save('labels_truncated12.npy', labels_truncated12)
np.save('num_sepsis12.npy', num_sepsis12)
np.save('num_control12.npy', num_control12)
np.save('transition_matrix_control12.npy', matrix_truncated_control12)
np.save('transition_matrix_sepsis12.npy', matrix_truncated_sepsis12)

clust_func.chord_diagram(matrix_truncated_sepsis12, 'Physionet_all_data/physionet_sepsis12.html')
clust_func.chord_diagram(matrix_truncated_control12, 'Physionet_all_data/physionet_control12.html')

#%%

clusters = 12

x = trajectory.clusterSeq.values
length = max(map(len, x))   
num_pats = len(trajectory)
hours_before = 6
prior = int(hours_before/3) - 1
labels_truncated6 = np.zeros((prior*num_pats, ))
corr_idx6 = np.zeros((prior*num_pats, ))
corr_pats6 = np.zeros((prior*num_pats, ))
corr_sepsis6 = np.zeros((prior*num_pats, ))

matrix_truncated_sepsis6 = np.zeros((clusters, clusters))
matrix_truncated_control6 = np.zeros((clusters, clusters))
num_sepsis6 = 0
num_control6 = 0

for idx, xi in enumerate(x):
    seq = np.int32(labels_truncated[idx*7: ((idx)*7) + prior])
    labels_truncated6[idx*prior: (idx+1)*prior] = seq
    
    if seq[0] != -1:
        if corr_sepsis[idx*7] == 1:
            num_sepsis6 += 1
            for j in range(len(seq)-1):
                matrix_truncated_sepsis6[seq[j], seq[j+1]] += 1
        elif corr_sepsis[idx*7] == 0:
            num_control6 += 1
            for j in range(len(seq)-1):
                matrix_truncated_control6[seq[j], seq[j+1]] += 1
        
np.save('labels_truncated6.npy', labels_truncated6)
np.save('num_sepsis6.npy', num_sepsis6)
np.save('num_control6.npy', num_control6)
np.save('transition_matrix_control6.npy', matrix_truncated_control6)
np.save('transition_matrix_sepsis6.npy', matrix_truncated_sepsis6)

#%%
clust_func.chord_diagram(matrix_truncated_sepsis6, 'Physionet_all_data/physionet_sepsis6.html')
clust_func.chord_diagram(matrix_truncated_control6, 'Physionet_all_data/physionet_control6.html')

#%%

from itertools import groupby 

plt.figure(figsize = (15,15))

trunc_seq = np.empty((7,0))
trunc_sepsis = []
transitions = []
for i in range(int(len(labels_truncated)/7)):
    seq = labels_truncated[i*7:(i*7)+7]
    if seq[0] != -1:
        trunc_seq = np.concatenate((trunc_seq, seq.reshape(-1,1)), axis = 1)
        if corr_sepsis[i*7] == 1:
            trunc_sepsis.append(1)
            plt.plot(seq, 'red')
        else:
            trunc_sepsis.append(0)
            plt.plot(seq, 'k')
        
        
plt.title('Spaghetti plot for 12 hrs prior to sepsis. Red- sepsis')
plt.legend(['Sepsis', 'control'])
plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11])
plt.show()
            

#%%

transition_matrix_sepsis = np.load('transition_matrix_sepsis24.npy')
transition_matrix_control = np.load('transition_matrix_control24.npy')

transition_matrix_sepsis_diag = np.eye(12)*transition_matrix_sepsis
transition_matrix_control_diag = np.eye(12)*transition_matrix_control

diff_state_transitions_sepsis = transition_matrix_sepsis - transition_matrix_sepsis_diag
diff_state_transitions_control = transition_matrix_control - transition_matrix_control_diag

total_diff_state_transitions = diff_state_transitions_sepsis.sum() + diff_state_transitions_control.sum()
print(total_diff_state_transitions)

sepsis_mat = diff_state_transitions_sepsis/diff_state_transitions_sepsis.sum()
control_mat = diff_state_transitions_control/diff_state_transitions_control.sum()

fig, a = plt.subplots(1,2,figsize = (25, 8))
im1 = a[1].matshow(sepsis_mat/sepsis_mat.max())
im2 = a[0].matshow(control_mat/control_mat.max())
a[1].set_xlabel('Cluster to', fontsize = 20)
a[1].set_ylabel('Cluster from', fontsize = 20)
a[1].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
a[1].set_yticks([0,1,2,3,4,5,6,7,8,9,10,11])
a[1].set_title('Ratio of Cluster Transitions for sepsis patients(24 hrs)', fontsize = 20)

a[0].set_xlabel('Cluster to', fontsize = 20)
a[0].set_ylabel('Cluster from', fontsize = 20)
a[0].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
a[0].set_yticks([0,1,2,3,4,5,6,7,8,9,10,11])
a[0].set_title('Ratio of Cluster Transitions Matrix for control patients (24 hrs)', fontsize = 20)

fig.colorbar(im1, ax = a[1])
fig.colorbar(im2, ax = a[0])



#%%
plt.figure(figsize = (15,10))
sns.heatmap(diff_state_transitions_control/diff_state_transitions_control.sum())
plt.xlabel('Cluster to', fontsize = 20)
plt.ylabel('Cluster from', fontsize = 20)
plt.title('Cluster Transition Matrix for control patients', fontsize = 20)













