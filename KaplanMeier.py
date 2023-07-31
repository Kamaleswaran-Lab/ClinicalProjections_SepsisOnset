# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:00:39 2023

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

#%%

trajectory_grady = pd.read_pickle('./Grady/trajectories_Grady_icu_discharge.pkl')
trajectory_grady_short_stay = trajectory_grady.loc[trajectory_grady.time_to_death < 24*21]

#%%

import lifelines
from lifelines import KaplanMeierFitter

import matplotlib.colors as mcolors

kmf = KaplanMeierFitter()
plt.style.use("ggplot")


km_data = trajectory_grady_short_stay.loc[trajectory_grady_short_stay.time_to_sepsis_subpat > 0]
#kmf.fit(km_data['time_to_sepsis'], km_data['Sepsis'], label='control')
#ax = kmf.plot_survival_function()
plt.figure(figsize = (20,10))

colours = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'yellow', 'magenta']

parameters = {'axes.labelsize': 25,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)

for cluster_num in range(12):
    index = (km_data.last_cluster_before_sepsis == cluster_num).values
    kmf.fit(km_data[index]['time_to_sepsis_subpat'], km_data[index]['Sepsis'], label='Cluster ' + str(cluster_num))
    ax = kmf.plot_survival_function(ci_show = False, color = colours[cluster_num], linewidth=2)
ax.set_title('Kaplan Meier Curves indicating sepsis onset times')
plt.xlabel('Time (hours) from ICU admission')
plt.ylabel('Ratio of patients who developed sepsis')
#plt.savefig('KP_Grady_sepsis.pdf', bbox_inches = "tight")

#%%

#km_data = trajectory_new.loc[trajectory_new['Sepsis'] == 1]
#kmf.fit(km_data['time_to_sepsis'], km_data['Sepsis'], label='control')


km_data = trajectory_grady_short_stay.loc[trajectory_grady_short_stay['Sepsis'] == 1]
#kmf.fit(km_data['time_to_sepsis'], km_data['Sepsis'], label='control')
#ax = kmf.plot_survival_function()
plt.figure(figsize = (20,10))

parameters = {'axes.labelsize': 25,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)

for cluster_num in range(12):
    index = (km_data.last_cluster_before_sepsis == cluster_num).values
    kmf.fit(km_data[index]['time_to_death']/24, km_data[index]['death'], label='Cluster ' + str(cluster_num))
    ax = kmf.plot_survival_function(ci_show = False, color = colours[cluster_num], linewidth=2)
ax.set_title('Kaplan Meier Curves Indicating Mortality')
ax.set_xticks([0,7,14,21])
plt.xlabel('Time (days) from Sepsis Onset')
plt.ylabel('Survival Probability')
#plt.savefig('KP_Grady_death_days.pdf', bbox_inches = "tight")



#%%

def trajectories(patID, clustering_labels_kmeans12, sepsisLabel):
    import collections
    pat_analyse = pd.DataFrame(columns = ['patid', 'clusterSeq', 'states_transition', 'counter', 
                                          'time_to_sepsis_subpat', 'total_icu_time_subpat',
                                          'last_cluster_before_sepsis', 'max_time_cluster'])
    
    temp_df = pd.DataFrame(columns = ['patid', 'labels', 'sepsislabels'])
    temp_df['patid'] = patID
    temp_df['labels'] = clustering_labels_kmeans12
    temp_df['sepsislabels'] = sepsisLabel
    temp_df_pats = [x for _, x in temp_df.groupby(temp_df['patid'])]
    
    for df in temp_df_pats:
        
        label_list = df['labels'].values
        counter = collections.Counter(label_list)
        pat_id = df['patid'].values[0]
        print(pat_id)
        
        total_icu_time = len(label_list)
        sepsis_time = np.where(df['sepsislabels'].values == 1)[0]
        if len(sepsis_time) != 0:
            time_to_sepsis = sepsis_time[0]
            last_cluster_before_sepsis = label_list[time_to_sepsis]
        else:
            time_to_sepsis = total_icu_time 
            last_cluster_before_sepsis = -1
            
        full_label_list = np.copy(label_list)
        l = len(label_list)
        i = 1
        while(i<l):
            if label_list[i] == label_list[i-1]:
                label_list = np.delete(label_list, i)
                l = l - 1
            else:
                i = i + 1
                
            
        pat_analyse.loc[len(pat_analyse)] = [pat_id, full_label_list, label_list, counter,
                                             time_to_sepsis, total_icu_time,
                                             last_cluster_before_sepsis, max(counter)]
    
    
    greater_than_25_clusters = pat_analyse['counter'].apply(lambda c: np.array(list(c))[np.where(list(c.values()) > sum(c)*0)[0]])
    pat_analyse['greater_than_25_perc'] = greater_than_25_clusters
    
    
    pats = pd.DataFrame(columns = ['SepsisLabel', 'patid'])
    pats['SepsisLabel'] = sepsisLabel 
    pats['patid'] = patID
    p = pats.groupby('patid').max()
    zip_iterator = zip(p.index, p['SepsisLabel'].values)

    sepsisp = dict(zip_iterator)
    pat_analyse['Sepsis'] = sepsisp.values()
    
    return pat_analyse

#%%

#%%
import collections

def get_sepsis_cluster(cluster_seqs, sepsis_subpats):
    clusters = []
    for i in range(len(cluster_seqs)):
        try:
            if sepsis_subpats[i] == -1:
                clusters.append(-1)
            elif sepsis_subpats[i] == 0:
                clusters.append(cluster_seqs[i][0])
            else:
                clusters.append(cluster_seqs[i][sepsis_subpats[i-1]])
        except:
            clusters.append(cluster_seqs[i][0])
            
    return np.array(clusters)

#%%


def create_trajectory_df_emory(emory_data):
    pat_analyse = pd.DataFrame(columns = ['patid', 'clusterSeq', 'states_transition', 'counter', 
                                          'time_to_sepsis_subpat', 'total_icu_time_subpat',
                                          'last_cluster_before_sepsis', 'max_time_cluster'])
    
    pat_analyse['patid'] = emory_data.patid 
    pat_analyse['clusterSeq'] = emory_data.cluster_seq 
    pat_analyse['states_transition'] = emory_data.cluster_seq.apply(np.unique)
    pat_analyse['counter'] = emory_data.cluster_seq.apply(lambda x: collections.Counter(x) )
    pat_analyse['time_to_sepsis_subpat'] = emory_data.sepsis_subpat
    pat_analyse['total_icu_time_subpat'] = emory_data.icu_len_subpat
    pat_analyse['last_cluster_before_sepsis'] = get_sepsis_cluster(emory_data.cluster_seq.values, emory_data.sepsis_subpat.values)
    pat_analyse['max_time_cluster'] = pat_analyse['counter'].apply(max)
    pat_analyse['Sepsis'] = np.int32(pat_analyse['time_to_sepsis_subpat'] != -1) 
    pat_analyse['death'] = np.int32(emory_data.dischage_status == 'EXPIRED')
    pat_analyse['time_to_death_subpat'] = emory_data.discharge_time_diff + emory_data.icu_len_subpat

    return pat_analyse

#%%
emory_data1 = pd.read_pickle('./Emory/icu_pats_subpats_corrected_2016.pkl')
trajectories1 = create_trajectory_df_emory(emory_data1)

emory_data2 = pd.read_pickle('./Emory/icu_pats_subpats_corrected_2017.pkl')
trajectories2 = create_trajectory_df_emory(emory_data2)

emory_data3 = pd.read_pickle('./Emory/icu_pats_subpats_corrected_2018.pkl')
trajectories3 = create_trajectory_df_emory(emory_data3)

emory_data4 = pd.read_pickle('./Emory/icu_pats_subpats_corrected_2019.pkl')
trajectories4 = create_trajectory_df_emory(emory_data4)

trajectories_emory = pd.concat((trajectories1, trajectories2, trajectories3, trajectories4))


#%%

kmf = KaplanMeierFitter()
plt.style.use("ggplot")


km_data = trajectories_emory.loc[(trajectories_emory.time_to_sepsis_subpat > 0) & (trajectories_emory.time_to_sepsis_subpat < 200)]
#kmf.fit(km_data['time_to_sepsis'], km_data['Sepsis'], label='control')
#ax = kmf.plot_survival_function()
plt.figure(figsize = (20,10))

colours = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'yellow', 'magenta']

parameters = {'axes.labelsize': 25,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 25,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)

for cluster_num in range(12):
    index = (km_data.last_cluster_before_sepsis == cluster_num).values
    kmf.fit(km_data[index]['time_to_sepsis_subpat'], km_data[index]['Sepsis'], label='Cluster ' + str(cluster_num))
    ax = kmf.plot_survival_function(ci_show = False, color = colours[cluster_num], linewidth=2)
ax.set_title('Kaplan Meier Curves indicating sepsis onset times')
plt.xlabel('Time (hours) from ICU admission')
plt.ylabel('Ratio of patients who developed sepsis')
#plt.savefig('KP_Emory_sepsis.pdf', bbox_inches = "tight")

#%%

#km_data = trajectory_new.loc[trajectory_new['Sepsis'] == 1]
#kmf.fit(km_data['time_to_sepsis'], km_data['Sepsis'], label='control')


km_data = trajectories_emory.loc[(trajectories_emory['Sepsis'] == 1) & (trajectories_emory['time_to_death_subpat']*3 < 21*24)]
#kmf.fit(km_data['time_to_sepsis'], km_data['Sepsis'], label='control')
#ax = kmf.plot_survival_function()
plt.figure(figsize = (20,10))

parameters = {'axes.labelsize': 25,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 20,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)

for cluster_num in range(12):
    index = (km_data.last_cluster_before_sepsis == cluster_num).values
    kmf.fit(km_data[index]['time_to_death_subpat']*3/24, km_data[index]['death'], label='Cluster ' + str(cluster_num))
    ax = kmf.plot_survival_function(ci_show = False, color = colours[cluster_num], linewidth=2)
ax.set_title('Kaplan Meier Curves Indicating Mortality')
ax.set_xticks([0,7,14,21])
plt.xlabel('Time (days) from Sepsis Onset')
plt.ylabel('Survival Probability')
#plt.savefig('KP_Emory_death_days.pdf', bbox_inches = "tight")

#%%

markov_model_clustering = pd.read_pickle('markov_model_clustering_emory.pickle')
markov_model_clustering.patid = markov_model_clustering.patid.astype('int64')
trajectories_emory.patid = trajectories_emory.patid.astype('int64')
sepsis_trajectories_emory_markov = markov_model_clustering.join(trajectories_emory.loc[trajectories_emory.Sepsis == 1], on = 'patid', how = 'left', lsuffix = 'markov_df', rsuffix = 'r')

#%%

trajectories_emory_markov = trajectories_emory.loc[trajectories_emory.patid.isin(markov_model_clustering.patid)]
trajectories_emory_markov['cluster'] = [float("nan")]*len(trajectories_emory_markov)

for i in range(len(trajectories_emory_markov)):
    patid = trajectories_emory_markov.iloc[i]['patid']
    trajectories_emory_markov.loc[trajectories_emory_markov.index == trajectories_emory_markov.index[i], 'cluster'] = markov_model_clustering.loc[markov_model_clustering.patid == patid]['cluster'].values[0]


#%%


km_data = trajectories_emory_markov.loc[trajectories_emory_markov['time_to_death_subpat']*3 < 21*24] #.loc[(trajectories_emory['Sepsis'] == 1) & (trajectories_emory['time_to_death_subpat']*3 < 21*24)]
#kmf.fit(km_data['time_to_sepsis'], km_data['Sepsis'], label='control')
#ax = kmf.plot_survival_function()
plt.figure(figsize = (20,10))

parameters = {'axes.labelsize': 25,
              'axes.titlesize': 35, 
              'figure.titlesize': 35,
              'legend.fontsize': 20,
              'legend.title_fontsize' : 25,
              'xtick.labelsize' : 20,
              'ytick.labelsize': 20}
plt.rcParams.update(parameters)

for cluster_num in range(5):
    index = (km_data.cluster == cluster_num).values
    kmf.fit(km_data[index]['time_to_death_subpat']*3/24, km_data[index]['death'], label='Cluster ' + str(cluster_num + 1))
    ax = kmf.plot_survival_function(ci_show = False, color = colours[cluster_num], linewidth=2)
ax.set_title('Kaplan Meier Curves Indicating Mortality')
ax.set_xticks([0,7,14,21])
plt.xlabel('Time (days) from Sepsis Onset')
plt.ylabel('Survival Probability')
#plt.savefig('KP_Emory_death_days.pdf', bbox_inches = "tight")











