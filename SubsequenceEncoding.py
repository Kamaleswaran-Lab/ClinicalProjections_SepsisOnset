# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:00:09 2023

@author: mehak
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import date
#import clustering_functions as clust_func
#import utils
import seaborn as sns

import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer,SilhouetteVisualizer

from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler

import collections
import lifelines
from lifelines import KaplanMeierFitter


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


def subsequence_matching_vector(trajectory, start, end, num_clusters, k):
    """
    Parameters
    ----------
    trajectory : TYPE - list of cluster transitions
        DESCRIPTION.
    start : TYPE - int
        DESCRIPTION. Index of the trajectory sequence to start subsequence frequency analysis
    end : TYPE - int
        DESCRIPTION. Index of the trajectory sequence to end subsequence frequency analysis
    num_clusters : TYPE - int
        DESCRIPTION. Number of clusters any given trajectory moves through
    k : TYPE - int
        DESCRIPTION. Length of the subsequence for matching

    Returns
    -------
    num_cluster^k dimensional vector with frequency of occurence of every subsequence

    """
    
    output_vector = np.zeros(num_clusters**k)
    if start == -1:
        start = 0
    
    if end == -1:
        end = len(trajectory)
    elif (end-start) > len(trajectory):
        end = len(trajectory)
    else:
        end = end + start
    trajectory_slice = trajectory[start:end+1]

    assert len(trajectory_slice)>=k, "Length of trajectory slice is lesser than subsequence length"
    
    
        
    for i in range(len(trajectory_slice)-k+1):
        subseq = list(trajectory_slice[i:i+k])
        index = 0
        j = 0
        while(len(subseq)>0):
            index += (num_clusters**j)*subseq.pop()
            j += 1
        output_vector[index] += 1
    
    return output_vector


def save_obj(object_var, folder, filename):
    
    if(filename[-3:] == 'pkl'):
        with open(folder + filename,'wb') as f:
            pickle.dump(object_var,f)
    else:
        with open(folder + filename + '.pkl','wb') as f:
            pickle.dump(object_var,f)
    
        
def load_obj(folder, filename):
    
    if(filename[-3:] == 'pkl'):
        with open(folder + filename,'rb') as f:
            object_var = pickle.load(f)
    else:
        with open(folder + filename + '.pkl','rb') as f:
            object_var = pickle.load(f)
        
    return object_var

#%%

kmeans = load_obj('./', 'feat_kmeans12_median.pkl')
trajectory_grady = pd.read_pickle('./Grady/trajectories_Grady_icu_discharge.pkl')
trajectory_grady_short_stay = trajectory_grady.loc[trajectory_grady.time_to_death < 24*21]

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

trajectories_emory_sepsis = trajectories_emory.loc[trajectories_emory.Sepsis == 1] 
trajectories_emory_sepsis_long = trajectories_emory_sepsis.loc[trajectories_emory_sepsis.total_icu_time_subpat > 3] 


trajectories_emory_sepsis_long['SubseqFreq_72hrspostsepsis'] = trajectories_emory_sepsis_long.apply(lambda x: subsequence_matching_vector(x.clusterSeq, x.time_to_sepsis_subpat, 72*3, 12, 3 ))

#%%

Xs = np.stack(trajectories_emory_sepsis_long['SubseqFreq_entireLOS'].values)

print(Xs.shape)

#%%
km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2,13))
 
visualizer.fit(Xs)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


#%%

from tqdm import tqdm
fig, ax = plt.subplots(4, 2, figsize=(15,8))
for i in tqdm([2,3,4,5,6,7,8,9]):
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(Xs) 


#%%

n_super_clusters = 6
km = KMeans(n_clusters = n_super_clusters, init='k-means++', n_init=10, max_iter=100, random_state=77)
km.fit(Xs)

#%%

clusters = km.predict(Xs)

trajectories_emory_sepsis_long['super_clusters'] = clusters

#%%


kmf = KaplanMeierFitter()
plt.style.use("ggplot")

km_data = trajectories_emory_sepsis_long #.loc[trajectories_emory_sepsis_long.total_icu_time_subpat*3 < 102]


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

for cluster_num in range(n_super_clusters):
    index = (km_data.super_clusters == cluster_num).values
    kmf.fit(km_data[index]['time_to_death_subpat']/24, km_data[index]['death'], label='Cluster ' + str(cluster_num))
    ax = kmf.plot_survival_function(ci_show = False, color = colours[cluster_num], linewidth=2)
ax.set_title('Kaplan Meier Curves survival curves: Entire LOS')
ax.set_xticks([0,7,14,21, 28, 35, 42])
ax.set_xlim([0, 50])
plt.xlabel('Time (Days) from ICU admission')
plt.ylabel('Ratio of survivers')



#%%


print(len(clusters[clusters == 0]))
print(len(clusters[clusters == 1]))
print(len(clusters[clusters == 2]))
print(len(clusters[clusters == 3]))
print(len(clusters[clusters == 4]))
print(len(clusters[clusters == 5]))



