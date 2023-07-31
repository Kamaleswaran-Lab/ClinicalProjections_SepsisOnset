# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 10:53:24 2022

@author: mehak
"""

import pandas as pd
import numpy as np
import os



data_path = '/labs/kamaleswaranlab/MODS/Encounter_Pickles/gr/PSV_FILES/'
patid_all_years = pd.read_pickle(os.path.join(data_path, 'patid_all_years_.pkl'))
patid = patid_all_years.astype(int)
unique_patid_str = np.unique(patid.astype(str).values)
unique_patid_counts = patid.astype(str).value_counts()
path_to_files = '/opt/bmi-585r/KLAB_SAIL/Grady_Data/Pickles/Encounter_Pickles/gr/20'

icu_time = pd.DataFrame(columns = ['patid', 'icu_start', 'icu_end', 'supertable_start'])
exception_pats = []
for pat_ in unique_patid_str:
    
    try:
        if pat_[0] == '3':
            path  = path_to_files + '20'
            pat = '10' + pat_
        else:
            path = path_to_files + '1' + pat_[0]
            pat = '1' + pat_[1:]
      
        file = pd.read_pickle(os.path.join(path, pat + '.pickle' ))
        if 'first_icu_start' in file['event_times'].keys():
            icu_time.loc[len(icu_time.index)] = [pat_, file['event_times']['first_icu_start'], file['event_times']['first_icu_end'], file['event_times']['start_index']]
        elif 'first_icu' in file['event_times'].keys():
            icu_time.loc[len(icu_time.index)] = [pat_, file['event_times']['first_icu'], None, file['event_times']['start_index']]
        else:
            icu_time.loc[len(icu_time.index)] = [pat_, None, None, file['event_times']['start_index']]


    except:
        print(pat_)
        exception_pats.append(pat_)
        
    
exception_pats = np.array(exception_pats)
np.save('/labs/kamaleswaranlab/MODS/SepsisProjectionProject/Clustering/Grady/exception_pats.npy', exception_pats)
icu_time.to_pickle('/labs/kamaleswaranlab/MODS/SepsisProjectionProject/Clustering/Grady/icu_times.pkl')