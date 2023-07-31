# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 00:15:14 2022

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


discharge_status = pd.DataFrame(columns = ['patid', 'discharge_status'])
for pat_ in unique_patid_str:
    if pat_[0] == '3':
        path  = path_to_files + '20'
        pat = '10' + pat_
    else:
        path = path_to_files + '1' + pat_[0]
        pat = '1' + pat_[1:]
    file = pd.read_pickle(os.path.join(path, pat + '.pickle' ))
    discharge_status.loc[len(discharge_status.index)] = [pat_, file['static_features']['discharge_status']]
    

discharge_status.to_pickle('/labs/kamaleswaranlab/MODS/SepsisProjectionProject/Clustering/Grady/discharge_status.pkl')