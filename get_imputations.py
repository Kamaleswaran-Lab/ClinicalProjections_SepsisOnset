# -*- coding: utf-8 -*-
"""
This is the entry point to the sepsis projections pipeline

"""

#!/usr/bin/env python3
import os
#os.dir('..\..\..\KamalLab\ai_sepsis-master')
import pipe.sepsis.sepsis as rd
import pipe.sepsis.settings as s

#%%

# TODO override parts of the settings array with command-line arguments

IMPORT_DIRS = ["input_data/training_setA", "input_data/training_setB"]
OUTPUT_DIR = "Imputed_12_9"

#%%

if __name__ == '__main__':
    df = rd.import_all(s.settings, IMPORT_DIRS, multicore=True)
    rd.write_summary_data(df, s.settings, OUTPUT_DIR)
