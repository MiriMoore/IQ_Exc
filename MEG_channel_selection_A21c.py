#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:54:33 2024

@author: Miri
"""

#%% import packages

import os
import re
import h5py
import gzip
import numpy as np
import pandas as pd


#%%

def extract_MEGcaseno(file):
    match = re.search(r'case_(\d+)', file)
    return match.group(1) if match else None

#%% set directories

os.chdir('/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/scripts')
dir_parent = '/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc'
dir_data = os.path.join(dir_parent, 'data/MEGs')
dir_figs = os.path.join(dir_parent, 'figs')
dir_procdata = os.path.join(dir_parent, 'proc_data')


#%% load metadata

df_l23_pIDMEG_metadata = pd.read_csv(os.path.join(dir_procdata, 'IQ_Excitability_L2L3_pID_MEGcase.csv'), dtype={'MEG_case#': str})

fs = 1250 # Hz

# eg_filepath = '/Users/Miri/Downloads/case_0054_spontaan1_raw_tsss_0.5_48Hz_BNA1_WITH_BNA2_WITH_BNA3_VE.asc'
# df_eg = pd.read_csv(eg_filepath, delimiter='\t', header=None)
# df_eg.columns = ['channel_' + str(i+1) for i in range(df_eg.shape[1])]
# df_eg_filt = df_eg.iloc[0:200, : ]

#%%

df_l23_pIDMEG = df_l23_pIDMEG_metadata.dropna(subset=['MEG_case#'])[['patient_id', 'MEG_case#', 'HemiRes', 'start-time MEG [sec]', 'T0 [sec]', 'Tend [sec]']]
ls_gzfiles = [f for f in os.listdir(dir_data) if f.endswith('.gz')]

ls_range = []
ls_minvals = []
hemimap = {'L' : '81', 'R' : '82'}
for index, row in df_l23_pIDMEG.iterrows():
    pID, caseID, hemires = row['patient_id'], str(row['MEG_case#']), row['HemiRes']
    T0, Tend = row['T0 [sec]'], row['Tend [sec]']
    S0, Send = int(T0 * fs), int(Tend * fs)
    for gz_file in ls_gzfiles:
        if f'case_{caseID}' in gz_file:
            print(f'File match found, processing Case {caseID}')
            gz_path = os.path.join(dir_data, gz_file)
            with gzip.open(gz_path, 'rt') as gzf:
                df = pd.read_csv(gzf, delimiter='\t', header=None)
                df.columns = [str(i+1) for i in range(df.shape[1])]
                hdf5_filename = f'{pID}_case{caseID}_{hemires}Resec_Tz{int(T0)}_Te{int(Tend)}.h5'
                hdf5_path = os.path.join(dir_data, f'MTG_A21c/{hdf5_filename}')
                with h5py.File(hdf5_path, 'w') as hdf:
                    for hemi, channel in hemimap.items():
                        if channel in df.columns:
                            array = df[channel].iloc[S0:Send].to_numpy()
                            ls_range.append(np.max(array)-np.min(array))
                            ls_minvals.append(np.min(array))
                            dataset_name = f'{hemi}{channel}' 
                            hdf.create_dataset(dataset_name, data=array)    
#%%

with h5py.File(os.path.join(dir_data, 'MTG_A21c/2017_01_25_case1547_LeftResec_Tz31_Te922.h5'), 'r') as hdf:
    for name, obj in hdf.items():
        if isinstance(obj, h5py.Dataset):
            print(name)
            
with h5py.File(os.path.join(dir_data, 'MTG_A21c/2017_01_25_case1547_LeftResec_Tz31_Te922.h5'), 'r') as hdf:
    left_data = np.array(hdf['L81'])
    right_data = np.array(hdf['R82'])
    print("Left Hemisphere Data:", left_data)
    print("Right Hemisphere Data:", right_data)
#%%
for index, row in df_l23_pIDMEG.iterrows():
    pID = row['patient_id']
    caseID = str(row['MEG_case#'])
    hemires = row['HemiRes']
    T0 = row['T0 [sec]']
    Tend = row['Tend [sec]']
    S0 = T0*fs
    Send = Tend*fs
    for gz_file in ls_gzfiles:
        if f'case_{caseID}' in gz_file:
            print(f'file match found, processing Case {caseID}')
            gz_path = os.path.join(dir_data, gz_file)
            with gzip.open(gz_path, 'rt') as gzf:
                df = pd.read_csv(gzf, delimiter='\t', header=None)
                df.columns = ['channel_' + str(i+1) for i in range(df.shape[1])]
                for hemi in hemimap:
                    channel_name = hemimap[hemi]
                    if channel_name in df.columns:
                        array = df[channel_name][S0:Send].to_numpy()

ls_casenos = []
for gz_file in ls_gzfiles:
    gz_path = os.path.join(dir_data, gz_file)
    case_number = extract_MEGcaseno(gz_file)
    ls_casenos.append(case_number)
    
    
    