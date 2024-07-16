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
import pickle
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
#%% pickle files:

df_l23_pIDMEG = df_l23_pIDMEG_metadata.dropna(subset=['MEG_case#'])[['patient_id', 'MEG_case#', 'HemiRes', 'start-time MEG [sec]', 'T0 [sec]', 'Tend [sec]']]
ls_gzfiles = [f for f in os.listdir(dir_data) if f.endswith('.gz')]


# ROIs = [str(r) for r in range(81, 103)] # for all TG regions

ROIs = [str(r) for r in range(81, 89)] # for all MTG regions
region_name = 'allMTGs'

# ROIs = [str(r) for r in range(75, 82)] + ['87', '88'] # for lateral MTG & STG regions
# region_name = 'lMTG_lSTG'





# df_COIdata = pd.DataFrame()
for index, row in df_l23_pIDMEG.iterrows():
    pID, caseID, hemires = row['patient_id'], str(row['MEG_case#']), row['HemiRes']
    T0, Tend = row['T0 [sec]'], row['Tend [sec]']
    S0, Send = int(T0 * fs), int(Tend * fs)
    ls_ROIarrays = []
    for gz_file in ls_gzfiles:
        if f'case_{caseID}' in gz_file:
            print(f'File match found, processing Case {caseID}')
            gz_path = os.path.join(dir_data, gz_file)
            try:
                with gzip.open(gz_path, 'rt') as gzf:
                    df = pd.read_csv(gzf, delimiter='\t', header=None)
                    df.columns = [str(i+1) for i in range(df.shape[1])]
                    ROI_cols = df.columns[df.columns.isin(ROIs)]
                    if not ROI_cols.empty:
                        try:
                            df_selected = df.loc[S0:Send, ROI_cols].astype(float)
                            ls_ROIarrays.append(df_selected)
                        except ValueError as e:
                            print(f'Error converting data to float in file: {gz_file}', e)
            except Exception as e:
                print(f'Error reading file: {gz_file}', e)
    if ls_ROIarrays:
        df_ROIarrays = pd.concat(ls_ROIarrays)
        pickle_filename = f'{pID}_case{caseID}_{hemires}Resec_Tz{int(T0)}_Te{int(Tend)}_R{ROIs[0]}R{ROIs[-1]}.pkl.gz'
        
###  change file path ###
        pickle_path = os.path.join(dir_data, region_name, f'{pickle_filename}')
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        
        
        with gzip.open(pickle_path, 'wb') as gzfile:
            pickle.dump(df_ROIarrays, gzfile)

#%% asc files

df_l23_pIDMEG = df_l23_pIDMEG_metadata.dropna(subset=['MEG_case#'])[['patient_id', 'MEG_case#', 'HemiRes', 'start-time MEG [sec]', 'T0 [sec]', 'Tend [sec]']]
ls_gzfiles = [f for f in os.listdir(dir_data) if f.endswith('.gz')]

# ROIs = [str(r) for r in range(81, 103)] # for all TG regions
ROIs = [str(r) for r in range(81, 89)] # for all MTG regions
region_name = 'allMTGs'

# ROIs = [str(r) for r in range(75, 82)] + ['87', '88'] # for lateral MTG & STG regions
# region_name = 'lMTG_lSTG'

# df_COIdata = pd.DataFrame()
for index, row in df_l23_pIDMEG.iterrows():
    pID, caseID, hemires = row['patient_id'], str(row['MEG_case#']), row['HemiRes']
    T0, Tend = row['T0 [sec]'], row['Tend [sec]']
    S0, Send = int(T0 * fs), int(Tend * fs)
    ls_ROIarrays = []
    for gz_file in ls_gzfiles:
        if f'case_{caseID}' in gz_file:
            print(f'File match found, processing Case {caseID}')
            gz_path = os.path.join(dir_data, gz_file)
            try:
                with gzip.open(gz_path, 'rt') as gzf:
                    df = pd.read_csv(gzf, delimiter='\t', header=None)
                    df.columns = [str(i+1) for i in range(df.shape[1])]
                    ROI_cols = df.columns[df.columns.isin(ROIs)]
                    if not ROI_cols.empty:
                        try:
                            df_selected = df.loc[S0:Send, ROI_cols].astype(float)
                            ls_ROIarrays.append(df_selected)
                        except ValueError as e:
                            print(f'Error converting data to float in file: {gz_file}', e)
            except Exception as e:
                print(f'Error reading file: {gz_file}', e)
    if ls_ROIarrays:
        df_ROIarrays = pd.concat(ls_ROIarrays)
        asc_filename = f'{pID}_case{caseID}_{hemires}Resec_Tz{int(T0)}_Te{int(Tend)}_R{ROIs[0]}R{ROIs[-1]}.asc.gz'
        asc_path = os.path.join(dir_data, region_name, f'ascfiles/{asc_filename}')
        
        os.makedirs(os.path.dirname(asc_path), exist_ok=True)
        
        with gzip.open(asc_path, 'wt') as gzfile:
            df_ROIarrays.to_csv(gzfile, sep='\t', index=False, header=False)

print("All files have been processed and saved.")

#%% presliced files:
filepaths = [f for f in os.listdir(os.path.join(dir_data, region_name, 'ascfiles')) if f.endswith('.asc.gz')]

df_subjectlist = pd.DataFrame(columns=['Path', 'Case_ID', 'Start', 'End'])
df_subjectlist['Path'] = filepaths

pattern = re.compile(r'_case(?P<caseID>\d+)_.*_Tz(?P<T0>\d+)_Te(?P<Tend>\d+)_.*\.asc\.gz')
data = []
dir_regiondata = os.path.join(dir_data, region_name, 'ascfiles')
for filename in os.listdir(dir_regiondata):
    if filename.endswith('.asc.gz'):
        match = pattern.search(filename)
        if match:
            caseID = str(match.group('caseID'))
            T0 = match.group('T0')
            Tend = match.group('Tend')
            data.append({
                'Path': dir_regiondata,
                'Case_ID': str(caseID),
                'Atlas': None,
                'MM': None,
                'Start': T0,
                'End': Tend,
                'Selection': None
            })
df_subjectlist = pd.DataFrame(data, columns=['Path', 'Case_ID', 'MM', 'Atlas', 'Start', 'End', 'Selection'])

df_subjectlist.to_csv(os.path.join(dir_regiondata, 'df_subjectlist_for_Pspec.csv'))


