#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:37:10 2024

@author: Miri
"""

#%% import packages

import os
import re
import math
import pyabf
import importlib
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from mat4py import loadmat
import statsmodels.api as sm
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from scipy.signal import butter, filtfilt, find_peaks


#%% set directories

os.chdir('/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/scripts')
dir_parent = '/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc'
dir_data = os.path.join(dir_parent, 'data')
dir_figs = os.path.join(dir_parent, 'figs')
dir_procdata = os.path.join(dir_parent, 'proc_data')

#%% import analysis and plotting modules

import IQexcitability_analysisfns
importlib.reload(IQexcitability_analysisfns)
from IQexcitability_analysisfns import * 

import IQexcitability_plotfns
importlib.reload(IQexcitability_plotfns)
from IQexcitability_plotfns import *

#%% load needed dataframes

df_eline = pd.read_excel('/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/data/metadata/Ephys_morph_old_data_NG_TV_AK.xlsx')
df_eline_new = pd.read_excel('/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/proc_data/240422_Eline_Femke_Master_ephys_morph_sheet.xlsx', sheet_name='Morph')
df_elife = pd.read_csv('/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/data/metadata/Data_eLife.csv')
df_eeg = pd.read_excel('/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/proc_data/EEG_ephys_morph.xlsx')
df_MEGanalysis = pd.read_csv(os.path.join(dir_procdata, 'df_MEGanalysis_DFA_fEI_BIS_HLP_allMTGs.csv'))

#%% load l23ccor create file metadata dataframe 

# define dat pattern to look for patient_id in file names
pat_date = re.compile(r'(\d{4})_(\d{2})_(\d{2})')
name_filecol = 'Ephys file name'
name_slicecol = 'Slice name/comments'


def process_metadata(dir_data, df_eline, df_elife, df_eeg, dir_procdata, metadata_filename='df_metadata_IQexcitability.csv', reload=False):
    # Check if the metadata file already exists and reload is False
    path_metadata = os.path.join(dir_procdata, metadata_filename)
    if not reload and os.path.exists(path_metadata):
        print(f"Metadata file '{metadata_filename}' already exists. Loading existing file.")
        return pd.read_csv(path_metadata)
    # create dataframe from files in the data directory with file identifiers
    ls_filecell = [
        {
            'file': file,
            'patient_id': '_'.join(match.groups()) if (match := pat_date.search(file)) else None,
            'cell_id': file[:-9]
        }
        for file in os.listdir(dir_data)
        if file.endswith('.mat') or file.endswith('.abf')
    ]
    df_filecell = pd.DataFrame(ls_filecell)
    # add in_eLife column for future back reference
    df_filecell['in_eLife'] = df_filecell['cell_id'].apply(lambda x: any(x in file for file in df_elife['File'] if pd.notna(file)))
    # create patient_id column in elife data for merging and merge elife columns to new metadata dataframe
    df_elife['File'] = df_elife['File'].replace('none', np.nan)
    df_elife['patient_id'] = pd.to_datetime(df_elife['Date'], format='%m/%d/%Y')
    df_elife['patient_id'] = df_elife['patient_id'].dt.strftime('%Y_%m_%d') 
    patientdata_elife = ['patient_id', 'Gender', 'AgeAtResection', 'AgeAtEpilepsyOnset', 'YearsEpilepsy', 'SeizuresMonth', 'DiseaseType', 'HemiRes', 'NewTIQ']
    patientdata_elife = df_elife[patientdata_elife]
    patientdata_elife = patientdata_elife.drop_duplicates(subset=['patient_id'], keep='first')
    df_metadata = pd.merge(df_filecell, patientdata_elife, left_on='patient_id', right_on='patient_id', how='left')
    # repeat merging process for eline dataframe
    filedata_elife = ['File', 'holdingcurrent']
    filedata_elife = df_elife[filedata_elife]
    df_metadata = pd.merge(df_metadata, filedata_elife, left_on='file', right_on='File', how='left')
    df_metadata = df_metadata.drop(columns='File')
    df_eline['patient_id'] = df_eline.apply(extract_patient_id, axis=1, file_column=name_filecol, cell_column=name_slicecol)
    # df_eline['patient_id'] = df_eline.apply(extract_patient_id, axis=1, file_column='EphysFileName', cell_column='SliceName_comments')
    regiondata_eline = ['patient_id', 'Region'] 
    regiondata_eline = df_eline[regiondata_eline]
    regiondata_eline = regiondata_eline.drop_duplicates(subset=['patient_id'], keep='first')
    df_metadata = pd.merge(df_metadata, regiondata_eline, left_on='patient_id', right_on='patient_id', how='left')
    # subset EEG detaframe and prpeare for merging
    df_eeg['patient_id'] = pd.to_datetime(df_eeg['Date of surgery'])
    df_eeg['patient_id'] = df_eeg['patient_id'].dt.strftime('%Y_%m_%d')
    eegdata_eeg = ['patient_id', 'EEG:'] 
    # add file specific data from all original dataframes, looped by row to allow matching to either file name or cell/slice name
    for index, row in df_metadata.iterrows():
        row_eeg = df_eeg[(df_eeg['patient_id']) == row['patient_id']]
        if not row_eeg.empty:
            df_metadata.at[index, 'EEG'] = row_eeg['EEG:'].iloc[0]
        # Find matching row in df_eline based on 'Ephys file name' or 'Slice name'
        row_eline = df_eline[(df_eline[name_filecol] == row['file']) | (df_eline[name_filecol] == row['cell_id'])]
        # row_eline = df_eline[(df_eline['EphysFileName'] == row['file']) | (df_eline['SliceName_comments'] == row['cell_id'])]
        if not row_eline.empty:
            df_metadata.at[index, 'layer'] = row_eline['Layer DAPI verified'].iloc[0]
            df_metadata.at[index, 'pia_WM'] = row_eline['Pia-WM'].iloc[0]
            df_metadata.at[index, 'pia_soma'] = row_eline['Pia-soma distance Neurolucida verified'].iloc[0]
            df_metadata.at[index, 'rel_depth_Cx'] = row_eline['Pia-soma distance Neurolucida verified'].iloc[0]/row_eline['Pia-WM'].iloc[0]
            df_metadata.at[index, 'cell_type'] = row_eline['MorphLabelStacey'].iloc[0]
            df_metadata.at[index, 'cortical_region'] = row_eline['Cortical Region'].iloc[0]
            df_metadata.at[index, 'TDL_eline'] = row_eline['TDL'].iloc[0]
            df_metadata.at[index, 'ADL_eline'] = row_eline['TDL apical'].iloc[0]
            df_metadata.at[index, 'TBranchPoints_eline'] = row_eline['Total Nodes'].iloc[0]
            df_metadata.at[index, 'AMaxPath_eline'] = row_eline['Apical pathlength'].iloc[0]
            df_metadata.at[index, 'TNTerminals_eline'] = row_eline['Total Ends'].iloc[0]
            df_metadata.at[index, 'ABranchPoints_eline'] = row_eline['Nodes Apical'].iloc[0]
            df_metadata.at[index, 'BBranchPoints_eline'] = row_eline['Nodes Basal'].iloc[0]
            df_metadata.at[index, 'SomaSA_eline'] = row_eline['Soma surface'].iloc[0]
            
            
        # Find matching row in df_eline based on 'Ephys file name' or 'Slice name'
        row_elife = df_elife[(df_elife['File'] == row['file']) | (df_elife['cellName'] == row['cell_id'])]
        column_mappings = {
                            'UserID': 'patcher_ID',
                            'CorticalThick': 'pia_WM',
                            'Depth': 'pia_soma',
                            'RelDepth': 'rel_depth_Cx',
                            'TDL': 'TDL',
                            'TBranchPoints': 'TBranchPoints',
                            'TNBranches': 'TNBranches',
                            'TNTrees': 'TNTrees',
                            'TNTerminals': 'TNTerminals',
                            'TMaxBrorder': 'TMaxBrorder',
                            'TMeanBrLen': 'TMeanBrLen',
                            'TVolume': 'TVolume',
                            'TArea': 'TArea',
                            'ADL': 'ADL',
                            'ABranchPoints': 'ABranchPoints',
                            'ANBranches': 'ANBranches',
                            'ANTrees': 'ANTrees',
                            'ANTerminals': 'ANTerminals',
                            'AMaxBrorder': 'AMaxBrorder',
                            'AMeanBrLen': 'AMeanBrLen',
                            'AVolume': 'AVolume',
                            'AArea': 'AArea',
                            'BDL': 'BDL',
                            'BBranchPoints': 'BBranchPoints',
                            'BNBranches': 'BNBranches',
                            'BNTrees': 'BNTrees',
                            'BNTerminals': 'BNTerminals',
                            'BMaxBrorder': 'BMaxBrorder',
                            'BMeanBrLen': 'BMeanBrLen',
                            'BVolume': 'BVolume',
                            'BArea': 'BArea',
                            'ThreshFrstAP': 'ThreshFAP',
                            'AmpFAPthresh': 'AmpFAPthresh',
                            'thresh11to50': 'thresh11to50',
                            'amp11to50': 'amp11to50',
                            'thresh21to50': 'thresh21to50',
                            'amp21to50': 'amp21to50',
                            'threshR11to50': 'threshR11to50',
                            'ampR11to50': 'ampR11to50',
                            'thresh11to50N': 'thresh11to50N',
                            'amp11to50N': 'amp11to50N'
                            }
        if not row_elife.empty:
            for source_col, target_col in column_mappings.items():
                df_metadata.at[index, target_col] = row_elife[source_col].iloc[0]

        # add ephys data for mat files
        file = row['file']
        path_file = os.path.join(dir_data, file)
        if file.endswith('.abf'):
            try:
                abf = pyabf.ABF(path_file)
                df_metadata.at[index, 'protocol'] = abf.protocol
                df_metadata.at[index, 'fs'] = abf.dataRate
            except NotImplementedError:
                print(index, file, 'Invalid ABF file format')
                continue
        elif file.endswith('.mat'):
            mat = loadmat(path_file)
            df_metadata.at[index, 'protocol'] = 'CC_Steps'
            df_metadata.at[index, 'fs'] = mat['abf']['sampleFrequency']
            df_metadata.at[index, 'idx_I_onset'] = mat['abf']['pulse_onset']
            df_metadata.at[index, 'idx_I_offset'] = mat['abf']['pulse_offset']
            df_metadata.at[index, 'I0_pA'] = mat['abf']['first_pulse']
            df_metadata.at[index, 'Idelta_pA'] = mat['abf']['pulse_delta']
    df_metadata.to_csv(path_metadata, index=False)
    print(f"Metadata saved to '{metadata_filename}'.")
    
    return df_metadata


df_filemeta2 = process_metadata(dir_data, df_eline_new, df_elife, df_eeg, dir_procdata, metadata_filename='df_metadata_IQexcitability_extramorphs.csv', reload=False)

#%% define global parameters

plt.rcParams.update({'font.size': 30})
plt.rcParams['figure.max_open_warning'] = 100

min_peak_threshold = 0
rmp_range = [-80, 58]

mV_to_V = 1e-3  # Conversion factor millivolts to volts
pA_to_A = 1e-12  # Conversion factor picoamperes to amperes
Ohm_to_MOhm = 1e-6 # Conversion factor Ohms to mega Ohms

#%% create metadata dataframe for CC Steps

ls_ccID_strs = ['TEP', '_HP_', 'IV']
ls_exclusion_strs = ['AP', 'Human', 'vanaf', 'depol', 'ecode', 'D50_IN2', 'IN2_50', 'STEP_1x_', 'min210pA', 'Long_STEP']

df_cc_filemeta2 = separate_metadata_by_protocol(df_filemeta2, ls_ccID_strs, ls_exclusion_strs)

#%% create metadata dataframe that only includes cells from L2/L3 or that are included in the eLife set

df_l23cc_filemeta2 = df_cc_filemeta2[(df_cc_filemeta2['in_eLife'] == True) | (df_cc_filemeta2['layer'] == 'L2 L3')]
df_l4cc_filemeta = df_cc_filemeta[(df_cc_filemeta['layer'] == 'L4')]
df_l56cc_filemeta = df_cc_filemeta[(df_cc_filemeta['layer'] == 'L5 L6')]
df_l456cc_filemeta = df_cc_filemeta[(df_cc_filemeta['layer'] == 'L4') | (df_cc_filemeta['layer'] == 'L5 L6')]
#%% LAYER 2/3

filepath_l23cc_sweepmeta = os.path.join(dir_procdata, 'df_l23cc_sweepmeta.csv')
filepath_l23cc_manual_Ionsets = os.path.join(dir_procdata, 'df_l23cc_manual_Ionsets.csv')

manual_Ionsets = []
manual_Ioffsets = []
click_count = 0

# some cells have multiple corresponding files. to avoid duplication - selection criteria: n_sweeps, rmp, max amp, stability, earliest protocol by file name
ls_l23cc_duplicate_protocols_for_inclusion = [
                                    '2017_04_19_0004.abf',
                                    '2017_01_25_0020.abf',
                                    '2015_12_02_0004.abf',
                                    '2015_11_04_0007.abf',
                                    '2015_03_11Cel01_0003.abf',
                                    '2015_03_11_0067.abf',
                                    '2015_02_18_0021.abf',
                                    '2015_02_11Cel01_0013.abf',
                                    '2015_01_28Cel08_0003.abf',
                                    '2015_01_28Cel06_0003.abf',
                                    '2015_01_28Cel05_0003.abf',
                                    '2015_01_28Cel01_0003.abf',
                                    '2015_01_28_0002.abf',
                                    '2014_11_27Cel07_0001.abf',
                                    '2014_11_27Cel01_0002.abf',
                                    '2014_09_10Cel05_0002.abf',
                                    '2014_08_27Cel06_0004.abf',
                                    '2014_08_27Cel04_0001.abf',
                                    '2014_08_20Cel06_0004.abf',
                                    '2014_06_25Cel02_0001.abf',
                                    '2013_12_04Cel05_0003.abf',
                                    '2013_10_30Cel01_0001.abf',
                                    '2013_08_21Cel06_0003.abf',
                                    '2013_08_21Cel03_0001.abf',
                                    '2013_08_21Cel02_0002.abf',
                                    '2013_06_12Cel10_0001.abf',
                                    '2013_06_17Cel07_0002.abf',
                                    '2013_06_12Cel05_0003.abf',
                                    '2013_06_12Cel01_0002.abf',
                                    '2013_05_08Cel05_0003.abf',
                                    '2013_03_20Cel04_0018.abf',
                                    '2013_02_20Cel03_0023.abf',
                                    '2013_03_20Cel01_0002.abf',
                                    '2013_03_13Cel10_0033.abf',
                                    '2013_03_13Cel07_0016.abf',
                                    '2013_03_13Cel06_0023.abf',
                                    '2013_03_13Cel05_0015.abf',
                                    '2013_03_13Cel04_0021.abf',
                                    '2013_03_13Cel03_0021.abf',
                                    '2013_03_13Cel02_0028.abf',
                                    '2013_03_13Cel01_0002.abf',
                                    '2013_03_13Cel07_0021.abf',
                                    '2013_03_13Cel06_0001.abf',
                                    '2013_02_27Cel08_0001.abf',
                                    '2013_02_27Cel07_0001.abf',
                                    '2013_02_27Cel06_0001.abf',
                                    '2013_02_27Cel01_0000.abf',
                                    '2013_02_20Cel01_0000.abf',
                                    '2013_01_16Cel09_0000.abf',
                                    '2013_01_16Cel02_0000.abf',
                                    '2013_01_16Cel01_0001.abf',
                                    '2013_01_06Cel10_0001.abf',
                                    '2013_01_09Cel04_0003.abf',
                                    '2013_03_07Cel05_0000.abf',
                                    '2012_01_11Cel04_0001.abf',
                                    '2011_01_25Cel08_0000.abf',
                                    '2011_01_25Cel06_0000.abf',
                                    '2011_01_25Cel01_0000.abf',
                                    '2010_10_20Cel11_0021.abf',
                                    '2010_09_29N_0000.abf',
                                    '2010_09_29Cel6_0000.abf',
                                    '2010_09_29Cel5_0001.abf',
                                    '2010_09_29Cel4_0000.abf',
                                    '2010_09_29Cel3_0002.abf',
                                    '2010_09_29Cel15_0001.abf',
                                    '2010_09_29Cel14_0001.abf',
                                    '2010_09_29Cel13_0003.abf',
                                    '2010_09_29Cel12_0005.abf',
                                    '2010_09_29Cel11_0000.abf',
                                    '2010_09_29Cel10_0000.abf',
                                    '2010_09_29Cel1_0000.abf',
                                    '2010_09_29_0000.abf',
                                    '2010_09_21CellN_0010.abf',
                                    '2010_09_21Cel18_0000.abf',
                                    '2010_09_21Cel17_0000.abf',
                                    '2010_09_21Cel16_0000.abf',
                                    '2010_09_21Cel14_0000.abf',
                                    '2010_09_21Cel12_0000.abf',
                                    '2010_08_24Cel8_0000.abf',
                                    '2010_08_24Cel7_0000.abf',
                                    '2010_08_24Cel5_0000.abf',
                                    '2010_08_24Cel4_0000.abf',
                                    '2010_08_24Cel2_0000.abf',
                                    '2010_08_24Cel14_0000.abf',
                                    '2010_08_24Cel1_0003.abf',
                                    '2010_06_22Cel6_0001.abf',
                                    '2010_06_22Cel21_0024.abf',
                                    '2010_06_22Cel20_0000.abf',
                                    '2010_06_22Cel2_0000.abf',
                                    '2010_06_22Cel11_0000.abf',
                                    '2010_06_22Cel10_0001.abf',
                                    '2010_06_22Cel1_0000.abf',
                                    '2010_06_08Cel9_0008.abf',
                                    '2010_06_08Cel2_0000.abf',
                                    '2010_06_08Cel15_0000.abf',
                                    '2010_06_08Cel13_0000.abf',
                                    '2010_06_08Cel12_0000.abf',
                                    '2010_06_08Cel1_0000.abf',
                                    '2010_04_13Cel6_0001.abf',
                                    '2010_03_16Cel16_0000.abf',
                                    '2010_03_16Cel15_0000.abf',
                                    '2009_11_11Cel1_0002.abf',
                                    '2009_11_11_0005.abf',
                                    '2009_11_10_0000.abf',
                                    ]


# After inspecting sweeps following files to be excluded:
ls_files_manually_exc = ['2010_03_16Cel16_0010.abf',
                            '2015_03_11Cel01_0005.abf',
                            '2013_12_04Cel02_0002.abf',
                            '2013_02_20Cel05_0018.abf',
                            '2013_02_20Cel05_0019.abf',
                            '2010_06_08Cel12_0011.abf',
                            '2013_10_30Cel01_0001.abf',
                            '2013_02_20Cel05_0018.abf',
                            '2013_10_30Cel01_0000.abf',
                            '2013_01_16Cel01_0016.abf',
                            '2015_03_11Cel01_0002.abf',
                            '2015_03_11Cel01_0003.abf',
                            '2010_04_13Cel6_0000.abf',
                            '2013_02_20Cel05_0017.abf',
                            '2015_03_11Cel01_0004.abf',
                            '2009_11_11_0005.abf',
                            '2010_09_21Cel1_0000.abf']

reload = False
if not os.path.exists(filepath_l23cc_sweepmeta) or reload:
    df_l23cc_sweepmeta, df_Iepoch_failed = create_df_sweep_metadata(df_l23cc_filemeta2, dir_data, dir_procdata)
    df_l23cc_sweepmeta, df_l23cc_manual_Iepochs = update_sweepmeta_with_manual_Iepochs(dir_procdata, df_Iepoch_failed, df_l23cc_sweepmeta) 
    df_l23cc_sweepmeta = detect_peaks(dir_data, df_l23cc_sweepmeta, min_peak_threshold)
    df_l23cc_sweepmeta.to_csv(filepath_l23cc_sweepmeta)
else:
    print(f"File {filepath_l23cc_sweepmeta} already exists. Skipping processing, loading file.")
    df_l23cc_sweepmeta = pd.read_csv(filepath_l23cc_sweepmeta)


df_l23cc_sweepmeta_inc2, df_l23cc_filemeta_inc2, df_l23cc_rhsweeps2 = exclude_files(dir_data, df_l23cc_filemeta2, df_l23cc_sweepmeta, ls_l23cc_duplicate_protocols_for_inclusion, ls_files_manually_exc, min_peak_threshold, rmp_range)

## currenlty dont know if/which mat files are duplicates - CHECK THIS
df_l23cc_filemeta_inc_nomat2 = df_l23cc_filemeta_inc2[df_l23cc_filemeta_inc2['file'].str.contains('.mat') == False]
df_l23cc_sweepmeta_inc_nomat = df_l23cc_sweepmeta_inc[df_l23cc_sweepmeta_inc['file'].str.contains('.mat') == False]
df_l23cc_sweepmeta_inc_nomat.to_csv(os.path.join(dir_procdata, 'fortjerk.csv'))

df_l23cc_sweepmeta_inc_nomat, df_l23cc_sweepmeta_negI = get_iRsag_values(dir_data, df_l23cc_filemeta_inc_nomat, df_l23cc_sweepmeta_inc_nomat, mV_to_V, pA_to_A, Ohm_to_MOhm)
df_l23cc_filemeta_inc_nomat = calculate_iR(df_l23cc_filemeta_inc_nomat, df_l23cc_sweepmeta_inc_nomat, plot=False)


#%%


#%% update with MEG analysis data

df_fEIsubset = df_MEGanalysis[['patient_ID', 'fEI_resec_band_alpha', 'fEI_ctrl_band_alpha']]
df_fEIsubset.rename(columns={'patient_ID': 'patient_id'}, inplace=True)
df_l23cc_filemeta_inc_nomat2 = pd.merge(df_l23cc_filemeta_inc_nomat2, df_fEIsubset, on='patient_id', how='left')

df_DFAsubset = df_MEGanalysis[['patient_ID', 'DFA_resec_band_alpha', 'DFA_ctrl_band_alpha', 'DFA_resec_band_beta', 'DFA_ctrl_band_beta']]
df_DFAsubset.rename(columns={'patient_ID': 'patient_id'}, inplace=True)
df_l23cc_filemeta_inc_nomat2 = pd.merge(df_l23cc_filemeta_inc_nomat2, df_DFAsubset, on='patient_id', how='left')

df_BISsubset = df_MEGanalysis[['patient_ID', 'BIS_resec_band_alpha', 'BIS_ctrl_band_alpha', 'BIS_resec_band_beta', 'BIS_ctrl_band_beta']]
df_BISsubset.rename(columns={'patient_ID': 'patient_id'}, inplace=True)
df_l23cc_filemeta_inc_nomat2 = pd.merge(df_l23cc_filemeta_inc_nomat2, df_BISsubset, on='patient_id', how='left')

df_HLPsubset = df_MEGanalysis[['patient_ID', 'HLP_resec_band_alpha', 'HLP_ctrl_band_alpha', 'HLP_resec_band_beta', 'HLP_ctrl_band_beta']]
df_HLPsubset.rename(columns={'patient_ID': 'patient_id'}, inplace=True)
df_l23cc_filemeta_inc_nomat2 = pd.merge(df_l23cc_filemeta_inc_nomat2, df_BISsubset, on='patient_id', how='left')

df_l23cc_filemeta_inc_L = df_l23cc_filemeta_inc_nomat2[df_l23cc_filemeta_inc_nomat2['HemiRes'] == 'Left']
df_l23cc_filemeta_inc_R = df_l23cc_filemeta_inc_nomat2[df_l23cc_filemeta_inc_nomat2['HemiRes'] == 'Right']

#%%

plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat2, 'l23cc_nomat', 'iR', 'iR (MOhm)')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_L, 'l23cc_nomat_L', 'iR', 'iR (MOhm)')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_R, 'l23cc_nomat_R', 'iR', 'iR (MOhm)')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'l23cc_nomat', 'rheobase', 'Rheobase (pA)')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_L, 'l23cc_nomat_L', 'rheobase', 'Rheobase (pA)')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_R, 'l23cc_nomat_R', 'rheobase', 'Rheobase (pA)')

plot_IQ_correlation_averaged_LMM(dir_figs, df_l23cc_filemeta_iR, 'L2L3', 'rheobase', 'Rheobase (pA)')
plot_IQ_correlation_averaged_LMM(dir_figs, df_l23cc_filemeta_iR, 'L2L3', 'iR', 'iR (MOhm)')
                                 
plot_IQ_correlation_averaged_LR(dir_figs, df_l23cc_filemeta_iR, 'L2L3', 'rheobase', 'Rheobase (pA)')
plot_IQ_correlation_averaged_LR(dir_figs, df_l23cc_filemeta_iR, 'L2L3', 'iR', 'iR (MOhm)')
plot_IQ_correlation_weighted_LR(dir_figs, df_l23cc_filemeta_iR, 'L2L3', 'rheobase', 'Rheobase (pA)')


#%%
biomarkers = ['DFA', 'fEI']
conditions = ['resec', 'ctrl']
fbands = ['alpha', 'beta']
cell_params = ['TArea', 'AArea', 'BArea']

for biomarker, condition, cell_param, fband in itertools.product(biomarkers, conditions, cell_params, fbands):
    try:
        plot_XY_LMM_MEGcorrelation_boxplot(dir_figs,df_l23cc_filemeta_inc_nomat2,f'{biomarker}_{condition}_band_{fband}',f'{biomarker} in {fband}, {condition} hemi',f'{cell_param}',f'{cell_param}','l23cc')
    except KeyError:
        print(f'KeyError within {biomarker}, {condition}, {cell_param}, {fband}')
        continue
    
#%%

cell_params = ['TArea', 'AArea', 'BArea']

for cell_param in cell_params:
    plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat2, 'L2L3', cell_param, cell_param)
#%%

plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'L2L3', 'ThreshFAP', 'Threshold First AP (mV)')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'L2L3', 'thresh21to50', 'Threshold IF 21-50 (mV)')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'L2L3', 'thresh11to50', 'Threshold IF 11-50 (mV)')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'L2L3', 'threshR11to50', 'Relative Threshold IF 11-50')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_L, 'L2L3', 'threshR11to50', 'Relative Threshold IF 11-50')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'L2L3', 'thresh11to50N', 'Normalised Threshold IF 11-50')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_L, 'L2L3_L', 'thresh11to50N', 'Normalised Threshold IF 11-50')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_R, 'L2L3_R', 'thresh11to50N', 'Normalised Threshold IF 11-50')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'L2L3', 'amp11to50', 'AP Amplitude IF 11-50 (mV)')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta, 'L2L3', 'amp21to50', 'AP Amplitude IF 21-50 (mV)')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta, 'L2L3', 'ampR11to50', 'Relative AP Amplitude IF 11-50')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta, 'L2L3', 'amp11to50N', 'Normalised AP Amplitude IF 11-50')

plot_XY_LMM_correlation(dir_figs, df_l23cc_filemeta, 'rheobase', 'Rheobase (pA)', 'amp11to50N', 'Normalised AP Amplitude IF 11-50', 'l23cc')


plot_XY_LMM_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'rheobase', 'Rheobase (pA)', 'dFdI', 'Slope Coefficient Initial FI Curve', 'l23cc_nomat')

#%% not sure how to include this
df_l23cc_rhsweeps_hero150 = get_rheo_and_hero_sweeps_150(dir_data, df_l23cc_sweepmeta_inc_nomat)
df_l23cc_filemeta_inc_nomat = pd.merge(df_l23cc_filemeta_inc_nomat, df_l23cc_rhsweeps_hero150, on='file', how='left')

filedata_cols = df_l23cc_sweepmeta_inc_nomat[['file', 'idx_I_onset', 'idx_I_offset', 't_I_dur', 'rmp']].drop_duplicates()

df_l23cc_filemeta_inc_nomat = update_filemetadata_with_epochs(df_l23cc_filemeta_inc_nomat, filedata_cols)
df_l23cc_filemeta_inc_nomat = calculate_ISI_properties(dir_data, df_l23cc_filemeta_inc_nomat, min_peak_threshold)
df_l23cc_filemeta_inc_nomat = calculate_iR(df_l23cc_filemeta_inc_nomat, df_l23cc_sweepmeta_inc_nomat, plot=False)


#%%

#%%

def calculate_sag(df_file_metadata, df_sweep_metadata, plot=True):
    filegroup = df_sweep_metadata.groupby('file')
    for file, group in filegroup:
        print(file)
        neg_sweepset = group[group['I_step'] < 0]
        if not neg_sweepset.empty:
            sag_ratios = neg_sweepset['sag_ratio'].values
            i_input = neg_sweepset['I_step'].values
            slope, intercept = np.polyfit(i_input, sag_ratios, 1)
            sag_ratio = slope
            if plot:
                plt.figure(figsize=(6, 4))
                plt.plot(i_input, sag_ratios, 'o')
                plt.plot(i_input, i_input * slope + intercept, 'r-', label=f'Sag Ratio = {sag_ratio:.6f}')
                plt.xlabel('Input Current (A)')
                plt.ylabel('Sag Ratio')
                plt.legend()
                plt.title(f'Sag Ratio for {file}') 
                plt.show()
            df_file_metadata.loc[df_file_metadata['file']==file, 'sag_ratio'] = sag_ratio
        else:
            print(f'Problem with SagRatio: Negative CC-Steps sweepset empty for {file}')
            
    return df_file_metadata
df_l23cc_filemeta_sagR = calculate_sag(df_l23cc_filemeta_inc_nomat, df_l23cc_sweepmeta_inc_nomat, plot=True)
df_l23cc_filemeta_sagR_filt = df_l23cc_filemeta_sagR[df_l23cc_filemeta_sagR['sag_ratio'] < 1]
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_sagR, 'l23cc_nomat', 'sag_ratio', 'Slope of Sag Ratio')

df_l23cc_sweepmeta_inc_nomat = detect_peaks(dir_data, df_l23cc_sweepmeta_inc_nomat, min_peak_threshold)
calculate_FI_slope(df_l23cc_sweepmeta_inc_nomat, df_l23cc_filemeta_inc_nomat, plot=False)
calculate_Lburst(df_l23cc_sweepmeta_inc_nomat, df_l23cc_filemeta_inc_nomat, plot=False)
df_l23cc_filemeta_inc_L = df_l23cc_filemeta_inc_nomat[df_l23cc_filemeta_inc_nomat['HemiRes'] == 'Left']
df_l23cc_filemeta_inc_R = df_l23cc_filemeta_inc_nomat[df_l23cc_filemeta_inc_nomat['HemiRes'] == 'Right']

df_l23cc_filemeta_inc_nomat.to_csv(os.path.join(dir_procdata, 'IQ_Excitability_L2L3_File_MetaData.csv'))


plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'l23cc_nomat', 'dBdI_500ms', 'dBdI of first 500ms')
plot_IQ_correlation_averaged_LR(dir_figs, df_l23cc_filemeta_inc_nomat, 'L2L3', 'dBdI_500ms', 'dBdI of first 500ms')

plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'l23cc_nomat', 'dFdI', 'Slope Coefficient Initial FI Curve')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_L, 'l23cc_L', 'dFdI', 'Slope Coefficient Initial FI Curve')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_R, 'l23cc_R', 'dFdI', 'Slope Coefficient Initial FI Curve')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'l23cc_nomat', 'TNBranches', 'No. Dendritic Branches')
plot_IQ_correlation_averaged_LR(dir_figs, df_l23cc_filemeta_inc_nomat, 'L2L3', 'dFdI', 'Slope Coefficient Initial FI Curve')
plot_IQ_correlation_averaged_LR(dir_figs, df_l23cc_filemeta, 'L2L3', 'TDL', 'Total Dendritic Length (um)')
plot_IQ_correlation_averaged_LR(dir_figs, df_l23cc_filemeta, 'L2L3', 'TNBranches', 'No. Dendritic Branches')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'l23cc_nomat', 'AgeAtResection', 'Age (y)')
# plot_rheosweeps_perpatient2(dir_data, df_l23cc_filemeta_hero150_nomat, df_l23cc_sweepmeta_inc, min_peak_threshold, fontsize=7)

plot_XY_LMM_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'iR', 'iR (MOhm)', 'rheobase', 'Rheobase', 'l23cc_nomat')
plot_XY_LMM_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'TDL', 'Total Dendritic Length (um)', 'iR', 'iR (MOhm)','l23cc_nomat')
plot_XY_LMM_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'TDL', 'Total Dendritic Length (um)', 'rheobase', 'Rheobase (pA)', 'l23cc_nomat')
plot_XY_LMM_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'rel_depth_Cx', 'Relative Cortical Depth', 'iR', 'iR (MOhm)', 'l23cc_nomat')

plot_XY_LMM_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'iR', 'iR (MOhm)', 'dFdI', 'Slope Coefficient Initial FI Curve', 'l23cc_nomat')
plot_XY_LMM_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'rheobase', 'Rheobase (pA)', 'dFdI', 'Slope Coefficient Initial FI Curve', 'l23cc_nomat')


# disease severity markers
plot_IQ_disease_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'l23cc_nomat', 'AgeAtEpilepsyOnset', 'Age at Epilepsy Onset (Years)')
plot_IQ_disease_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'l23cc_nomat', 'YearsEpilepsy', 'Years of Epilepsy')
plot_IQ_disease_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'l23cc_nomat', 'SeizuresMonth', 'Number of Monthly Seizures')
plot_XY_LMM_correlation_boxplot(dir_figs, df_l23cc_filemeta_inc_nomat, 'AgeAtEpilepsyOnset', 'Age at Epilepsy Onset (Years)', 'iR', 'iR (MOhm)', 'l23cc_nomat', box_width=0.35)
plot_XY_LMM_correlation_boxplot(dir_figs, df_l23cc_filemeta_inc_nomat, 'YearsEpilepsy', 'Years of Epilepsy', 'iR', 'iR (MOhm)', 'l23cc_nomat', box_width=0.35)
plot_XY_LMM_correlation_boxplot(dir_figs, df_l23cc_filemeta_inc_nomat, 'SeizuresMonth', 'Number of Monthly Seizures', 'iR', 'iR (MOhm)', 'l23cc_nomat', box_width=2)
plot_XY_LMM_correlation_boxplot(dir_figs, df_l23cc_filemeta_inc_nomat, 'AgeAtEpilepsyOnset', 'Age at Epilepsy Onset (Years)', 'rheobase', 'Rheobase (pA)', 'l23cc_nomat', box_width=0.35)
plot_XY_LMM_correlation_boxplot(dir_figs, df_l23cc_filemeta_inc_nomat, 'YearsEpilepsy', 'Years of Epilepsy', 'rheobase', 'Rheobase (pA)', 'l23cc_nomat', box_width=0.25)
plot_XY_LMM_correlation_boxplot(dir_figs, df_l23cc_filemeta_inc_nomat, 'SeizuresMonth', 'Number of Monthly Seizures', 'rheobase', 'Rheobase (pA)', 'l23cc_nomat', box_width=2)
plot_XY_LMM_correlation_boxplot(dir_figs, df_l23cc_filemeta_inc_nomat, 'AgeAtEpilepsyOnset', 'Age at Epilepsy Onset (Years)', 'dFdI_100ms', 'dFdI of first 100ms', 'l23cc_nomat', box_width=0.35)
plot_XY_LMM_correlation_boxplot(dir_figs, df_l23cc_filemeta_inc_nomat, 'YearsEpilepsy', 'Years of Epilepsy', 'dFdI_100ms', 'dFdI of first 100ms', 'l23cc_nomat', box_width=0.25)
plot_XY_LMM_correlation_boxplot(dir_figs, df_l23cc_filemeta_inc_nomat, 'SeizuresMonth', 'Number of Monthly Seizures', 'dFdI_100ms', 'dFdI of first 100ms', 'l23cc_nomat', box_width=2)

plot_XY_LMM_correlation(dir_figs, df_l23cc_filemeta_inc_nomat, 'rel_depth_Cx', 'Relative Cortical Depth', 'iR', 'iR (MOhm)', 'l23cc_nomat')
# what difference do MAT files make??? are there mat files that are not duplicates of abf files??


#%% calculate sag

#%%
#%% LAYERS 4, 5 and 6

filepath_l456cc_sweepmeta = os.path.join(dir_procdata, 'df_l456cc_sweepmeta.csv')
filepath_l456cc_manual_Ionsets = os.path.join(dir_procdata, 'df_l456cc_manual_Ionsets.csv')

manual_Ionsets = []
manual_Ioffsets = []
ls_duplicate_protocols_for_inclusion = []

click_count = 0
ls_l456_manually_exc_files = []

reload = False
if not os.path.exists(filepath_l456cc_sweepmeta) or reload:
    df_l456cc_sweepmeta, df_l456_Iepoch_failed = create_df_sweep_metadata(df_l456cc_filemeta, dir_data, dir_procdata)
    df_l456cc_sweepmeta, df_l456cc_manual_Iepochs = update_sweepmeta_with_manual_Iepochs(dir_procdata, df_Iepoch_failed, df_l456cc_sweepmeta) 
    df_l456cc_sweepmeta = detect_peaks(dir_data, df_l456cc_sweepmeta, min_peak_threshold)
    df_l456cc_sweepmeta.to_csv(filepath_l456cc_sweepmeta)
else:
    print(f"File {filepath_l23cc_sweepmeta} already exists. Skipping processing, loading file.")
    df_l456cc_sweepmeta = pd.read_csv(filepath_l456cc_sweepmeta)

# Filter cells that are too deep for L2/L3
ls_files_depth_criteria = df_file_metadata[(df_file_metadata['rel_depth_Cx'].isna()) | 
                           (round(df_file_metadata['rel_depth_Cx'], 2) <= 1.0)]['file']



# ls_l456_selected_multicell_files = select_file_for_multifile_cells(df_l456cc_filemeta, ls_duplicate_protocols_for_inclusion)     
df_l456cc_sweepmeta_inc, df_l456cc_filemeta_inc, df_l456cc_rhsweeps = exclude_files(dir_data, df_l456cc_filemeta, df_l456cc_sweepmeta, ls_l456_selected_multicell_files, ls_l456_manually_exc_files, min_peak_threshold, rmp_range)


def process_files(dir_data, df_file_metadata, df_sweep_metadata, ls_duplicate_protocols_for_inclusion, ls_files_manually_exc, min_peak_threshold, rmp_range):
    # Remove spontaneous spiking cells
    ls_spontaneous_spikers = [file for file, group in df_l456cc_sweepmeta.groupby('file') if any((group['I_step'] <= 0) & (group['n_peaks'] > 0))]
    
    exc_criteria = (df_sweep_metadata['file'].isin(ls_spontaneous_spikers)) | (df_sweep_metadata['rmp'] < rmp_range[0]) | (df_sweep_metadata['rmp'] > rmp_range[1])
    # df_sweep_metadata_inc = df_sweep_metadata[~df_sweep_metadata['file'].isin(ls_spontaneous_spikers)]
    
    df_sweep_metadata_inc = df_sweep_metadata[~exc_criteria]
    print(df_file_metadata['file'].isin(df_sweep_metadata_inc['file']).value_counts())

    df_file_metadata_inc = df_file_metadata[df_file_metadata['file'].isin(df_sweep_metadata_inc['file'])]
    df_rheohero_sweeps = get_rheo_and_hero_sweeps_150(dir_data, df_sweep_metadata_inc)
    df_file_metadata_inc = pd.merge(df_file_metadata_inc, df_rheohero_sweeps, on='file', how='left')
    #filedata_cols = df_sweep_metadata_inc[['file', 'idx_I_onset', 'idx_I_offset', 't_I_dur', 'rmp']].drop_duplicates()
    filedata_cols = df_sweep_metadata_inc[['file', 'idx_I_onset', 'idx_I_offset', 'rmp']].drop_duplicates()
    df_file_metadata_inc = update_filemetadata_with_epochs(df_file_metadata_inc, filedata_cols)
    df_file_metadata_inc = calculate_ISI_properties(dir_data, df_file_metadata_inc, min_peak_threshold)

    return df_sweep_metadata_inc, df_file_metadata_inc, df_rheohero_sweeps


df_l456cc_sweepmeta_inc, df_l456cc_filemeta_inc, df_l456cc_rhsweeps = process_files(dir_data, df_l456cc_filemeta, df_l456cc_sweepmeta, ls_l456_selected_multicell_files, ls_l456_manually_exc_files, min_peak_threshold, rmp_range)

## currenlty dont know if/which mat files are duplicates - CHECK THIS
df_l456cc_filemeta_inc_nomat = df_l456cc_filemeta_inc[df_l456cc_filemeta_inc['file'].str.contains('.mat') == False]
df_l456cc_sweepmeta_inc_nomat = df_l456cc_sweepmeta_inc[df_l456cc_sweepmeta_inc['file'].str.contains('.mat') == False]

df_l456cc_sweepmeta_inc_nomat, df_l456cc_sweepmeta_negI = get_iRsag_values(df_l456cc_filemeta_inc_nomat, df_l456cc_sweepmeta_inc_nomat)
df_l456cc_filemeta_iR = calculate_iR(df_l456cc_filemeta_inc_nomat, df_l456cc_sweepmeta_inc_nomat, plot=False)

plot_IQ_correlation(dir_figs, df_l456cc_filemeta_iR, 'l456cc_nomat', 'iR', 'iR (MOhm)')
plot_IQ_correlation(dir_figs, df_l456cc_filemeta_iR, 'l456cc_nomat', 'rheobase', 'Rheobase')

#%%
#%% ALL LAYERS

df_l23456cc_filemeta = pd.concat([df_l23cc_filemeta_inc_nomat, df_l456cc_filemeta_iR], ignore_index=True)
plot_IQ_correlation(dir_figs, df_l23456cc_filemeta, 'l23456cc_nomat', 'iR', 'iR (MOhm)')
plot_IQ_correlation(dir_figs, df_l23456cc_filemeta, 'l23456cc_nomat', 'rheobase', 'Rheobase')
plot_XY_LMM_correlation(dir_figs, df_l23456cc_filemeta, 'iR', 'iR (MOhm)', 'rheobase', 'Rheobase', 'l23456cc_nomat')


plot_XY_LMM_correlation(dir_figs, df_l23456cc_filemeta, 'AgeAtEpilepsyOnset', 'Age at Epilepsy Onset (Years)', 'iR', 'iR (MOhm)', 'l23456cc_nomat')
plot_XY_LMM_correlation(dir_figs, df_l23456cc_filemeta, 'YearsEpilepsy', 'Years of Epilepsy', 'iR', 'iR (MOhm)', 'l23456cc_nomat')
plot_XY_LMM_correlation(dir_figs, df_l23456cc_filemeta ,'SeizuresMonth', 'Number of Monthly Seizures', 'iR', 'iR (MOhm)', 'l23456cc_nomat')

plot_XY_LMM_correlation(dir_figs, df_l23456cc_filemeta, 'AgeAtEpilepsyOnset', 'Age at Epilepsy Onset (Years)', 'rheobase', 'Rheobase (pA)', 'l23456cc_nomat')
plot_XY_LMM_correlation(dir_figs, df_l23456cc_filemeta, 'YearsEpilepsy', 'Years of Epilepsy', 'rheobase', 'Rheobase (pA)', 'l23456cc_nomat')
plot_XY_LMM_correlation(dir_figs, df_l23456cc_filemeta ,'SeizuresMonth', 'Number of Monthly Seizures', 'rheobase', 'Rheobase (pA)', 'l23456cc_nomat')

plot_XY_LMM_correlation(dir_figs, df_l23456cc_filemeta,'rel_depth_Cx', 'Relative Cortical Depth', 'iR', 'iR (MOhm)', 'l23456cc_nomat')
plot_XY_LMM_correlation(dir_figs, df_l23456cc_filemeta,'rel_depth_Cx', 'Relative Cortical Depth', 'rheobase', 'Rheobase (pA)', 'l23456cc_nomat')
#%%
#%% ONLY eLife 

df_l23cc_filemeta_elife = df_l23cc_filemeta[df_l23cc_filemeta['in_eLife']==True]
df_l23cc_sweepmeta_elife = df_l23cc_sweepmeta[df_l23cc_sweepmeta['file'].isin(df_l23cc_filemeta_elife['file'])]

exc_criteria = (pd.isna(df_l23cc_sweepmeta_elife['idx_I_onset'])) | (df_l23cc_sweepmeta_elife['idx_I_onset'] == 0) | (df_l23cc_sweepmeta_inc2['file'].isin(ls_spontaneous_spikers)) | (df_l23cc_sweepmeta_inc2['file'].isin(ls_files_exc)) 
df_l23cc_sweepmeta_elife = df_l23cc_sweepmeta_elife [~exc_criteria]      
  
df_rhsweeps_elife = get_rheo_and_hero_sweeps_150(dir_data, df_l23cc_sweepmeta_elife)
df_l23cc_filemeta_elife = pd.merge(df_l23cc_filemeta_elife, df_rhsweeps_elife, on='file', how='left')

filedata_cols = df_l23cc_sweepmeta_elife[['file', 'idx_I_onset', 'idx_I_offset', 't_I_dur', 'rmp']].drop_duplicates()

df_l23cc_filemeta_elife = update_filemetadata_with_epochs(df_l23cc_filemeta_elife, filedata_cols)
df_l23cc_filemeta_elife = calculate_ISI_properties(dir_data, df_l23cc_filemeta_elife, min_peak_threshold)

plot_IQ_correlation(dir_figs, df_l23cc_filemeta_elife, 'rheobase', 'Rheobase (pA)')
plot_IQ_correlation(dir_figs, df_l23cc_filemeta_elife, 'rmp', 'RMP (mV)')

# with or without additonal exclusion criteria, dont get the same trend as in eLife
#%% find the rheobase and hero sweeps for ISI and AI calculations


#%%

df_rhsweeps_heroisrheoplus75min3peaks = get_rheo_and_hero_sweeps_75x3(df_l23cc_sweepmeta_inc)
df_l23cc_filemeta_hero75x3 = pd.merge(df_l23cc_filemeta_inc, df_rhsweeps_heroisrheoplus75min3peaks, on='file', how='left')
df_l23cc_filemeta_hero75x3 = update_filemetadata_with_epochs(df_l23cc_filemeta_hero75x3, filedata_cols)
df_l23cc_filemeta_hero75x3 = calculate_ISI_properties(dir_data, df_l23cc_filemeta_hero75x3, min_peak_threshold)


df_rhsweeps_heroisrheoplus250 = get_rheo_and_hero_sweeps_250(df_l23cc_sweepmeta_inc)
df_l23cc_filemeta_hero250 = pd.merge(df_l23cc_filemeta_inc, df_rhsweeps_heroisrheoplus250, on='file', how='left')
df_l23cc_filemeta_hero250 = update_filemetadata_with_epochs(df_l23cc_filemeta_hero250, filedata_cols)
df_l23cc_filemeta_hero250 = calculate_ISI_properties(dir_data, df_l23cc_filemeta_hero250, min_peak_threshold)


#%% plot 

# files with high iR:
# 2009_11_10Cel09N_0050.abf
# 2013_05_08Cel04_0000.abf

example_ccsteps = plot_sweeps_from_file(dir_data, df_l23cc_sweepmeta_inc, '2013_05_08Cel04_0000.abf')

plot_herosweeps_perpatient(dir_data, df_l23cc_filemeta_inc_nomat, df_l23cc_sweepmeta_inc_nomat, min_peak_threshold)

#%% checking excluded files for inclusion

ls_dodgyfiles = ['2015_02_18_0016.abf',
'2015_12_02_0058.abf',
'2013_09_04Cel09_0001.abf',
'2013_02_20Cel05_0017.abf',
'2015_03_11Cel01_0001.abf',
'2013_01_16Cel09_0001.abf',
'2015_12_02_0076.abf',
'2013_05_08_Cel05_0000.abf',
'2012_03_07Cel05_0011.abf',
'2013_06_12Cel07_0006.abf',
'2013_10_30Cel01_0001.abf',
'2013_01_16Cel01_0016.abf',
'2013_01_16Cel01_0017.abf',
'2015_11_04_0035.abf',
'2010_06_08_Cel13_0000.abf',
'2011_01_25_Cel01_0012.abf',
'2009_11_10Cel7_0000.abf',
'2014_08_27Cel06_0004.abf',
'2011_01_25Cel01_0013.abf',
'2015_03_11_0023.abf',
'2013_02_20Cel05_0018.abf',
'2015_11_04_0095.abf',
'2015_12_02_0068.abf', 
'2015_02_11Cel01_0013.abf',
'2010_09_29Cel4_0011.abf', 
'2013_01_16Cel09_0001.abf',
'2013_10_30Cel01_0000.abf',
'2009_11_10_0050.abf',
'2013_02_27Cel08_0023.abf',
'2013_03_20Cel01_0002.abf',
'2010_06_08_Cel15_0011.abf',
'2009_11_10Cel09N_0050',
'2013_02_27Cel08_0024.abf',
'2013_03_16_Cell2N_0005.abf',
'2010_06_05Cel12_0012.abf',
'2010_06_08Cel12_0010.abf']


for dodgy_file in ls_dodgyfiles:
    matching_rows = df_l23cc_sweepmeta[df_l23cc_sweepmeta['file'] == dodgy_file]
    if not matching_rows.empty:
        n_plots = len(matching_rows)
        rows, cols = subplot_split(n_plots)
        fig, axs = plt.subplots(rows, cols, figsize=(10, 5))
        plt.rcParams.update({'font.size': 7})# Adjust figsize as necessary
        fig.suptitle(dodgy_file, fontsize=10)
        if n_plots > 1:
            axs = axs.ravel()
        else:
            axs = [axs]
        for idx, row in enumerate(matching_rows.itertuples()):
            file_path, file_ext = get_filepath_and_ext(dir_data, dodgy_file)
            abf = pyabf.ABF(file_path)
            abf.setSweep(row.sweep_number)
            axs[idx].plot(abf.sweepX, abf.sweepY)
            axs[idx].set_title(f"Sweep Number: {row.sweep_number}", fontsize=10)
        plt.tight_layout()
        plt.show()

    
#%% remove files that are duplicates of cells
grouped = df_l23cc_filemeta.groupby('cell_id')
multi_file_cells = grouped.filter(lambda x: len(x) > 1)

      
for cell_id, group in multi_file_cells.groupby('cell_id'):
    n_files = len(group)  # Number of files for the cell_id
    rows, cols = subplot_split(n_files)
    fig, axs = plt.subplots(rows, cols, figsize=(10, 5), squeeze=False)  # Ensure axs is always 2D
    plt.rcParams.update({'font.size': 7})
    fig.suptitle(f"Cell ID: {cell_id}", fontsize=10)

    axs = axs.ravel()  # Flatten axs array for easy indexing
    
    for idx, (index, row) in enumerate(group.iterrows()):
        file = row['file']
        file_path, file_ext = get_filepath_and_ext(dir_data, file)
        if file_ext == '.abf':
            abf = pyabf.ABF(file_path)
            for n_sweep in range(abf.sweepCount):
                abf.setSweep(n_sweep)
                axs[idx].plot(abf.sweepX, abf.sweepY)
            max_sweep_num = abf.sweepCount - 1  # Maximum sweep number for this file
            legend_label = f"Max Sweep: {max_sweep_num}"
            axs[idx].legend([legend_label])
            axs[idx].set_title(f"File: {file}")
    plt.tight_layout()
    plt.show()





#%%
ls_dfs = [df_l23cc_filemeta_hero75x3, df_l23cc_filemeta_hero150, df_l23cc_filemeta_hero250]




for df in ls_dfs[1:2]:
    plot_rheosweeps_perpatient2(dir_data, df, df_l23cc_sweepmeta_inc, min_peak_threshold, fontsize=7)
for df in ls_dfs[1:2]: 
    plot_herosweeps_perpatient2(dir_data, df, df_l23cc_sweepmeta_inc, min_peak_threshold, fontsize=7)


for df in ls_dfs[2:3]:
    plot_IQ_correlation(dir_figs, df, 'isi0_ms', 'Initial ISI (ms)')
    plot_IQ_correlation(dir_figs, df, 'isi_avg_ms', 'Mean ISI (ms)')    
    plot_IQ_correlation(dir_figs, df, 'isi_ai', 'Adaptation Index')
    plot_IQ_correlation(dir_figs, df, 'isi_cv', 'ISI Coefficient of Variance')
    
for df in ls_dfs[2:3]:    
    plot_IQ_correlation(dir_figs, df, 'TDL', 'Total Dendritic Length')
    plot_XY_LMM_correlation(dir_figs, df, 'TDL', 'TDL', 'rheobase', 'Rheobase (pA)')
        
for df in ls_dfs[2:3]:
    plot_IQ_correlation(dir_figs, df, 'rheobase', 'Rheobase (pA)')
    plot_IQ_correlation(dir_figs, df, 'rmp', 'RMP (mV)')
    
for df in ls_dfs[2:3]:
    plot_IQ_correlation(dir_figs, df, 'n_hero_peaks', 'Peaks in Hero Sweep 0.5s')
    plot_IQbin_correlation_bar(dir_figs, df, 'n_hero_IFbursts', 'Number of Bursts >75Hz in Hero Sweep')
    
for df in ls_dfs[2:3]:
    plot_XY_LMM_correlation(dir_figs, df, 'rel_depth_Cx', 'Relative Cortical Depth', 'rheobase', 'Rheobase')
    plot_XY_LMM_correlation(dir_figs, df, 'AgeAtEpilepsyOnset', 'Age at Epilepsy Onset (Years)', 'rheobase', 'Rheobase')
    plot_XY_LMM_correlation(dir_figs, df, 'AgeAtEpilepsyOnset', 'Age at Epilepsy Onset (Years)', 'rheobase', 'Rheobase')
    plot_XY_LMM_correlation(dir_figs, df, 'rel_depth_Cx', 'Relative Cortical Depth', 'rmp', 'RMP')
    
    
for df in ls_dfs[2:3]:  
    plot_IQ_correlation(dir_figs, df, 'AgeAtResection', 'Age at Resection (Years)')
    plot_IQ_correlation(dir_figs, df, 'AgeAtEpilepsyOnset', 'Age at Epilepsy Onset (Years)') 
    plot_IQ_correlation(dir_figs, df, 'YearsEpilepsy', 'Years of Epilepsy')
    plot_IQ_correlation(dir_figs, df, 'SeizuresMonth', 'Seizures per Month')
    
for df in ls_dfs[2:3]: 
    plot_XY_LMM_correlation(dir_figs, df, 'AgeAtEpilepsyOnset', 'Age Onset', 'YearsEpilepsy', 'Year sEpilepsy')









