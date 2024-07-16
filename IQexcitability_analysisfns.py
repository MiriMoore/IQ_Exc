#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:44:07 2024

Functions needed to analyse .abf and .mat files for the IQ vs Intrinsic Excitability project

@author: Miri
"""
#%% import packages

import os
import re
import math
import pyabf
import importlib
import numpy as np
import pandas as pd
from scipy import stats
from mat4py import loadmat
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.signal import butter, filtfilt, find_peaks

import IQexcitability_plotfns
importlib.reload(IQexcitability_plotfns)
from IQexcitability_plotfns import *

#%%

mV_to_V = 1e-3  # Conversion factor millivolts to volts
pA_to_A = 1e-12  # Conversion factor picoamperes to amperes
Ohm_to_MOhm = 1e-6 # Conversion factor Ohms to mega Ohms

#%% process metadata from various data sources into one dataframe for comparisons

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
    # df_eline['patient_id'] = df_eline.apply(extract_patient_id, axis=1, file_column='EphysFileName', cell_column='Slice name')
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
        row_eline = df_eline[(df_eline[name_filecol] == row['file']) | (df_eline[name_slicecol] == row['cell_id'])]
        if not row_eline.empty:
            df_metadata.at[index, 'layer'] = row_eline['Layer DAPI verified'].iloc[0]
            df_metadata.at[index, 'pia_WM'] = row_eline['Pia-WM'].iloc[0]
            df_metadata.at[index, 'pia_soma'] = row_eline['Pia-soma distance Neurolucida verified'].iloc[0]
            df_metadata.at[index, 'rel_depth_Cx'] = row_eline['Pia-soma distance Neurolucida verified'].iloc[0]/row_eline['Pia-WM'].iloc[0]
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
#%% 

def extract_patient_id(row, file_column, cell_column):
    # Try to extract date from the file column
    if pd.notna(row[file_column]):
        # print(type(row[file_column]))
        match = pat_date.search(str(row[file_column]))
        if match:
            return '_'.join(match.groups())
    # If file column is NaN or no date found, try cell column
    if pd.notna(row[cell_column]):
        match = pat_date.search(str(row[cell_column]))
        if match:
            return '_'.join(match.groups())
    # If both are NaN or no date found in either, return NaN
    return np.nan

def separate_metadata_by_protocol(df_metadata, protocolID_strs, exclusion_strs):
    protocolIDs = '|'.join(protocolID_strs)
    exclusions = '|'.join(exclusion_strs)

    # Filter the DataFrame based on the pattern, excluding certain patterns
    df_protocol_filemeta = df_metadata[(df_metadata['protocol'].str.contains(protocolIDs, case=False, na=False)) &
                       (~df_metadata['protocol'].str.contains(exclusions, case=False, na=False))]
    return df_protocol_filemeta
        
def lowpass_filter(sweepY, fs, cutoff=750, order=1):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, sweepY)


def get_filepath_and_ext(dir_data, file_name):
    file_path = os.path.join(dir_data, file_name)
    file_ext = os.path.splitext(file_path)[1]
    return file_path, file_ext

    
def process_abf_sweepEpochs(epoch_info, fs):
    elements = re.findall("Step ([\-0-9.]+) \[([0-9]+):([0-9]+)\]", str(epoch_info))
    for element in elements:
        I_step = float(element[0])
        # Check if I_step is not 0
        if I_step != 0:
            # Convert to pA if necessary
            if -1 < I_step < 1:
                I_step *= 1000
            if I_step == -0:
                I_step = abs(I_step)
            I_onset = int(element[1])
            I_offset = int(element[2])
            return I_step, I_onset, I_offset
    return None, None, None


def process_abf_headerText(header_text):
    """ 
    attempted to extract epoch data using headerText 
    epoch onset and offset are not clear from the data provided
    compare to moving window defined epochs when possible 
    NOT YET DONE
    """
    lines = header_text.split('\n')
    ls_lines = []
    for line in lines:
        if 'fEpochInitLevel' in line:
            ls_lines.append(line)
            fEpochInitLevel = eval(line.split('=')[1].strip())
            I_step0 = [x for x in fEpochInitLevel if x != 0][0]
        if 'fEpochLevelInc' in line:
            ls_lines.append(line)
            fEpochLevelInc = eval(line.split('=')[1].strip())
            I_delta = [x for x in fEpochLevelInc if x != 0][0]
        if 'lEpochInitDuration' in line:
            ls_lines.append(line)
            lEpochInitDuration = eval(line.split('=')[1].strip())
            if len(lEpochInitDuration) > 1: 
                I_dur = ([x for x in lEpochInitDuration if x != 0][1]) 
                I_onset = ([x for x in lEpochInitDuration if x != 0][0] + 1) 
                I_offset = I_onset + I_dur 
            else: 
                I_dur = I_onset = I_offset = None
    return ls_lines, I_step0, I_delta, I_onset, I_offset, I_dur


def process_abf_sweep_with_window(v, t, fs):
    # find window initiation of input I
    window = int(0.005*fs)
    min_initVdelta = 1
    min_endVdelta = 2
    idx_I_onset = None
    idx_I_offset = None
    v = lowpass_filter(v, fs, cutoff=75, order=1)
    v_early = v[(t >= 0.0) & (t < 0.4)]
    # print(v_early)
    idx_v_early_onset = np.where(t >= 0.0)[0][0]
    v_late = v[t >= 0.4]
    # print(v_late)
    idx_v_late_onset = np.where(t >= 0.4)[0][0]
    for i in range(len(v_early) - window):
        if v_early[i] - v_early[i + window] >= min_initVdelta:
            idx_I_onset = i + np.argmax(v_early[i:i+window]) + idx_v_early_onset
            # print(f'POSITIVE Input')
            break
        if v_early[i] - v_early[i + window] <= -min_initVdelta:
            idx_I_onset = i + np.argmin(v_early[i:i+window]) + idx_v_early_onset
            # print(f'NEGATIVE Input')
            break
    rmp = np.mean(v[int(idx_I_onset - (0.02 * fs)):int(idx_I_onset)]) if idx_I_onset is not None else None
    ## choose section where Y returns to base within 0.1ms ##
    for i in range(len(v_late) - window):
        if v_late[i] - v_late[i + window] >= min_endVdelta:
            idx_I_offset = i + np.argmax(v_late[i:i+window]) + idx_v_late_onset
            if idx_I_onset is not None and idx_I_offset > idx_I_onset + (0.5 * fs):
                break
            else:
                break
        if v_late[i] - v_late[i + window] <= -min_endVdelta:
            idx_I_offset = i + np.argmin(v_late[i:i+window]) + idx_v_late_onset
            if idx_I_onset is not None and idx_I_offset > idx_I_onset + (0.5 * fs):
                break
            else:
                break
    return rmp, idx_I_onset, idx_I_offset


def create_df_sweep_metadata(df_file_metadata, dir_data, dir_procdata):
    """
    Processes ephys data from files (.mat or .abf) and compiles metadata for each sweep.

    Iterates over files listed in df_filemetadata, extracting key sweep information such as sampling frequency, 
    stimulus parameters, and peak counts.

    Parameters:
    - df_filemetadata: DataFrame with file metadata.
    - dir_data: Directory path with data files.
    - min_peak_threshold: Threshold for peak detection.

    Returns:
    - DataFrame with compiled sweep metadata.
    """
    ls_sweeps = []
    ls_Iepoch_failed = []
    # Iterate over each file in the file metadata DataFrame
    for file in df_file_metadata['file']:
        print(file)
        # Get the complete file path and extension
        file_path, file_ext = get_filepath_and_ext(dir_data, file)
        file_date = df_file_metadata[df_file_metadata['file'] == file]['patient_id'].iloc[0]
        # Handling for .mat files
        if file_ext == '.mat':
            # Load data from the .mat file
            mat = loadmat(file_path)
            # Extract necessary parameters from the file
            fs = mat['abf']['sampleFrequency']
            I_step0 = mat['abf']['first_pulse']
            I_delta = mat['abf']['pulse_delta']
            I_onset = mat['abf']['pulse_onset'] / 1000 # ms to s
            I_offset = mat['abf']['pulse_offset'] / 1000 # ms to s
            I_dur = mat['abf']['pulse_duration']
            # Transpose sweep data and convert it into a DataFrame
            sweep_data = mat['abf']['data']
            sweeps = np.transpose(np.array(sweep_data))
            df_filesweeps = pd.DataFrame(sweeps)
            # Process each sweep in the file
            for n_sweep in range(len(df_filesweeps)):
                if n_sweep == 0:
                    v = df_filesweeps[n_sweep]
                    rmp = np.mean(v[int(round((I_onset * fs) - (0.02 * fs))):int(round(I_onset * fs))])
                ls_sweeps.append({'patient_id': file_date,
                                  'file': file, 
                                  'sweep_number': n_sweep, 
                                  'fs': fs,
                                  'rmp': rmp,
                                  'idx_I_onset': round(I_onset * fs), 
                                  't_I_onset': I_onset, 
                                  'idx_I_offset': round(I_offset * fs),
                                  't_I_offset': I_offset, 
                                  't_I_dur': I_dur,
                                  'I_step': I_step0 + (I_delta * n_sweep)})
        # Handling for .abf files
        elif file_ext == '.abf':
            # Read .abf file using pyabf library
            abf = pyabf.ABF(file_path)
            ls_Isteps = []
            fs = abf.dataRate
            abf.setSweep(0)
            v = abf.sweepY
            t = abf.sweepX
            patcher_id = df_file_metadata[df_file_metadata['file'] == file]['patcher_ID'].iloc[0]
            rmp, sw_idxIon, sw_idxIoff = process_abf_sweep_with_window(v, t, fs)
            
            if pd.isna(sw_idxIon) or pd.isna(sw_idxIoff):
                ls_Iepoch_failed.append({'file': file, 'filepath': file_path})

            for n_sweep in range(abf.sweepCount):
                abf.setSweep(n_sweep)
                epoch_info = abf.sweepEpochs
                # Extract I_step, I_onset, and I_offset from epoch info
                I_step, _, _ = process_abf_sweepEpochs(epoch_info, fs)
                # Calculate I_dur and store in ls_Isteps
                # Append sweep metadata to the list
                ls_sweeps.append({'patient_id': file_date,
                                  'file': file, 
                                  'sweep_number': n_sweep, 
                                  'fs': fs,
                                  'rmp': rmp,
                                  'idx_I_onset': sw_idxIon,
                                  't_I_onset': sw_idxIon / fs if sw_idxIon is not None else None, 
                                  'idx_I_offset': sw_idxIoff, 
                                  't_I_offset': sw_idxIoff / fs if sw_idxIoff is not None else None, 
                                  'I_step': I_step})
                
            # Process header if I_step values are insufficient
            try:
                if any(x is None or np.isnan(x) for x in ls_Isteps) or max(ls_Isteps, default=float('-inf')) <= 0:
                    abf.setSweep(0)
                    header_text = abf.headerText
                    _, I_step0, I_delta, _, _, _ = process_abf_headerText(header_text)
                    # Update sweeps list with new information
                    for idx, sweep in enumerate(ls_sweeps):
                        if sweep['file'] == file:
                            sweep['I_step'] = I_step0 + (I_delta * sweep['sweep_number'])
            except IndexError:
                print(file, 'HeaderText IndexError')
                continue
    
    # Convert sweeps list to DataFrame and return
    df_sweep_metadata = pd.DataFrame(ls_sweeps)
    # Correct I_step values when values extraction fails
    tolerance = 1e-10  # Adjust the tolerance as necessary
    df_sweep_metadata['I_step'] = df_sweep_metadata['I_step'].apply(
        lambda x: 0 if abs(x + 3.72529e-09) < tolerance or abs(x + 3.72529e-06) < tolerance else (x * 1000 if -1 < x < 1 else x)
    )
    # convert numeric columns to numeric values to avoid issues in furhter analysis
    for col in ['t_I_onset', 't_I_offset', 'idx_I_onset', 'idx_I_offset']:
        df_sweep_metadata[col] = pd.to_numeric(df_sweep_metadata[col], errors='coerce')
        
    df_Iepoch_failed = pd.DataFrame(ls_Iepoch_failed)
    print('Sweep Metadata Created')
    return df_sweep_metadata, df_Iepoch_failed


def update_filemetadata_with_epochs(df_file_metadata, filedata_cols):
    temp = filedata_cols.copy()
    if 'file' in temp.columns:
        temp.set_index('file', inplace=True)
    if 'file' in df_file_metadata.columns and df_file_metadata.index.name != 'file':
        df_file_metadata.set_index('file', inplace=True)
    for column in temp.columns:
        if column in df_file_metadata.columns:
            df_file_metadata[column].update(temp[column])
        else:
            df_file_metadata[column] = temp[column]
    df_file_metadata.reset_index(inplace=True)
    return df_file_metadata

def onclick_select(event):
    if event.inaxes:  # Check if the click was within an axes
        title = event.inaxes.get_title()  # Get the title of the subplot
        ls_duplicate_protocols_for_inclusion.append({'file': title})
        plt.close(event.canvas.figure)
    return ls_duplicate_protocols_for_inclusion
        
def select_file_for_multifile_cells(df_file_metadata, ls_duplicate_protocols_for_inclusion):      
    grouped = df_file_metadata.groupby('cell_id')
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
        fig.canvas.mpl_connect('button_press_event', onclick_select)
        plt.tight_layout()
        plt.ion()
        plt.pause(5)  # 5 seconds to chooose value before new figure is plotted
    return ls_duplicate_protocols_for_inclusion


def exclude_files(dir_data, df_file_metadata, df_sweep_metadata, ls_duplicate_protocols_for_inclusion, ls_files_manually_exc, min_peak_threshold, rmp_range):
    """ currently for this function, ls_duplicate_protocols_for_inclusion must be done separately and manually. 
    this should be folded into this function"""
    
    # Filter cells that are too deep for L2/L3
    ls_files_inc = df_file_metadata[(df_file_metadata['rel_depth_Cx'].isna()) | 
                               (round(df_file_metadata['rel_depth_Cx'], 2) <= 0.50)]['file']

    # Remove spontaneous spiking cells
    ls_spontaneous_spikers = []
    for file, group in df_sweep_metadata.groupby('file'):
        if any((group['I_step'] <= 0) & (group['n_peaks'] > 0)):
            ls_spontaneous_spikers.append(file)

    # Remove files that are duplicate protocols from the same cell
    grouped = df_file_metadata.groupby('cell_id')
    multi_file_cells = grouped.filter(lambda x: len(x) > 1)
    ls_files_exc = []
    for cell_id, group in multi_file_cells.groupby('cell_id'):
        for _, row in group.iterrows():
            file = row['file']
            if file not in ls_duplicate_protocols_for_inclusion: 
                ls_files_exc.append(file)
    
    ls_sweeps_inc = df_sweep_metadata['file'].isin(ls_files_inc)
    df_sweep_metadata_inc = df_sweep_metadata[df_sweep_metadata['file'].isin(ls_files_inc)]
    
    exc_criteria = (df_sweep_metadata_inc['file'].isin(ls_files_manually_exc))| \
            (df_sweep_metadata_inc['file'].isin(ls_spontaneous_spikers)) | \
            (df_sweep_metadata_inc['file'].isin(ls_files_exc)) | \
            (df_sweep_metadata_inc['rmp'] < rmp_range[0]) | (df_sweep_metadata_inc['rmp'] > rmp_range[1]) | \
            (pd.isna(df_sweep_metadata_inc['idx_I_onset'])) 
    
    df_sweep_metadata_inc = df_sweep_metadata_inc[~exc_criteria]
    df_file_metadata_inc = df_file_metadata[df_file_metadata['file'].isin(df_sweep_metadata_inc['file'])]
    df_rheohero_sweeps = get_rheo_and_hero_sweeps_150(dir_data, df_sweep_metadata_inc)
    df_file_metadata_inc = pd.merge(df_file_metadata_inc, df_rheohero_sweeps, on='file', how='left')
    #filedata_cols = df_sweep_metadata_inc[['file', 'idx_I_onset', 'idx_I_offset', 't_I_dur', 'rmp']].drop_duplicates()
    filedata_cols = df_sweep_metadata_inc[['file', 'idx_I_onset', 'idx_I_offset', 'rmp']].drop_duplicates()
    df_file_metadata_inc = update_filemetadata_with_epochs(df_file_metadata_inc, filedata_cols)
    df_file_metadata_inc = calculate_ISI_properties(dir_data, df_file_metadata_inc, min_peak_threshold)

    return df_sweep_metadata_inc, df_file_metadata_inc, df_rheohero_sweeps


def onclick(event):
    global manual_Ionsets, manual_Ioffsets, click_count
    if event.xdata is not None:  # Ensure the click was within the axes
        file = event.canvas.figure.file  # Access the file property from the figure
        if click_count % 2 == 0:  # First click (even count including 0) for onset
            manual_Ionsets.append({'file': file, 'idx_I_onset': round(event.xdata)})
        else:  # Second click (odd count) for offset
            manual_Ioffsets.append({'file': file, 'idx_I_offset': round(event.xdata)})
            plt.close(event.canvas.figure)  # Close the figure after the second click
        click_count += 1

def update_sweepmeta_with_manual_Iepochs(dir_procdata, df_Iepoch_failed, df_sweep_metadata):
    global manual_Ionsets, manual_Ioffsets, click_count
    manual_Ionsets = []
    manual_Ioffsets = []
    df_manual_Iepochs = pd.DataFrame()
    if not df_Iepoch_failed.empty:
        for _, row in df_Iepoch_failed.iterrows():
            click_count = 0  # Reset click count for each figure
            filepath = row['filepath']
            file = row['file']
            abf = pyabf.ABF(filepath)
            abf.setSweep(0)
            fig, ax = plt.subplots()
            fig.file = file  # Set a custom property on the figure object to hold the file name
            ax.plot(abf.sweepY)
            ax.set_title(file)
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.ion()
            plt.pause(5)  # 5 seconds to chooose value before new figure is plotted
    
        # Merge manual_Ionsets and manual_Ioffsets into a single DataFrame
        df_manual_Ionsets = pd.DataFrame(manual_Ionsets)
        df_manual_Ioffsets = pd.DataFrame(manual_Ioffsets)
        df_manual_Iepochs = pd.merge(df_manual_Ionsets, df_manual_Ioffsets, on="file", how="outer")
    
        # Update df_sweep_metadata with the new onset and offset times
        for _, row in df_manual_Iepochs.iterrows():
            file = row['file']
            if file in df_sweep_metadata['file'].values:
                if 'idx_I_onset' in row:
                    df_sweep_metadata.loc[df_sweep_metadata['file'] == file, 'idx_I_onset'] = row['idx_I_onset']
                if 'idx_I_offset' in row:
                    df_sweep_metadata.loc[df_sweep_metadata['file'] == file, 'idx_I_offset'] = row['idx_I_offset']
        
    return df_sweep_metadata, df_manual_Iepochs


def get_iRsag_values(dir_data, df_file_metadata, df_sweep_metadata, mV_to_V, pA_to_A, Ohm_to_MOhm):
    df_sweepset = df_sweep_metadata[df_sweep_metadata['I_step'] < 0]
    filegroup = df_sweepset.groupby('file')
    ls_sweep_metadata_negI = []
    for file, group in filegroup:
        print(file)
        file_path, file_ext = get_filepath_and_ext(dir_data, file)  # Assuming you have this function defined
        abf = pyabf.ABF(file_path)
        for _, row in group.iterrows():
            n_sweep = row['sweep_number']
            i_step = row['I_step']
            idx_Ionset = int(row['idx_I_onset']) if pd.notna(row['idx_I_onset']) else np.nan
            idx_Ioffset = int(row['idx_I_offset']) if pd.notna(row['idx_I_offset']) else np.nan
            fs = int(row['fs'])
            v_rest, idx_vsag, v_sag, v_ss, sag_defl, ss_defl, sag_ratio, iR = [np.nan] * 8
            
            if pd.notna(idx_Ionset) and pd.notna(idx_Ioffset):
                abf.setSweep(n_sweep)
                v, t = abf.sweepY, abf.sweepX
                v_rest = np.mean(v[(idx_Ionset - int(0.04 * fs)) : idx_Ionset])
                idx_vsag = idx_Ionset + np.argmin(v[idx_Ionset : idx_Ionset + int(0.1 * fs)])
                v_sag = v[idx_vsag]
                v_ss = np.mean(v[idx_Ioffset - int(0.1 * fs) : idx_Ioffset])
                sag_defl = v_sag - v_rest
                if pd.isna(sag_defl):
                    plt.figure()
                    plt.plot(t, v)
                    plt.title(file)
                    plt.show()
                ss_defl = v_sag - v_ss
                sag_ratio = ss_defl / sag_defl if sag_defl != 0 else np.nan
                iR_defl = (v_sag - v_rest) * mV_to_V
                iR_inputI = i_step * pA_to_A
                iR = (iR_defl / iR_inputI) * Ohm_to_MOhm
                ls_sweep_metadata_negI.append({
                    'file': file,
                    'sweep_number': n_sweep,
                    'v_rest': v_rest,
                    'idx_vsag': idx_vsag,
                    'v_sag': v_sag,
                    'v_ss': v_ss,
                    'sag_defl': sag_defl,
                    'ss_defl': ss_defl,
                    'sag_ratio': sag_ratio,
                    'iR': iR
                })
            else:
                print('idx_Ionset or idx_Ioffset = NaN')
    df_sweep_metadata_negI = pd.DataFrame(ls_sweep_metadata_negI)
    for idx, updated_row in df_sweep_metadata_negI.iterrows():
        file = updated_row['file']
        sweep_number = updated_row['sweep_number']
        match_condition = (df_sweep_metadata['file'] == file) & (df_sweep_metadata['sweep_number'] == sweep_number)
        for col in ['v_rest', 'idx_vsag', 'v_sag', 'v_ss', 'sag_defl', 'ss_defl', 'sag_ratio', 'iR']:
            df_sweep_metadata.loc[match_condition, col] = updated_row[col]
    
    return df_sweep_metadata, df_sweep_metadata_negI


def calculate_iR(df_file_metadata, df_sweep_metadata, plot=True):
    filegroup = df_sweep_metadata.groupby('file')
    for file, group in filegroup:
        print(file)
        neg_sweepset = group[group['I_step'] < 0]
        if not neg_sweepset.empty:
            v_response = (neg_sweepset['sag_defl'].values) * mV_to_V
            i_input = (neg_sweepset['I_step'].values) * pA_to_A 
            slope, intercept = np.polyfit(i_input, v_response, 1)
            iR = slope * Ohm_to_MOhm
            if plot:
                plt.figure(figsize=(6, 4))
                plt.plot(i_input, v_response, 'o')
                plt.plot(i_input, i_input * slope + intercept, 'r-', label=f'iR = {iR:.2f} MOhm')
                plt.xlabel('Input Current (A)')
                plt.ylabel('Voltage Response (V)')
                plt.legend()
                plt.title(f'Input Resistance for {file}') 
                plt.show()
            df_file_metadata.loc[df_file_metadata['file']==file, 'iR'] = iR
        else:
            print(f'Problem with iR: Negative CC-Steps sweepset empty for {file}')
            
    return df_file_metadata

def calculate_sag(df_file_metadata, df_sweep_metadata, plot=True):
    filegroup = df_sweep_metadata.groupby('file')
    for file, group in filegroup:
        print(file)
        neg_sweepset = group[group['I_step'] < 0]
        if not neg_sweepset.empty:
            sag_ratios = neg_sweepset['sag_ratio'].values
            i_input = neg_sweepset['I_step'].values
            slope, intercept = np.polyfit(i_input, sag_ratio, 1)
            sag_ratio = slope
            if plot:
                plt.figure(figsize=(6, 4))
                plt.plot(i_input, sag_ratios, 'o')
                plt.plot(i_input, i_input * slope + intercept, 'r-', label=f'Sag Ratio = {sag_ratio:.2f}')
                plt.xlabel('Input Current (A)')
                plt.ylabel('Sag Ratio')
                plt.legend()
                plt.title(f'Sag Ratio for {file}') 
                plt.show()
            df_file_metadata.loc[df_file_metadata['file']==file, 'sag_ratio'] = sag_ratio
        else:
            print(f'Problem with SagRatio: Negative CC-Steps sweepset empty for {file}')
            
    return df_file_metadata


def detect_peaks(dir_data, df_sweep_metadata, min_peak_threshold):
    filegroup = df_sweep_metadata.groupby('file')
    for file, group in filegroup:
        file_path, file_ext = get_filepath_and_ext(dir_data, file)
        # Handling for .mat files
        if file_ext == '.mat':
            # Load data from the .mat file
            mat = loadmat(file_path)
            # Transpose sweep data and convert it into a DataFrame
            sweep_data = mat['abf']['data']
            sweeps = np.transpose(np.array(sweep_data))
            df_filesweeps = pd.DataFrame(sweeps)
            for index, row in group.iterrows():
                n_sweep = row['sweep_number']
                fs = row['fs']
                idx_I_onset = int(row['idx_I_onset']) if not np.isnan(row['idx_I_onset']) else None
                idx_I_offset = int(row['idx_I_offset']) if not np.isnan(row['idx_I_offset']) else None
                epoch_100ms = int(0.1 * fs)
                epoch_500ms = int(0.5 * fs)
                v = df_filesweeps.iloc[n_sweep]
                t = np.arange(0, len(v)) / fs
                # Detect peaks in the sweep
                peaks, _ = find_peaks(v[idx_I_onset:idx_I_offset], height=min_peak_threshold) # replace _ with 'properties' for other AP information
                peaks_100ms, _ = find_peaks(v[idx_I_onset:idx_I_onset + epoch_100ms], height=min_peak_threshold)
                peaks_500ms, _ = find_peaks(v[idx_I_onset:idx_I_onset + epoch_500ms], height=min_peak_threshold)
                fpeaks100_Hz = len(peaks_100ms) / 0.1
                fpeaks500_Hz = len(peaks_500ms) / 0.5
                n_peaks = len(peaks)
                df_sweep_metadata.at[index, 'n_peaks'] = n_peaks
                df_sweep_metadata.at[index, 'fpeaksin100ms_Hz'] = fpeaks100_Hz
                df_sweep_metadata.at[index, 'fpeaksin500ms_Hz'] = fpeaks500_Hz
        # Handling for .abf files
        elif file_ext == '.abf':
            abf = pyabf.ABF(file_path)
            for index, row in group.iterrows():
                n_sweep = row['sweep_number']
                fs = row['fs']
                abf.setSweep(n_sweep)
                epoch_100ms = int(0.1 * fs)
                epoch_500ms = int(0.5 * fs)
                v = abf.sweepY
                idx_I_onset = int(row['idx_I_onset']) if not np.isnan(row['idx_I_onset']) else None
                idx_I_offset = int(row['idx_I_offset']) if not np.isnan(row['idx_I_offset']) else None
                peaks, _ = find_peaks(v[idx_I_onset:idx_I_offset], height=min_peak_threshold)
                n_peaks = len(peaks)
                
                peaks_100ms, _ = find_peaks(v[idx_I_onset:idx_I_onset + epoch_100ms], height=min_peak_threshold)
                peaks_500ms, _ = find_peaks(v[idx_I_onset:idx_I_onset + epoch_500ms], height=min_peak_threshold)
                fpeaks100_Hz = len(peaks_100ms) / 0.1
                fpeaks500_Hz = len(peaks_500ms) / 0.5
                
                isis = np.diff(peaks_500ms)
                isis_ms = (isis / fs) * 1000
                IF_bursts = count_IF_bursts(isis_ms)
                
                df_sweep_metadata.at[index, 'n_peaks'] = n_peaks
                df_sweep_metadata.at[index, 'fpeaksin500ms_Hz'] = fpeaks500_Hz
                df_sweep_metadata.at[index, 'fpeaksin100ms_Hz'] = fpeaks100_Hz
                df_sweep_metadata.at[index, 'n_burstsin500ms'] = IF_bursts
    return(df_sweep_metadata)


def update_filemetadata_with_epochs(df_file_metadata, filedata_cols):
    temp = filedata_cols.copy()
    if 'file' in temp.columns:
        temp.set_index('file', inplace=True)
    if 'file' in df_file_metadata.columns and df_file_metadata.index.name != 'file':
        df_file_metadata.set_index('file', inplace=True)
    for column in temp.columns:
        if column in df_file_metadata.columns:
            df_file_metadata[column].update(temp[column])
        else:
            df_file_metadata[column] = temp[column]
    df_file_metadata.reset_index(inplace=True)
    return df_file_metadata

def get_rheo_and_hero_sweeps_150(dir_data, df_sweep_metadata):
    dict_rhsweeps = {}
    for file, group in df_sweep_metadata.groupby('file'):
        rheobase = rheo_sweep = hero = hero_sweep = None
        min_diff = float('inf')
        for idx, row in group.iterrows():
            if row['n_peaks'] > 0 and rheobase is None:
                rheobase = row['I_step']
                rheo_sweep = row['sweep_number']
                break
        # if rheobase is None:
        #     print(f'{file}, no rheobase found')
        #     plot_sweeps_from_file(dir_data, df_sweep_metadata, file)
        if rheobase is not None and rheo_sweep > 0:
            hero_target = rheobase + 150
            hero_min = rheobase + 99
            hero_max = rheobase + 201
            for idx, row in group.iterrows():
                if hero_min <= row['I_step'] <= hero_max:
                    diff = abs(row['I_step'] - hero_target)
                    if diff < min_diff:
                        hero = row['I_step']
                        hero_sweep = row['sweep_number']
                        hero_npeaks = row['n_peaks']
                        min_diff = diff
        else:
            hero = hero_sweep = None
        dict_rhsweeps[file] = {'rheobase': rheobase,
                               'rheobase_sweep': rheo_sweep,
                               'hero': hero,
                               'hero_sweep': hero_sweep,
                               'hero_npeaks': hero_npeaks}
    df_rhsweeps = pd.DataFrame.from_dict(dict_rhsweeps, orient='index').reset_index()
    df_rhsweeps.rename(columns={'index': 'file'}, inplace=True)
    return df_rhsweeps

def get_rheo_and_hero_sweeps_250(df_sweep_metadata):
    dict_rhsweeps = {}
    for file, group in df_sweep_metadata.groupby('file'):
        rheobase = rheo_sweep = hero = hero_sweep = None
        min_diff = float('inf')
        for idx, row in group.iterrows():
            if row['n_peaks'] > 0 and rheobase is None:
                rheobase = row['I_step']
                rheo_sweep = row['sweep_number']
                break
        if rheobase is not None and rheo_sweep > 0:
            hero_target = rheobase + 250
            hero_min = rheobase + 199
            hero_max = rheobase + 301
            for idx, row in group.iterrows():
                if hero_min <= row['I_step'] <= hero_max:
                    diff = abs(row['I_step'] - hero_target)
                    if diff < min_diff:
                        hero = row['I_step']
                        hero_sweep = row['sweep_number']
                        hero_npeaks = row['n_peaks']
                        min_diff = diff
        else:
            hero = hero_sweep = None
        dict_rhsweeps[file] = {'rheobase': rheobase,
                               'rheobase_sweep': rheo_sweep,
                               'hero': hero,
                               'hero_sweep': hero_sweep,
                               'hero_npeaks': hero_npeaks}
    df_rhsweeps = pd.DataFrame.from_dict(dict_rhsweeps, orient='index').reset_index()
    df_rhsweeps.rename(columns={'index': 'file'}, inplace=True)
    return df_rhsweeps


def get_rheo_and_hero_sweeps_75x3(df_sweep_metadata):
    dict_rhsweeps = {}
    for file, group in df_sweep_metadata.groupby('file'):
        rheobase = rheo_sweep = hero = hero_sweep = None
        min_diff = float('inf')
        for idx, row in group.iterrows():
            if row['n_peaks'] > 0 and rheobase is None:
                rheobase = row['I_step']
                rheo_sweep = row['sweep_number']
                break
        if rheobase is not None and rheo_sweep > 0:
            hero_target = rheobase + 75
            hero_min = rheobase + 49
            hero_max = rheobase + 101
            for idx, row in group.iterrows():
                if hero_min <= row['I_step'] <= hero_max and row['n_peaks'] >= 3:
                    diff = abs(row['I_step'] - hero_target)
                    if diff < min_diff:
                        hero = row['I_step']
                        hero_sweep = row['sweep_number']
                        hero_npeaks = row['n_peaks']
                        min_diff = diff
        else:
            hero = hero_sweep = None
        dict_rhsweeps[file] = {'rheobase': rheobase,
                               'rheobase_sweep': rheo_sweep,
                               'hero': hero,
                               'hero_sweep': hero_sweep,
                               'hero_npeaks': hero_npeaks}
    df_rhsweeps = pd.DataFrame.from_dict(dict_rhsweeps, orient='index').reset_index()
    df_rhsweeps.rename(columns={'index': 'file'}, inplace=True)
    return df_rhsweeps


def update_filemetadata_with_epochs(df_file_metadata, filedata_cols):
    # Make a copy of filedata_cols to avoid modifying it in place
    temp = filedata_cols.copy()

    # Set 'file' column as index for filedata_cols_copy DataFrame if 'file' is in its columns
    if 'file' in temp.columns:
        temp.set_index('file', inplace=True)

    # Iterate over columns in filedata_cols_copy
    for column in temp.columns:
        # Check if the column exists in df_file_metadata
        if column in df_file_metadata.columns:
            # Update existing column
            df_file_metadata[column].update(temp[column])
        else:
            # Add new column
            df_file_metadata[column] = temp[column]

    # Reset the index
    df_file_metadata.reset_index(inplace=True)

    return df_file_metadata



def count_doublets_and_bursts(isis_ms):
    # Initialize counts
    doublets = 0
    bursts = 0
    consecutive_count = 0

    for i, isi in enumerate(isis_ms):
        if isi <= 10:
            consecutive_count += 1
            if consecutive_count == 1 and i < len(isis_ms) - 1 and isis_ms[i + 1] > 10:
                # Count as a doublet if the next ISI is greater than 10ms
                doublets += 1
            if consecutive_count >= 2:
                # Count as a burst if at least two consecutive ISIs are <= 10 ms
                bursts += 1
        else:
            consecutive_count = 0

    return doublets, bursts


def count_IF_bursts(isis_ms):
    bursts = 0
    in_burst = False

    for isi in isis_ms:
        if isi <= round((1/75)*1000, 2):  # Threshold for 75Hz firing rate
            if not in_burst:
                bursts += 1
                in_burst = True
        else:
            in_burst = False

    return bursts


def calculate_ISI_properties(dir_data, df_file_metadata, min_peak_threshold):
    for index, row in df_file_metadata.iterrows():
        if pd.notna(row['hero_sweep']):
            file = row['file']
            print(file)
            file_path, file_ext = get_filepath_and_ext(dir_data, file)
            n_hsweep = int(row['hero_sweep'])
            if pd.notna(row['idx_I_onset']):
                idx_I_onset = int(row['idx_I_onset'])
                if file_ext == '.mat':
                    mat = loadmat(file_path)
                    sweep_data = mat['abf']['data']
                    fs = mat['abf']['sampleFrequency']
                    sweeps = np.transpose(np.array(sweep_data))
                    df_filesweeps = pd.DataFrame(sweeps)
                    v = df_filesweeps.iloc[n_hsweep]
                    min_epoch = round(0.5 * fs)
                    # Detect peaks in the sweep
                    peaks, _ = find_peaks(v[idx_I_onset:idx_I_onset+min_epoch], height=min_peak_threshold) 
                    # peaks, _ = find_peaks(v, height=min_peak_threshold) 
            
                elif file_ext == '.abf':
                    # Read .abf file using pyabf library
                    abf = pyabf.ABF(file_path)
                    abf.setSweep(n_hsweep)
                    v = abf.sweepY
                    fs = abf.sampleRate
                    min_epoch = int(round(0.5 * fs))
                    peaks, _ = find_peaks(v[idx_I_onset:idx_I_onset+min_epoch], height=min_peak_threshold) # replace _ with 'properties' for other AP information
                
                isis = np.diff(peaks)
                isis_ms = (isis / fs) * 1000
                if len(isis_ms) >= 1:
                    isi0 = isis_ms[0]
                    isi_avg = np.mean(isis_ms)
                    isi_cv = np.std(isis_ms) / isi_avg if isi_avg != 0 else None
                    isi_ai = np.sum(np.abs(np.diff(isis_ms)) / (isis_ms[:-1] + isis_ms[1:])) / len(isis_ms) if len(isis_ms) > 1 else None
                    doublets, bursts = count_doublets_and_bursts(isis_ms)
                    IF_bursts = count_IF_bursts(isis_ms)
                else:
                    None
                df_file_metadata.at[index, 'isi0_ms'] = isi0 if len(isis) >=2 else np.nan
                df_file_metadata.at[index, 'isi_avg_ms'] = isi_avg
                df_file_metadata.at[index, 'isi_cv'] = isi_cv
                df_file_metadata.at[index, 'isi_ai'] = isi_ai
                df_file_metadata.at[index, 'n_hero_peaks'] = len(peaks)
                df_file_metadata.at[index, 'n_hero_doublets'] = doublets
                df_file_metadata.at[index, 'n_hero_bursts'] = bursts
                df_file_metadata.at[index, 'n_hero_IFbursts'] = IF_bursts
    return df_file_metadata
  

def calculate_FI_slope(df_sweep_metadata, df_file_metadata, plot=False):
    ## find the maximum I_step range that spans all files
    rheobase_per_file = df_sweep_metadata[df_sweep_metadata['n_peaks'] > 0].groupby('file')['I_step'].min()
    max_common_range = (df_sweep_metadata.groupby('file')['I_step'].max() - rheobase_per_file).min()
    print(f'max common range for slope: {max_common_range}')
    # Iterate over each file, applying the common range to determine the analysis subset
    for file, group in df_sweep_metadata.groupby('file'):
        print(file)
        if file in rheobase_per_file:
            max_I_step = rheobase_per_file[file] + 100
            if group['I_step'].max() < max_I_step:
                print(f'does not reach minimum range value, skipping {file}')
                avg_dFdI500 = np.nan
                continue
            slope_sweepset = group[(group['n_peaks'] > 0) & (group['I_step'] <= max_I_step)]
            if slope_sweepset.empty or len(slope_sweepset['I_step']) <= 1:
                print(f'Error file {file}, no dFdI calculated')
                continue 
            fpeaks500 = slope_sweepset['fpeaksin500ms_Hz'].values
            i_input = slope_sweepset['I_step'].values
            dFdI500 = np.gradient(fpeaks500, i_input)
            avg_dFdI500 = np.mean(dFdI500) 
            if plot:
                plt.figure(figsize=(6, 4))
                plt.plot(i_input, fpeaks500, label=f'dFdI 0-500ms:{avg_dFdI500:.2f}', color='b')
                plt.title(file)
                # plt.plot(i_input, fpeaks100, label=f'dFdI 0-100ms:{avg_dFdI100:.2f}', color='r')
                plt.xlim([0, None]) 
                plt.xlabel('Input Current (A)')
                plt.ylabel('Frequency of Peaks (Hz)')
                plt.legend(fontsize = 8)
                plt.show()
            df_file_metadata.loc[df_file_metadata['file']==file, 'dFdI'] = avg_dFdI500
        else:
            print(f"No rheobase found for file {file}, skipping.")
            
            
def calculate_Lburst(df_sweep_metadata, df_file_metadata, plot=False):
    filegroup = df_sweep_metadata.groupby('file')
    for file, group in filegroup:
        print(file)
        pos_sweepset = group[group['n_peaks'] > 0]
        if not pos_sweepset.empty:
            nbursts500 = pos_sweepset['n_burstsin500ms'].values
            i_input = pos_sweepset['I_step'].values
            if len(i_input) > 1:  # Check if i_input is not empty
                dbdI500 = np.gradient(nbursts500, i_input)
                avg_dbdI500 = np.mean(dbdI500)
            else:
                avg_dbdI500 = np.nan
                print(f'Error file {file}, no dFdI calculated')
            
            if plot:
                plt.figure(figsize=(6, 4))
                plt.plot(i_input, nbursts500, label=f'dBdI 0-500ms:{avg_dbdI500:.2f}', color='b')
                plt.xlim([0, None]) 
                plt.xlabel('Input Current (A)')
                plt.ylabel('Number of Bursts')
                plt.legend()
                plt.show()
            df_file_metadata.loc[df_file_metadata['file']==file, 'dBdI_500ms'] = avg_dbdI500


def calculate_Lburst(df_sweep_metadata, df_file_metadata, plot=False):
    filegroup = df_sweep_metadata.groupby('file')
    for file, group in filegroup:
        # Check if any sweep in the file has more than one burst
        if group['n_burstsin500ms'].max() > 1:
            print(file)
            pos_sweepset = group[group['n_peaks'] > 0]
            if not pos_sweepset.empty:
                nbursts500 = pos_sweepset['n_burstsin500ms'].values
                i_input = pos_sweepset['I_step'].values
                if len(i_input) > 1:  # Ensure i_input is not empty
                    dbdI500 = np.gradient(nbursts500, i_input)
                    avg_dbdI500 = np.mean(dbdI500)
                else:
                    avg_dbdI500 = np.nan
                    print(f'Error file {file}, no dBdI calculated')
                
                if plot:
                    plt.figure(figsize=(6, 4))
                    plt.plot(i_input, nbursts500, label=f'dBdI 0-500ms:{avg_dbdI500:.2f}', color='b')
                    plt.xlim([0, None])
                    plt.xlabel('Input Current (A)')
                    plt.ylabel('Number of Bursts')
                    plt.legend()
                    plt.show()
                df_file_metadata.loc[df_file_metadata['file'] == file, 'dBdI_500ms'] = avg_dbdI500
        else:
            df_file_metadata.loc[df_file_metadata['file'] == file, 'dBdI_500ms'] = np.nan
            print(f'File {file} skipped, does not have more than one burst in any sweep.')
            
            
                    
def find_abf_epochs(df_file_metadata, start_index = None, end_index=None):
    ls_abf_epochs = []  # List to store data for DataFrame
    for index, row in df_file_metadata[start_index:end_index].iterrows():
        file = row['file']
        file_path, file_ext = get_filepath_and_ext(dir_data, file)
        plt.figure()
        if file_ext == '.abf':
            print(file)
            abf = pyabf.ABF(file_path)
            fs = abf.dataRate
            abf.setSweep(0)
            v = abf.sweepY
            t = abf.sweepX
            # sweep epoch s
            epoch_info = abf.sweepEpochs
            ei_Istep0, ei_idxIon, ei_idxIoff = process_abf_sweepEpochs(epoch_info, fs)
            # # header text
            header_info = abf.headerText
            ls_lines, ht_Istep0, ht_Idelta, ht_idxIon, ht_idxIoff, ht_I_dur = process_abf_headerText(header_info)
            # sweep window
            _, sw_idxIon, sw_idxIoff = process_abf_sweep_with_window(v, t, fs)
            # save to list
            ls_abf_epochs.append([file, fs, epoch_info, ei_Istep0, ei_idxIon, ei_idxIoff, ls_lines, ht_Istep0, ht_idxIon, ht_idxIoff, sw_idxIon, sw_idxIoff])
            # ls_abf_epochs.append([file, fs, epoch_info, ei_Istep0, ei_idxIon, ei_idxIoff, sw_idxIon, sw_idxIoff])
            
            plt.plot(t, v, color='k')
            if ei_idxIon is not None:
                plt.axvline(t[ei_idxIon], linestyle='-', color='b', label='epoch info')
            if ei_idxIoff is not None:
                plt.axvline(t[ei_idxIoff], linestyle='-', color='b')
            if sw_idxIon is not None:
                plt.axvline(t[sw_idxIon], linestyle='--', color='r', label='sweep window on')
            if sw_idxIoff is not None:
                plt.axvline(t[sw_idxIoff], linestyle='--', color='c', label='sweep window off')
        plt.xlim([0,5])
        plt.title (file)
        plt.legend(fontsize=7)

    # convert to df
    df_abf_epochs = pd.DataFrame(ls_abf_epochs, columns= ['file', 'fs', 'epoch_info', 'ei_Istep0', 'ei_idxIon', 'ei_idxIoff', 'sw_idxIon', 'sw_idxIoff'])
    return df_abf_epochs


def multivariate_regression(df, dependent_var, independent_vars):
    """
    Perform multiple regression analysis.

    Parameters:
    df (DataFrame): The data frame containing the data.
    dependent_var (str): The name of the column to be used as the dependent variable.
    independent_vars (list): A list of column names to be used as independent variables.

    Returns:
    A summary of the regression results.
    """

    # Ensure the dependent variable is in the DataFrame
    if dependent_var not in df.columns:
        raise ValueError(f"Dependent variable '{dependent_var}' not found in DataFrame.")

    # Ensure all independent variables are in the DataFrame
    for var in independent_vars:
        if var not in df.columns:
            raise ValueError(f"Independent variable '{var}' not found in DataFrame.")

    df_cleaned = df.dropna(subset=[dependent_var] + independent_vars)

    # Selecting the dependent and independent variables from the DataFrame
    X = df_cleaned[independent_vars]
    y = df_cleaned[dependent_var]

    # Adding a constant to the model (for the intercept)
    X = sm.add_constant(X)

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Return the summary of the model
    return model.summary()