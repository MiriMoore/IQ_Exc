#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:48:23 2024

Functions needed to plot data from .abf and .mat files for the IQ vs Intrinsic Excitability project

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
import seaborn as sns
from scipy import stats
from mat4py import loadmat
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.signal import butter, filtfilt, find_peaks

import IQexcitability_analysisfns
importlib.reload(IQexcitability_analysisfns)
from IQexcitability_analysisfns import * 

#%%

date = datetime.now().strftime('%Y%m%d')
#%% define plot functions

def close():
    plt.close('all')
    
    
def browse_sweeps(event):
    if event.key == 'right':
        plt.close()
        
        
def subplot_split(num):
    sqrt_num = int(math.sqrt(num))
    rows = sqrt_num if sqrt_num * sqrt_num == num else sqrt_num + 1
    cols = math.ceil(num / rows)
    return rows, cols
    
    
def plot_sweeps_from_file(dir_data, df_sweep_metadata, file_name, ax=None, show_legend=True, fontsize=None):
    file_path, file_ext = get_filepath_and_ext(dir_data, file_name)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    if file_ext == '.mat':
        mat = loadmat(file_path)
        fs = df_sweep_metadata[df_sweep_metadata['file'] == file_name].iloc[0]['fs']
        sweep_data = mat['abf']['data']
        sweeps = np.transpose(np.array(sweep_data))
        df_filesweeps = pd.DataFrame(sweeps)
        idx_I_onset, idx_I_offset = int(df_sweep_metadata[df_sweep_metadata['file'] == file_name].iloc[0]['idx_I_onset']), int(df_sweep_metadata[df_sweep_metadata['file'] == file_name].iloc[0]['idx_I_offset'])
        t_I_onset, t_I_offset = df_sweep_metadata[df_sweep_metadata['file'] == file_name].iloc[0]['t_I_onset'], df_sweep_metadata[df_sweep_metadata['file'] == file_name].iloc[0]['t_I_offset']
        for idx_sweep in range(len(df_filesweeps)):
            v = df_filesweeps.iloc[idx_sweep]
            t = np.arange(0, len(v)) / fs  # Scale time with sampling frequency
            v_Iepoch = v[idx_I_onset : idx_I_offset]
            t_Iepoch = t[idx_I_onset : idx_I_offset]
            # peaks, _ = find_peaks(v_Iepoch, height=min_peak_threshold)
            # Iepoch_peaks = peaks + round(t_I_onset*fs)
            ax.plot(t, v, alpha=0.8, linewidth=0.5)
            # ax.plot(t[Iepoch_peaks], v[Iepoch_peaks], 'x', color='r', label=f"{idx_sweep}: {len(peaks)}pks")
            # if idx_sweep == len(df_filesweeps) - 1:
            #     ax.axvline(t[round(t_I_onset*fs)], linestyle='--', color='b')
            #     ax.axvline(t[round(t_I_offset*fs)], linestyle='--', color='b')
    elif file_ext == '.abf':
        abf = pyabf.ABF(file_path)
        fs = df_sweep_metadata[df_sweep_metadata['file'] == file_name].iloc[0]['fs']
        # idx_I_onset, idx_I_offset = int(df_sweep_metadata[df_sweep_metadata['file'] == file_name].iloc[0]['idx_I_onset']), int(df_sweep_metadata[df_sweep_metadata['file'] == file_name].iloc[0]['idx_I_offset'])
        for idx, row in df_sweep_metadata[df_sweep_metadata['file'] == file_name].iterrows():
            abf.setSweep(row['sweep_number'])
            v = abf.sweepY
            t = abf.sweepX
            # v_Iepoch = v[idx_I_onset:idx_I_offset]
            # t_Iepoch = t[idx_I_onset:idx_I_offset]
            # peaks, _ = find_peaks(v_Iepoch, height=min_peak_threshold)
            # Iepoch_peaks = peaks + idx_I_onset
            ax.plot(t, v, alpha=0.8, linewidth=0.5)
            # ax.plot(t[Iepoch_peaks], v[Iepoch_peaks], 'x', color='r', label=f"{row['sweep_number']}: {row['I_step']}pA, {len(peaks)}pks")
            # if idx == len(df_sweep_metadata[df_sweep_metadata['file'] == file_name]) -1:
            #     ax.axvline(t[idx_I_onset], linestyle='--', color='b')
            #     ax.axvline(t[idx_I_offset], linestyle='--', color='b')
    if fontsize is not None:
        ax.set_xlabel("Time (s)", fontsize=fontsize)
        ax.set_ylabel("Voltage (mV)", fontsize=fontsize)
        ax.set_title(file_name, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
    if show_legend:
            ax.legend(fontsize=7)            
        

def plot_sweeps_perfile_perpatient(dir_data, df_sweep_metadata, fontsize=7): 
    for date, date_group in df_sweep_metadata.groupby('patient_id'):
        num_files = len(date_group['file'].unique())
        rows, cols = subplot_split(num_files)
        fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
        fig.suptitle(f"Patient: {date}")
        axs = axs.flatten() if num_files > 1 else [axs]
        # Group by 'file' within each patient_id
        for idx, (file, file_group) in enumerate(date_group.groupby('file')):
            ax = axs[idx]
            plot_sweeps_from_file(dir_data, file_group, file, ax, show_legend=False, fontsize=fontsize)
        for idx in range(num_files, rows * cols):
            axs[idx].axis('off')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
                    
                    
def plot_herosweeps_perpatient(dir_data, df_file_metadata, df_sweep_metadata, min_peak_threshold, fontsize=7):
    for date, date_group in df_sweep_metadata.groupby('patient_id'):
        num_files = len(date_group['file'].unique())
        rows, cols = subplot_split(num_files)
        fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
        fig.suptitle(f"Patient: {date}")
        axs = axs.flatten() if num_files > 1 else [axs]
        for idx, (file, file_group) in enumerate(date_group.groupby('file')):
            file_path, file_ext = get_filepath_and_ext(dir_data, file)
            file_meta = df_file_metadata[df_file_metadata['file'] == file]
            if not pd.isna(file_meta['hero_sweep'].iloc[0]):
                n_sweep = int(file_meta['hero_sweep'].iloc[0])
                ax = axs[idx]
                if file_ext == '.abf':
                    abf = pyabf.ABF(file_path)
                    abf.setSweep(n_sweep)
                    v = abf.sweepY
                    t = abf.sweepX
                    ax.plot(t, v)
                    ax.set_xlim([0, 3])
                    ax.tick_params(axis='both', labelsize=fontsize) 
                    ax.set_title(f"File: {file}", fontsize=fontsize)
                    ax.set_xlabel("Time (s)", fontsize=fontsize)
                    ax.set_ylabel("Voltage (mV)", fontsize=fontsize)
        plt.tight_layout()
        plt.show()


def plot_rheosweeps_perpatient2(dir_data, df_file_metadata, df_sweep_metadata, min_peak_threshold, fontsize=7):
    for date, date_group in df_sweep_metadata.groupby('patient_id'):
        # Initialize figure
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f"Patient: {date}", fontsize=fontsize)

        for file, file_group in date_group.groupby('file'):
            file_path, file_ext = get_filepath_and_ext(dir_data, file)
            file_meta = df_file_metadata[df_file_metadata['file'] == file]
            
            if not file_meta.empty and not pd.isna(file_meta['rheobase_sweep'].iloc[0]):
                n_sweep = int(file_meta['rheobase_sweep'].iloc[0])
                if file_ext == '.abf':
                    abf = pyabf.ABF(file_path)
                    abf.setSweep(n_sweep)
                    v = abf.sweepY
                    t = abf.sweepX
                    ax.plot(t, v, label=f"File: {file}")  # Add a label for each file
        
        # Only call these once since we're only using one set of axes
        ax.set_xlim([0, 3])
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.set_xlabel("Time (s)", fontsize=fontsize)
        ax.set_ylabel("Voltage (mV)", fontsize=fontsize)
        ax.legend(fontsize=fontsize)  # Display the legend
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
        plt.show()
        
        
def plot_herosweeps_perpatient2(dir_data, df_file_metadata, df_sweep_metadata, min_peak_threshold, fontsize=7):
    for date, date_group in df_sweep_metadata.groupby('patient_id'):
        # Initialize figure
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f"Patient: {date}", fontsize=fontsize)

        for file, file_group in date_group.groupby('file'):
            file_path, file_ext = get_filepath_and_ext(dir_data, file)
            file_meta = df_file_metadata[df_file_metadata['file'] == file]
            
            if not file_meta.empty and not pd.isna(file_meta['hero_sweep'].iloc[0]):
                n_sweep = int(file_meta['hero_sweep'].iloc[0])
                if file_ext == '.abf':
                    abf = pyabf.ABF(file_path)
                    abf.setSweep(n_sweep)
                    v = abf.sweepY
                    t = abf.sweepX
                    ax.plot(t, v, label=f"File: {file}")  # Add a label for each file
        
        # Only call these once since we're only using one set of axes
        ax.set_xlim([0, 3])
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.set_xlabel("Time (s)", fontsize=fontsize)
        ax.set_ylabel("Voltage (mV)", fontsize=fontsize)
        ax.legend(fontsize=fontsize)  # Display the legend
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
        plt.show()

  
def plot_XY_LMM_correlation(dir_figs, df_file_metadata, Xcorrelate, Xcorrelate_label, Ycorrelate, Ycorrelate_label, box_width=0.005):
    df_file_metadata = df_file_metadata.dropna(subset=[Ycorrelate, Xcorrelate])
    df_file_metadata[Xcorrelate] = pd.to_numeric(df_file_metadata[Xcorrelate], errors='coerce')

    plt.figure(figsize=(20, 10))

    # Scatter plot colored by 'patient_id'
    unique_pIDs = df_file_metadata['patient_id'].unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_pIDs)))
    for pID, color in zip(unique_pIDs, colors):
        subset = df_file_metadata[df_file_metadata['patient_id'] == pID]
        x = np.random.normal(subset[Xcorrelate], 0.04)  # Adjust spread of points
        plt.scatter(x, subset[Ycorrelate], color=color, alpha=0.7, s=20, label=pID)

    plt.xlabel(Xcorrelate_label)
    plt.ylabel(Ycorrelate_label)
    # plt.legend(title='patient_id', bbox_to_anchor=(1.05, 1), loc='upper left')

    # LMM fitting
    md = smf.mixedlm(f"{Ycorrelate} ~ {Xcorrelate}", df_file_metadata, groups=df_file_metadata["patient_id"])
    mdf = md.fit()
    print(mdf.summary())
    p_value = mdf.pvalues[Xcorrelate]
    predictions = mdf.predict(df_file_metadata.sort_values(Xcorrelate))
    plt.plot(df_file_metadata[Xcorrelate].sort_values(), predictions, color='red', linestyle='--', linewidth=1.5)

    # Annotate p-value
    if p_value < 0.00005:
        plt.annotate(f'p = {p_value:.2e}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.0005:
        plt.annotate(f'p = {round(p_value, 5)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.005:
        plt.annotate(f'p = {round(p_value, 4)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.05:
        plt.annotate(f'p = {round(p_value, 3)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    else:
        plt.annotate(f'p={round(p_value, 2)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    print(p_value)
    plt.show()
    plt.savefig(os.path.join(dir_figs, f'{Ycorrelate}_{Xcorrelate}_LMM_{date}.png'))
    plt.savefig(os.path.join(dir_figs, f'{Ycorrelate}_{Xcorrelate}_LMM_{date}.eps'))
              

def plot_XY_LMM_correlation_boxplot(dir_figs, df_file_metadata, Xcorrelate, Xcorrelate_label, Ycorrelate, Ycorrelate_label, df_code, box_width=0.005):
    df_file_metadata = df_file_metadata.dropna(subset=[Ycorrelate, Xcorrelate])
    df_file_metadata[Xcorrelate] = pd.to_numeric(df_file_metadata[Xcorrelate], errors = 'coerce')
    # Collect boxplot data by patient and calculate positions with jitter
    grouped = df_file_metadata.groupby([Xcorrelate, 'patient_id']) 
    box_data = []
    positions = []
    for (xcorr, _), group in grouped:
        box_data.append(group[Ycorrelate].values)
        # jitter = np.random.uniform(-0.3, 0.3)
        positions.append(xcorr)
    
    plt.figure(figsize=(8, 10))
    bp = plt.boxplot(box_data, positions=positions, vert=1, showfliers=False, patch_artist=True, widths=box_width)
    # Scatter individual data points
    for i, y in enumerate(box_data):
        x = np.random.normal(positions[i], 0.04, size=len(y))  # Adjust spread of points as needed
        plt.scatter(x, y, color='k', alpha=0.7, s=20)
    # Set color and other properties for each box
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor('lightgrey')
        patch.set_alpha(0.2)
    plt.xlabel(Xcorrelate_label)
    plt.ylabel(Ycorrelate_label)
    # Set x-ticks and annotations
    xticks = np.linspace(min(positions), max(positions), num=5)
    xticks = np.round(xticks, 1)
    plt.xticks(ticks=xticks, labels=[f"{x:.1f}" for x in xticks])
    # fit LMM
    md = smf.mixedlm(f"{Ycorrelate} ~ {Xcorrelate}", df_file_metadata, groups = df_file_metadata["patient_id"])
    mdf=md.fit()
    print(mdf.summary())
    p_value = mdf.pvalues[Xcorrelate]
    predictions = mdf.predict(df_file_metadata.sort_values(Xcorrelate))
    plt.plot(df_file_metadata[Xcorrelate].sort_values(), predictions, color='red', linestyle='--', linewidth=1.5)
    plt.xlim([min(positions) - box_width, max(positions) + box_width])
    # Annotate p-value
    if p_value < 0.0005:
        plt.annotate(f'p = {round(p_value, 5)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.005:
        plt.annotate(f'p = {round(p_value, 4)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.05:
        plt.annotate(f'p = {round(p_value, 3)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    else:
        plt.annotate(f'p={round(p_value, 2)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    print(p_value)
    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, f'{Ycorrelate}_{Xcorrelate}_bpLMM_{df_code}_{date}.png'))
    plt.savefig(os.path.join(dir_figs, f'{Ycorrelate}_{Xcorrelate}_bpLMM_{df_code}_{date}.eps'))


def plot_XY_LMM_MEGcorrelation_boxplot(dir_figs, df_file_metadata, Xcorrelate, Xcorrelate_label, Ycorrelate, Ycorrelate_label, df_code, box_width=0.005):
    df_file_metadata = df_file_metadata.dropna(subset=[Ycorrelate, Xcorrelate])
    df_file_metadata[Xcorrelate] = pd.to_numeric(df_file_metadata[Xcorrelate], errors = 'coerce')
    # Collect boxplot data by patient and calculate positions with jitter
    grouped = df_file_metadata.groupby([Xcorrelate, 'patient_id']) 
    box_data = []
    positions = []
    for (xcorr, _), group in grouped:
        box_data.append(group[Ycorrelate].values)
        # jitter = np.random.uniform(-0.3, 0.3)
        positions.append(xcorr)
    
    plt.figure(figsize=(8, 10))
    bp = plt.boxplot(box_data, positions=positions, vert=1, showfliers=False, patch_artist=True, widths=box_width)
    # Scatter individual data points
    for i, y in enumerate(box_data):
        x = np.random.normal(positions[i], 0.0004, size=len(y))  # Adjust spread of points as needed
        plt.scatter(x, y, color='k', alpha=0.7, s=20)
    # Set color and other properties for each box
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor('lightgrey')
        patch.set_alpha(0.2)
    plt.xlabel(Xcorrelate_label)
    plt.ylabel(Ycorrelate_label)
    # Set x-ticks and annotations
    xticks = np.linspace(min(positions), max(positions), num=5)
    xticks = np.round(xticks, 1)
    plt.xticks(ticks=xticks, labels=[f"{x:.1f}" for x in xticks])
    # fit LMM
    md = smf.mixedlm(f"{Ycorrelate} ~ {Xcorrelate}", df_file_metadata, groups = df_file_metadata["patient_id"])
    mdf=md.fit()
    print(mdf.summary())
    p_value = mdf.pvalues[Xcorrelate]
    predictions = mdf.predict(df_file_metadata.sort_values(Xcorrelate))
    plt.plot(df_file_metadata[Xcorrelate].sort_values(), predictions, color='red', linestyle='--', linewidth=1.5)
    plt.xlim([min(positions) - box_width, max(positions) + box_width])
    # Annotate p-value
    if p_value < 0.0005:
        plt.annotate(f'p = {round(p_value, 5)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.005:
        plt.annotate(f'p = {round(p_value, 4)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.05:
        plt.annotate(f'p = {round(p_value, 3)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    else:
        plt.annotate(f'p={round(p_value, 2)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    print(p_value)
    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, f'{Ycorrelate}_{Xcorrelate}_bpLMM_{df_code}_{date}.png'))
    plt.savefig(os.path.join(dir_figs, f'{Ycorrelate}_{Xcorrelate}_bpLMM_{df_code}_{date}.eps'))
    
    
                
def plot_IQ_correlation(dir_figs, df_file_metadata, df_code, correlate, correlate_label):
    df_file_metadata = df_file_metadata.dropna(subset=[correlate, 'NewTIQ'])
    df_file_metadata['NewTIQ'] = pd.to_numeric(df_file_metadata['NewTIQ'], errors = 'coerce')
    # Collect boxplot data by patient and calculate positions with jitter
    grouped = df_file_metadata.groupby(['NewTIQ', 'patient_id']) 
    box_data = []
    positions = []
    for (newtiq, _), group in grouped:
        box_data.append(group[correlate].values)
        jitter = np.random.uniform(-0.3, 0.3)
        positions.append(newtiq + jitter)
    
    plt.figure(figsize=(20, 10))
    bp = plt.boxplot(box_data, positions=positions, vert=1, showfliers=False, patch_artist=True, widths=0.4)
    # Scatter individual data points
    for i, y in enumerate(box_data):
        x = np.random.normal(positions[i], 0.04, size=len(y))  # Adjust spread of points as needed
        plt.scatter(x, y, color='k', alpha=0.7, s=20)
    # Set color and other properties for each box
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor('lightgrey')
        patch.set_alpha(0.2)
    for whisker in bp['whiskers']:
        whisker.set_color('grey')
    for cap in bp['caps']:
        cap.set_color('grey')
    plt.xlabel('Patient IQ')
    plt.ylabel(correlate_label)
    # Set x-ticks and annotations
    xticks = np.arange(min(positions), max(positions) + 1, 10)  # Adjust step size as needed
    plt.xticks(ticks=xticks, labels=xticks.astype(int))
    # fit LMM
    md = smf.mixedlm(f"{correlate} ~ NewTIQ", df_file_metadata, groups = df_file_metadata["patient_id"])
    mdf=md.fit()
    print(mdf.summary())
    p_value = mdf.pvalues['NewTIQ']
    predictions = mdf.predict(df_file_metadata.sort_values('NewTIQ'))
    plt.plot(df_file_metadata['NewTIQ'].sort_values(), predictions, color='red', linestyle='--', linewidth=1.5)
    
    # Annotate p-value
    if p_value < 0.0005:
        plt.annotate(f'p = {round(p_value, 5)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.005:
        plt.annotate(f'p = {round(p_value, 4)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.05:
        plt.annotate(f'p = {round(p_value, 3)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    else:
        plt.annotate(f'p={round(p_value, 2)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    plt.show()
    plt.savefig(os.path.join(dir_figs, f'{correlate}_IQ_LMM_{df_code}_{date}.png'))
    plt.savefig(os.path.join(dir_figs, f'{correlate}_IQ_LMM_{df_code}_{date}.eps'))
    

def plot_IQ_correlation_bar(dir_figs, df_file_metadata, correlate, correlate_label):
    # Drop rows with NaN in 'NewTIQ' or the correlated variable
    df_file_metadata = df_file_metadata.dropna(subset=[correlate, 'NewTIQ'])
    df_file_metadata['NewTIQ'] = pd.to_numeric(df_file_metadata['NewTIQ'], errors='coerce')

    # Group by NewTIQ and calculate mean of correlate
    grouped = df_file_metadata.groupby('NewTIQ')[correlate].mean()

    # Plotting
    plt.figure(figsize=(15, 10))
    grouped.plot(kind='bar', color='lightgrey', edgecolor='black')
    plt.xlabel('Patient IQ')
    plt.ylabel(correlate_label)

    plt.show()
    plt.savefig(os.path.join(dir_figs, f'{correlate}_IQ_Bar_{date}.png'))
    plt.savefig(os.path.join(dir_figs, f'{correlate}_IQ_Bar_{date}.eps'))
    
    
def plot_IQbin_correlation_bar(dir_figs, df_file_metadata, correlate, correlate_label):
    # Drop rows with NaN in 'NewTIQ' or the correlated variable
    df_file_metadata = df_file_metadata.dropna(subset=[correlate, 'NewTIQ'])
    df_file_metadata['NewTIQ'] = pd.to_numeric(df_file_metadata['NewTIQ'], errors='coerce')

    # Define IQ bins
    bins = [0, 70, 79, 89, 109, 119, 129, np.inf]
    labels = ['<70', '70-79', '80-89', '90-109', '110-119', '120-129', '130+']
    #Below 70: Intellectual disability 70-79: Borderline 80-89: Low average 90-109: Average 110-119: High average 120-129: Superior 130 and above: Very superior or gifted

    df_file_metadata['IQ_Category'] = pd.cut(df_file_metadata['NewTIQ'], bins=bins, labels=labels, right=False)

    # Group by IQ category and calculate mean of correlate
    grouped = df_file_metadata.groupby('IQ_Category')[correlate].mean()
   
    # Plotting
    plt.figure(figsize=(15, 10))
    grouped.plot(kind='bar', color='lightgrey', edgecolor='black')
    plt.xlabel('IQ Category')
    plt.ylabel(correlate_label)
    
    plt.show()
    plt.savefig(os.path.join(dir_figs, f'{correlate}_IQbin_barchart_{date}.png'))
    plt.savefig(os.path.join(dir_figs, f'{correlate}_IQbin_barchart_{date}.eps'))
    
    
def plot_IQbin_patient_count(dir_figs, df_file_metadata):
    # Drop rows with NaN in 'NewTIQ'
    df_file_metadata = df_file_metadata.dropna(subset=['NewTIQ'])
    df_file_metadata['NewTIQ'] = pd.to_numeric(df_file_metadata['NewTIQ'], errors='coerce')

    # Define IQ bins
    bins = [0, 70, 79, 89, 109, 119, 129, np.inf]
    labels = ['<70', '70-79', '80-89', '90-109', '110-119', '120-129']

    df_file_metadata['IQ_Category'] = pd.cut(df_file_metadata['NewTIQ'], bins=bins, labels=labels, right=False)

    # Group by IQ category and count unique dates
    patient_counts = df_file_metadata.groupby('IQ_Category')['patient_id'].nunique()

    # Plotting
    plt.figure(figsize=(15, 10))
    patient_counts.plot(kind='bar', color='lightgrey', edgecolor='black')
    plt.xlabel('IQ Category')
    plt.ylabel('Number of Unique Patients')

    plt.show()
    plt.savefig(os.path.join(dir_figs, 'IQbin_patient_count_barchart_{date}.png'))
    plt.savefig(os.path.join(dir_figs, 'IQbin_patient_count_barchart_{date}.eps'))
    

def plot_IQ_correlation_averaged_LMM(dir_figs, df_file_metadata, df_code, correlate, correlate_label):
    # Drop rows with NaN values in 'correlate' and 'NewTIQ' columns
    df_file_metadata = df_file_metadata.dropna(subset=[correlate, 'NewTIQ'])
    df_file_metadata['NewTIQ'] = pd.to_numeric(df_file_metadata['NewTIQ'], errors='coerce')
    
    # Group by 'patient_id' and calculate mean and standard error of 'correlate' for each patient
    grouped = df_file_metadata.groupby('patient_id')[correlate].agg(['mean', 'sem']).reset_index()
    newtiq_values = df_file_metadata.groupby('patient_id')['NewTIQ'].first().values
    
    # Plotting
    plt.figure(figsize=(15, 10))
    plt.errorbar(newtiq_values, grouped['mean'], yerr=grouped['sem'], fmt='o', color='k', alpha=0.7, markersize=5, capsize=5)
    
    # fit LMM
    md = smf.mixedlm(f"{correlate} ~ NewTIQ", df_file_metadata, groups=df_file_metadata["patient_id"])
    mdf = md.fit()
    print(mdf.summary())
    
    p_value = mdf.pvalues['NewTIQ']
    predictions = mdf.predict(df_file_metadata.sort_values('NewTIQ'))
    plt.plot(df_file_metadata['NewTIQ'].sort_values(), predictions, color='red', linestyle='--', linewidth=1.5)
    
    # Annotate p-value
    if p_value < 0.0005:
        plt.annotate('p<0.0005', xy=(0.9, 0.9), xycoords='axes fraction')
    elif p_value < 0.005:
        plt.annotate('p<0.005', xy=(0.9, 0.9), xycoords='axes fraction')
    elif p_value < 0.05:
        plt.annotate('p<0.05', xy=(0.85, 0.9), xycoords='axes fraction')
    else:
        plt.annotate(f'p={round(p_value, 2)}', xy=(0.85, 0.9), xycoords='axes fraction')

    plt.xlabel('Patient IQ')
    plt.ylabel(correlate_label)
    plt.title(f'{df_code} IQ-{correlate.capitalize()} Correlation with Linear Mixed Model')
    plt.savefig(os.path.join(dir_figs, f'{correlate}_IQ_LMM_{df_code}_averaged_{date}.png'))
    plt.savefig(os.path.join(dir_figs, f'{correlate}_IQ_LMM_{df_code}_averaged_{date}.eps'))
    plt.show()
    

def plot_IQ_correlation_averaged_LR(dir_figs, df_file_metadata, df_code, correlate, correlate_label):
    """
    Plot correlation with linear regression using Seaborn, for patient averages.
    
    Parameters:
    - dir_figs: Directory where the figure will be saved.
    - df_file_metadata: DataFrame containing the data.
    - df_code: A unique code or identifier for the dataset.
    - correlate: Name of the column in df_file_metadata to correlate with IQ.
    - correlate_label: The label to use for the correlate on the y-axis.
    """
    
    # Drop rows with NaN values in 'correlate' and 'NewTIQ' columns
    df_clean = df_file_metadata.dropna(subset=[correlate, 'NewTIQ'])
    # Convert 'NewTIQ' to numeric, coercing errors
    df_clean['NewTIQ'] = pd.to_numeric(df_clean['NewTIQ'], errors='coerce')
    # Group by 'patient_id' and calculate mean and SEM of 'correlate' for each patient
    df_grouped = df_clean.groupby('patient_id').agg({
        'NewTIQ': 'first',  # Assuming each patient has only one NewTIQ value
        correlate: ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]  # SEM for correlate
    }).reset_index()
    # Flatten MultiIndex columns
    df_grouped.columns = ['patient_id', 'NewTIQ', f'{correlate}_mean', f'{correlate}_SEM']

    # Calculate linear regression using statsmodels for detailed statistics
    X = sm.add_constant(df_grouped['NewTIQ'])  # adding a constant
    y = df_grouped[f'{correlate}_mean']
    model = sm.OLS(y, X).fit()
    intercept, slope = model.params
    r_squared = model.rsquared
    p_value = model.pvalues[1]  # p-value for slope
    
    # Calculate the correlation coefficient (r-value)
    r_value, _ = stats.pearsonr(df_grouped['NewTIQ'], df_grouped[f'{correlate}_mean'])
    
    # Plotting
    plt.figure(figsize=(15, 10))
    ax = sns.scatterplot(x='NewTIQ', y=f'{correlate}_mean', data=df_grouped, color ='k')
    # Error bars for correlate
    for _, row in df_grouped.iterrows():
        plt.errorbar(row['NewTIQ'], row[f'{correlate}_mean'], yerr=row[f'{correlate}_SEM'], fmt='none', ecolor='gray', alpha=0.7)
    # Linear regression with seaborn
    sns.regplot(x='NewTIQ', y=f'{correlate}_mean', data=df_grouped, ci=95, scatter=False, ax=ax, color='blue', line_kws={'lw': 1, 'ls': '--'}, scatter_kws={'s': 40, 'alpha': 1})
    # Annotate p-value
    if p_value < 0.0005:
        plt.annotate(f'R = {r_value:.2f}\nR² = {r_squared:.2f}\np = {round(p_value, 5)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.005:
        plt.annotate(f'R = {r_value:.2f}\nR² = {r_squared:.2f}\np = {round(p_value, 4)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.05:
        plt.annotate(f'R = {r_value:.2f}\nR² = {r_squared:.2f}\np = {round(p_value, 3)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    else:
        plt.annotate(f'R = {r_value:.2f}\nR² = {r_squared:.2f}\np={round(p_value, 2)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', bbox=dict(boxstyle="round", alpha=0.5, color='w'))

    
    # Setting plot labels and title
    plt.xlabel('Patient IQ')
    plt.ylabel(f'Mean {correlate_label}')
    # plt.title(f'{df_code} IQ-{correlate} Patient Averages')

    # Ensure the directory exists
    os.makedirs(dir_figs, exist_ok=True)
    # Save the plot
    fig_path = os.path.join(dir_figs, f'{correlate}_IQ_LR_{df_code}_averaged_with_SEM_and_stats_{date}.png')
    plt.savefig(fig_path)
    
    # Show the plot
    plt.show()
    

def plot_IQ_correlation_weighted_LR(dir_figs, df_file_metadata, df_code, correlate, correlate_label):
    """
    Plot correlation with linear regression using Seaborn, for patient averages, weighted by the number of files per patient.
    """
    
    # Drop rows with NaN values in 'correlate' and 'NewTIQ' columns
    df_clean = df_file_metadata.dropna(subset=[correlate, 'NewTIQ'])
    df_clean['NewTIQ'] = pd.to_numeric(df_clean['NewTIQ'], errors='coerce')
    
    # Calculate the number of files per patient and the corresponding weights
    files_count = df_clean.groupby('patient_id').size()
    weights = 1 / files_count
    weights.name = 'weight'
    
    # Join the weights back to the original DataFrame
    df_weighted = df_clean.join(weights, on='patient_id')
    
    # Group by 'patient_id' and calculate weighted mean of 'correlate'
    df_grouped = df_weighted.groupby('patient_id').apply(
        lambda x: np.average(x[correlate], weights=x['weight'])
    ).reset_index(name=f'{correlate}_mean')
    
    # Join back the NewTIQ and weights for plotting and regression
    df_grouped = df_grouped.join(df_weighted[['patient_id', 'NewTIQ', 'weight']].drop_duplicates().set_index('patient_id'), on='patient_id')
    
    # Plotting
    plt.figure(figsize=(15, 10))
    ax = sns.scatterplot(x='NewTIQ', y=f'{correlate}_mean', data=df_grouped, size='weight', alpha=0.7)
    
    # Fit weighted linear regression using statsmodels
    X = sm.add_constant(df_grouped['NewTIQ'])
    y = df_grouped[f'{correlate}_mean']
    wls_model = sm.WLS(y, X, weights=df_grouped['weight']).fit()
    predictions = wls_model.predict(X)
    
    # Plot regression line
    plt.plot(df_grouped['NewTIQ'], predictions, color='blue', linestyle='--', linewidth=1.5)
    
    # Calculate the correlation coefficient (r-value) and p-value for plotting
    r_value, p_value = stats.pearsonr(df_grouped['NewTIQ'], df_grouped[f'{correlate}_mean'])
    
    # Annotate plot with r-value and p-value
    plt.annotate(f'R = {r_value:.2f}\np = {p_value:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', bbox=dict(boxstyle="round", alpha=0.5, color='w'))

    plt.xlabel('Patient IQ')
    plt.ylabel(f'Weighted Mean {correlate_label}')
    
    # Ensure the directory exists
    os.makedirs(dir_figs, exist_ok=True)
    
    # Save the plot
    fig_path = os.path.join(dir_figs, f'Weighted_{correlate}_IQ_LR_{df_code}_{date}.png')
    plt.savefig(fig_path)
    
    plt.show()
    
    
def plot_IQ_disease_correlation(dir_figs, df_file_metadata, df_code, correlate, correlate_label):
    """
    Plot correlation between IQ and the specified disease correlate for each patient, considering the first record for each.
    
    Parameters:
    - dir_figs: Directory where the figure will be saved.
    - df_file_metadata: DataFrame containing the data.
    - df_code: A unique code or identifier for the dataset.
    - correlate: Name of the column in df_file_metadata for the disease correlate.
    - correlate_label: The label to use for the disease correlate on the y-axis.
    """
    # Filter out rows with NaN in 'NewTIQ' and the correlate column, then select the first entry for each patient
    df_filtered = df_file_metadata.dropna(subset=['NewTIQ', correlate])
    df_first_rows = df_filtered.groupby('patient_id').first().reset_index()

    # Ensure numeric types for analysis
    df_first_rows['NewTIQ'] = pd.to_numeric(df_first_rows['NewTIQ'], errors='coerce')
    df_first_rows[correlate] = pd.to_numeric(df_first_rows[correlate], errors='coerce')

    # Plotting
    plt.figure(figsize=(20, 10))
    ax = sns.scatterplot(x='NewTIQ', y=correlate, data=df_first_rows, color='k', alpha=0.7)

    # Fit linear regression using statsmodels
    X = sm.add_constant(df_first_rows['NewTIQ'])  # Add constant for intercept
    y = df_first_rows[correlate]
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    r_squared = model.rsquared
    
    # Plot regression line
    sns.regplot(x='NewTIQ', y=f'{correlate}', data=df_first_rows, ci=95, scatter=False, ax=ax, color='blue', line_kws={'lw': 1, 'ls': '--'}, scatter_kws={'s': 40, 'alpha': 1})
    # Calculate the correlation coefficient (r-value) and p-value
    r_value, p_value = stats.pearsonr(df_first_rows['NewTIQ'], df_first_rows[correlate])
    
     # Annotate p-value
    if p_value < 0.0005:
        plt.annotate(f'R = {r_value:.2f}\nR² = {r_squared:.2f}\np = {round(p_value, 5)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.005:
        plt.annotate(f'R = {r_value:.2f}\nR² = {r_squared:.2f}\np = {round(p_value, 4)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    elif p_value < 0.05:
        plt.annotate(f'R = {r_value:.2f}\nR² = {r_squared:.2f}\np = {round(p_value, 3)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', bbox=dict(boxstyle="round", alpha=0.5, color='w'))
    else:
        plt.annotate(f'R = {r_value:.2f}\nR² = {r_squared:.2f}\np={round(p_value, 2)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', bbox=dict(boxstyle="round", alpha=0.5, color='w'))

    
    # Setting plot labels and title
    plt.xlabel('IQ')
    plt.ylabel(correlate_label)
    # plt.title(f'{df_code} IQ vs. {correlate_label} Correlation')

    # Ensure the directory exists
    os.makedirs(dir_figs, exist_ok=True)
    # Save the plot
    fig_path_png = os.path.join(dir_figs, f'IQ_vs_{correlate}_{df_code}_{date}.png')
    plt.savefig(fig_path_png)
    
    # Show the plot
    plt.show()
    