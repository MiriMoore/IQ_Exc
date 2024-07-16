#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:11:15 2024

@author: Miri
"""
#%% import packages

import os
import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
from statsmodels.stats.anova import anova_lm
from scipy.stats import shapiro, ttest_rel, ttest_ind, wilcoxon





#%% set directories

os.chdir('/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/scripts')
dir_parent = '/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc'
dir_data = os.path.join(dir_parent, 'data')
dir_figs = os.path.join(dir_parent, 'figs')
dir_procdata = os.path.join(dir_parent, 'proc_data')

#%% load dataframes

df_filemeta = pd.read_csv(os.path.join(dir_procdata, 'IQ_Excitability_L2L3_File_Metadata.csv'))

#%% define variables

fbins_labels = [[1.0, 4.0], #delta 2-4
         [4.0, 5.1], [5.1, 6.5], #theta 4-7
         [6.5, 8.3], [8.3, 10.5], [10.5, 13.4], #alpha 8-13
         [13.4, 17.0], [17.0, 21.7], [21.7, 27.6], #beta 16-25
         [27.6, 35.2],[35.2, 44.8]] #gamma 30-50


def format_p(p_value):
    """Format p-value with usual significance notation."""
    if p_value < 0.00005:
        return "< 0.00005"
    elif p_value < 0.0005:
        return "< 0.0005"
    elif p_value < 0.005:
        return "< 0.005"
    elif p_value < 0.05:
        return "< 0.05"
    else:
        return f"= {round(p_value, 2)}"
    
    
#%% set colours

def cmyk_to_rgb(c, m, y, k):
    c, m, y, k = [x / 100.0 for x in [c, m, y, k]]  # Convert to the range 0 to 1
    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)
    return (r / 255, g / 255, b / 255)  # Normalize RGB values to 0-1 range

# Define your CMYK colors
cmyk_colours = {
    'ctrl': (100, 86, 37, 27),
    'resec': (0, 56, 81, 0),
    'Left': (50, 17, 7, 0),
    'Right': (7, 11, 80, 0)
}

rgb_colours = {key: cmyk_to_rgb(*value) for key, value in cmyk_colours.items()}
rgb_colours = {
                'ctrl': (64/255, 214/255, 89/255),
                'resec': (122/255, 59/255, 128/255),
                'Left': (57/255, 188/255, 217/255),
                'Right': (219/255, 21/255,74/255)
            }
def format_fbins(fbins):
    """Formats the frequency bins as strings with one decimal place."""
    return [f"{round(f[0], 1)}-{round(f[1], 1)}" for f in fbins]


#%% create dictionary from MEG analysis data
ROI = 'allMTGs'

dict_MEGanalysis = {}
dir_MEGfiles = os.path.join(dir_procdata, f'MEGanalysis_{ROI}')
# dir_MEGfiles = '/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/dir_crosci/analysis/'
for filename in os.listdir(dir_MEGfiles):
    if filename.endswith('.pkl'):
        filepath = os.path.join(dir_MEGfiles, filename)
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            fbins = data['frequency_bins']
        
        left_data = {}
        right_data = {}
        print(filename)
        for paramkey in ['DFA', 'fEI', 'BIS', 'HLP']:
            print(paramkey)
            param_data = data[paramkey]
            indices = np.arange(param_data.shape[0])
            channels_L = param_data[indices % 2 != 0]
            channels_R = param_data[indices % 2 == 0]
            avg_L = np.nanmean(channels_L, axis=0)
            avg_R = np.nanmean(channels_R, axis=0)
            
            left_data[paramkey] = channels_L
            left_data[f'{paramkey}_ROIavg'] = avg_L
            right_data[paramkey] = channels_R
            right_data[f'{paramkey}_ROIavg'] = avg_R
            
        dict_MEGanalysis[filename] = {'Left': left_data, 'Right': right_data}
        
# PROBLEM runtimewarning: mean of empty slice => some rows likely to be empty - find which ones and why
#%%

fbin_labels = format_fbins(fbins)
date = datetime.now().strftime('%Y%m%d')
#%% create analysis df with fEI and DFA

# df_name = 'df_IQExcitability_DFA_fEI_LRA21c.csv'
# df_name = 'df_IQExcitability_DFA_fEI_allTGs.csv'
df_name = f'df_IQExcitability_DFA_fEI_BIS_HLP_{ROI}.csv'
biomarkers = ['DFA', 'fEI', 'BIS', 'HLP']

def create_MEGanalysis_dataframe(dict_MEGanalysis, df_filemeta, dir_procdata, df_name, reload=False):
    df_path = os.path.join(dir_procdata, df_name)
    if not os.path.exists(df_path) or reload == True:
        patient_iq_map = df_filemeta.set_index('patient_id')['NewTIQ'].to_dict()
        patt_pID = r'^(\d{4}_\d{2}_\d{2})'
        data_list = []
        for filename, data in dict_MEGanalysis.items():
            resec_hemi = 'Left' if 'LeftResec' in filename else 'Right'
            ctrl_hemi = 'Right' if 'LeftResec' in filename else 'Left'
            row_dict = {
                        'file': filename,
                        'patient_ID': pd.NA,
                        'resection_hemisphere': resec_hemi,
                        'NewTIQ': pd.NA
                        }
            match = re.match(patt_pID, filename)
            if match:
                pID = match.group(1)
                if pID in patient_iq_map:
                    row_dict['NewTIQ'] = patient_iq_map[pID]
                    row_dict['patient_ID'] = pID
                for i in range(len(data[resec_hemi]['DFA_ROIavg'])):
                    for biomarker in biomarkers:
                        row_dict.update({
                            f'{biomarker}_resec_bin{i}': data[resec_hemi][f'{biomarker}_ROIavg'][i],
                            f'{biomarker}_ctrl_bin{i}': data[ctrl_hemi][f'{biomarker}_ROIavg'][i]
                        })
                        # Calculate the average for bins 4 and 5
                        row_dict[f'{biomarker}_resec_band_alpha'] = (data[resec_hemi][f'{biomarker}_ROIavg'][4] + data[resec_hemi][f'{biomarker}_ROIavg'][5]) / 2
                        row_dict[f'{biomarker}_ctrl_band_alpha'] = (data[ctrl_hemi][f'{biomarker}_ROIavg'][4] + data[ctrl_hemi][f'{biomarker}_ROIavg'][5]) / 2
                    
                        # Calculate the average for bins 6, 7, 8
                        row_dict[f'{biomarker}_resec_band_beta'] = (data[resec_hemi][f'{biomarker}_ROIavg'][6] + data[resec_hemi][f'{biomarker}_ROIavg'][7] + data[resec_hemi][f'{biomarker}_ROIavg'][8]) / 3
                        row_dict[f'{biomarker}_ctrl_band_beta'] = (data[ctrl_hemi][f'{biomarker}_ROIavg'][6] + data[ctrl_hemi][f'{biomarker}_ROIavg'][7] + data[ctrl_hemi][f'{biomarker}_ROIavg'][8]) / 3
                    
                data_list.append(row_dict)
            df_MEGanalysis = pd.DataFrame(data_list)
            df_MEGanalysis.to_csv(os.path.join(dir_procdata, df_name), index=False)
    else:
        df_MEGanalysis = pd.read_csv(df_path)
    return data_list, df_MEGanalysis


data_list, df_MEGanalysis = create_MEGanalysis_dataframe(dict_MEGanalysis, df_filemeta, dir_procdata, df_name, reload=True)

#%% create dictionary with ephys, morph and MEG data

def create_patient_dict(df_MEGanalysis, df_filemeta):
    patient_dict = {}

    meg_grouped = df_MEGanalysis.groupby('patient_ID')
    ephys_grouped = df_filemeta.groupby('patient_id')

    for patient_ID, group in meg_grouped:
        meg_data = group.drop(columns=['patient_ID', 'NewTIQ', 'resection_hemisphere'])
        if patient_ID not in patient_dict:
            patient_dict[patient_ID] = {
                'NewTIQ': group['NewTIQ'].iloc[0],
                'resection_hemisphere': group['resection_hemisphere'].iloc[0],
                'MEG_data': meg_data
            }
        else:
            patient_dict[patient_ID]['MEG_data'] = meg_data

    for patient_ID, group in ephys_grouped:
        patient_data = ['Gender', 'AgeAtResection', 'AgeAtEpilepsyOnset', 'YearsEpilepsy', 'SeizuresMonth', 'DiseaseType', 'Region', 'EEG']
        ephys_data = group.drop(columns=['patient_id', 'NewTIQ', 'HemiRes'])
        if patient_ID not in patient_dict:
            patient_dict[patient_ID] = {}
        for col in patient_data:
            patient_dict[patient_ID][col] = group[col].iloc[0]
        patient_dict[patient_ID]['Ephys_data'] = ephys_data
    
    return patient_dict


dict_meg_ephys = create_patient_dict(df_MEGanalysis, df_filemeta)

with open(os.path.join(dir_procdata, 'metadata_pID_MEG_ephys_morph.pkl'), 'wb') as f:
    pickle.dump(dict_meg_ephys, f)
    
with open(os.path.join(dir_procdata, 'metadata_pID_MEG_ephys_morph.pkl'), 'rb') as f:
    dict_MEGephys = pickle.load(f)
    
#%% analyse DFA vs TDL

for patient_ID in dict_MEGephys:
    patient_data = dict_MEGephys[patient_ID]
    if 'MEG_data' in patient_data:
        DFA_resec_alpha = patient_data['MEG_data']['DFA_resec_band_alpha'].iloc[0]
        DFA_ctrl_alpha = patient_data['MEG_data']['DFA_ctrl_band_alpha'].iloc[0]
        DFA_resec_beta = patient_data['MEG_data']['DFA_resec_band_beta'].iloc[0]
        DFA_ctrl_beta = patient_data['MEG_data']['DFA_ctrl_band_beta'].iloc[0]
        TDL = patient_data['Ephys_data']['TDL']
        
    plt.figure()
    plt.plot()


# Initialize a list to collect data
data = []

# Collect data
for patient_ID in dict_meg_ephys:
    patient_data = dict_meg_ephys[patient_ID]
    if 'MEG_data' in patient_data:
        DFA_resec_alpha = patient_data['MEG_data']['DFA_resec_band_alpha'].iloc[0]
        TDL = patient_data['Ephys_data']['TDL']
        
        # Ensure TDL is a Series or list
        if isinstance(TDL, pd.Series) or isinstance(TDL, list):
            for tdl_value in TDL:
                data.append({'TDL': tdl_value, 'DFA_resec_alpha': DFA_resec_alpha})
        else:
            data.append({'TDL': TDL, 'DFA_resec_alpha': DFA_resec_alpha})

# Convert to DataFrame
df = pd.DataFrame(data)

# Create box plot
plt.figure(figsize=(10, 6))
df.boxplot(column='TDL', by='DFA_resec_alpha', grid=False, vert=False)
plt.title('TDL by DFA Resec Alpha')
plt.suptitle('')  # Suppress the default title to avoid duplication
plt.xlabel('DFA Resec Alpha')
plt.ylabel('TDL')
plt.show()


#%% summarise LMM results

def save_lmm_summary_to_file(biomarker, result_hemisphere, result_resec_ctrl, filepath):
    with open(filepath, 'w') as file:
        file.write(f"Mixed Linear Model Regression Results for Left vs. Right Hemisphere ({biomarker}):\n")
        file.write("="*100 + "\n")
        file.write(result_hemisphere.summary().as_text() + "\n")
        file.write("\n")
        file.write("Summary:\n")
        
        file.write(f"1. The difference between {biomarker} values of the right and left hemispheres ")
        p_value = result_hemisphere.pvalues[f"hemisphere[T.{biomarker}_right]"]
        if p_value < 0.05:
            file.write(f"is significant (p = {p_value:.3f}).\n")
        elif p_value < 0.1:
            file.write(f"is marginally significant (p = {p_value:.3f}).\n")
        else:
            file.write(f"is not significant (p = {p_value:.3f}).\n")

        file.write(f"2. The interaction between hemisphere and bin ")
        p_value = result_hemisphere.pvalues[f"hemisphere[T.{biomarker}_right]:bin"]
        if p_value < 0.05:
            file.write(f"is significant (p = {p_value:.3f}).\n")
        elif p_value < 0.1:
            file.write(f"is marginally significant (p = {p_value:.3f}).\n")
        else:
            file.write(f"is not significant (p = {p_value:.3f}).\n")

        file.write(f"3. The interaction between hemisphere and NewTIQ ")
        p_value = result_hemisphere.pvalues[f"hemisphere[T.{biomarker}_right]:NewTIQ"]
        if p_value < 0.05:
            file.write(f"is significant (p = {p_value:.3f}).\n")
        elif p_value < 0.1:
            file.write(f"is marginally significant (p = {p_value:.3f}).\n")
        else:
            file.write(f"is not significant (p = {p_value:.3f}).\n")
        
        file.write(f"4. The three-way interaction between hemisphere, bin, and NewTIQ ")
        p_value = result_hemisphere.pvalues[f"hemisphere[T.{biomarker}_right]:bin:NewTIQ"]
        if p_value < 0.05:
            file.write(f"is significant (p = {p_value:.3f}).\n")
        elif p_value < 0.1:
            file.write(f"is marginally significant (p = {p_value:.3f}).\n")
        else:
            file.write(f"is not significant (p = {p_value:.3f}).\n")
        
        file.write("="*100 + "\n\n")
        
        file.write(f"Mixed Linear Model Regression Results for Resection vs. Control Hemisphere ({biomarker}):\n")
        file.write("="*100 + "\n")
        file.write(result_resec_ctrl.summary().as_text() + "\n")
        file.write("\n")
        file.write("Summary:\n")
        
        file.write(f"1. The difference between {biomarker} values of the resection and control hemispheres ")
        p_value = result_resec_ctrl.pvalues[f"hemisphere[T.{biomarker}_resec]"]
        if p_value < 0.05:
            file.write(f"is significant (p = {p_value:.3f}).\n")
        elif p_value < 0.1:
            file.write(f"is marginally significant (p = {p_value:.3f}).\n")
        else:
            file.write(f"is not significant (p = {p_value:.3f}).\n")

        file.write(f"2. The interaction between resection hemisphere and bin ")
        p_value = result_resec_ctrl.pvalues[f"hemisphere[T.{biomarker}_resec]:bin"]
        if p_value < 0.05:
            file.write(f"is significant (p = {p_value:.3f}).\n")
        elif p_value < 0.1:
            file.write(f"is marginally significant (p = {p_value:.3f}).\n")
        else:
            file.write(f"is not significant (p = {p_value:.3f}).\n")

        file.write(f"3. The interaction between resection hemisphere and NewTIQ ")
        p_value = result_resec_ctrl.pvalues[f"hemisphere[T.{biomarker}_resec]:NewTIQ"]
        if p_value < 0.05:
            file.write(f"is significant (p = {p_value:.3f}).\n")
        elif p_value < 0.1:
            file.write(f"is marginally significant (p = {p_value:.3f}).\n")
        else:
            file.write(f"is not significant (p = {p_value:.3f}).\n")

        file.write(f"4. The three-way interaction between resection hemisphere, bin, and NewTIQ ")
        p_value = result_resec_ctrl.pvalues[f"hemisphere[T.{biomarker}_resec]:bin:NewTIQ"]
        if p_value < 0.05:
            file.write(f"is significant (p = {p_value:.3f}).\n")
        elif p_value < 0.1:
            file.write(f"is marginally significant (p = {p_value:.3f}).\n")
        else:
            file.write(f"is not significant (p = {p_value:.3f}).\n")
        
        file.write("="*100 + "\n")
    return file

#%% LMM LvsR and RvsC

def check_normality(data):
    stat, p = shapiro(data)
    return p > 0.05  # If p > 0.05, the data is normally distributed

def test_stat_differences_LvR_RvC(df_MEGanalysis, biomarker, output_csv):
    df_long_hemisphere = pd.DataFrame()
    df_long_resec_ctrl = pd.DataFrame()

    for fbin in range(11):
        temp_df = df_MEGanalysis.copy()
        temp_df['bin'] = fbin
        temp_df[f'{biomarker}_left'] = temp_df.apply(
            lambda row: row[f'{biomarker}_resec_bin{fbin}'] if row['resection_hemisphere'] == 'Left' else row[f'{biomarker}_ctrl_bin{fbin}'], axis=1
        )
        temp_df[f'{biomarker}_right'] = temp_df.apply(
            lambda row: row[f'{biomarker}_ctrl_bin{fbin}'] if row['resection_hemisphere'] == 'Left' else row[f'{biomarker}_resec_bin{fbin}'], axis=1
        )
        temp_df[f'{biomarker}_resec'] = temp_df[f'{biomarker}_resec_bin{fbin}']
        temp_df[f'{biomarker}_ctrl'] = temp_df[f'{biomarker}_ctrl_bin{fbin}']
        
        temp_hemisphere = temp_df[['patient_ID', 'bin', 'NewTIQ', f'{biomarker}_left', f'{biomarker}_right']]
        temp_resec_ctrl = temp_df[['patient_ID', 'bin', 'NewTIQ', f'{biomarker}_resec', f'{biomarker}_ctrl']]
        
        df_long_hemisphere = pd.concat([df_long_hemisphere, temp_hemisphere], ignore_index=True)
        df_long_resec_ctrl = pd.concat([df_long_resec_ctrl, temp_resec_ctrl], ignore_index=True)

    # Convert to long format
    df_long_hemisphere = pd.melt(df_long_hemisphere, id_vars=['patient_ID', 'bin', 'NewTIQ'], 
                                 value_vars=[f'{biomarker}_left', f'{biomarker}_right'],
                                 var_name='hemisphere', value_name=biomarker)
    df_long_resec_ctrl = pd.melt(df_long_resec_ctrl, id_vars=['patient_ID', 'bin', 'NewTIQ'], 
                                 value_vars=[f'{biomarker}_resec', f'{biomarker}_ctrl'],
                                 var_name='hemisphere', value_name=biomarker)
    df_long_hemisphere = df_long_hemisphere.dropna(subset=[biomarker])
    df_long_resec_ctrl = df_long_resec_ctrl.dropna(subset=[biomarker])
    
    median_TIQ = df_long_hemisphere['NewTIQ'].median()
    df_long_hemisphere['IQ_group'] = df_long_hemisphere['NewTIQ'].apply(lambda x: 'High IQ' if x > median_TIQ else 'Low IQ')
    df_long_resec_ctrl['IQ_group'] = df_long_resec_ctrl['NewTIQ'].apply(lambda x: 'High IQ' if x > median_TIQ else 'Low IQ')
    
    # Check normality for differences in each bin
    # print(f"Normality check for {biomarker}:")
    for fbin in range(11):
        left = df_long_hemisphere[(df_long_hemisphere['bin'] == fbin) & (df_long_hemisphere['hemisphere'] == f'{biomarker}_left')][biomarker]
        right = df_long_hemisphere[(df_long_hemisphere['bin'] == fbin) & (df_long_hemisphere['hemisphere'] == f'{biomarker}_right')][biomarker]
        resec = df_long_resec_ctrl[(df_long_resec_ctrl['bin'] == fbin) & (df_long_resec_ctrl['hemisphere'] == f'{biomarker}_resec')][biomarker]
        ctrl = df_long_resec_ctrl[(df_long_resec_ctrl['bin'] == fbin) & (df_long_resec_ctrl['hemisphere'] == f'{biomarker}_ctrl')][biomarker]
        # print(f'Bin {fbin}: Left vs. Right normality:', check_normality(left - right))
        # print(f'Bin {fbin}: Resection vs. Control normality:', check_normality(resec - ctrl))

    # Fit the mixed-effects models
    model_hemisphere = smf.mixedlm(f"{biomarker} ~ hemisphere * bin", df_long_hemisphere, 
                                   groups=df_long_hemisphere["patient_ID"], re_formula="~hemisphere*bin")
    result_hemisphere = model_hemisphere.fit()
    model_resec_ctrl = smf.mixedlm(f"{biomarker} ~ hemisphere * bin", df_long_resec_ctrl, 
                                   groups=df_long_resec_ctrl["patient_ID"], re_formula="~hemisphere*bin")
    result_resec_ctrl = model_resec_ctrl.fit()

    print("\nMixed Linear Model Regression Results for Left vs. Right Hemisphere:")
    print(result_hemisphere.summary())

    print("\nMixed Linear Model Regression Results for Resection vs. Control Hemisphere:")
    print(result_resec_ctrl.summary())

    return df_long_hemisphere, df_long_resec_ctrl


df_long_hemisphere_fEI, df_long_resecctrl_fEI = test_stat_differences_LvR_RvC(df_MEGanalysis, 'fEI', 'LMM_results_fEI.csv')
df_long_hemisphere_DFA, df_long_resecctrl_DFA = test_stat_differences_LvR_RvC(df_MEGanalysis, 'DFA', 'LMM_results_DFA.csv')
df_long_hemisphere_BIS, df_long_resecctrl_BIS = test_stat_differences_LvR_RvC(df_MEGanalysis, 'BIS', 'LMM_results_BIS.csv')
df_long_hemisphere_HLP, df_long_resecctrl_HLP = test_stat_differences_LvR_RvC(df_MEGanalysis, 'HLP', 'LMM_results_HLP.csv')

#%% LMM LvsR and RvsC Bin

def LMM_FE_hemibinIQ(df_MEGanalysis, biomarker):
    df_long_hemisphere = pd.DataFrame()
    df_long_resec_ctrl = pd.DataFrame()

    for fbin in range(11):
        temp_df = df_MEGanalysis.copy()
        temp_df['bin'] = fbin
        temp_df[f'{biomarker}_left'] = temp_df.apply(
            lambda row: row[f'{biomarker}_resec_bin{fbin}'] if row['resection_hemisphere'] == 'Left' else row[f'{biomarker}_ctrl_bin{fbin}'], axis=1
        )
        temp_df[f'{biomarker}_right'] = temp_df.apply(
            lambda row: row[f'{biomarker}_ctrl_bin{fbin}'] if row['resection_hemisphere'] == 'Left' else row[f'{biomarker}_resec_bin{fbin}'], axis=1
        )
        temp_df[f'{biomarker}_resec'] = temp_df[f'{biomarker}_resec_bin{fbin}']
        temp_df[f'{biomarker}_ctrl'] = temp_df[f'{biomarker}_ctrl_bin{fbin}']
        
        temp_hemisphere = temp_df[['patient_ID', 'bin', 'NewTIQ', f'{biomarker}_left', f'{biomarker}_right']]
        temp_resec_ctrl = temp_df[['patient_ID', 'bin', 'NewTIQ', f'{biomarker}_resec', f'{biomarker}_ctrl']]
        
        df_long_hemisphere = pd.concat([df_long_hemisphere, temp_hemisphere], ignore_index=True)
        df_long_resec_ctrl = pd.concat([df_long_resec_ctrl, temp_resec_ctrl], ignore_index=True)

    # Convert to long format
    df_long_hemisphere = pd.melt(df_long_hemisphere, id_vars=['patient_ID', 'bin', 'NewTIQ'], 
                                 value_vars=[f'{biomarker}_left', f'{biomarker}_right'],
                                 var_name='hemisphere', value_name=biomarker)
    df_long_resec_ctrl = pd.melt(df_long_resec_ctrl, id_vars=['patient_ID', 'bin', 'NewTIQ'], 
                                 value_vars=[f'{biomarker}_resec', f'{biomarker}_ctrl'],
                                 var_name='hemisphere', value_name=biomarker)
    df_long_hemisphere = df_long_hemisphere.dropna()
    df_long_resec_ctrl = df_long_resec_ctrl.dropna()
    
    median_TIQ = df_long_hemisphere['NewTIQ'].median()
    df_long_hemisphere['IQ_group'] = df_long_hemisphere['NewTIQ'].apply(lambda x: 'High IQ' if x > median_TIQ else 'Low IQ')
    df_long_resec_ctrl['IQ_group'] = df_long_resec_ctrl['NewTIQ'].apply(lambda x: 'High IQ' if x > median_TIQ else 'Low IQ')
    
    # Check normality for differences in each bin
    # print(f"Normality check for {biomarker}:")
    for fbin in range(11):
        left = df_long_hemisphere[(df_long_hemisphere['bin'] == fbin) & (df_long_hemisphere['hemisphere'] == f'{biomarker}_left')][biomarker]
        right = df_long_hemisphere[(df_long_hemisphere['bin'] == fbin) & (df_long_hemisphere['hemisphere'] == f'{biomarker}_right')][biomarker]
        resec = df_long_resec_ctrl[(df_long_resec_ctrl['bin'] == fbin) & (df_long_resec_ctrl['hemisphere'] == f'{biomarker}_resec')][biomarker]
        ctrl = df_long_resec_ctrl[(df_long_resec_ctrl['bin'] == fbin) & (df_long_resec_ctrl['hemisphere'] == f'{biomarker}_ctrl')][biomarker]
        # print(f'Bin {fbin}: Left vs. Right normality:', check_normality(left - right))
        # print(f'Bin {fbin}: Resection vs. Control normality:', check_normality(resec - ctrl))

    # Fit the mixed-effects models
    model_hemisphere = smf.mixedlm(f"{biomarker} ~ hemisphere * bin * NewTIQ", df_long_hemisphere, 
                                   groups=df_long_hemisphere["patient_ID"], re_formula="~1 + bin + hemisphere * bin")
    result_hemisphere = model_hemisphere.fit()
    model_resec_ctrl = smf.mixedlm(f"{biomarker} ~ hemisphere * bin * NewTIQ", df_long_resec_ctrl, 
                                   groups=df_long_resec_ctrl["patient_ID"], re_formula="~1 + bin + hemisphere * bin")
    result_resec_ctrl = model_resec_ctrl.fit()

    print("\nMixed Linear Model Regression Results for Left vs. Right Hemisphere:")
    print(result_hemisphere.summary())

    print("\nMixed Linear Model Regression Results for Resection vs. Control Hemisphere:")
    print(result_resec_ctrl.summary())

    # Save the summary to a text file
    txt_lmm_summary = save_lmm_summary_to_file(biomarker, result_hemisphere, result_resec_ctrl, f'LMM_results_summary_{biomarker}.txt')
    
    return txt_lmm_summary, df_long_hemisphere, df_long_resec_ctrl

LMM_FE_hemibinIQ(df_MEGanalysis, 'DFA')
LMM_FE_hemibinIQ(df_MEGanalysis, 'fEI')
LMM_FE_hemibinIQ(df_MEGanalysis, 'BIS')
LMM_FE_hemibinIQ(df_MEGanalysis, 'HLP')

#%% check model fit
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import shapiro, chi2
import matplotlib.pyplot as plt
import seaborn as sns
import os

def prepare_long_format(df_MEGanalysis, biomarker):
    df_long_hemisphere = pd.DataFrame()
    df_long_resec_ctrl = pd.DataFrame()

    for fbin in range(11):
        temp_df = df_MEGanalysis.copy()
        temp_df['bin'] = fbin
        temp_df[f'{biomarker}_left'] = temp_df.apply(
            lambda row: row[f'{biomarker}_resec_bin{fbin}'] if row['resection_hemisphere'] == 'Left' else row[f'{biomarker}_ctrl_bin{fbin}'], axis=1
        )
        temp_df[f'{biomarker}_right'] = temp_df.apply(
            lambda row: row[f'{biomarker}_ctrl_bin{fbin}'] if row['resection_hemisphere'] == 'Left' else row[f'{biomarker}_resec_bin{fbin}'], axis=1
        )
        temp_df[f'{biomarker}_resec'] = temp_df[f'{biomarker}_resec_bin{fbin}']
        temp_df[f'{biomarker}_ctrl'] = temp_df[f'{biomarker}_ctrl_bin{fbin}']
        
        temp_hemisphere = temp_df[['patient_ID', 'bin', 'NewTIQ', f'{biomarker}_left', f'{biomarker}_right']]
        temp_resec_ctrl = temp_df[['patient_ID', 'bin', 'NewTIQ', f'{biomarker}_resec', f'{biomarker}_ctrl']]
        
        df_long_hemisphere = pd.concat([df_long_hemisphere, temp_hemisphere], ignore_index=True)
        df_long_resec_ctrl = pd.concat([df_long_resec_ctrl, temp_resec_ctrl], ignore_index=True)

    df_long_hemisphere = pd.melt(df_long_hemisphere, id_vars=['patient_ID', 'bin', 'NewTIQ'], 
                                 value_vars=[f'{biomarker}_left', f'{biomarker}_right'],
                                 var_name='hemisphere', value_name=biomarker)
    df_long_resec_ctrl = pd.melt(df_long_resec_ctrl, id_vars=['patient_ID', 'bin', 'NewTIQ'], 
                                 value_vars=[f'{biomarker}_resec', f'{biomarker}_ctrl'],
                                 var_name='hemisphere', value_name=biomarker)
    df_long_hemisphere = df_long_hemisphere.dropna()
    df_long_resec_ctrl = df_long_resec_ctrl.dropna()
    
    median_TIQ = df_long_hemisphere['NewTIQ'].median()
    df_long_hemisphere['IQ_group'] = df_long_hemisphere['NewTIQ'].apply(lambda x: 'High IQ' if x > median_TIQ else 'Low IQ')
    df_long_resec_ctrl['IQ_group'] = df_long_resec_ctrl['NewTIQ'].apply(lambda x: 'High IQ' if x > median_TIQ else 'Low IQ')
    
    return df_long_hemisphere, df_long_resec_ctrl

def verify_random_effects(df_MEGanalysis, biomarker):
    df_long_hemisphere, df_long_resec_ctrl = prepare_long_format(df_MEGanalysis, biomarker)

    # Fit the mixed-effects models with different random effects structures
    model_simple_hemisphere = smf.mixedlm(f"{biomarker} ~ hemisphere * bin * NewTIQ", df_long_hemisphere,
                                          groups=df_long_hemisphere["patient_ID"], re_formula="~1")
    result_simple_hemisphere = model_simple_hemisphere.fit(reml=False)

    model_complex_hemisphere = smf.mixedlm(f"{biomarker} ~ hemisphere * bin * NewTIQ", df_long_hemisphere,
                                           groups=df_long_hemisphere["patient_ID"], re_formula="~1 + bin + hemisphere * bin")
    result_complex_hemisphere = model_complex_hemisphere.fit(reml=False)

    model_simple_resec_ctrl = smf.mixedlm(f"{biomarker} ~ hemisphere * bin * NewTIQ", df_long_resec_ctrl,
                                          groups=df_long_resec_ctrl["patient_ID"], re_formula="~1")
    result_simple_resec_ctrl = model_simple_resec_ctrl.fit(reml=False)

    model_complex_resec_ctrl = smf.mixedlm(f"{biomarker} ~ hemisphere * bin * NewTIQ", df_long_resec_ctrl,
                                           groups=df_long_resec_ctrl["patient_ID"], re_formula="~1 + bin + hemisphere * bin")
    result_complex_resec_ctrl = model_complex_resec_ctrl.fit(reml=False)

    # Compare models using AIC and BIC
    print("AIC for Simple Hemisphere Model:", result_simple_hemisphere.aic)
    print("AIC for Complex Hemisphere Model:", result_complex_hemisphere.aic)
    print("BIC for Simple Hemisphere Model:", result_simple_hemisphere.bic)
    print("BIC for Complex Hemisphere Model:", result_complex_hemisphere.bic)

    print("AIC for Simple Resection vs Control Model:", result_simple_resec_ctrl.aic)
    print("AIC for Complex Resection vs Control Model:", result_complex_resec_ctrl.aic)
    print("BIC for Simple Resection vs Control Model:", result_simple_resec_ctrl.bic)
    print("BIC for Complex Resection vs Control Model:", result_complex_resec_ctrl.bic)

    # Likelihood ratio tests
    lr_stat_hemisphere = 2 * (result_complex_hemisphere.llf - result_simple_hemisphere.llf)
    p_value_hemisphere = chi2.sf(lr_stat_hemisphere, df=result_complex_hemisphere.df_modelwc - result_simple_hemisphere.df_modelwc)
    print("Likelihood Ratio Test p-value for Hemisphere Model:", p_value_hemisphere)

    lr_stat_resec_ctrl = 2 * (result_complex_resec_ctrl.llf - result_simple_resec_ctrl.llf)
    p_value_resec_ctrl = chi2.sf(lr_stat_resec_ctrl, df=result_complex_resec_ctrl.df_modelwc - result_simple_resec_ctrl.df_modelwc)
    print("Likelihood Ratio Test p-value for Resection vs Control Model:", p_value_resec_ctrl)

    # Plot random effects
    random_effects_hemisphere = result_complex_hemisphere.random_effects
    df_random_effects_hemisphere = pd.DataFrame(random_effects_hemisphere).transpose()

    plt.figure(figsize=(10, 6))
    for col in df_random_effects_hemisphere.columns:
        sns.histplot(df_random_effects_hemisphere[col], kde=True, label=col)
    plt.title('Distribution of Random Intercepts for Hemisphere Model')
    plt.xlabel('Random Intercept')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    random_effects_resec_ctrl = result_complex_resec_ctrl.random_effects
    df_random_effects_resec_ctrl = pd.DataFrame(random_effects_resec_ctrl).transpose()

    plt.figure(figsize=(10, 6))
    for col in df_random_effects_resec_ctrl.columns:
        sns.histplot(df_random_effects_resec_ctrl[col], kde=True, label=col)
    plt.title('Distribution of Random Intercepts for Resection vs Control Model')
    plt.xlabel('Random Intercept')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Residuals plot for the complex model
    plt.figure(figsize=(10, 6))
    sns.residplot(x=result_complex_hemisphere.fittedvalues, y=result_complex_hemisphere.resid)
    plt.title('Residuals vs Fitted Values for Hemisphere Model')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

    sm.qqplot(result_complex_hemisphere.resid, line ='45')
    plt.title('QQ Plot of Residuals for Hemisphere Model')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.residplot(x=result_complex_resec_ctrl.fittedvalues, y=result_complex_resec_ctrl.resid)
    plt.title('Residuals vs Fitted Values for Resection vs Control Model')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

    sm.qqplot(result_complex_resec_ctrl.resid, line ='45')
    plt.title('QQ Plot of Residuals for Resection vs Control Model')
    plt.show()

# Example usage
verify_random_effects(df_MEGanalysis, 'DFA')
verify_random_effects(df_MEGanalysis, 'fEI')


#%%
def plot_binvsbiomarker(df_long_hemisphere, df_long_resec_ctrl, biomarker):
    plt.figure(figsize=(14, 6))

    # Plot Left vs. Right Hemisphere
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df_long_hemisphere, x='bin', y=biomarker, hue='hemisphere', ci=95)
    plt.title('Left vs. Right Hemisphere')
    plt.xlabel('Frequency Bin')
    plt.ylabel(f'{biomarker} Value')
    plt.legend(title='Hemisphere')
    plt.grid(False)

    # Plot Resection vs. Control Hemisphere
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df_long_resec_ctrl, x='bin', y=biomarker, hue='hemisphere', ci=95)
    plt.title('Resection vs. Control Hemisphere')
    plt.xlabel('Frequency Bin')
    plt.ylabel(f'{biomarker} Value')
    plt.legend(title='Condition')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, f'hemisphere_comparison_{biomarker}_{date}.eps'))
    plt.savefig(os.path.join(dir_figs, f'hemisphere_comparison_{biomarker}_{date}.png'))
    



#%%
def plot_IQ_groups_averaged(df_long_hemisphere, biomarker, dir_figs):
    # Calculate the average of {biomarker}_left and {biomarker}_right for each patient, bin, and IQ group
    df_long_hemisphere['average_biomarker'] = df_long_hemisphere.groupby(['patient_ID', 'bin', 'IQ_group'])[biomarker].transform('mean')

    # Drop duplicate rows to avoid plotting the same average value multiple times
    df_avg = df_long_hemisphere[['patient_ID', 'bin', 'IQ_group', 'average_biomarker']].drop_duplicates()

    plt.figure(figsize=(7, 6))
    sns.lineplot(data=df_avg, x='bin', y='average_biomarker', hue='IQ_group', ci=95)
    plt.title('High vs. Low IQ')
    plt.xlabel('Frequency Bin')
    plt.xticks(ticks=range(len(fbin_labels)), labels=fbin_labels, rotation=45, ha='right')

    plt.ylabel(f'{biomarker} Value')
    plt.legend(title='IQ')
    plt.grid(False)
    plt.savefig(os.path.join(dir_figs, f'IQgroup_comparison_{biomarker}_{date}.eps'))
    plt.savefig(os.path.join(dir_figs, f'IQgroup_comparison_{biomarker}_{date}.png'))
    plt.show()


#%%
def plot_IQ_HvL_groups(df_long_hemisphere, biomarker):
    plt.figure(figsize=(14, 6))

    # Plot High vs. Low IQ for Left Hemisphere
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df_long_hemisphere[df_long_hemisphere['hemisphere'] == f'{biomarker}_left'], 
                 x='bin', y=biomarker, hue='IQ_group', ci=95)
    plt.title('High vs. Low IQ: Left Hemisphere')
    plt.xlabel('Frequency Bin')
    plt.xticks(ticks=range(len(fbin_labels)), labels=fbin_labels, rotation=45, ha='right')
    plt.ylabel(f'{biomarker} Value')
    plt.legend(title='IQ Group')
    plt.grid(False)

    # Plot High vs. Low IQ for Right Hemisphere
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df_long_hemisphere[df_long_hemisphere['hemisphere'] == f'{biomarker}_right'], 
                 x='bin', y=biomarker, hue='IQ_group', ci=95)
    plt.title('High vs. Low IQ: Right Hemisphere')
    plt.xticks(ticks=range(len(fbin_labels)), labels=fbin_labels, rotation=45, ha='right')
    plt.xlabel('Frequency Bin')
    plt.ylabel(f'{biomarker} Value')
    plt.legend(title='IQ Group')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, f'IQgrouphemisphere_comparison_{biomarker}_{date}.eps'))
    plt.savefig(os.path.join(dir_figs, f'IQgrouphemisphere_comparison_{biomarker}_{date}.png'))
    plt.show()
    


def plot_IQ_patients(df_long_hemisphere, biomarker, fbins, dir_figs):
    plt.figure(figsize=(14, 6))
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    # Plot High vs. Low IQ for Left Hemisphere
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df_long_hemisphere[df_long_hemisphere['hemisphere'] == f'{biomarker}_left'], 
                 x='bin', y=biomarker, hue='NewTIQ', marker='o', palette=cmap, ci=None, legend=False)
    plt.title('IQ: Left Hemisphere')
    plt.xlabel('Frequency Bin')
    plt.xticks(ticks=range(len(fbin_labels)), labels=fbin_labels, rotation=45, ha='right')
    plt.ylabel(f'{biomarker} Value')
    plt.grid(False)

    # Plot High vs. Low IQ for Right Hemisphere
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df_long_hemisphere[df_long_hemisphere['hemisphere'] == f'{biomarker}_right'], 
                 x='bin', y=biomarker, hue='NewTIQ', marker='o', palette=cmap, ci=None, legend=False)
    plt.title('IQ: Right Hemisphere')
    plt.xlabel('Frequency Bin')
    plt.xticks(ticks=range(len(fbin_labels)), labels=fbin_labels, rotation=45, ha='right')
    plt.ylabel(f'{biomarker} Value')
    plt.grid(False)
    plt.tight_layout()
    
    # Add a color bar
    norm = plt.Normalize(df_long_hemisphere['NewTIQ'].min(), df_long_hemisphere['NewTIQ'].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gcf().axes, orientation='horizontal', fraction=0.02, pad=0.05, label='NewTIQ')
    
    # Save the figure
    plt.savefig(os.path.join(dir_figs, f'IQpatienthemisphere_comparison_{biomarker}_{date}.eps'))
    plt.savefig(os.path.join(dir_figs, f'IQpatienthemisphere_comparison_{biomarker}_{date}.png'))
    plt.show()
    

# plot_IQ_patients(df_long_hemisphere_fEI, 'fEI')
# plot_IQ_patients(df_long_hemisphere_DFA, 'DFA', fbin_labels, dir_figs)

    #%%
    
def test_differences_in_groups(df, resec_cond, ctrl_cond):
    # Filter necessary data, removing NaN values
    df_filtered = df.dropna(subset=[resec_cond, ctrl_cond])
    
    # Shapiro-Wilk Normality Test for both groups
    sw_stat_resec, p_value_resec_sw = stats.shapiro(df_filtered[resec_cond])
    sw_stat_ctrl, p_value_ctrl_sw = stats.shapiro(df_filtered[ctrl_cond])
    # print(f"Shapiro-Wilk Normality Test for Resection: Statistic={sw_stat_resec}, p-value={p_value_resec_sw}")
    # print(f"Shapiro-Wilk Normality Test for Control: Statistic={sw_stat_ctrl}, p-value={p_value_ctrl_sw}")
    
    # Decision based on normality test for using parametric or non-parametric tests
    if p_value_resec_sw > 0.05 and p_value_ctrl_sw > 0.05:
        print("Both groups passed normality test, using paired t-test.")
        t_stat, p_value_ttest = stats.ttest_rel(df_filtered[resec_cond], df_filtered[ctrl_cond])
        print(f"Paired t-test results: t-statistic = {t_stat}, p-value = {p_value_ttest}")
    else:
        print("At least one group failed normality test, using Wilcoxon signed-rank test.")
        stat, p_value_wilcoxon = stats.wilcoxon(df_filtered[resec_cond], df_filtered[ctrl_cond])
        print(f"Wilcoxon signed-rank test results: statistic = {stat}, p-value = {p_value_wilcoxon}")

    return {
        "normality_resection": (sw_stat_resec, p_value_resec_sw),
        "normality_control": (sw_stat_ctrl, p_value_ctrl_sw),
        "ttest": ("N/A" if p_value_resec_sw <= 0.05 or p_value_ctrl_sw <= 0.05 else (t_stat, p_value_ttest)),
        "wilcoxon": ("N/A" if p_value_resec_sw > 0.05 and p_value_ctrl_sw > 0.05 else (stat, p_value_wilcoxon))
    }
    
    
    
def plot_IQ_correlation_resecctrl(dir_figs, df_meg_data, df_code, correlate, fbin):
    plt.figure(figsize=(15, 7))
    resec_cond = f'{correlate}_resec_bin{fbin}'
    ctrl_cond = f'{correlate}_ctrl_bin{fbin}'
    df_filtered = df_meg_data.dropna(subset=['NewTIQ', 'resection_hemisphere', resec_cond, ctrl_cond])
    colors = ['red', 'blue']
    ycoords = [0.85, 0.95]
    for i, condition in enumerate([resec_cond, ctrl_cond]):
        # Scatter and regression for Resection Condition
        sns.scatterplot(x='NewTIQ', y=condition, data=df_filtered, color=colors[i], alpha=0.7)
        sns.regplot(x='NewTIQ', y=condition, data=df_filtered, ci=95, scatter=False, color=colors[i], line_kws={'lw': 2})
    
        X = sm.add_constant(df_filtered['NewTIQ'])
        y = df_filtered[condition]
        model = sm.OLS(y, X).fit()
        r_value, p_value = stats.pearsonr(df_filtered['NewTIQ'], y)
        plt.annotate(f'Resection Hemisphere: R = {r_value:.2f}, p = {p_value:.4f}', xy=(0.05, ycoords[i]), xycoords='axes fraction', ha='left', va='top')
    
    plt.xlabel('IQ')
    plt.ylabel(f'{correlate} values')
    plt.title(f'Resection vs Control Hemisphere Analysis')
      
    fig_path_png = os.path.join(dir_figs, f'IQ_vs_{correlate}_resecctrl_bin{fbin}_{df_code}_{date}.png')
    os.makedirs(dir_figs, exist_ok=True)
    plt.savefig(fig_path_png)
    plt.show()
    
    # Call the function within your existing framework after plotting or in a separate analysis section
    results = test_differences_in_groups(df_filtered, resec_cond, ctrl_cond)

def plot_IQ_correlation_resecctrl_barplot(dir_figs, df_MEGanalysis, df_code, correlate, fbins):
    # Create labels for the frequency bins, rounded to one decimal point
    bin_labels = [f'{round(b[0], 1)}-{round(b[1], 1)} Hz' for b in fbins]
    
    correlations = {}
    # Iterate over each frequency bin index and the associated labels
    for idx, (fbin, label) in enumerate(zip(fbins, bin_labels)):
        resec_cond = f'{correlate}_resec_bin{idx}'
        ctrl_cond = f'{correlate}_ctrl_bin{idx}'
    
        df_filtered = df_MEGanalysis.dropna(subset=['NewTIQ', 'resection_hemisphere', resec_cond, ctrl_cond])
        
        # Calculate Pearson correlation for resection
        r_value_resec, p_value_resec = stats.pearsonr(df_filtered['NewTIQ'], df_filtered[resec_cond])
        correlations[f'fei_res_{label}'] = r_value_resec
    
        # Calculate Pearson correlation for control
        r_value_ctrl, p_value_ctrl = stats.pearsonr(df_filtered['NewTIQ'], df_filtered[ctrl_cond])
        correlations[f'fei_cont_{label}'] = r_value_ctrl
    
    # Prepare DataFrame for plotting
    data = []
    for idx, (key, value) in enumerate(correlations.items()):
        # Extract the label from the key directly
        bin_label = key.split('_')[-1]
        condition = 'Resection' if 'res' in key else 'Control'
        data.append({
            'Correlation': value,
            'Frequency Bin': bin_label,
            'Condition': condition
        })
    
    corr_df = pd.DataFrame(data)
    # Plotting with grouped bars
    plt.figure(figsize=(14, 9))
    sns.barplot(x='Frequency Bin', y='Correlation', hue='Condition', data=corr_df, palette={'Resection': 'red', 'Control': 'blue'}, alpha=0.6, dodge=True)
    
    plt.title('Correlation of fEI values with IQ across Frequency Bins')  # Title
    plt.xlabel('Frequency Bins (Hz)')  # X-axis label
    plt.ylabel('Correlation Coefficient')  # Y-axis label
    plt.axhline(0, color='grey', linestyle='--')  # Reference line at zero
    plt.legend(title='Condition', loc='upper right')  # Legend configuration
    
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()  # Display the plot
    
    # Ensure directory exists and save the plot
    fig_path_png = os.path.join(dir_figs, f'IQ_vs_{correlate}_resecctrl_barplot_{df_code}_{date}.png')
    os.makedirs(dir_figs, exist_ok=True)
    plt.savefig(fig_path_png)
    
    
def plot_IQ_correlation_combined(dir_figs, df_meg_data, df_code, correlate, fbins, rgb_colours):
    # Setup figure for multiple subplots
    plt.figure(figsize=(20, 10))
    # plt.subplots_adjust(hspace=0.4, wspace=0.4)
    # Create labels for the frequency bins
    bin_labels = [f'{round(b[0], 1)}-{round(b[1], 1)} Hz' for b in fbins]
    # Initialize storage for correlation results
    correlations = []
    # Process each frequency bin
    for idx, (fbin, label) in enumerate(zip(fbins, bin_labels)):
        resec_cond = f'{correlate}_resec_bin{idx}'
        ctrl_cond = f'{correlate}_ctrl_bin{idx}'
        # Filter data for current bin
        df_filtered = df_meg_data.dropna(subset=['NewTIQ', 'resection_hemisphere', resec_cond, ctrl_cond])
        # Calculate Pearson correlation for both conditions and store results
        if len(df_filtered) >= 2:
            r_value_resec, _ = stats.pearsonr(df_filtered['NewTIQ'], df_filtered[resec_cond])
            r_value_ctrl, _ = stats.pearsonr(df_filtered['NewTIQ'], df_filtered[ctrl_cond])
        else:
            r_value_resec, r_value_ctrl = None, None  # Use None for insufficient data
        correlations.append({
            'Frequency Bin': label, 'Condition': 'Resection', 'Correlation': r_value_resec
        })
        correlations.append({
            'Frequency Bin': label, 'Condition': 'Control', 'Correlation': r_value_ctrl
        })
        # Scatter and regression plot for current bin
        ax = plt.subplot(2, len(fbins), idx + 1)
        sns.scatterplot(x='NewTIQ', y=resec_cond, data=df_filtered, color=rgb_colours['resec'], alpha=0.7, ax=ax)
        sns.scatterplot(x='NewTIQ', y=ctrl_cond, data=df_filtered, color=rgb_colours['ctrl'], alpha=0.7, ax=ax)
        sns.regplot(x='NewTIQ', y=resec_cond, data=df_filtered, scatter=False, color=rgb_colours['resec'], ax=ax)
        sns.regplot(x='NewTIQ', y=ctrl_cond, data=df_filtered, scatter=False, color=rgb_colours['ctrl'], ax=ax)
        # ax.set_title(f'Scatter & Regression for {label}')
        ax.set_ylim([0.37, 1.34])
        ax.set_xlim([61, 127])
        ax.set_xlabel('IQ', fontsize=16)
        if idx == 0:
            ax.set_ylabel(f'{correlate}', fontsize=24)  # Only set for the first subplot
        else:
            ax.get_yaxis().set_visible(False)  # Hide y-axis labels and ticks for other subplots
    # Convert correlation data to DataFrame
    df_correlations = pd.DataFrame(correlations)
    # Bar plot for correlations across all bins
    ax2 = plt.subplot(2, 1, 2)
    sns.barplot(x='Frequency Bin', y='Correlation', hue='Condition', data=df_correlations,
                palette={'Resection': rgb_colours['resec'], 'Control': rgb_colours['ctrl']}, ax=ax2)
    # ax2.set_title('Correlation Coefficients Accross Frequency Bins')
    ax2.set_xlabel('Frequency Bins (Hz)', fontsize=24)
    ax2.set_ylabel('Pearson R', fontsize=24)
    ax2.axhline(0, color='grey', linestyle='--')
    ax2.legend_.remove()
    # ax2.legend(title='Condition')
    # plt.suptitle(f'Correlation of {correlate} values with IQ across Frequency Bins')
    plt.tight_layout()
    # Save the figure
    fig_path_png = os.path.join(dir_figs, f'test_IQ_vs_{correlate}_combined_{df_code}_{date}.png')
    os.makedirs(dir_figs, exist_ok=True)
    plt.savefig(fig_path_png)
    plt.show()
    
# def compare_coefficients(dir_figs, df_meg_data, df_code, correlate, fbins, rgb_colours):
#     plt.figure(figsize=(20, 10))
#     bin_labels = [f'{round(b[0], 1)}-{round(b[1], 1)} Hz' for b in fbins]
#     Lr, Lc, Rr, Rc = [], [], [], []
#     groups = [Lr, Lc, Rr, Rc]
#     IQ_Lr, IQ_Lc, IQ_Rr, IQ_Rc = [], [], [], []
#     group_IQs = [IQ_Lr, IQ_Lc, IQ_Rr, IQ_Rc]
#     for _, row in df_meg_data.iterrows():
#         for fbin in fbins:
#             if row['resection_hemisphere'] == 'Left':
#                 Lr.append(row[['{correlate}_resec_bin{fbin}']].values)
#                 Lc.append(row[['{correlate}_resec_bin{fbin}']].values)
#                 IQ_Lr.append(row['NewTIQ'])
#                 IQ_Lc.append(row['NewTIQ'])
#             elif row['resection_hemisphere'] == 'Right':
#                 Rr.append(row[['{correlate}_resec_bin{fbin}']].values)
#                 Rc.append(row[['{correlate}_resec_bin{fbin}']].values)
#                 IQ_Rr.append(row['NewTIQ'])
#                 IQ_Rc.append(row['NewTIQ'])
#     for group in groups:
#         group = np.array(group).flatten()
    



def add_stat_annotation(ax, x1, x2, y, text, line_height=0.02):
    """Add statistical annotation to the plot"""
    line_offset = line_height / 2
    ax.plot([x1, x1, x2, x2], [y, y + line_height, y + line_height, y], lw=1.5, color='k')
    ax.text((x1 + x2) * .5, y + line_height + line_offset, text, ha='center', va='bottom', color='k')

# def compare_coefficients_anova(dir_figs, df_meg_data, correlate, fbins, rgb_colours):
#     data = []

#     # Combine data into a single DataFrame for ANOVA
#     for _, row in df_meg_data.iterrows():
#         for i, fbin in enumerate(fbins):
#             bin_label = f'{round(fbin[0], 1)}-{round(fbin[1], 1)} Hz'
#             if row['resection_hemisphere'] == 'Left':
#                 data.append([row['patient_ID'], 'Left', 'Resected', row[f'{correlate}_resec_bin{i}'], row['NewTIQ'], bin_label, 'Lr'])
#                 data.append([row['patient_ID'], 'Left', 'Control', row[f'{correlate}_ctrl_bin{i}'], row['NewTIQ'], bin_label, 'Lc'])
#             elif row['resection_hemisphere'] == 'Right':
#                 data.append([row['patient_ID'], 'Right', 'Resected', row[f'{correlate}_resec_bin{i}'], row['NewTIQ'], bin_label, 'Rr'])
#                 data.append([row['patient_ID'], 'Right', 'Control', row[f'{correlate}_ctrl_bin{i}'], row['NewTIQ'], bin_label, 'Rc'])

#     df_anova = pd.DataFrame(data, columns=['patient_ID', 'Hemisphere', 'Condition', correlate, 'IQ', 'Frequency_Bin', 'Group'])

#     # Fit regression models for each group
#     formula = f'{correlate} ~ IQ'
#     groups = ['Lr', 'Lc', 'Rr', 'Rc']
#     regression_results = {}

#     for group in groups:
#         group_data = df_anova[df_anova['Group'] == group]
#         model = ols(formula, data=group_data).fit()
#         regression_results[group] = model
#         print(f'\nRegression results for {group}:\n', model.summary())

#     # Create a summary DataFrame of the regression coefficients
#     summary_data = []
#     for group, model in regression_results.items():
#         coeff = model.params['IQ']
#         se = model.bse['IQ']
#         summary_data.append([group, coeff, se])

#     df_summary = pd.DataFrame(summary_data, columns=['Group', 'Coefficient', 'SE'])

#     # Plot the regression coefficients
#     plt.figure(figsize=(10, 6))
#     ax = sns.barplot(x='Group', y='Coefficient', data=df_summary, capsize=0.1)
#     ax.errorbar(x=df_summary['Group'], y=df_summary['Coefficient'], yerr=df_summary['SE'], fmt='none', c='black', capsize=5)

#     # Adding statistical annotations manually
#     comparisons = [('Lr', 'Lc'), ('Rr', 'Rc'), ('Lr', 'Rr'), ('Lr', 'Rc'), ('Lc', 'Rr'), ('Lc', 'Rc')]
#     max_y = df_summary['Coefficient'].max() + df_summary['SE'].max() * 1.5
    
#     for group1, group2 in comparisons:
#         coeff1 = df_summary[df_summary['Group'] == group1]['Coefficient'].values[0]
#         coeff2 = df_summary[df_summary['Group'] == group2]['Coefficient'].values[0]
#         se1 = df_summary[df_summary['Group'] == group1]['SE'].values[0]
#         se2 = df_summary[df_summary['Group'] == group2]['SE'].values[0]
#         z = (coeff1 - coeff2) / np.sqrt(se1**2 + se2**2)
#         p_value = 2 * (1 - norm.cdf(abs(z)))
#         if p_value < 0.05:
#             add_stat_annotation(ax, groups.index(group1), groups.index(group2), max_y, '*' if p_value < 0.05 else 'ns')
    
#     plt.title(f'Regression Coefficients of {correlate} ~ IQ by Group')
#     plt.tight_layout()
#     plt.savefig(os.path.join(dir_figs, f'Regression_Coefficients_{correlate}_IQ_by_Group_{date}.png'))
#     plt.show()

#     return df_summary

# df_coeffsummary = compare_coefficients_anova(dir_figs, df_MEGanalysis, 'fEI', fbins, rgb_colours)


def add_stat_annotation(ax, x1, x2, y, text, line_height=0.02):
    """Add statistical annotation to the plot"""
    line_offset = line_height / 2
    ax.plot([x1, x1, x2, x2], [y, y + line_height, y + line_height, y], lw=1.5, color='k')
    ax.text((x1 + x2) * .5, y + line_height + line_offset, text, ha='center', va='bottom', color='k')

def compare_coefficients_anova(dir_figs, df_meg_data, correlate, fbins, rgb_colours):
    data = []

    # Combine data into a single DataFrame for ANOVA
    for _, row in df_meg_data.iterrows():
        for i, fbin in enumerate(fbins):
            bin_label = f'{round(fbin[0], 1)}-{round(fbin[1], 1)} Hz'
            if row['resection_hemisphere'] == 'Left':
                data.append([row['patient_ID'], 'Left', 'Resected', row[f'{correlate}_resec_bin{i}'], row['NewTIQ'], bin_label, 'Lr'])
                data.append([row['patient_ID'], 'Left', 'Control', row[f'{correlate}_ctrl_bin{i}'], row['NewTIQ'], bin_label, 'Lc'])
            elif row['resection_hemisphere'] == 'Right':
                data.append([row['patient_ID'], 'Right', 'Resected', row[f'{correlate}_resec_bin{i}'], row['NewTIQ'], bin_label, 'Rr'])
                data.append([row['patient_ID'], 'Right', 'Control', row[f'{correlate}_ctrl_bin{i}'], row['NewTIQ'], bin_label, 'Rc'])

    df_anova = pd.DataFrame(data, columns=['patient_ID', 'Hemisphere', 'Condition', correlate, 'IQ', 'Frequency_Bin', 'Group'])

    # Perform two-way ANOVA
    formula = f'{correlate} ~ C(Hemisphere) * C(Condition) * C(Frequency_Bin) + IQ'
    model = smf.ols(formula, data=df_anova).fit()
    anova_results = anova_lm(model, typ=2)

    print(anova_results)

    # Post-hoc comparisons with Tukey HSD
    mc = multi.MultiComparison(df_anova[correlate], df_anova['Group'])
    tukey_result = mc.tukeyhsd()

    print(tukey_result)

    # Plotting (optional)
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='Frequency_Bin', y=correlate, hue='Group', data=df_anova, showfliers=False)
    sns.stripplot(x='Frequency_Bin', y=correlate, hue='Group', data=df_anova, dodge=True, alpha=0.8)

    # Adding statistical annotations using Tukey HSD results
    frequency_bins = df_anova['Frequency_Bin'].unique()
    groups = list(df_anova['Group'].unique())  # Convert to list to use index method
    group_pairs = tukey_result._results_table.data[1:]  # Skip the header

    for pair in group_pairs:
        group1, group2, meandiff, p_adj, lower, upper, reject = pair
        if reject:
            for i, bin_label in enumerate(frequency_bins):
                group1_data = df_anova[(df_anova['Frequency_Bin'] == bin_label) & (df_anova['Group'] == group1)][correlate]
                group2_data = df_anova[(df_anova['Frequency_Bin'] == bin_label) & (df_anova['Group'] == group2)][correlate]
                max_y = max(group1_data.max(), group2_data.max()) * 1.1
                if p_adj < 0.05:
                    add_stat_annotation(ax, i - 0.2 + groups.index(group1) * 0.1, i - 0.2 + groups.index(group2) * 0.1, max_y, '*' if p_adj < 0.05 else 'ns')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.title(f'{correlate} by Frequency Bin and Group')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, f'ANOVA_{correlate}_frequency_bins_{date}.png'))
    plt.show()

    return df_anova
    
df_anova = compare_coefficients_anova(dir_figs, df_MEGanalysis, 'DFA', fbins, rgb_colours)
df_anova = compare_coefficients_anova(dir_figs, df_MEGanalysis, 'HLP', fbins, rgb_colours)
df_anova = compare_coefficients_anova(dir_figs, df_MEGanalysis, 'BIS', fbins, rgb_colours)

def plot_IQ_correlation_combined_sig(dir_figs, df_meg_data, df_code, correlate, fbins, rgb_colours):
    plt.figure(figsize=(20, 10))
    bin_labels = [f'{round(b[0], 1)}-{round(b[1], 1)} Hz' for b in fbins]
    correlations = []

    for idx, (fbin, label) in enumerate(zip(fbins, bin_labels)):
        resec_cond = f'{correlate}_resec_bin{idx}'
        ctrl_cond = f'{correlate}_ctrl_bin{idx}'
        df_filtered = df_meg_data.dropna(subset=['NewTIQ', 'resection_hemisphere', resec_cond, ctrl_cond])

        if len(df_filtered) >= 2:
            r_value_resec, p_value_resec = stats.pearsonr(df_filtered['NewTIQ'], df_filtered[resec_cond])
            r_value_ctrl, p_value_ctrl = stats.pearsonr(df_filtered['NewTIQ'], df_filtered[ctrl_cond])
        else:
            r_value_resec, p_value_resec = None, None
            r_value_ctrl, p_value_ctrl = None, None

        correlations.append({
            'Frequency Bin': label, 'Condition': 'Resection', 'Correlation': r_value_resec, 'P-Value': p_value_resec
        })
        correlations.append({
            'Frequency Bin': label, 'Condition': 'Control', 'Correlation': r_value_ctrl, 'P-Value': p_value_ctrl
        })
        
        ax = plt.subplot(2, len(fbins), idx + 1)
        sns.scatterplot(x='NewTIQ', y=resec_cond, data=df_filtered, color=rgb_colours['resec'], alpha=0.7, ax=ax)
        sns.scatterplot(x='NewTIQ', y=ctrl_cond, data=df_filtered, color=rgb_colours['ctrl'], alpha=0.7, ax=ax)
        sns.regplot(x='NewTIQ', y=resec_cond, data=df_filtered, scatter=False, color=rgb_colours['resec'], ax=ax)
        sns.regplot(x='NewTIQ', y=ctrl_cond, data=df_filtered, scatter=False, color=rgb_colours['ctrl'], ax=ax)
        
        if p_value_resec is not None:  # Corrected condition check
            fp_resec = format_p(p_value_resec)
            ax.annotate(f'p {fp_resec}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9, ha='left', va='top', color=rgb_colours['resec'])
        if p_value_ctrl is not None:
            fp_ctrl = format_p(p_value_ctrl)
            ax.annotate(f'p {fp_ctrl}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9, ha='left', va='top', color=rgb_colours['ctrl'])
         
        # Determine y-axis limits
        ymin = min(df_filtered[resec_cond].min(), df_filtered[ctrl_cond].min())
        ymax = max(df_filtered[resec_cond].max(), df_filtered[ctrl_cond].max())
        y_range = ymax - ymin
        ax.set_ylim([ymin - 0.1 * y_range, ymax + 0.1 * y_range])
        ax.set_xlim([61, 127])
        ax.set_xlabel('IQ', fontsize=16)
        if idx == 0:
            ax.set_ylabel(correlate, fontsize=24)
        else:
            ax.get_yaxis().set_visible(False)

    df_correlations = pd.DataFrame(correlations)
    ax2 = plt.subplot(2, 1, 2)
    sns.barplot(x='Frequency Bin', y='Correlation', hue='Condition', data=df_correlations, palette={'Resection': rgb_colours['resec'], 'Control': rgb_colours['ctrl']}, ax=ax2)
    ax2.set_xlabel('Frequency Bins (Hz)', fontsize=24)
    ax2.set_ylabel('Pearson R', fontsize=24)
    ax2.set_ylim(-0.85, 0.9)
    ax2.axhline(0, color='grey', linestyle='--')
    plt.suptitle(f'{correlate} vs IQ {df_code}')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, f'{df_code}_combined_{correlate}_{date}.png'))
    plt.show()


def plot_IQ_correlation_bands(dir_figs, df_meg_data, df_code, correlate, fband, rgb_colours):
    correlations = []
    resec_cond = f'{correlate}_resec_band_{fband}'
    ctrl_cond = f'{correlate}_ctrl_band_{fband}'
    df_filtered = df_meg_data.dropna(subset=['NewTIQ', 'resection_hemisphere', resec_cond, ctrl_cond])
    if len(df_filtered) >= 2:
            r_value_resec, p_value_resec = stats.pearsonr(df_filtered['NewTIQ'], df_filtered[resec_cond])
            r_value_ctrl, p_value_ctrl = stats.pearsonr(df_filtered['NewTIQ'], df_filtered[ctrl_cond])
    else:
        r_value_resec, p_value_resec = None, None
        r_value_ctrl, p_value_ctrl = None, None

    correlations.append({
        'Frequency Bin': fband, 'Condition': 'Resection', 'Correlation': r_value_resec, 'P-Value': p_value_resec
    })
    correlations.append({
        'Frequency Bin': fband, 'Condition': 'Control', 'Correlation': r_value_ctrl, 'P-Value': p_value_ctrl
    })
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sns.scatterplot(x='NewTIQ', y=resec_cond, data=df_filtered, color=rgb_colours['resec'], alpha=0.7, ax=ax)
    sns.scatterplot(x='NewTIQ', y=ctrl_cond, data=df_filtered, color=rgb_colours['ctrl'], alpha=0.7, ax=ax)
    sns.regplot(x='NewTIQ', y=resec_cond, data=df_filtered, scatter=False, color=rgb_colours['resec'], ax=ax)
    sns.regplot(x='NewTIQ', y=ctrl_cond, data=df_filtered, scatter=False, color=rgb_colours['ctrl'], ax=ax)
    
    if p_value_resec is not None:
        fp_resec = format_p(p_value_resec)
        ax.annotate(f'p {fp_resec}, r={r_value_resec:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9, ha='left', va='top', color=rgb_colours['resec'])
    if p_value_ctrl is not None:
        fp_ctrl = format_p(p_value_ctrl)
        ax.annotate(f'p {fp_ctrl}, r={r_value_ctrl:.3f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9, ha='left', va='top', color=rgb_colours['ctrl'])
        
    # Determine y-axis limits
    ymin = min(df_MEGanalysis[resec_cond].min(), df_MEGanalysis[ctrl_cond].min())
    ymax = max(df_MEGanalysis[resec_cond].max(), df_MEGanalysis[ctrl_cond].max())
    y_range = ymax - ymin
    ax.set_ylim([ymin - 0.1 * y_range, ymax + 0.1 * y_range])
    ax.set_xlim([61, 127])
    ax.set_title(f'Frequency Band {fband}, {df_code} Hemisphere(s)')
    ax.set_ylabel(correlate, fontsize=24)
    ax.set_xlabel('IQ', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, f'{df_code}_{fband}_band_{correlate}_regression_{ROI}_{date}.png'))
    plt.show()

    
        
        


def plot_IQ_correlation_combined_RCavg_sig(dir_figs, df_meg_data, df_code, correlate, fbins, rgb_colours):
    plt.figure(figsize=(20, 10))
    bin_labels = [f'{round(b[0], 1)}-{round(b[1], 1)} Hz' for b in fbins]
    correlations = []

    for idx, (fbin, label) in enumerate(zip(fbins, bin_labels)):
        resec_cond = f'{correlate}_resec_bin{idx}'
        ctrl_cond = f'{correlate}_ctrl_bin{idx}'
        
        # Calculate the average value for resection and control for each patient
        df_filtered = df_meg_data.dropna(subset=['NewTIQ', 'resection_hemisphere', resec_cond, ctrl_cond]).copy()
        df_filtered[f'{correlate}_left'] = df_filtered.apply(
            lambda row: row[resec_cond] if row['resection_hemisphere'] == 'Left' else row[ctrl_cond], axis=1
        )
        df_filtered[f'{correlate}_right'] = df_filtered.apply(
            lambda row: row[ctrl_cond] if row['resection_hemisphere'] == 'Left' else row[resec_cond], axis=1
        )
        
        if len(df_filtered) >= 2:
            r_value_left, p_value_left = stats.pearsonr(df_filtered['NewTIQ'], df_filtered[f'{correlate}_left'])
            r_value_right, p_value_right = stats.pearsonr(df_filtered['NewTIQ'], df_filtered[f'{correlate}_right'])
        else:
            r_value_left, p_value_left = None, None
            r_value_right, p_value_right = None, None

        correlations.append({
            'Frequency Bin': label, 'Hemisphere': 'Left', 'Correlation': r_value_left, 'P-Value': p_value_left
        })
        correlations.append({
            'Frequency Bin': label, 'Hemisphere': 'Right', 'Correlation': r_value_right, 'P-Value': p_value_right
        })
        
        ax = plt.subplot(2, len(fbins), idx + 1)
        sns.scatterplot(x='NewTIQ', y=f'{correlate}_left', data=df_filtered, color=rgb_colours['left'], alpha=0.7, ax=ax)
        sns.regplot(x='NewTIQ', y=f'{correlate}_left', data=df_filtered, scatter=False, color=rgb_colours['left'], ax=ax)
        sns.scatterplot(x='NewTIQ', y=f'{correlate}_right', data=df_filtered, color=rgb_colours['right'], alpha=0.7, ax=ax)
        sns.regplot(x='NewTIQ', y=f'{correlate}_right', data=df_filtered, scatter=False, color=rgb_colours['right'], ax=ax)
        
        if p_value_left is not None:
            ax.annotate(f'p={p_value_left:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9, ha='left', va='top', color=rgb_colours['left'])
        if p_value_right is not None:
            ax.annotate(f'p={p_value_right:.3f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9, ha='left', va='top', color=rgb_colours['right'])
            
        ax.set_ylim([0.37, 1.34])
        ax.set_xlim([61, 127])
        ax.set_xlabel('IQ', fontsize=16)
        if idx == 0:
            ax.set_ylabel(correlate, fontsize=24)
        else:
            ax.get_yaxis().set_visible(False)

    df_correlations = pd.DataFrame(correlations)
    ax2 = plt.subplot(2, 1, 2)
    sns.barplot(x='Frequency Bin', y='Correlation', hue='Hemisphere', data=df_correlations, palette={'Left': rgb_colours['left'], 'Right': rgb_colours['right']}, ax=ax2)
    ax2.set_xlabel('Frequency Bins (Hz)', fontsize=24)
    ax2.set_ylabel('Pearson R', fontsize=24)
    ax2.set_ylim([-0.3, 0.9])
    ax2.axhline(0, color='grey', linestyle='--')
    plt.suptitle(f'{correlate} vs IQ {df_code}')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, f'{df_code}_combined_{correlate}_{date}.png'))
    plt.show()



def plot_regression_by_hemisphere(correlate, condition, df, fbins, fbin):
    cond_code = f'{correlate}_{condition}_bin{fbin}'
    # Dropping rows with missing values in specified columns
    df_cond = df.dropna(subset=['NewTIQ', 'resection_hemisphere', cond_code])
    # Separating data by hemisphere
    df_Lplot = df_cond[df_cond['resection_hemisphere'] == 'Left']
    df_Rplot = df_cond[df_cond['resection_hemisphere'] == 'Right']
    # Fit the model for Left Hemisphere
    X_L = sm.add_constant(df_Lplot['NewTIQ'])  # Adding a constant for intercept
    model_L = sm.OLS(df_Lplot[cond_code], X_L)
    results_L = model_L.fit()
    # Fit the model for Right Hemisphere
    X_R = sm.add_constant(df_Rplot['NewTIQ'])  # Adding a constant for intercept
    model_R = sm.OLS(df_Rplot[cond_code], X_R)
    results_R = model_R.fit()
    # Plotting
    plt.figure(figsize=(20, 10))
    # Left hemisphere plots
    sns.scatterplot(x='NewTIQ', y=cond_code, data=df_Lplot, color=rgb_colours['Left'])
    sns.regplot(x='NewTIQ', y=cond_code, data=df_Lplot, scatter=False, color=rgb_colours['Left'], label=f'Left Hemisphere R²={results_L.rsquared:.2f}, p={results_L.pvalues[1]:.2f}')
    # Right hemisphere plots
    sns.scatterplot(x='NewTIQ', y=cond_code, data=df_Rplot, color=rgb_colours['Right'])
    sns.regplot(x='NewTIQ', y=cond_code, data=df_Rplot, scatter=False, color=rgb_colours['Right'], label=f'Right Hemisphere R²={results_R.rsquared:.2f}, p={results_R.pvalues[1]:.2f}')

    # plt.title(f'{condition} Condition in Left and Right Hemispheres')
    plt.xlim([62.6, 125.4])
    plt.ylim([0.45, 1.23])
    plt.xlabel('IQ', fontsize=24)
    plt.ylabel(f'fEI in Range {round(fbins[fbin][0], 1)} - {round(fbins[fbin][1], 1)} Hz', fontsize=24) # should be changed by bin but no time
    plt.legend(fontsize=16)
    # Save the figure
    fig_path_png = os.path.join(dir_figs, f'IQ_vs_{correlate}_{condition}condition_bin{fbin}_{date}.png')
    os.makedirs(dir_figs, exist_ok=True)
    plt.savefig(fig_path_png)
    plt.show()


def plot_regression_by_hemisphere_avgctrlresec(correlate, df, fbins, fbin, dir_figs, rgb_colours):
    resec_code = f'{correlate}_resec_bin{fbin}'
    ctrl_code = f'{correlate}_ctrl_bin{fbin}'
    avg_code = f'{correlate}_avg_bin{fbin}'
    
    # Check and create new column with average of both conditions
    df[avg_code] = df[[resec_code, ctrl_code]].mean(axis=1)
    
    # Dropping rows with missing values in specified columns
    df_cond = df.dropna(subset=['NewTIQ', 'resection_hemisphere', avg_code])
    
    # Separating data by hemisphere
    df_Lplot = df_cond[df_cond['resection_hemisphere'] == 'Left']
    df_Rplot = df_cond[df_cond['resection_hemisphere'] == 'Right']
    
    # Fit the model for Left Hemisphere
    X_L = sm.add_constant(df_Lplot['NewTIQ'])  # Adding a constant for intercept
    model_L = sm.OLS(df_Lplot[avg_code], X_L)
    results_L = model_L.fit()
    
    # Fit the model for Right Hemisphere
    X_R = sm.add_constant(df_Rplot['NewTIQ'])  # Adding a constant for intercept
    model_R = sm.OLS(df_Rplot[avg_code], X_R)
    results_R = model_R.fit()
    
    # Plotting
    plt.figure(figsize=(20, 10))
    
    # Left hemisphere plots
    sns.scatterplot(x='NewTIQ', y=avg_code, data=df_Lplot, color=rgb_colours['Left'])
    sns.regplot(x='NewTIQ', y=avg_code, data=df_Lplot, scatter=False, color=rgb_colours['Left'], label=f'Left Hemisphere: R²={results_L.rsquared:.2f}, p={results_L.pvalues[1]:.3f}')
    
    # Right hemisphere plots
    sns.scatterplot(x='NewTIQ', y=avg_code, data=df_Rplot, color=rgb_colours['Right'])
    sns.regplot(x='NewTIQ', y=avg_code, data=df_Rplot, scatter=False, color=rgb_colours['Right'], label=f'Right Hemisphere: R²={results_R.rsquared:.2f}, p={results_R.pvalues[1]:.3f}')
    
    #plt.title(f'{condition} Condition in Left and Right Hemispheres')
    plt.xlim([62.6, 125.4])
    plt.ylim([0.45, 1.23])
    plt.xlabel('IQ', fontsize=24)
    plt.ylabel(f'{correlate} in Range {round(fbins[fbin][0], 1)} - {round(fbins[fbin][1], 1)} Hz', fontsize=24)
    plt.legend(fontsize=20)
    plt.tight_layout()
    
    # Save the figure
    fig_path_png = os.path.join(dir_figs, f'IQ_vs_{correlate}_avgctrlresec_bin{fbin}_{date}.png')
    os.makedirs(dir_figs, exist_ok=True)
    plt.savefig(fig_path_png)
    plt.show()

def plot_biomarkerbyfbin_colouredIQ(correlate, df, fbins, dir_figs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for fbin in range(len(fbins)):
        resec_code = f'{correlate}_resec_bin{fbin}'
        ctrl_code = f'{correlate}_ctrl_bin{fbin}'
        avg_code = f'{correlate}_avg_bin{fbin}'

        df[avg_code] = df[[resec_code, ctrl_code]].mean(axis=1)
        df_cond = df.dropna(subset=['NewTIQ', 'resection_hemisphere', avg_code])

        df_Lplot = df_cond[df_cond['resection_hemisphere'] == 'Left']
        df_Rplot = df_cond[df_cond['resection_hemisphere'] == 'Right']

        # Normalize NewTIQ for color mapping
        norm = plt.Normalize(df_cond['NewTIQ'].min(), df_cond['NewTIQ'].max())
        cmap = plt.get_cmap('coolwarm')

        for _, row in df_Lplot.iterrows():
            color = cmap(norm(row['NewTIQ']))
            axes[0].plot(fbin, row[avg_code], color=color)
        for _, row in df_Rplot.iterrows():
            color = cmap(norm(row['NewTIQ']))
            axes[1].plot(fbin, row[avg_code], color=color)

    axes[0].set_title('Left Hemisphere')
    axes[1].set_title('Right Hemisphere')

    # Add colorbars
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='horizontal', label='NewTIQ')

    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, f'{correlate}_fbin_colored_IQ_{date}.png'))
    plt.show()

#%%

def plot_IQ_correlation_LR_resecctrl(dir_figs, df_meg_data, df_code, correlate, fbin):
    plt.figure(figsize=(15, 7))
    resec_cond = f'{correlate}_resec_bin{fbin}'
    ctrl_cond = f'{correlate}_ctrl_bin{fbin}'
    df_resec = df_meg_data.dropna(subset=['NewTIQ', 'resection_hemisphere', resec_cond])
    df_ctrl = df_meg_data.dropna(subset=['NewTIQ', 'resection_hemisphere', ctrl_cond])
    for i, hemisphere in enumerate(['Left', 'Right']):
        opposite_hemisphere = 'Right' if hemisphere == 'Left' else 'Left'
        ax = plt.subplot(1, 2, i+1)
        df_plotresec = df_resec[df_resec['resection_hemisphere'] == hemisphere]
        df_plotctrl = df_ctrl[df_ctrl['resection_hemisphere'] == opposite_hemisphere]
    
        # Scatter and regression for Resection Condition
        sns.scatterplot(x='NewTIQ', y=resec_cond, data=df_plotresec, color='red', alpha=0.7, ax=ax)
        sns.regplot(x='NewTIQ', y=resec_cond, data=df_plotresec, ci=95, scatter=False, ax=ax, color='red', line_kws={'lw': 2})
    
        # Scatter and regression for Control Condition
        sns.scatterplot(x='NewTIQ', y=ctrl_cond, data=df_plotctrl, color='blue', alpha=0.5, ax=ax)
        sns.regplot(x='NewTIQ', y=ctrl_cond, data=df_plotctrl, ci=95, scatter=False, ax=ax, color='blue', line_kws={'lw': 2})
    
        # Statistical annotations for Resection
        X = sm.add_constant(df_plotresec['NewTIQ'])
        y = df_plotresec[resec_cond]
        model = sm.OLS(y, X).fit()
        r_value, p_value = stats.pearsonr(df_plotresec['NewTIQ'], y)
        plt.annotate(f'Resection R = {r_value:.2f}, p = {p_value:.4f}', xy=(0.05, 0.85), xycoords='axes fraction', ha='left', va='top', bbox=dict(boxstyle="round", alpha=0.5))
    
        # Statistical annotations for Control
        X = sm.add_constant(df_plotctrl['NewTIQ'])
        y = df_plotctrl[ctrl_cond]
        model = sm.OLS(y, X).fit()
        r_value, p_value = stats.pearsonr(df_plotctrl['NewTIQ'], y)
        plt.annotate(f'Control R = {r_value:.2f}, p = {p_value:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', bbox=dict(boxstyle="round", alpha=0.5))
    
        # Set labels and title for subplot
        plt.xlabel('IQ')
        plt.ylabel(f'{correlate} values')
        plt.title(f'{hemisphere} Hemisphere Analysis')
    
    # General figure settings
    plt.suptitle(f'Correlation of IQ with {correlate} bin {fbin}')
    fig_path_png = os.path.join(dir_figs, f'IQ_vs_{correlate}_LRresecctrl_bin{fbin}_{df_code}_{date}.png')
    os.makedirs(dir_figs, exist_ok=True)
    plt.savefig(fig_path_png)
    plt.show()
    
    
def plot_IQ_correlation_LR_resecctrl_barplot(dir_figs, df_meg_data, df_code, correlate, fbins):
    plt.figure(figsize=(18, 8))  # Adjust size to accommodate both subplots
    
    bin_labels = [f'{round(b[0], 1)}-{round(b[1], 1)} Hz' for b in fbins]
    results = []

    for idx, label in enumerate(bin_labels):
        resec_cond = f'{correlate}_resec_bin{idx}'
        ctrl_cond = f'{correlate}_ctrl_bin{idx}'

        for hemisphere in ['Left', 'Right']:
            df_hemi = df_meg_data[(df_meg_data['resection_hemisphere'] == hemisphere) & 
                                  df_meg_data[resec_cond].notna() & 
                                  df_meg_data[ctrl_cond].notna() & 
                                  df_meg_data['NewTIQ'].notna()]

            if len(df_hemi) >= 2:
                # Correlation for resection
                r_value_resec, _ = stats.pearsonr(df_hemi['NewTIQ'], df_hemi[resec_cond])
                # Correlation for control
                r_value_ctrl, _ = stats.pearsonr(df_hemi['NewTIQ'], df_hemi[ctrl_cond])
            else:
                r_value_resec, r_value_ctrl = None, None

            results.extend([
                {'Hemisphere': hemisphere, 'Condition': 'Resection', 'Correlation': r_value_resec, 'Frequency Bin': label},
                {'Hemisphere': hemisphere, 'Condition': 'Control', 'Correlation': r_value_ctrl, 'Frequency Bin': label}
            ])

    df_results = pd.DataFrame(results)

    # Plot each hemisphere's data in a subplot
    for i, hemisphere in enumerate(['Left', 'Right']):
        ax = plt.subplot(1, 2, i + 1)
        sns.barplot(x='Frequency Bin', y='Correlation', hue='Condition',
                    data=df_results[df_results['Hemisphere'] == hemisphere], palette={'Resection': 'red', 'Control': 'blue'}, alpha=0.6, ax=ax)
        ax.set_title(f'{hemisphere} Hemisphere')
        ax.axhline(0, color='grey', linestyle='--')
        ax.set_xlabel('Frequency Bins (Hz)')
        ax.set_ylabel('Correlation Coefficient')
        ax.legend(loc='upper right')

    plt.suptitle(f'Correlation of {correlate} values with IQ across Frequency Bins')
    plt.tight_layout()

    fig_path_png = os.path.join(dir_figs, f'IQ_vs_{correlate}_LRresecctrl_barplot_{df_code}_{date}.png')
    plt.savefig(fig_path_png)
    plt.show()
    return df_results


#%% correlate 1/f to IQ

## forgot to split by left and right when avging slope and offset!!!
df_1favgs = pd.read_csv('/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/procdata/MEGs/MTG_allchannels/1foffset_1fslope_allMTGs_avg_20240702.csv')
if 'Unnamed: 0' in df_1favgs.columns:
    df_1favgs.set_index('Unnamed: 0', inplace=True)

case_numbers = [re.search(r'(\d{4})', col).group(1) for col in df_1favgs.columns if re.search(r'(\d{4})', col)]

# Initialize new columns in df_MEGanalysis
new_columns = ['1fslope_resec', '1fslope_ctrl', '1foffset_resec', '1foffset_ctrl']
for col in new_columns:
    df_MEGanalysis[col] = pd.NA

# Fill in the new columns with data from df_1favgs
for case in case_numbers:
    # Extract the relevant data for the current case from df_1favgs
    slope_L = df_1favgs[f'{case}_avg_L'].loc['Slope']
    slope_R = df_1favgs[f'{case}_avg_R'].loc['Slope']
    offset_L = df_1favgs[f'{case}_avg_L'].loc['Offset']
    offset_R = df_1favgs[f'{case}_avg_R'].loc['Offset']
    
    # Update df_MEGanalysis with the extracted data
    mask = df_MEGanalysis['file'].str.contains(f'_case{case}_')
    for idx in df_MEGanalysis[mask].index:
        if 'LeftResec' in df_MEGanalysis.loc[idx, 'file']:
            df_MEGanalysis.loc[idx, '1fslope_resec'] = slope_L
            df_MEGanalysis.loc[idx, '1fslope_ctrl'] = slope_R
            df_MEGanalysis.loc[idx, '1foffset_resec'] = offset_L
            df_MEGanalysis.loc[idx, '1foffset_ctrl'] = offset_R
        else:
            df_MEGanalysis.loc[idx, '1fslope_resec'] = slope_R
            df_MEGanalysis.loc[idx, '1fslope_ctrl'] = slope_L
            df_MEGanalysis.loc[idx, '1foffset_resec'] = offset_R
            df_MEGanalysis.loc[idx, '1foffset_ctrl'] = offset_L

def plot_IQ_correlation_1f_sig(dir_figs, df_meg_data, df_code, f_correlate, rgb_colours):
    plt.figure(figsize=(10, 6))
    correlations = []

    resec_cond = f'{f_correlate}_resec'
    ctrl_cond = f'{f_correlate}_ctrl'
    
    df_filtered = df_meg_data.dropna(subset=['NewTIQ', 'resection_hemisphere', resec_cond, ctrl_cond])
    # Ensure that the data is numeric
    df_filtered['NewTIQ'] = pd.to_numeric(df_filtered['NewTIQ'], errors='coerce')
    df_filtered[resec_cond] = pd.to_numeric(df_filtered[resec_cond], errors='coerce')
    df_filtered[ctrl_cond] = pd.to_numeric(df_filtered[ctrl_cond], errors='coerce')
    if len(df_filtered) >= 2:
        r_value_resec, p_value_resec = stats.pearsonr(df_filtered['NewTIQ'], df_filtered[resec_cond])
        r_value_ctrl, p_value_ctrl = stats.pearsonr(df_filtered['NewTIQ'], df_filtered[ctrl_cond])
    else:
        r_value_resec, p_value_resec = None, None
        r_value_ctrl, p_value_ctrl = None, None

    correlations.append({
        'Condition': 'Resection', 'Correlation': r_value_resec, 'P-Value': p_value_resec
    })
    correlations.append({
        'Condition': 'Control', 'Correlation': r_value_ctrl, 'P-Value': p_value_ctrl
    })
    
    ax = plt.subplot(1, 1, 1)  # Corrected indexing for a single plot
    sns.scatterplot(x='NewTIQ', y=resec_cond, data=df_filtered, color=rgb_colours['resec'], alpha=0.7, ax=ax)
    sns.scatterplot(x='NewTIQ', y=ctrl_cond, data=df_filtered, color=rgb_colours['ctrl'], alpha=0.7, ax=ax)
    sns.regplot(x='NewTIQ', y=resec_cond, data=df_filtered, scatter=False, color=rgb_colours['resec'], ax=ax)
    sns.regplot(x='NewTIQ', y=ctrl_cond, data=df_filtered, scatter=False, color=rgb_colours['ctrl'], ax=ax)
    
    if p_value_resec is not None:  # Corrected condition check
        fp_resec = format_p(p_value_resec)
        ax.annotate(f'p {fp_resec}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9, ha='left', va='top', color=rgb_colours['resec'])
    if p_value_ctrl is not None:
        fp_ctrl = format_p(p_value_ctrl)
        ax.annotate(f'p {fp_ctrl}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9, ha='left', va='top', color=rgb_colours['ctrl'])
     
    # Determine y-axis limits
    ymin = min(df_filtered[resec_cond].min(), df_filtered[ctrl_cond].min())
    ymax = max(df_filtered[resec_cond].max(), df_filtered[ctrl_cond].max())
    y_range = ymax - ymin
    ax.set_ylim([ymin - 0.1 * y_range, ymax + 0.1 * y_range])
    ax.set_xlim([61, 127])
    ax.set_xlabel('IQ', fontsize=16)
    ax.set_ylabel(f_correlate, fontsize=24)

    df_correlations = pd.DataFrame(correlations)

    plt.suptitle(f'{f_correlate} vs IQ {df_code}')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, f'{df_code}_combined_{f_correlate}_{date}.png'))
    plt.show()



#%%
for fbin in range(len(fbins)):
    # plot_IQ_correlation_LR_resecctrl(dir_figs, df_MEGanalysis, 'MEG_LRA21c', 'fEI', fbin)
    # plot_IQ_correlation_LR_resecctrl(dir_figs, df_MEGanalysis, 'MEG_LRA21c', 'DFA', fbin)
    plot_IQ_correlation_resecctrl(dir_figs, df_MEGanalysis, 'MEG_LRA21c', 'fEI', fbin)
    
plot_IQ_correlation_resecctrl_barplot(dir_figs, df_MEGanalysis, 'MEG_LRA21c', 'fEI', fbins)

#%%
ROI_code = 'allMTGs'
df_MEGanalysis.to_csv(os.path.join(dir_procdata, f'df_MEGanalysis_DFA_fEI_BIS_HLP_{ROI_code}.csv'))

df_MEGanalysis_L = df_MEGanalysis[df_MEGanalysis['resection_hemisphere'] == 'Left']
df_MEGanalysis_R = df_MEGanalysis[df_MEGanalysis['resection_hemisphere'] == 'Right']

plot_binvsbiomarker(df_long_hemisphere_fEI, df_long_resecctrl_fEI, 'fEI')
plot_binvsbiomarker(df_long_hemisphere_DFA, df_long_resecctrl_DFA, 'DFA')
plot_binvsbiomarker(df_long_hemisphere_BIS, df_long_resecctrl_BIS, 'BIS')
plot_binvsbiomarker(df_long_hemisphere_HLP, df_long_resecctrl_HLP, 'HLP')

plot_IQ_groups_averaged(df_long_hemisphere_DFA, 'DFA', dir_figs)
plot_IQ_groups_averaged(df_long_hemisphere_fEI, 'fEI', dir_figs)
plot_IQ_groups_averaged(df_long_hemisphere_BIS, 'BIS', dir_figs)
plot_IQ_groups_averaged(df_long_hemisphere_HLP, 'HLP', dir_figs)

plot_IQ_HvL_groups(df_long_hemisphere_fEI, 'fEI')
plot_IQ_HvL_groups(df_long_hemisphere_DFA, 'DFA')
plot_IQ_HvL_groups(df_long_hemisphere_BIS, 'BIS')
plot_IQ_HvL_groups(df_long_hemisphere_HLP, 'HLP')


# plot LR hemispheres combined:
for biomarker in biomarkers:
    plot_IQ_correlation_combined_sig(dir_figs, df_MEGanalysis, f'LR_MEG_{ROI_code}', biomarker, fbins, rgb_colours)
    for fband in ['alpha', 'beta']:
        plot_IQ_correlation_bands(dir_figs, df_MEGanalysis, 'LR', biomarker, fband, rgb_colours)
plot_IQ_correlation_1f_sig(dir_figs, df_MEGanalysis, 'LR', '1fslope', rgb_colours)
plot_IQ_correlation_1f_sig(dir_figs, df_MEGanalysis, 'LR', '1foffset', rgb_colours)

# Plot L hemisphere:
for biomarker in biomarkers:
    plot_IQ_correlation_combined_sig(dir_figs, df_MEGanalysis_L, f'L_MEG_{ROI_code}', biomarker, fbins, rgb_colours)   
    for fband in ['alpha', 'beta']:
        plot_IQ_correlation_bands(dir_figs, df_MEGanalysis_L, 'L', biomarker, fband, rgb_colours)
plot_IQ_correlation_1f_sig(dir_figs, df_MEGanalysis_L, 'L', '1fslope', rgb_colours)
plot_IQ_correlation_1f_sig(dir_figs, df_MEGanalysis_L, 'L', '1foffset', rgb_colours)


# Plot R hemisphere:
for biomarker in biomarkers:
    plot_IQ_correlation_combined_sig(dir_figs, df_MEGanalysis_R, f'R_MEG_{ROI_code}', biomarker, fbins, rgb_colours)
    for fband in ['alpha', 'beta']:
        plot_IQ_correlation_bands(dir_figs, df_MEGanalysis, 'R', biomarker, fband, rgb_colours)
plot_IQ_correlation_1f_sig(dir_figs, df_MEGanalysis_R, 'R', '1fslope', rgb_colours)
plot_IQ_correlation_1f_sig(dir_figs, df_MEGanalysis_R, 'R', '1foffset', rgb_colours)





plot_IQ_correlation_combined_RCavg_sig(dir_figs, df_MEGanalysis_L, f'L_MEG_{ROI_code}', 'DFA', fbins, rgb_colours)
plot_IQ_correlation_combined_RCavg_sig(dir_figs, df_MEGanalysis_R, f'R_MEG_{ROI_code}', 'DFA', fbins, rgb_colours)
plot_IQ_correlation_combined_sig(dir_figs, df_MEGanalysis_R, f'R_MEG_{ROI_code}', 'DFA', fbins, rgb_colours)








plot_IQ_correlation_1f_sig(dir_figs, df_MEGanalysis, 'LR', '1foffset', rgb_colours)
