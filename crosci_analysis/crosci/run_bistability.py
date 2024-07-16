from scipy.io import loadmat,savemat
from bistability import get_bistability_index
import os
import numpy as np
from scipy.signal import resample
import re
from mne.io import read_raw_fif,RawArray
import mne
import sys
sys.path.insert(0,'//home/arthur/workspace/git/NBT2/')
from nbt.signal_container import SignalContainer
from nbt.preprocessing import BandpassFilterStep,AmplitudeEnvelopeStep,DownsampleStep,TrimSignalStep
from nbt.serialization import write_serialized
import matplotlib.pyplot as plt
from crosci.biomarkers import get_frequency_bins,get_DFA_fitting_interval,DFA,fEI,bistability_index
from PyAstronomy.pyasl import generalizedESD

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from build_database_bis import build_database
import itertools
from joblib import Parallel, delayed
    
# specify the number of histogram bins used to estimate the bistability index
num_bins = 200

# whether to display histograms along with bi-exponential fits
plot_histogram = False

def write_dict_to_property_file(dictionary, filename):
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}={value}\n")

def find_peak_indices(data):
    # Compute the difference between consecutive elements
    differences = np.diff(data)
    # Find indices where the difference changes sign from positive to negative
    peak_indices = np.where((differences[:-1] > 0) & (differences[1:] < 0))[0] + 1
    return peak_indices


def run_bistability_on_amp(amp, outlier_removal=True, num_bins=200):
    power = np.power(amp,2)
    
    median_power = np.median(power)
    power = power/median_power

    bistability_index,bistab_objective,monoM_peak_log_power,monoM_peak_height_norm,biM_peak_log_power,biM_peak_height_norm,delta,gamma1,gamma2,b_dist_exp,b_dist_biexp,num_switches  = get_bistability_index(power, num_bins, False)
    bistab_output = {}
    bistab_output['BiS']=bistability_index
    bistab_output['median_power']=median_power
    bistab_output['gamma1']=gamma1
    bistab_output['gamma2']=gamma2
    bistab_output['delta']=delta
    bistab_output['b_dist_exp']=b_dist_exp
    bistab_output['b_dist_biexp']=b_dist_biexp
    bistab_output['good_signal_ratio']=good_signal_ratio
    bistab_output['num_switches']=num_switches

    if biM_peak_log_power.size==2:
        bistab_output['bi_peak_log_power_peak1']=biM_peak_log_power[0]
        bistab_output['bi_peak_height_norm_peak1']=biM_peak_height_norm[0]
        bistab_output['bi_peak_log_power_peak2']=biM_peak_log_power[1]
        bistab_output['bi_peak_height_norm_peak2']=biM_peak_height_norm[1]
    else:
        bistab_output['bi_peak_log_power_peak1']=biM_peak_log_power.item()
        bistab_output['bi_peak_height_norm_peak1']=biM_peak_height_norm.item()
        bistab_output['bi_peak_log_power_peak2']=None
        bistab_output['bi_peak_height_norm_peak2']=None

    return bistab_output

def calc_bistab_and_write(crt_file_name,data_folder,analysis_folder):
    print(crt_file_name)

    filename_without_extension = os.path.splitext(crt_file_name)[0]
    analysis_file_path = os.path.join(analysis_folder,filename_without_extension+"_BiS.nbt")

    if os.path.exists(analysis_file_path):
        return

    signal_container = SignalContainer(os.path.join(data_folder,crt_file_name))
    signal_container.load()
    DownsampleStep(250).apply(signal_container)
    signal_container.raw.apply_proj()

    fs = signal_container.raw.info['sfreq']
    ch_names = signal_container.raw.info.ch_names
    
    frequency_bins = get_frequency_bins([1,45])

    BiS = np.zeros((len(frequency_bins),len(ch_names)))
    delta_array = np.zeros((len(frequency_bins),len(ch_names)))
    b_dist_exp_array = np.zeros((len(frequency_bins),len(ch_names)))
    b_dist_biexp_array = np.zeros((len(frequency_bins),len(ch_names)))
    good_signal_ratio = np.zeros((len(frequency_bins),len(ch_names)))

    for idx_frequency,frequency_bin in enumerate(frequency_bins):
        signal_container_copy = signal_container.copy()

        BandpassFilterStep(frequency_bin[0], frequency_bin[1]).apply(signal_container_copy)
        AmplitudeEnvelopeStep().apply(signal_container_copy)
        TrimSignalStep().apply(signal_container_copy)
        
        data = signal_container_copy.raw[:][0]

        for i in range(len(ch_names)):
            amp = data[i,:]
            bistab_output = run_bistability_on_amp(amp,outlier_removal=False)

            BiS[idx_frequency,i] = bistab_output['BiS']
            delta_array[idx_frequency,i] = bistab_output['delta']
            b_dist_exp_array[idx_frequency,i] = bistab_output['b_dist_exp']
            b_dist_biexp_array[idx_frequency,i] = bistab_output['b_dist_biexp']
            good_signal_ratio[idx_frequency,i] = bistab_output['good_signal_ratio']
        
        print('Finished analysis for frequency bin',frequency_bin)
    
    bistab_results = {
        'BiS': BiS,
        'delta': delta_array,
        'b_dist_exp': b_dist_exp_array,
        'b_dist_biexp': b_dist_biexp_array,
        'good_signal_ratio': good_signal_ratio
    }

    write_serialized(bistab_results,analysis_file_path)

def process_folder(data_folder,analysis_folder,db_name):
    num_parallel_workers=14

    files = [f for f in os.listdir(data_folder) if re.match(r'.+.fif', f)]

    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)

    analysis_folder_results = os.path.join(analysis_folder,'')
    db_folder = os.path.join(analysis_folder,'databases')

    # Using Joblib's Parallel and delayed
    Parallel(n_jobs=num_parallel_workers)(
        delayed(calc_bistab_and_write)(file, data_folder, analysis_folder_results) 
        for file, data_folder, analysis_folder_results, in zip(files, itertools.repeat(data_folder), itertools.repeat(analysis_folder_results))
    )

    # with ProcessPoolExecutor(max_workers=num_parallel_workers) as executor:
    #     results = list(executor.map(process_file, files))
       
    # for result in results:
    #     if isinstance(result, dict) and "error" in result:
    #         print(f"Error processing {result['file']}: {result['error']}") 

    # build_database(data_folder,analysis_folder_results,db_folder,db_name)