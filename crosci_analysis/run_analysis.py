import h5py
import os
import numpy as np
import mne
from mne.filter import next_fast_len
from scipy.signal import hilbert
from joblib import Parallel, delayed
import multiprocessing
from crosci.biomarkers import get_frequency_bins,get_DFA_fitting_interval,DFA,fEI,bistability_index
import pickle
from joblib import Parallel, delayed

eeg_path = '../data/MTG_A21c/'
eeg_files = os.listdir(eeg_path)
eeg_files = [eeg_file for eeg_file in eeg_files if '.h5' in eeg_file]

analysis_path = 'analysis'
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)

frequency_bins = get_frequency_bins([1,45])
sampling_frequency = 1250

biomarkers_to_compute = ["DFA","fEI","BIS"]

def process_file(eeg_file_name):
    eeg_file_path = os.path.join(eeg_path,eeg_file_name)
    analysis_file_name = eeg_file_name.replace("h5","pkl")
    analysis_file_path = os.path.join(analysis_path,analysis_file_name)

    #skip if file has already been analyzed
    if os.path.exists(analysis_file_path):
        return
    
    with h5py.File(eeg_file_path, "r") as f:
        data1 = f.get("L81")[:].copy()
        data2 = f.get("R82")[:].copy()
        signal_matrix = np.vstack((data1,data2)).astype(float)

        num_channels = np.shape(signal_matrix)[0]
        num_timepoints = np.shape(signal_matrix)[1]

        output = {}
        output['frequency_bins'] = frequency_bins
        if 'DFA' or 'fEI' in biomarkers_to_compute:
            output['DFA']=np.zeros((num_channels,len(frequency_bins)))
        if 'fEI' in biomarkers_to_compute:
            output['fEI']=np.zeros((num_channels,len(frequency_bins)))
        if 'BIS' in biomarkers_to_compute:
            output['BIS']=np.zeros((num_channels,len(frequency_bins)))
            output['HLP']=np.zeros((num_channels,len(frequency_bins)))

         # Parameters
        fEI_window_seconds = 5
        fEI_overlap = 0.8

        DFA_overlap = True

        for idx_frequency,frequency_bin in enumerate(frequency_bins):

            # Get fit interval
            fit_interval = get_DFA_fitting_interval(frequency_bin)
            DFA_compute_interval = fit_interval

            # Filter signal in the given frequency bin
            filtered_signal = mne.filter.filter_data(data=signal_matrix,sfreq=sampling_frequency,
                                                    l_freq=frequency_bin[0],h_freq=frequency_bin[1],
                                                    filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                                                    fir_window='hamming',phase='zero',fir_design="firwin",
                                                    pad='reflect_limited', verbose=0)

            filtered_signal = filtered_signal[:,1*sampling_frequency:filtered_signal.shape[1]-1*sampling_frequency]
            # Compute amplitude envelope
            n_fft = next_fast_len(num_timepoints)
            amplitude_envelope = Parallel(n_jobs=num_cores,backend='threading',verbose=0)(delayed(hilbert)
                                                                                (filtered_signal[idx_channel,:],n_fft)
                                                                                for idx_channel in range(num_channels))
            amplitude_envelope = np.abs(np.array(amplitude_envelope))

            if 'DFA' in biomarkers_to_compute or 'fEI' in biomarkers_to_compute:
                print("Computing DFA for frequency range: %.2f - %.2f Hz" % (frequency_bin[0], frequency_bin[1]))
                (dfa_array,window_sizes,fluctuations,dfa_intercept) = DFA(amplitude_envelope,sampling_frequency,fit_interval,
                                                                            DFA_compute_interval,DFA_overlap)
                output['DFA'][:,idx_frequency] = dfa_array

            if 'fEI' in biomarkers_to_compute:
                print("Computing fEI for frequency range: %.2f - %.2f Hz" % (frequency_bin[0], frequency_bin[1]))
                (fEI_outliers_removed,fEI_val,num_outliers,wAmp,wDNF) = fEI(amplitude_envelope,sampling_frequency,
                                                                            fEI_window_seconds,fEI_overlap,dfa_array)
                output['fEI'][:,idx_frequency] = np.squeeze(fEI_outliers_removed)
            
            if 'BIS' in biomarkers_to_compute:
                print("Computing BIS for frequency range: %.2f - %.2f Hz" % (frequency_bin[0], frequency_bin[1]))
                (BIS,HLP) = bistability_index(amplitude_envelope)
                output['BIS'][:,idx_frequency] = np.squeeze(BIS)
                output['HLP'][:,idx_frequency] = np.squeeze(HLP)

            with open(analysis_file_path, 'wb') as fp:
                pickle.dump(output, fp)

num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(process_file)(eeg_file_name) for eeg_file_name in eeg_files)