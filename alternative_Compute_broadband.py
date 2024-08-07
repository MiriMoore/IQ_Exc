#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Computes broadband power, offset and slope of power spectrum

Based on selected epochs (e.g. ASCIIS) in the list of files a power spectrum
is computed. Based on this power spectrum the broadband power is calculated,
followed by the offset and slope using the FOOOF algorithm.

Reference paper FOOOF: Haller M, Donoghue T, Peterson E, Varma P, Sebastian P,
Gao R, Noto T, Knight RT, Shestyuk A, Voytek B (2018) Parameterizing Neural
Power Spectra. bioRxiv, 299859. doi: https://doi.org/10.1101/299859
reference Github: https://fooof-tools.github.io/fooof/index.html

"""

__author__ = "Tianne Numan"
__contact__ = "t.numan@amsterdamumc.nl"  # or l.douw@amsterdamumc.nl
__date__ = "2020/09/14"  ### Date it was created
__status__ = "Finished"


####################
# Review History   #
####################

# Reviewed and updated by Eduarda Centeno 20201030
# Reviewed and updated by Mona Zimmermann in 20210421


####################
# Libraries        #
####################

# Standard imports
import time
import os
import glob
import ast
import gzip
from datetime import date

# Third party imports
import numpy as np  # version 1.19.1
import matplotlib.pyplot as plt  # version 3.3.0
import pandas as pd  # version 1.1.0
from scipy import signal  # version 1.4.1
from fooof import FOOOF  # version 0.1.3


# Define Functions ------------------------------------------------------------


def find_paths(main_dir, subject, extension, **kwargs):
    """Flexible way to find files in subdirectories based on keywords

    Parameters
    ----------
    main_dir: str
        Give the main directory where the subjects' folders are stored
    subject: str
        Give the name of the subject to be analyzed
    extension: str
        Give the extension type
    **kwargs: str
        Give keywords that will be used in the filtering of paths

        !Important!
        It is possible to use the kwargs 'start' & 'end' (int) OR
        'selection' (list or str) for selecting epochs. The 'selection'
        list should contain the exact way in which the Tr is written, e.g.
        Tr01, or Tr_1, etc.

    Examples
    -------
    Ex.1
    find_paths(main_dir='/data/KNW/NO-cohorten/Scans/',
               subject='sub-9690',
               extension='.asc',
               key1='T1',
               key2='BNA',
               key3='Tr07')

    This example will result in a list with a single path:
        ['.../T1/BNA/1_100_WITH_200_WITH_246_VE_89.643to102.750_Tr_7.asc']

    Ex2.
    find_paths(main_dir='/data/KNW/NO-cohorten/Scans/',
               subject='sub-9690',
               extension='.asc',
               key1='T1',
               key2='BNA',
               start=20,
               end=23)

    This example will result in a list with several paths:
        ['.../T1/BNA/1_100_WITH_200_WITH_246_VE_260.037to273.143_Tr_20.asc',
         '.../T1/BNA/1_100_WITH_200_WITH_246_VE_273.144to286.250_Tr_21.asc',
         '.../T1/BNA/1_100_WITH_200_WITH_246_VE_286.251to299.358_Tr_22.asc',
         '.../T1/BNA/1_100_WITH_200_WITH_246_VE_299.358to312.465_Tr_23.asc']

    Returns
    -------
    updatedfilter: list
        List with path strings

    Notes
    -------
    Be careful that final slicing for 'start' & 'end' is done assuming that
    the sorting step was correct. Thus, it is based on index not on finding the
    specific start-end values in the string. This was done because the tested
    paths had various ways of using Tr (e.g. Tr_1 or Tr_01, or Tr1 or Tr_01) -
    what caused inconsistencies in the output.

    """

    # Check if arguments are in the correct type
    assert isinstance(main_dir, str), "Argument must be str"
    assert isinstance(subject, str), "Argument must be str"
    assert isinstance(extension, str), "Argument must be str"

    if extension != ".asc.gz":
        print("-------------------------------------------------")
        print(
            "!IMPORTANT!\n find_paths is optimized to .asc files.\n Other formats might leave to reading errors."
        )
        print("-------------------------------------------------")
    # Filtering step based on keywords
    # firstfilter = glob.glob(main_dir + subject + "/**/*" + extension, recursive=True)
    print(subject)
    firstfilter = glob.glob(main_dir + "/**/*" + extension, recursive=True)
    print(f'filter0 = {firstfilter}')
    updatedfilter = [path for path in firstfilter if f'case{subject}' in os.path.basename(path)]
    print(f'filter1: {updatedfilter}')
    print("\n..............NaN keys will be printed.................")
    start = None
    end = None
    selection = None
    for key, value in kwargs.items():
        # In case the key value is NaN (possible in subjects dataframe)
        if not isinstance(value, list) and pd.isnull(value):
            print(key + "= NaN")
            continue
        elif key == "start":
            assert isinstance(
                value, (int, str, float)
            ), "Argument must be int or number str"
            start = int(value)
        elif key == "end":
            assert isinstance(
                value, (int, str, float)
            ), "Argument must be int or number str"
            end = int(value)
        elif key == "selection":
            if isinstance(value, list):
                selection = value
            elif isinstance(value, str):
                selection = value.replace(
                    ";", ","
                )  # Step that convert ; to , (used in example.csv)
                selection = ast.literal_eval(selection)
            assert isinstance(
                selection, list
            ), "Argument should end up being a list of Tr numbers strings"
            assert all(
                isinstance(item, str) for item in selection
            ), "Argument must be a list of of Tr numbers strings"
        else:
            start = None
            end = None
            selection = None
            # Update list accoring to key value
            updatedfilter = list(filter(lambda path: value in path, updatedfilter))
    print(f'filter2: {updatedfilter}')
    # Check if too many arguments were passed!
    print("\n..............Checking if input is correct!.................")
    # print(start, end, selection)
    if (start and end) != None and selection != None:
        raise RuntimeError("User should use Start&End OR Selection")
    else:
        print("All good to continue! \n")
        pass
    # To find index of Tr (last appearance)
    location = updatedfilter[0].rfind("Tr")
    # Sort list according to Tr* ending (+1 was necessary to work properly)
    updatedfilter.sort(
        key=lambda path: int("".join(filter(str.isdigit, path[location + 1 :])))
    )

    # After the list is sorted, slice by index.
    if (start and end) != None:
        print(
            "Start&End were given. \n"
            + "-- Start is: "
            + str(start)
            + "\n--End is: "
            + str(end)
        )
        updatedfilter = updatedfilter[start - 1 : end]
    # After the list is sorted, interesect with selection.
    elif selection != None:
        print(
            "\nA selection of values was given."
            + "\nThe selection was: "
            + str(selection)
        )
        updatedlist = []
        for item in selection:
            updatedlist += list(
                filter(lambda path: item + extension in path[location:], updatedfilter)
            )
        updatedfilter = updatedlist
    print(f'updatedfilter = {updatedfilter}')
    return updatedfilter


def make_csv(csv_path, output_path, extension=".asc"):
    """Function to insert the number of epochs to include in analysis into csv.
    Number of epochs is calculated by comparing the number of epochs available
    for each subject and including the minimum amount.

    Parameters
    ----------
    csv_path : str,
        path to the csv containing information on subjects to include

    output_path: str,
        complete path to output new csv (e.g. '/path/to/folder/new_csv.csv')

    extension : str,
        file extension of meg files (e.g. '.asc')
        default = '.asc'

    Returns
    -------
    None
    saves the extended csv to the same directory where found old csv
    (i.e. overwrites old csv)

    epochs_df: pandas DataFrame,
        dataframe containing the filepaths to the epochs included for every subject

    """

    df = pd.read_csv(csv_path, delimiter=",", header=0)

    nr_epochs = []
    for index, row in df.iterrows():
        asc_paths = find_paths(
            main_dir=row["Path"],
            subject=str(row["Case_ID"]),
            extension=extension,
            timepoint=row["MM"],
            atlas=row["Atlas"],
        )
        print(f'asc_paths = {asc_paths}')
        # store nr of epochs available for each subject
        nr_epochs.append(len(asc_paths))
    # find smallest number of epochs available
    min_nr_epochs = min(nr_epochs)

    # add start and stop epochs to df
    df["Start"] = np.repeat(1, len(df["Path"]))
    df["End"] = np.repeat(min_nr_epochs, len(df["Path"]))

    # save new csv file that includes the epochs to analyse
    df.to_csv(output_path, index=False, sep=",")

    # load new csv file with start and end epochs
    new_csv = pd.read_csv(output_path)
    subs = []
    paths = []

    # search for asc files between start and end epoch range specified in csv
    for index, row in new_csv.iterrows():
        subs.append(row["Case_ID"])
        asc_paths = find_paths(
            main_dir=row["Path"],
            subject=str(row["Case_ID"]),
            extension=extension,
            timepoint=row["MM"],
            atlas=row["Atlas"],
            start=row["Start"],
            end=row["End"],
        )
        # append list of asc_paths for subject to list
        paths.append(asc_paths)
    # store lists of asc_paths (for every subject) in dataframe
    epochs_df = pd.DataFrame(paths)

    # index rows to subject IDs
    epochs_df.set_index([pd.Index(subs)], "Subs", inplace=True)

    return epochs_df


def cal_power_spectrum(
    timeseries,
    nr_rois=np.arange(92),
    fs=1250,
    window="hamming",
    nperseg=4096,
    scaling="spectrum",
    plot_figure=False,
    title_plot="average power spectrum",
):
    """Calculate (and plot) power spectrum of timeseries

    Parameters
    ----------
    timeseries: DataFrame with ndarrays
        Rows are timepoints, columns are rois/electrodes
        Give list with rois/electrodes you want to include,
        default=np.arange(92)
    fs: int, optional
        Sample frequency, default=1250
    window: str or tuple, optional
        Type of window you want to use, check spectral.py for details,
        default='hamming'
    nperseg : int, optional
        Length of each segment, default=4096
    scaling : str, optional
        'density' calculates the power spectral density (V**2/Hz), 'spectrum'
        calculates the power spectrum (V**2), default='spectrum'
    plot_figure: bool
        Creates a figure of the mean + std over all rois/electrodes,
        default=False
    title_plot: str
        Give title of the plot, default='average power spectrum'

    Returns
    -------
    f: ndarray
        Array with sample frequencies (x-axis of power spectrum plot)
    pxx: ndarray
        Columns of power spectra for each roi/VE

    """
    # print(timeseries.head())

    pxx = np.empty([int(nperseg / 2 + 1), np.size(nr_rois)])

    i = 0
    for roi in nr_rois:
        (f, pxx[:, i]) = signal.welch(
            timeseries[roi].values, fs, window, nperseg, scaling=scaling
        )
        i = i + 1
    if plot_figure == True:
        plt.figure()
        plt.plot(f, np.mean(pxx, 1), color="teal")
        plt.plot(f, np.mean(pxx, 1) + np.std(pxx, 1), color="teal", linewidth=0.7)
        plt.plot(f, np.mean(pxx, 1) - np.std(pxx, 1), color="teal", linewidth=0.7)
        plt.fill_between(
            f,
            np.mean(pxx, 1) + np.std(pxx, 1),
            np.mean(pxx, 1) - np.std(pxx, 1),
            color="teal",
            alpha=0.2,
        )
        plt.xlim(0, 100)
        plt.xlabel("Frequency (Hz)")
        plt.title(title_plot)
        plt.show()
    return f, pxx


def find_nearest(array, value):
    """Find nearest value of interest in array (used for frequencies,
    no double value issues)

    Parameters
    ----------
    array: array
        Give the array in which you want to find index of value nearest-by
    value: int or float
        The value of interest

    Return
    ------
    idx: int
        Index of value nearest by value of interest

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def cal_FOOOF_parameters(pxx, f, freq_range=[0.5, 48]):
    """Obtain slope and offset using the FOOOF algorithm
    Reference paper: Haller M, Donoghue T, Peterson E, Varma P, Sebastian P,
    Gao R, Noto T, Knight RT, Shestyuk A, Voytek B (2018) Parameterizing Neural
    Power Spectra. bioRxiv, 299859. doi: https://doi.org/10.1101/299859

    Reference Github: https://fooof-tools.github.io/fooof/index.html

    Parameters
    ----------
    pxx: ndarray
        Column of power spectra
    f: 1d-array
        Array with sample frequencies (x-asix of power spectrum plot)
    freq_range: list, optional
        Gives the upper and lower boundaries to calculate the broadband power,
        default=[0.5, 48]

    Returns
    -------
    FOOOF_offset: float
        Offset
    FOOOF_slope: float
        Slope

    """

    # initialize FOOOF oject
    fm = FOOOF()
    # create variables
    fm.fit(f, pxx, freq_range)
    FOOOF_offset = fm.background_params_[0]
    FOOOF_slope = fm.background_params_[1]
    time.sleep(1)  # heavy algorithm

    return FOOOF_offset, FOOOF_slope


def run_loop_powerspectrum(
    subject_list,
    extension=".asc",
    nr_rois=np.arange(92),
    Fs=1250,
    window_length=4096,
    freq_range=[0.5, 48],
    plot_figure=False,
):

    """Calculate power spectrum for all cases within the subject_list

    Parameters
    ----------
    subject_list: string
        String with the file path.
    extension: str, optional
        Give the extension of ASCIIs, '.txt' or '.asc', default='.asc',
    nr_rois: int, optional
        Give list with rois you want to analyse, default=np.arange(92)
    Fs: int, optional
        Sample frequency, default=1250
    window_length: int, optional
        Window length to calculate power spectrum, default=4096
    freq_range: list, optional
        Gives the upper and lower boundaries to calculate the broadband power,
        default=[0.5, 48]
    plot_figure: bool
        To plot average power spectrum plot per epoch or not, default=False

    Return
    ------
    mean_pxx: ndarray (size: len(subjects), len(power spectrum), nr_rois)
        Power spectum for all subjects, and all rois/VE, averaged over
        nr_epochs
    broadband_power : ndarray (size: len(subjects), nr_rois)
        Broadband power between freq_range
    f: ndarray
        Array with sample frequencies (x-axis of power spectrum plot)

    """
    print("\n____STARTING TO COMPUTE POWER SPECTRUM!____")
    # subjects = pd.read_csv(subject_list, delimiter=",", header=0)
    print("\nThis is the content of the subjects_list file: \n" + str(subjects))
    mean_pxx = np.empty([len(subjects), int(window_length / 2 + 1), np.size(nr_rois)])
    broadband_power = np.empty([len(subjects), np.size(nr_rois)])
    freq = np.empty([len(subjects), int(window_length / 2 + 1)])
    for index, row in subjects.iterrows():
        print(
            "\n\n//////////// Subject " + str(index) + " on subject_list ////////////"
        )
        files_list = find_paths(
            main_dir=main_dir,
            subject=str(row["Case_ID"]),
            extension=extension,
            # timepoint=row["MM"],
            # atlas=row["Atlas"],
            # start=row["Start"],
            # end=row["End"],
            # selection=row["Selection"],
        )
        # files_list = [row['Path']]
        # print("\nThe paths found are: \n" + str(files_list))
        if len(files_list) == 0:
            print("No ASCIIs available for: " + row)
            continue
        elif len(files_list) == 1:
            print('single subject ascii found')
            single_ascii = files_list[0]
            with gzip.open(single_ascii, 'rt') as gzfile:
                timeseries = pd.read_csv(gzfile, index_col=False, header=None, delimiter="\t")
                # print(timeseries.head())
                f, pxx = cal_power_spectrum(
                    timeseries,
                    nr_rois=nr_rois,
                    fs=Fs,
                    plot_figure=plot_figure,
                    title_plot="power spectrum",
                )
                mean_pxx[index, :, :] = pxx
                broadband_power[index, :] = np.sum(
                    pxx[find_nearest(f, freq_range[0]):find_nearest(f, freq_range[1]), :], axis=0,)
                freq[index, :] = f
            # mean_pxx = pxx
            # broadband_power = np.sum(
            #     mean_pxx[
            #         find_nearest(f, freq_range[0]) : find_nearest(f, freq_range[1]), :
            #     ],
            #     axis=0,
            # )
            # freq = f
        else:
            print('multiple subject asciis found')
            sub_pxx = np.zeros(
                (len(files_list), int(window_length / 2 + 1), np.size(nr_rois))
            )
            # mean_pxx[index,:,:] = 'nan'
            # broadband_power[index,:] = 'nan'
            for file, name in zip(range(len(files_list)), files_list):
                location = name.rfind("Tr")
                print(f'file: {file} {name}')
                with gzip.open(name, 'rt') as gzfile:
                    timeseries = pd.read_csv(gzfile, index_col=False, header=None, delimiter="\t")
                    # print(timeseries.head())
                # timeseries = pd.read_csv(
                #     files_list[file], index_col=False, header=None, delimiter="\t"
                # )
                # Compute power spectrum
                f, pxx = cal_power_spectrum(
                    timeseries,
                    nr_rois=nr_rois,
                    fs=Fs,
                    plot_figure=plot_figure,
                    title_plot="avg power spectrum - epoch: "
                    + "".join(filter(str.isdigit, name[location:])),
                )
                sub_pxx[file, :, :] = pxx
            freq[index, :] = f
            mean_pxx[index, :, :] = np.nanmean(sub_pxx, axis=0)
            broadband_power[index, :] = np.sum(
                mean_pxx[
                    index,
                    find_nearest(f, freq_range[0]) : find_nearest(f, freq_range[1]),
                    :,
                ],
                axis=0,
            )
    return mean_pxx, broadband_power, freq

# -----------------------------------------------------------------------------

###########################
# Settings                #
###########################
# set nice level to 10, especially FOOOF algorithm is heavy!
os.nice(10)

# 1. Create correctly your list of subjects you want to process
# an example is given here: 'example_MEG_list.csv'
subject_list = "/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/data/MEGs/allMTGs/ascfiles/df_subjectlist_for_Pspec.csv"

# 2. Define the type of file extension your are looking for
extension = ".asc.gz"  # extension type

# 3. Select which roi or rois you want to analyze
# if you want to analyze 1 roi, specify its number (nr_rois = (10,))
nr_rois = [r for r in range(0, 8)]  # (10,) if only run for roi 11 # note that python indexes at 0!
# if you want to analyze multiple rois, create list with these rois
# (for example nr_rois = np.arange(78) for all 78 cortical AAL rois)

# 4. Set sample frequency (1250 Hz for Elekta data)
Fs = 1250  # sample frequency

# 5. Set frequency range you want to study
freq_range = [1, 44.8]  # frequency range you want to analyze

# 6. Give output directory
dir_output = "/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/procdata/MEGs/MTG_allchannels/power_spec"
os.makedirs(dir_output, exist_ok=True)
# 7. Do you want to see the plots?
plot_choice = True

# 7a. Do you want to save the output?
save_output = True  # you can save output

###########################
# Run analysis            #
###########################
main_dir="/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/data/MEGs/allMTGs/ascfiles"
subjects = pd.read_csv(subject_list, delimiter=",", header=0, dtype={'Case_ID': str})


# mean_pxx contains the average power spectra over nr_epochs, for all subjects
# for all rois, broadband_power is the area under the average power spectra
# over the frequency range for all subjects and rois, f gives the frequencies
# of the power spectrum (can be useful when plotting power spectrum)
mean_pxx, broadband_power, f = run_loop_powerspectrum(
    subject_list,
    extension,
    nr_rois=nr_rois,
    Fs=Fs,
    window_length=4096,
    freq_range=freq_range,
    plot_figure=plot_choice,
)

print(broadband_power)



# save output
if save_output == True:
    # subjects = pd.read_csv(subject_list, delimiter=",", header=0, dtype={'Case_ID': str})
    # nr_rois = subjects.columns.tolist()
    print(f'nr_rois = {nr_rois}')
    print("\n.....Saving power spectra and frequency data......")
    for index, row in subjects.iterrows():
        if len(mean_pxx.shape) > 2:
            df_pxx_f = pd.DataFrame(mean_pxx[index, :, :])
            df_pxx_f.columns = [roi for roi in nr_rois]
            df_pxx_f["Frequency"] = f[index, :]
            # columns will be roi numbers + Frequency as last col
            df_pxx_f.to_csv(
                path_or_buf=dir_output
                + row["Case_ID"]
                + "_"
                # + str(row["MM"])
                # + "_"
                # + str(row["Atlas"])
                # + "_"
                + str(date.today().strftime("%Y%m%d"))
                + "_"
                + "pxxfreq"
                + ".csv",
                header=True,
                index=False,
            )
        elif len(mean_pxx.shape) == 2:
            df_pxx_f = pd.DataFrame(mean_pxx)
            df_pxx_f.columns = [roi for roi in nr_rois]
            df_pxx_f["Frequency"] = f
            # columns will be roi numbers + Frequency as last col
            df_pxx_f.to_csv(
                path_or_buf=dir_output
                + row["Case_ID"]
                + "_"
                # + str(row["MM"])
                # + "_"
                # + str(row["Atlas"])
                # + "_"
                + str(date.today().strftime("%Y%m%d"))
                + "_"
                + "pxxfreq"
                + ".csv",
                header=True,
                index=False,
            )
###### Compute FOOOOF offset and slope ######


# create empty arrays to store offset and slope values
offset = np.empty([mean_pxx.shape[0], mean_pxx.shape[-1]])
# index 0 -> number of subjects, index 2 -> number of rois
slope = np.empty([mean_pxx.shape[0], mean_pxx.shape[-1]])

print("\n.........................Running FOOOF.........................")

# run across all subjects in your list
for index, row in subjects.iterrows():
    print("row : " + str(row))  # print which subject is analyzed
    if len(mean_pxx.shape) > 2:
        if np.isnan(mean_pxx[index, 1, 0]):
            # if there is no mean_pxx for a subject, set offset/slope to nan
            offset[
                index,
            ] = np.nan
            slope[
                index,
            ] = np.nan
        else:
            for roi in np.arange(len(nr_rois)):
                # roi-=1
                print("roi : " + str(nr_rois[roi]))
                offset[index, roi], slope[index, roi] = cal_FOOOF_parameters(
                    mean_pxx[index, :, roi], f[index, :], freq_range=[0.5, 48]
                )
            time.sleep(0.05)
            # by having a pause it should limit the %CPU continuously
            # might be relevant when computing FOOOF for large dataset
    elif len(mean_pxx.shape) == 2:
        i = 0
        offset = np.empty(mean_pxx.shape[1])
        slope = np.empty(mean_pxx.shape[1])
        for roi in nr_rois:
            print("roi : " + str(roi))
            offset[i], slope[i] = cal_FOOOF_parameters(
                mean_pxx[:, i], f, freq_range=[0.5, 48]
            )
            i = i + 1
            time.sleep(0.05)
# save output
if save_output == True:
    print("\n.....Saving broadband power, slope, and offset......")
    for index, row in subjects.iterrows():
        if len(mean_pxx.shape) > 2:
            df_pxx_slope = pd.DataFrame(slope[index]).T
            df_pxx_offset = pd.DataFrame(offset[index]).T
            df_bbpower = pd.DataFrame(broadband_power[index]).T
            df_all = pd.concat([df_pxx_slope, df_pxx_offset, df_bbpower])
            df_all.index = ["Slope", "Offset", "Broadband Power"]
            df_all.columns = [roi for roi in nr_rois]  # columns will be roi numbers
            df_all.to_csv(
                path_or_buf=dir_output
                + row["Case_ID"]
                + "_"
                # + str(row["MM"])
                # + "_"
                # + str(row["Atlas"])
                # + "_"
                + str(date.today().strftime("%Y%m%d"))
                + "_"
                + "bbp_offset_slope"
                + ".csv",
                header=True,
                index=True,
            )
        elif len(mean_pxx.shape) == 2:
            df_pxx_slope = pd.DataFrame(slope).T
            df_pxx_offset = pd.DataFrame(offset).T
            df_bbpower = pd.DataFrame(broadband_power).T
            df_all = pd.concat([df_pxx_slope, df_pxx_offset, df_bbpower])
            df_all.index = ["Slope", "Offset", "Broadband Power"]
            df_all.columns = [roi for roi in nr_rois]  # columns will be roi numbers
            df_all.to_csv(
                path_or_buf=dir_output
                + row["Case_ID"]
                + "_"
                # + str(row["MM"])
                # + "_"
                # + str(row["Atlas"])
                # + "_"
                + str(date.today().strftime("%Y%m%d"))
                + "_"
                + "bbp_offset_slope"
                + ".csv",
                header=True,
                index=True,
            )

#%% create df for ROI averages -> IQ correlation
ROI = 'allMTGs'
dir_procdata = '/Users/Miri/Documents/Research/EPhys/DataAnalysis/IQ_Exc/procdata/MEGs/MTG_allchannels/'
ls_powerspec_summs = os.listdir(dir_procdata)
ls_summfiles = []
for file in ls_powerspec_summs:
    if 'offset_slope' in file:
        ls_summfiles.append(file)

ls_biomarkers = ['1f_slope', '1f_offset']
ls_avgs = []
for filename in ls_summfiles:
    file = pd.read_csv(os.path.join(dir_procdata, filename), index_col=0)
    file['avg_L'] = file.iloc[:, [0, 2, 4, 6]].mean(axis=1)
    file['avg_R'] = file.iloc[:, [1, 3, 5, 6]].mean(axis=1)
    file['avg_LR'] = file.mean(axis=1)
    
    ls_avgs.append(file[['avg_L', 'avg_R', 'avg_LR']])
    print(file[['avg_L', 'avg_R', 'avg_LR']])

# Combine all averages into a single DataFrame
combined_avgs = pd.concat(ls_avgs, axis=1)

# Use only the first 14 characters of the base filename as the column name
combined_avgs.columns = [f"{os.path.basename(f)[10:14]}_avg_L" for f in ls_summfiles] + \
                        [f"{os.path.basename(f)[10:14]}_avg_R" for f in ls_summfiles] + \
                        [f"{os.path.basename(f)[10:14]}_avg_LR" for f in ls_summfiles]

# Save the combined averages to a new CSV file
combined_output_file = os.path.join(dir_procdata, f'1foffset_1fslope_{ROI}_avg_{date.today().strftime("%Y%m%d")}.csv')
combined_avgs.to_csv(combined_output_file)

print("Processing completed.")


#%%
# Generate sample data
fs = 1250  # Sampling frequency
t = np.arange(0, 10, 1/fs)  # Time vector
# Simulate a signal with 1/f characteristics (e.g., pink noise)
signal_data = np.random.normal(0, 1, t.shape)

# Calculate power spectrum
f, pxx = signal.welch(signal_data, fs, nperseg=4096, scaling='spectrum')

# Plot power spectrum
plt.figure()
plt.plot(f, pxx)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectrum')
plt.show()

# Fit FOOOF model
freq_range = [1, 44.8]
fm = FOOOF()
fm.fit(f, pxx, freq_range)
fm.report()

# Extract and print FOOOF parameters
FOOOF_offset = fm.background_params_[0]
FOOOF_slope = fm.background_params_[1]
print(f'FOOOF Offset: {FOOOF_offset}')
print(f'FOOOF Slope: {FOOOF_slope}')
