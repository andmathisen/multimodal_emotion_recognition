import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time
from itertools import count
import sys
import pandas as pd
from sklearn import preprocessing
import numpy as np
from scipy.signal import find_peaks
import os
import json
import neurokit2 as nk

def eda_custom_process(eda_signal, sampling_rate=4, method="neurokit"): # Method to extract phasic(signals['EDA_phasic']) and tonic(signals['EDA_tonic']) component from EDA signal, 
    

    eda_signal = nk.signal_sanitize(eda_signal)
    
    # Series check for non-default index
    if type(eda_signal) is pd.Series and type(eda_signal.index) != pd.RangeIndex:
        eda_signal = eda_signal.reset_index(drop=True)
    
    # Preprocess
    eda_cleaned = eda_signal  #Add your custom cleaning module here or skip cleaning
    eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate)

    # Find peaks
    peak_signal, info = nk.eda_peaks(
        eda_decomposed["EDA_Phasic"].values,
        sampling_rate=sampling_rate,
        method=method,
        amplitude_min=0.2,
    )
    info['sampling_rate'] = sampling_rate  # Add sampling rate in dict info

    # Store
    signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned})

    signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)

    return signals, info


def normalize_dataframe_values(dataframe, column_name):
    nparray = dataframe.values
    if column_name == 'ACC':
        nparray = nparray.reshape(-1,1)
    scaler = preprocessing.StandardScaler()
    min_max_scaler = preprocessing.MinMaxScaler()
    nparray_scaled = min_max_scaler.fit_transform(nparray)
    # nparray /= np.max(np.abs(nparray), axis=0)
    df_scaled = pd.DataFrame(nparray_scaled)
    df_scaled.rename(columns={df_scaled.columns[0]: column_name}, inplace=True)
    return df_scaled


def transform_bvp(dataframe, column_name):
    nparray = dataframe.values

    nparray /= np.max(np.abs(nparray), axis=0)

    nparray = nparray.clip(min=0)

    nparray = nparray[:, 0]
    peaks, _ = find_peaks(nparray, distance=40)
    single_peak_values_with_min = []
    cur_val = 0.1
    start = 0
    for i in range(0, len(nparray)):
        if i in peaks:
            new_val = nparray[i]
            if new_val >= 0.1:
                cur_val = new_val

            for j in range(start, i+1):
                single_peak_values_with_min.append(cur_val)
            start = i+1

        if i == len(nparray)-1:
            for j in range(start, i+1):
                single_peak_values_with_min.append(cur_val)

    df_scaled = pd.DataFrame(single_peak_values_with_min)
    df_scaled.rename(columns={df_scaled.columns[0]: column_name}, inplace=True)
    
    return df_scaled

def transform_bvp_amplitudes(dataframe, column_name):
    nparray = dataframe.values

    nparray /= np.max(np.abs(nparray), axis=0)


    nparray = nparray.clip(min=0)

    nparray = nparray[:, 0]
    peaks, _ = find_peaks(nparray, distance=40)
    single_peak_values_with_min = []
    cur_val = 0.1
    start = 0
    for i in range(0, len(nparray)):
        if i in peaks:
            new_val = nparray[i]
            if new_val >= 0.1:
                cur_val = new_val

            for j in range(start, i+1):
                single_peak_values_with_min.append(cur_val)
            start = i+1

        if i == len(nparray)-1:
            for j in range(start, i+1):
                single_peak_values_with_min.append(cur_val)

    df_scaled = pd.DataFrame(single_peak_values_with_min)
  
    df_scaled.rename(columns={df_scaled.columns[0]: column_name}, inplace=True)

    
    
    df_BVP_avg = df_scaled.rolling(64).max()
    df_BVP_avg = df_BVP_avg.iloc[::64, :]


    df_BVP_avg = df_BVP_avg.iloc[1:, :]  # Remove NaN from first index
    df_BVP_avg = df_BVP_avg.reset_index()
    df_BVP_avg = df_BVP_avg.drop("index", 1)
    df_BVP_avg = df_BVP_avg['BVP'].round(decimals=3)
    df_BVP_avg = df_BVP_avg.to_frame("BVP")
    return df_BVP_avg
