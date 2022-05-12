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
import seaborn
import neurokit2 as nk
import librosa 
import librosa.display

def eda_custom_process(eda_signal, sampling_rate=4, method="neurokit"):
    

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

def create_emotion_df(path):
    emotions = []
    for i in range(0, 2100):
        json_file = json.load(open(path+"/prediction.json"))
        emotion = json_file[str(i)]["DeepFace"][0]
        # if (
        # json_file[str(i)]["DeepFace"][1] > json_file[str(i)]["ResNet"][1]) else json_file[str(i)]["ResNet"][0]  # Sets emotion to value with highest confidence
        emotions.append(emotion)
    df = pd.DataFrame(emotions, columns=["Emotion"])
    return df


def nr_similar_predictions(path):
    tot_similar = 0
    tot_confident_predictions = 0
    imgs_predicted_similarly = []
    for i in range(0, 2100):
        json_file = json.load(open(path+"/prediction.json"))
        emotion_deepface = json_file[str(i)]["DeepFace"][0]
        pred_score_deepface = json_file[str(i)]["DeepFace"][1]
        emotion_resnet = json_file[str(i)]["ResNet"][0]
        pred_score_resnet = json_file[str(i)]["ResNet"][1]
        if emotion_deepface == emotion_resnet:
            tot_similar = tot_similar + 1
            imgs_predicted_similarly.append(i)
        if pred_score_deepface > 90 or pred_score_resnet > 90:
            tot_confident_predictions = tot_confident_predictions + 1
    print("Total images that are predicted to be same class: " +
          str(tot_similar)+"/2100")
    print("Total images that are predicted with a probability percentage over 90: " +
          str(tot_confident_predictions)+"/2100")
    return imgs_predicted_similarly


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

def z_score_normalize(df,column_name):
    df_std = df.copy()
    # apply the z-score method
    df_std[column_name] = (df_std[column_name] - df_std[column_name].mean()) / df_std[column_name].std()
        
    return df_std


def transform_bvp(dataframe, column_name):
    nparray = dataframe.values
    # dataframe.plot()
    nparray /= np.max(np.abs(nparray), axis=0)
    # df = pd.DataFrame(nparray)
    # df.plot()
    nparray = nparray.clip(min=0)
    # df = pd.DataFrame(nparray)
    # df.plot()
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
    # df_scaled.plot()
    df_scaled.rename(columns={df_scaled.columns[0]: column_name}, inplace=True)
    # plt.show()
    return df_scaled


def set_phys_in_json(path, phys_df, phys_type):

    json_file = json.load(open(path+"/prediction.json", "r"))
    for i in range(0, 2100):
        json_file[str(i)][phys_type] = phys_df.iloc[i, 0]
    json.dump(json_file, open(path+"/prediction.json", "w"))


participant = "participant_8"

path = "/Users/andreas/Desktop/master/toadstool/participants/" + \
    participant+"/images"

# emotion_df = create_emotion_df(path)
# imgs_predicted_similarly = nr_similar_predictions(path)


df_HR = pd.read_csv("/Users/andreas/Desktop/master/toadstool/participants/" +
                    participant+"/"+participant+"_sensor/HR_sync_video.csv")

df_EDA = pd.read_csv("/Users/andreas/Desktop/master/toadstool/participants/" +
                     participant+"/"+participant+"_sensor/EDA_sync_video.csv")

df_BVP = pd.read_csv("/Users/andreas/Desktop/master/toadstool/participants/" +
                     participant+"/"+participant+"_sensor/BVP_sync_video.csv")

df_ACC = pd.read_csv("/Users/andreas/Desktop/master/toadstool/participants/" +
                     participant+"/"+participant+"_sensor/ACC_sync_video.csv",sep=';')

# print(df_ACC)
# new_df = df_ACC.apply(lambda r: np.sqrt((r['x']**2)+(r['y']**2)+(r['z']**2)),axis=1)
# norm_acc = normalize_dataframe_values(new_df, 'ACC')
#new_df.plot()
# norm_acc.plot()
# transformed_df_BVP = transform_bvp(df_BVP, "BVP")

# df_BVP_max = transformed_df_BVP.rolling(64).max()
# df_BVP_max = df_BVP_max.iloc[::64, :]

# df_BVP_max = df_BVP_max.iloc[1:, :]  # Remove NaN from first index

# df_BVP_max = df_BVP_max.reset_index()
# df_BVP_max = df_BVP_max.drop("index", 1)

# df_BVP_max = df_BVP_max['BVP'].round(decimals=3)
# df_BVP_max = df_BVP_max.to_frame("BVP")

# df_BVP_avg = transformed_df_BVP.rolling(64).mean()
# df_BVP_avg = df_BVP_avg.iloc[::64, :]


# df_BVP_avg = df_BVP_avg.iloc[1:, :]  # Remove NaN from first index
# df_BVP_avg = df_BVP_avg.reset_index()
# df_BVP_avg = df_BVP_avg.drop("index", 1)
# df_BVP_avg = df_BVP_avg['BVP'].round(decimals=3)
# df_BVP_avg = df_BVP_avg.to_frame("BVP")

# # df_BVP_avg.plot()
# # df_BVP_max.plot()
# # plt.show()


# df_HR_scaled = normalize_dataframe_values(df_HR, "HR")


# arr = np.repeat(df_EDA['EDA'].to_numpy(),16)

# mfccs = librosa.feature.mfcc(arr,n_mfcc=8, sr=64)
# print(mfccs.shape)
# np.save(participant+'_mfccs.npy', mfccs, allow_pickle=True)

signals, info = eda_custom_process(df_EDA["EDA"])
tonic = signals["EDA_Tonic"]
ax = tonic.plot(label="Tonic EDA Level")

phasic = signals["EDA_Phasic"]


phasic.to_csv(
     "/Users/andreas/Desktop/master/toadstool/participants/"+participant+"/"+participant+"_sensor/EDA_Phasic.csv", header="EDA")
# features = [info["SCR_Peaks"]]
#plot = nk.events_plot(features, phasic, color=['blue'])

#df_EDA_norm = z_score_normalize(df_EDA, "EDA")
#df_EDA_norm.plot()



# set_phys_in_json(path, df_BVP_avg, "BVP")

# df_BVP_max.to_csv(
#     "/Users/andreas/Desktop/master/toadstool/participants/"+participant+"/"+participant+"_sensor/transformed_bvp.csv", header="BVP")

# df_BVP_max.plot(kind="line", y="BVP")

# seaborn.set(style='ticks')
# _emotions = [
#     'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt'
# ]

# df_bvp_emotion = df_BVP_max[:2100].join(emotion_df)

# sim_samples = df_bvp_emotion.iloc[imgs_predicted_similarly, :]


# print(sim_samples.groupby('Emotion').count())
# print(sim_samples.groupby('Emotion').mean())
# seaborn.relplot(data=sim_samples.reset_index(), x='index',
#                 y='BVP', hue='Emotion', hue_order=_emotions, aspect=1.61)


# df_eda_emotion = df_EDA_scaled[:2100].join(emotion_df)
# df_hr_emotion = df_HR_scaled[:2100].join(emotion_df)

# seaborn.relplot(data=df_bvp_emotion.reset_index(), x='index',
#                 y='BVP', hue='Emotion', hue_order=_emotions, aspect=1.61)

# seaborn.relplot(data=df_eda_emotion.reset_index(), x='index',
#                 y='EDA', hue='Emotion', hue_order=_emotions, aspect=1.61)

# seaborn.relplot(data=df_hr_emotion.reset_index(), x='index',
#                 y='HR', hue='Emotion', hue_order=_emotions, aspect=1.61)

# print(df_bvp_emotion.groupby('Emotion').count())
# print(df_bvp_emotion.groupby('Emotion').mean())
# print(df_eda_emotion.groupby('Emotion').mean())
# print(df_hr_emotion.groupby('Emotion').mean())


