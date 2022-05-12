from transform_phys import transform_bvp,normalize_dataframe_values
import numpy as np
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os 
from random import uniform,shuffle,sample,randrange
import cv2
from helpers import *
from statistics import mean
from PIL import Image





classes = {
    'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6
}


class CKDataset_2D(Dataset):

    def __init__(self, rootdir, transforms):
        self.dataset = []
        for dirs in os.listdir(rootdir):
            path = rootdir + "/" + dirs
            if os.path.isdir(path):

                label = classes[dirs]
                for file in os.listdir(path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                         # load the image
                        #print(path+'/'+file)
                        image = Image.open(path+'/'+file)
                        # convert image to numpy array
                        
                        data = np.asarray(image).astype(
                            np.uint8)
    
                        self.dataset.append((data, label))

        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        transformed = self.transforms(self.dataset[i][0])
        return (transformed, self.dataset[i][1])


class CKDataset(Dataset):

    def __init__(self, dataset, transforms, seq_len):
        # function to get evenly distributed indexes over image sequence
        self.transforms = transforms
        self.dataset = dataset
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        transformed_seq = []

        for j in range(0, self.seq_len):
            transformed_seq.append(self.transforms(self.dataset[i][0][j])) #runs spatial transform
       
        if randrange(0, 100) < 20: #With certain probability(20%) run temporal cropping of the sequence
            transformed_seq = temporal_random_crop(transformed_seq,2)
        return torch.stack(transformed_seq, 0).permute(1, 0, 2, 3), self.dataset[i][1]

class Toadstool(Dataset):

    def __init__(self, dataset, transforms, seq_len):
        self.transforms = transforms
        self.dataset = dataset
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        transformed_seq = []

        for j in range(0, self.seq_len):
            transformed_seq.append(self.transforms(self.dataset[i][0][j])) #runs spatial transform
        return torch.stack(transformed_seq, 0).permute(1, 0, 2, 3), self.dataset[i][1]




class Toadstool_LSTM(Dataset):
    def __init__(self, seq_arr, transforms, seq_len):
        self.dataset = seq_arr
        self.transforms = transforms
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        transformed_seq = []
        for j in range(0, self.seq_len):
            transformed_seq.append(self.transforms(self.dataset[i][0][j]))
        return torch.stack(transformed_seq, 0), self.dataset[i][1]

class ToadstoolPhys(Dataset):

    def __init__(self, dataset):
        
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):

        t = torch.Tensor(self.dataset[i][0])
        
        return t, self.dataset[i][1]

class ToadstoolMultimodal(Dataset):

    def __init__(self, dataset, transforms, seq_len):
       
        self.transforms = transforms
        self.dataset = dataset
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        transformed_seq = []

        phys = torch.Tensor(self.dataset[i][0][1])
        for j in range(0, self.seq_len):
            transformed_seq.append(self.transforms(self.dataset[i][0][0][j])) #runs spatial transform
       
        return [torch.stack(transformed_seq, 0).permute(1, 0, 2, 3),phys,self.dataset[i][0][2]], self.dataset[i][1]

def get_ntop_sequences(jsonfile,ntop,label):
    with open(jsonfile, 'r') as fp:
        preds = json.load(fp)
        
        seqs = [p for p in preds.items() if p[1]['Emotion Label'] == label]
        
        seqs.sort(key = lambda x: x[1]['Probability score'])
        seqs = seqs[-ntop:]
        seqs = [int(seq[0].split('-')[0]) for seq in seqs] #Gets start index of each sequence
        return seqs



def temporal_random_crop(seq,crop_size):
    indices = sample(range(9), crop_size)
    for i in indices:
        seq.pop(i)
    duplicates = [val for val in seq[-crop_size:] for _ in (0, 1)]
    seq = seq[:-crop_size]
    seq.extend(duplicates)
    return seq

def create_face_bvp_arr(frames, df_bvp,seq_len,step_size):
    all_seq = []
    frame_indices = [2,2,3,3]
    for start in range(0,len(frames),step_size):
        jump_index = 0
        seq = []
        if (start+seq_len)>len(frames):
            break
        for index in frame_indices:
            for frame in frames[int(start+jump_index):int(start+jump_index+index)]:
                seq.append(frame)
            jump_index += 3

        all_seq.append((seq, round(mean(df_bvp.iloc[int(start/3):int(start/3)+4]["BVP"].tolist()), 4)))
    return all_seq

def create_augmented_data(label, all_seq, nr_rotate, nr_flip, nr_contrast,rotate,flip):
    
    aug_all_seq = []
    for seq in all_seq[:nr_rotate]:  # Adds Rotation augmentation
        augmented_seq = []
        for data in seq:
            cp_data = data.copy()
            aug_data = rotate(cp_data)
            aug_data = np.asarray(aug_data).astype(
                                np.uint8)
            augmented_seq.append(aug_data)
        aug_all_seq.append((augmented_seq, label))

    for seq in all_seq[-nr_flip:]:
        augmented_seq = []
        for data in seq:
            cp_data = data.copy()
            aug_data= flip(cp_data)
            aug_data = np.asarray(aug_data).astype(
                                np.uint8)
            augmented_seq.append(aug_data)
        aug_all_seq.append((augmented_seq, label))

    for seq in all_seq[(int(len(all_seq)/2)):((int(len(all_seq)/2))+nr_contrast)]:
        augmented_seq = []
        contrast = uniform(0.3, 1.8)
        for data in seq:
            cp_data = data.copy()
            aug_data = 127 + contrast * (cp_data[:]-127)
            aug_data = np.asarray(aug_data).astype(
                                np.uint8)
            augmented_seq.append(aug_data)
        aug_all_seq.append((augmented_seq, label))
    return aug_all_seq



def create_ck_arr(rootdir, seq_len, rotate, flip):
    dataset = []
    for dirs in os.listdir(rootdir):
        path = rootdir + "/" + dirs
        print("Creating datapoints in folder:" + path)
        all_seq = []
        peak_sequences = 0
        if os.path.isdir(path):
            label = classes[dirs]
            for seq in os.listdir(path):
                img_seq = {}  # need to keep track of file names so that images are added to img_seq in correct order
                if not seq.startswith('.'):
                    files = next(os.walk(path+"/"+seq))[2]
                    n_files = len(files)
                   

                    for img in os.listdir(path+"/"+seq):

                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):

                           
                            if dirs == 'neutral':  # When iterating through directory the image files will not come in sequencial order therefor we have to retrieve the image nr
                                img_nr = img.split(
                                    '_')[-1].split('-')[0].lstrip('0')
                                img_nr = int(img_nr)
                            else:
                                img_nr = img.split(
                                    '_')[-1].split('.')[0].lstrip('0')
                                img_nr = int(img_nr)

                            img = cv2.imread(path+"/"+seq+'/'+img)
                            # convert image to numpy array
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                             
                            face = fdc.detect_face(img)
                            
                            data = np.asarray(face).astype(
                                np.uint8)
                            img_seq[img_nr] = data

                    img_seq = dict(sorted(img_seq.items()))
                    img_seq = list(img_seq.values())

                    if dirs == "neutral":
                        shuffle(img_seq)
                    if len(img_seq) >= seq_len:
                        img_seq = img_seq[:seq_len]
                        dataset.append((img_seq, label))
                        all_seq.append(img_seq)
                    else:
                        nr_dup = seq_len-n_files
                        duplicates = [val for val in img_seq[-nr_dup:] for _ in (0, 1)]
                        img_seq = img_seq[:-nr_dup]
                        img_seq.extend(duplicates)
                        dataset.append((img_seq, label))
                        all_seq.append(img_seq)
                    if dirs != "neutral" and peak_sequences<5:
                        new_img_seq = [val for val in img_seq[-5:] for _ in (0, 1)]
                        dataset.append((new_img_seq,label))
                        all_seq.append(new_img_seq)
                    peak_sequences += 1
        
        if len(all_seq) > 0:  # Adds a number of augmented sequences to each class to get a more balanced dataset
            if dirs == "sadness":
                augmented = create_augmented_data(label, all_seq, 5, 4, 4,rotate,flip)
                dataset.extend(augmented)
            elif dirs == "neutral":
                augmented = create_augmented_data(label, all_seq, 1, 1 ,1,rotate,flip)
                dataset.extend(augmented)
            elif dirs == "happy":
                augmented = create_augmented_data(label, all_seq, 1, 1, 1,rotate,flip)
                dataset.extend(augmented)
            elif dirs == "fear":
                augmented = create_augmented_data(label, all_seq, 5, 4, 4,rotate,flip)
                dataset.extend(augmented)
            elif dirs == "disgust":
                augmented = create_augmented_data(label, all_seq, 1, 1, 1,rotate,flip)
                dataset.extend(augmented)
            elif dirs == "anger":
                augmented = create_augmented_data(label, all_seq, 5, 4, 4,rotate,flip)
                dataset.extend(augmented)
            elif dirs == "surprise":
                augmented = create_augmented_data(label, all_seq, 1, 1, 1,rotate,flip)
                dataset.extend(augmented)
    return dataset




def create_multimodal_dataset(json,frames,bvp_df,hr_df,eda_df,acc_df,nr_of_each_label):
    dataset_arr = []
    frame_indices = [2,2,3,3]
    hr_df = normalize_dataframe_values(hr_df,'HR')
    acc_df = normalize_dataframe_values(acc_df,'ACC')
    eda_df = normalize_dataframe_values(eda_df,"EDA_Phasic")
    bvp_df = normalize_dataframe_values(bvp_df,"BVP")
    #mfccs = mfccs / mfccs.max(axis=0)
    classes = {
        'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6
    }

    for c in classes.items():
        label = c[0]
        label_i = c[1]
        nr_labels = nr_of_each_label[label_i]
        seqs = get_ntop_sequences(json,nr_labels,label)
        for seq_start in seqs:
            seq = []
            #mfcc = []
            jump_index = 0
            for index in frame_indices:
                for frame in frames[int(jump_index+(seq_start*3)):int(jump_index+index+(seq_start*3))]:
                    seq.append(frame)
                jump_index += 3
            
            bvp_seq = bvp_df.iloc[seq_start*64:(seq_start+4)*64]['BVP'].tolist()
            eda_seq = eda_df.iloc[(seq_start)*4:((seq_start)+4)*4]['EDA_Phasic'].tolist()
            hr_seq = hr_df.iloc[seq_start:seq_start+4]['HR'].tolist()
            acc_seq = acc_df.iloc[seq_start*32:(seq_start+4)*32]['ACC'].tolist()
            
            
            eda_seq = np.repeat(eda_seq,16)
            hr_seq = np.repeat(hr_seq,64)
            acc_seq = np.repeat(acc_seq,2)
            # for i in range(4):
            #     mfcc.append(mfccs[:,int(((seq_start+i)/2097*len(mfccs)))])
            # mfcc = np.mean(mfcc,axis=0)
            # mfcc = np.repeat(mfcc,32)
            eda_max = np.max(eda_seq)
            eda_max = np.repeat(eda_max,256)
            bvp_max = np.max(bvp_seq)
            bvp_max = np.repeat(bvp_max,256)
            acc_max = np.max(acc_seq)
            acc_max = np.repeat(acc_max,256)
            eda_mean = np.max(eda_seq)
            eda_mean = np.repeat(eda_mean,256)
            bvp_mean = np.max(bvp_seq)
            bvp_mean = np.repeat(bvp_mean,256)
            acc_mean = np.max(acc_seq)
            acc_mean = np.repeat(acc_mean,256)
            
            fused = [bvp_seq,eda_seq,hr_seq,acc_seq,bvp_max,bvp_mean,eda_max,eda_mean,acc_max,acc_mean]
            
            dataset_arr.append(([seq,fused,seq_start],label_i))
    return dataset_arr

def create_dataset_svm(json,bvp_df,hr_df,eda_df,acc_df,mfccs,nr_of_each_label):
    dataset_arr = []
    
    hr_df = normalize_dataframe_values(hr_df,'HR')
    acc_df = normalize_dataframe_values(acc_df,'ACC')
    eda_df = normalize_dataframe_values(eda_df,"EDA_Phasic")
    bvp_df = normalize_dataframe_values(bvp_df,"BVP")
    #mfccs = mfccs / mfccs.max(axis=0)
    classes = {
        'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6
    }

    for c in classes.items():
        label = c[0]
        label_i = c[1]
        nr_labels = nr_of_each_label[label_i]
        seqs = get_ntop_sequences(json,nr_labels,label)
        for seq_start in seqs:
            seq = []
            mfcc = []
            
            for i in range(4):
                seq.append(np.mean(bvp_df.iloc[(seq_start+i)*64:((seq_start+i)*64)+64]['BVP'].tolist()))
                seq.append(np.mean(eda_df.iloc[(seq_start+i)*4:((seq_start+i)*4)+4]['EDA_Phasic'].tolist()))
                seq.append(hr_df.iloc[seq_start+i]['HR'].tolist())
                seq.append(np.mean(acc_df.iloc[(seq_start+i)*32:((seq_start+i)*32)+32]['ACC'].tolist()))
                seq.append(np.max(bvp_df.iloc[(seq_start+i)*64:((seq_start+i)*64)+64]['BVP'].tolist()))
                seq.append(np.max(eda_df.iloc[(seq_start+i)*4:((seq_start+i)*4)+4]['EDA_Phasic'].tolist()))
                seq.append(np.max(acc_df.iloc[(seq_start+i)*32:((seq_start+i)*32)+32]['ACC'].tolist()))
                #mfcc.append(mfccs[:,int(((seq_start+i)/2097*len(mfccs)))])
            # mfcc = np.mean(mfcc,axis=0)
            # seq.extend(mfcc)
            dataset_arr.append((seq,label_i))
    return dataset_arr

def create_dataset(json,frames,bvp_df,hr_df,eda_df,acc_df,mfccs,nr_of_each_label):
    dataset_arr = []
    bvp_dataset = []
    frame_indices = [2,2,3,3]
    hr_df = normalize_dataframe_values(hr_df,'HR')
    acc_df = normalize_dataframe_values(acc_df,'ACC')
    eda_df = normalize_dataframe_values(eda_df,"EDA_Phasic")
    bvp_df = normalize_dataframe_values(bvp_df,"BVP")
    #mfccs = mfccs / mfccs.max(axis=0)
    classes = {
        'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6
    }

    for c in classes.items():
        label = c[0]
        label_i = c[1]
        nr_labels = nr_of_each_label[label_i]
        seqs = get_ntop_sequences(json,nr_labels,label)
        for seq_start in seqs:
            seq = []
            mfcc = []
            jump_index = 0
            for index in frame_indices:
                for frame in frames[int(jump_index+(seq_start*3)):int(jump_index+index+(seq_start*3))]:
                    seq.append(frame)
                jump_index += 3
            
            bvp_seq = bvp_df.iloc[seq_start*64:(seq_start+4)*64]['BVP'].tolist()
            eda_seq = eda_df.iloc[seq_start*4:(seq_start+4)*4]['EDA_Phasic'].tolist()
            hr_seq = hr_df.iloc[seq_start:seq_start+4]['HR'].tolist()
            acc_seq = acc_df.iloc[seq_start*32:(seq_start+4)*32]['ACC'].tolist()
            
            
            # for i in range(4):
            #     mfcc.append(mfccs[:,int(((seq_start+i)/2097*len(mfccs)))])
            # mfcc = np.mean(mfcc,axis=0)
            # mfcc = np.repeat(mfcc,32)
            eda_seq = np.repeat(eda_seq,16)
            hr_seq = np.repeat(hr_seq,64)
            acc_seq = np.repeat(acc_seq,2)
            eda_max = np.max(eda_seq)
            eda_max = np.repeat(eda_max,256)
            bvp_max = np.max(bvp_seq)
            bvp_max = np.repeat(bvp_max,256)
            acc_max = np.max(acc_seq)
            acc_max = np.repeat(acc_max,256)
            eda_mean = np.max(eda_seq)
            eda_mean = np.repeat(eda_mean,256)
            bvp_mean = np.max(bvp_seq)
            bvp_mean = np.repeat(bvp_mean,256)
            acc_mean = np.max(acc_seq)
            acc_mean = np.repeat(acc_mean,256)
            fused = [bvp_seq,eda_seq,hr_seq,acc_seq,bvp_max,bvp_mean,eda_max,eda_mean,acc_max,acc_mean] 
            bvp_dataset.append((fused,label_i))
            dataset_arr.append((seq,label_i))
    return dataset_arr,bvp_dataset

def create_unprocessed_multimodal_dataset(json,frames,bvp_df,hr_df,eda_df,acc_df,nr_of_each_label):
    dataset_arr = []
    
    classes = {
        'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6
    }

    for c in classes.items():
        label = c[0]
        label_i = c[1]
        nr_labels = nr_of_each_label[label_i]
        seqs = get_ntop_sequences(json,nr_labels,label)
        for seq_start in seqs:
            seq = []
            
            for frame in frames[int((seq_start*3)):int((seq_start+4)*3)]:
                seq.append(frame)
                
            
            bvp_seq = bvp_df.iloc[seq_start*64:(seq_start+4)*64]['BVP'].tolist()
            eda_seq = eda_df.iloc[(seq_start)*4:((seq_start)+4)*4]['EDA'].tolist()
            hr_seq = hr_df.iloc[seq_start:seq_start+4]['HR'].tolist()
            
            acc_seq = acc_df.iloc[seq_start*32:(seq_start+4)*32]['x;y;z'].tolist()
            
            fused = [bvp_seq,eda_seq,hr_seq,acc_seq]
            dataset_arr.append(([seq,fused,seq_start],label_i))
    return dataset_arr 

def main():
    sample_distribution = {0:[27,101,74,13,	0,4,1760],1:[196,1190,0,97,0,0,614],2:[8,0,122,537,42,9,1379],3:[72,0,4,17,201,1,1802],4:[32,98,61,248,69,124,1465],5:[255,123,35,698,80,328,578],6:[41,310,799,315,22,9,601],7:[690,232,0,9,167,100,899],8:[0,0,1065,6,262,0,764],9:[722,66,13,105,161,129,901]}
    for i in range(10):
        
        participant = "participant_" + str(i)
        json = participant + "_prediction4sec.json"
        frames_file = participant + "_3FramesPerSec.npy"
        frames = np.load(frames_file, allow_pickle=True)
        bvp_df = pd.read_csv(participant + "/BVP_sync_video.csv")
        hr_df = pd.read_csv(participant + "/HR_sync_video.csv")
        eda_df = pd.read_csv(participant + "/EDA_sync_video.csv")
        acc_df = pd.read_csv(participant + "/ACC_sync_video.csv",sep=';')
        #mfccs = np.load(participant+'/'+participant+'_mfccs.npy', allow_pickle=True)
        trfm_acc_df = acc_df.apply(lambda r: np.sqrt((r['x']**2)+(r['y']**2)+(r['z']**2)),axis=1)
        
        transformed_df_BVP = transform_bvp(bvp_df, "BVP")
        nr_of_each_label = sample_distribution[i] # Number of how many samples to be extracted from each label category
        
        print("Creating dataset for: " + participant)
        if not os.path.exists(participant+'_toadstool_multimodal.npy'):
            dataset_arr = create_multimodal_dataset(json,frames,transformed_df_BVP,hr_df,eda_df,trfm_acc_df,nr_of_each_label)
            dataset_arr = np.asarray(dataset_arr)
            np.save(participant+'_toadstool_multimodal.npy', dataset_arr, allow_pickle=True)

if __name__ == '__main__':
    main()