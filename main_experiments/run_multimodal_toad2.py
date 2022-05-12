import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
import json
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os 
import numpy as np
from resnext import C3D 
from sklearn.svm import SVC
from sklearn.utils import class_weight
from sklearn.model_selection import KFold 
import copy
from helpers import *

grayscale = transforms.Compose(
    [

        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
        
    ])


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    losses = Loss()
    top1 = Metrics()
    
    for (inputs, targets) in data_loader:

        optimizer.zero_grad()

        outputs = model(inputs)
        softmax = F.softmax(outputs,dim=1)
        loss = criterion(softmax, targets)
        prec1,prec,recall,f1_score,support = calculate_metrics(outputs.data, targets.data)
        top1.update(prec1,prec,recall,f1_score,support, inputs.size(0))
        losses.update(loss.item(), inputs[0].size(0))

        loss.backward()
        optimizer.step()

    return losses.avgLoss, top1.avgAcc

def train_epoch_multimodal(modelFER,modelPER, data_loader, criterion, optimizerFER,optimizerPER, device):
    modelFER.train()
    modelPER.train()
    weightFER = 0.7
    weightPER = 0.3
    losses = Loss()
    metrics = Metrics()
    metricsFER = Metrics()
    metricsPER = Metrics()

    
    for (inputs, targets) in data_loader:

        
        optimizerFER.zero_grad()
        optimizerPER.zero_grad()

        outputsFER = modelFER(inputs[0])
        outputsPER = modelPER(inputs[1])
        
        softmaxFER = F.softmax(outputsFER,dim=1)
        softmaxPER = F.softmax(outputsPER,dim=1)
        softFusion = (weightFER * softmaxFER) + (weightPER * softmaxPER)

        lossFER = criterion(softmaxFER, targets)
        lossPER = criterion(softmaxPER, targets)

        acc,prec,recall,f1_score,support = calculate_metrics(softFusion.data, targets.data)
        accFER,precFER,recallFER,f1_scoreFER,supportFER = calculate_metrics(softmaxFER.data, targets.data)
        accPER,precPER,recallPER,f1_scorePER,supportPER = calculate_metrics(softmaxPER.data, targets.data)

        
        metrics.update(acc,prec,recall,f1_score,support, inputs[0].size(0))
        metricsFER.update(accFER,precFER,recallFER,f1_scoreFER,supportFER, inputs[0].size(0))
        metricsPER.update(accPER,precPER,recallPER,f1_scorePER,supportPER, inputs[0].size(0))

        losses.update(lossFER.item(), inputs[0].size(0))

        lossFER.backward()
        optimizerFER.step()
        lossPER.backward()
        optimizerPER.step()
        

    return losses.avgLoss, metrics.avgAcc, metricsFER.avgAcc, metricsPER.avgAcc

def val_epoch_multimodal(modelFER,modelPER, data_loader, criterion, device):
    weightFER = 0.7
    weightPER = 0.3
    with torch.no_grad():
        modelFER.eval()
        modelPER.eval()
        losses = Loss()
        metrics = Metrics()
        metricsFER = Metrics()
        metricsPER = Metrics()
        mer = []
        fer = []
        per = []
        targets_list = []
        sequences = []
        predicted = []
        confusion_matrix_MER = torch.zeros(7, 7)
        confusion_matrix_FER = torch.zeros(7, 7)
        confusion_matrix_PER = torch.zeros(7, 7)

        for (inputs, targets) in data_loader:

            outputsFER = modelFER(inputs[0])
            outputsPER = modelPER(inputs[1])
            softmaxFER = F.softmax(outputsFER,dim=1)
            softmaxPER = F.softmax(outputsPER,dim=1)
            softFusion = (weightFER * softmaxFER) + (weightPER * softmaxPER)

        
            loss = criterion(softFusion, targets)
            acc,prec,recall,f1_score,support = calculate_metrics(softFusion.data, targets.data)
            accFER,precFER,recallFER,f1_scoreFER,supportFER = calculate_metrics(softmaxFER.data, targets.data)
            accPER,precPER,recallPER,f1_scorePER,supportPER = calculate_metrics(softmaxPER.data, targets.data)

            metrics.update(acc,prec,recall,f1_score,support, inputs[0].size(0))
            metricsFER.update(accFER,precFER,recallFER,f1_scoreFER,supportFER, inputs[0].size(0))
            metricsPER.update(accPER,precPER,recallPER,f1_scorePER,supportPER, inputs[0].size(0))

            _, preds = torch.max(softFusion, 1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix_MER[t.long(), p.long()] += 1
            
            _, preds_fer = torch.max(softmaxFER, 1)
            for t, p in zip(targets.view(-1), preds_fer.view(-1)):
                confusion_matrix_FER[t.long(), p.long()] += 1

            _, preds_per = torch.max(softmaxPER, 1)
            for t, p in zip(targets.view(-1), preds_per.view(-1)):
                confusion_matrix_PER[t.long(), p.long()] += 1
            
            losses.update(loss.item(), inputs[0].size(0))
            if acc>accFER:
                mer.append(softFusion.tolist())
                mer.append("\n")
                fer.append(softmaxFER.tolist())
                fer.append("\n")
                per.append(softmaxPER.tolist())
                per.append("\n")
                targets_list.append(targets.tolist())
                sequences.append(inputs[2].tolist())
                predicted.append([torch.argmax(softFusion,dim=1),torch.argmax(softmaxFER,dim=1),torch.argmax(softmaxPER,dim=1)])

    return losses.avgLoss, metrics.avgAcc, metricsFER.avgAcc, metricsPER.avgAcc, mer,fer,per,targets_list,sequences,confusion_matrix_MER,confusion_matrix_FER,confusion_matrix_PER,predicted

def val_epoch(model, data_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        losses = Loss()
        top1 = Metrics()
        confusion_matrix = torch.zeros(7, 7)
        for (inputs, targets) in data_loader:

            outputs = model(inputs)
            softmax = F.softmax(outputs,dim=1)
            loss = criterion(softmax, targets)
            prec1,prec,recall,f1_score,support = calculate_metrics(outputs.data, targets.data)
            top1.update(prec1,prec,recall,f1_score,support, inputs.size(0))
            losses.update(loss.item(), inputs.size(0))
            _, preds = torch.max(outputs, 1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)
    
    return losses.avgLoss, top1.avgAcc, acc_per_class.tolist()


def run_svm():
    participant = "participant_4"
    json = participant + "_prediction.json"
    frames_file = participant + "_3frames_histmatched.npy"
    frames = np.load(frames_file, allow_pickle=True)
    bvp_df = pd.read_csv(participant + "/BVP_sync_video.csv")
    hr_df = pd.read_csv(participant + "/HR_sync_video.csv")
    eda_df = pd.read_csv(participant + "/EDA_Phasic.csv")
    acc_df = pd.read_csv(participant + "/ACC_sync_video.csv",sep=';')
    mfccs = np.load(participant+'/'+participant+'_mfccs.npy', allow_pickle=True)
    trfm_acc_df = acc_df.apply(lambda r: np.sqrt((r['x']**2)+(r['y']**2)+(r['z']**2)),axis=1)
    
    transformed_df_BVP = transform_bvp(bvp_df, "BVP")
    nr_of_each_label = [32,45,45,45,45,45,45] # Number of how many samples to be extracted from each label category

    dataset = create_dataset_svm(json,frames,transformed_df_BVP,hr_df,eda_df,trfm_acc_df,mfccs,nr_of_each_label)
    train, val = train_test_split(dataset,shuffle=True)

    
    X_train = [i[0] for i in train]
    y_train = [i[1] for i in train]

    X_val = [i[0] for i in val]
    y_val = [i[1] for i in val]
    
            
    svm = SVC(kernel='rbf',gamma=6.5 ,C=6.5 ,random_state=0)
    svm.fit(X_train, y_train)
    print(svm.score(X_val,y_val))
    

def run_multimodal():
    optim_phys = {
        "lr_rate": 0.0001,
        "weight_decay": 0.00001,
        "n_epochs": 350
    }
    optim = {
        "lr_rate": 0.0001,
        "weight_decay": 0.00001,
        "n_epochs": 200
    }

    np.seterr(divide='ignore', invalid='ignore')
    device = get_default_device()
    participant = "participant_4"
    json = participant + "_prediction.json"
    frames_file = participant + "_3frames_histmatched.npy"
    frames = np.load(frames_file, allow_pickle=True)
    bvp_df = pd.read_csv(participant + "/BVP_sync_video.csv")
    hr_df = pd.read_csv(participant + "/HR_sync_video.csv")
    eda_df = pd.read_csv(participant + "/EDA_Phasic.csv")
    acc_df = pd.read_csv(participant + "/ACC_sync_video.csv",sep=';')
    mfccs = np.load(participant+'/'+participant+'_mfccs.npy', allow_pickle=True)
    trfm_acc_df = acc_df.apply(lambda r: np.sqrt((r['x']**2)+(r['y']**2)+(r['z']**2)),axis=1)
    
    transformed_df_BVP = transform_bvp(bvp_df, "BVP")
    nr_of_each_label = [32,45,45,45,45,45,45] # Number of how many samples to be extracted from each label category
    k = 3
    kf = KFold(n_splits=k,shuffle=True,random_state=1)
    
    if not os.path.exists(participant+'_toadstool_multimodal.npy'):
        dataset_arr = create_multimodal_dataset(json,frames,transformed_df_BVP,hr_df,eda_df,trfm_acc_df,mfccs,nr_of_each_label)
        dataset_arr = np.asarray(dataset_arr)
        np.save(participant+'_toadstool_multimodal.npy', dataset_arr, allow_pickle=True)

    else:
        dataset_arr = np.load(participant+'_toadstool_multimodal.npy', allow_pickle=True)
    
    dataset = ToadstoolMultimodal(dataset_arr,grayscale,10)
    best_accs = 0
    best_accs_FER = 0
    best_accs_PER = 0
    confusion_matrices_MER = []
    confusion_matrices_FER = []
    confusion_matrices_PER = []

    
    for fold,(train_ids,test_ids) in enumerate(kf.split(dataset)):
        batch_num = 16
        modelPER = Net1D()
        to_device(modelPER,device)

        state1 = torch.load("trained_model_weights/"+participant+"_Net1D_weights0.pt")
        state2 = torch.load("trained_model_weights/"+participant+"_Net1D_weights1.pt")
        state3 = torch.load("trained_model_weights/"+participant+"_Net1D_weights2.pt")
        for key in state1:
            state1[key] = (state1[key] + state2[key] + state3[key]) / 3.

        modelPER.load_state_dict(state1,strict=False)
        
        PER_params = list(modelPER.parameters())
        optimizerPER = torch.optim.Adam(
            PER_params, lr=optim_phys["lr_rate"], weight_decay=optim_phys["weight_decay"])

        modelFER = C3D(sample_size=112,
                    sample_duration=10,
                    num_classes=7)

        to_device(modelFER,device)
        FER_params = list(modelFER.parameters())
        optimizerFER = torch.optim.Adam(
            FER_params, lr=optim["lr_rate"], weight_decay=optim["weight_decay"])

    
        criterion = nn.CrossEntropyLoss() 

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_num, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_num, sampler=test_subsampler)
        

        train_dl = DeviceDataLoader(trainloader, device)
        val_dl = DeviceDataLoader(testloader, device)
        best_acc = 0
        best_acc_FER = 0
        best_acc_PER = 0
        
        best_loss = 10000
        

        for epoch in range(1, optim["n_epochs"] + 1):
            train_loss, train_acc,train_acc_FER,train_acc_PER= train_epoch_multimodal(
                modelFER,modelPER, train_dl, criterion, optimizerFER,optimizerPER, device)

            val_loss, val_acc,val_acc_FER,val_acc_PER, softFusion,softmaxFER,softmaxPER,targets,sequences,confusion_matrix_MER,confusion_matrix_FER,confusion_matrix_PER,predicted = val_epoch_multimodal(modelFER,modelPER, val_dl, criterion, device)

            # if val_acc>=0.8 and val_acc>val_acc_FER:
            #     with open(participant+'_analysis.txt', 'a') as f:
            #         f.write(str(softFusion))
            #         f.write("\n")
            #         f.write(str(softmaxFER))
            #         f.write("\n")
            #         f.write(str(softmaxPER))
            #         f.write("\n")
            #         f.write(str(targets))
            #         f.write("\n")
            #         f.write(str(sequences))
            #         f.write("\n")
            #         f.write(str(predicted))
            #         f.write("\n")
            if val_acc>best_acc: 
                best_acc = val_acc
                
                best_val_confusion_MER = confusion_matrix_MER.tolist()
            if val_acc_FER > best_acc_FER:
                best_acc_FER = val_acc_FER
                
                best_val_confusion_FER = confusion_matrix_FER.tolist()

            if val_acc_PER > best_acc_PER:
                best_acc_PER = val_acc_PER
                
                best_val_confusion_PER = confusion_matrix_PER.tolist()
            if val_loss < best_loss:
                early_stopping_patience = 20
                best_loss = val_loss
            else:
                early_stopping_patience -= 1
            if early_stopping_patience == 0:
                 break
        
            print("Epoch nr "+str(epoch)+":\n      Training loss:" + str(train_loss) + ", Validation loss: " +
                str(val_loss) + "\n      Training acc:" + str(train_acc) + ", Validation acc: " + str(val_acc) + 
                "\n      Training acc FER:" + str(train_acc_FER) + ", Validation acc FER: " + str(val_acc_FER) + 
                "\n      Training acc PER:" + str(train_acc_PER) + ", Validation acc PER: " + str(val_acc_PER))
        
        confusion_matrices_MER.append(best_val_confusion_MER)
        confusion_matrices_FER.append(best_val_confusion_FER)
        confusion_matrices_PER.append(best_val_confusion_PER)


        best_accs += best_acc
        best_accs_FER += best_acc_FER
        best_accs_PER += best_acc_PER
        

    print("Average best accuracy:", str(best_accs/k))
    print("Average best accuracy FER:", str(best_accs_FER/k))
    print("Average best accuracy PER:", str(best_accs_PER/k))
    precision_mer,recall_mer,f1_score_mer = metrics_from_cm(confusion_matrices_MER)
    precision_fer,recall_fer,f1_score_fer = metrics_from_cm(confusion_matrices_FER)
    precision_per,recall_per,f1_score_per = metrics_from_cm(confusion_matrices_PER)
    print("Precision per class MER: ", str(precision_mer))
    print("Precision per class FER: ", str(precision_fer))
    print("Precision per class PER: ", str(precision_per))
    print("Average Precision MER: ", str(np.mean(precision_mer)))
    print("Average Precision FER: ", str(np.mean(precision_fer)))
    print("Average Precision PER: ", str(np.mean(precision_per)))
    print("Recall per class MER: ", str(recall_mer))
    print("Recall per class FER: ", str(recall_fer))
    print("Recall per class PER: ", str(recall_per))
    print("Average Recall MER: ", str(np.mean(recall_mer)))
    print("Average Recall FER: ", str(np.mean(recall_fer)))
    print("Average Recall PER: ", str(np.mean(recall_per)))
    print("F1-Score per class MER: ", str(f1_score_mer))
    print("F1-Score per class FER: ", str(f1_score_fer))
    print("F1-Score per class PER: ", str(f1_score_per))
    print("Average F1-Score MER: ", str(np.mean(f1_score_mer)))
    print("Average F1-Score FER: ", str(np.mean(f1_score_fer)))
    print("Average F1-Score PER: ", str(np.mean(f1_score_per)))
    print(str(confusion_matrices_MER))
    print(str(confusion_matrices_FER))
    print(str(confusion_matrices_PER))


    


def run_1DCNN():
    optim_phys = {
        "lr_rate": 0.0001,
        "weight_decay": 0.00001,
        "n_epochs": 350
    }
    

    np.seterr(divide='ignore', invalid='ignore')
    device = get_default_device()
    participant = "participant_4"
    json = participant + "_prediction.json"
    frames_file = participant + "_3frames_histmatched.npy"
    frames = np.load(frames_file, allow_pickle=True)
    bvp_df = pd.read_csv(participant + "/BVP_sync_video.csv")
    hr_df = pd.read_csv(participant + "/HR_sync_video.csv")
    eda_df = pd.read_csv(participant + "/EDA_Phasic.csv")
    acc_df = pd.read_csv(participant + "/ACC_sync_video.csv",sep=';')
    mfccs = np.load(participant+'/'+participant+'_mfccs.npy', allow_pickle=True)
    trfm_acc_df = acc_df.apply(lambda r: np.sqrt((r['x']**2)+(r['y']**2)+(r['z']**2)),axis=1)
    
    transformed_df_BVP = transform_bvp(bvp_df, "BVP")
    nr_of_each_label = [32,45,45,45,45,45,45] # Number of how many samples to be extracted from each label category
    k = 3
    kf = KFold(n_splits=k,shuffle=True,random_state=1)

    
    if not os.path.exists(participant+'_toadstool.npy') or not os.path.exists(participant+'_toadstool_Physfused.npy'):
        dataset_arr,phys_dataset = create_dataset(json,frames,transformed_df_BVP,hr_df,eda_df,trfm_acc_df,mfccs,nr_of_each_label)
        dataset_arr = np.asarray(dataset_arr)
        phys_dataset = np.asarray(phys_dataset)
        np.save(participant+'_toadstool.npy', dataset_arr, allow_pickle=True)
        np.save(participant+'_toadstool_Physfused.npy', phys_dataset, allow_pickle=True)

    else:
        dataset_arr = np.load(participant+'_toadstool.npy', allow_pickle=True)
        phys_dataset = np.load(participant+'_toadstool_Physfused.npy', allow_pickle=True)

    dataset = ToadstoolPhys(phys_dataset)
    best_accs = 0
    overall_best_acc = 0
    for fold,(train_ids,test_ids) in enumerate(kf.split(dataset)):
        batch_num = 12
        model = Net1D()
        model.cuda()
        crnn_params = list(model.parameters())
        optimizer = torch.optim.Adam(
            crnn_params, lr=optim_phys["lr_rate"], weight_decay=optim_phys["weight_decay"])

        device = get_default_device()
        class_weights=class_weight.compute_class_weight('balanced',classes=np.unique(phys_dataset[:,1]),y=phys_dataset[:,1])
        class_weights=torch.tensor(class_weights,dtype=torch.float).cuda()
        
        criterion = nn.CrossEntropyLoss() #weight=class_weights,reduction='mean'

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_num, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_num, sampler=test_subsampler)
        

        train_dl = DeviceDataLoader(trainloader, device)
        val_dl = DeviceDataLoader(testloader, device)
        best_acc = 0
        best_loss = 10000
        for epoch in range(1, optim_phys["n_epochs"] + 1):
            train_loss, train_acc = train_epoch(
                model, train_dl, criterion, optimizer, device)

            val_loss, val_acc, acc_per_class = val_epoch(model, val_dl, criterion, device)
            
            if val_acc>best_acc: #any(i<val_acc for i in best_accs)
                best_acc = val_acc
                best_acc_per_class = acc_per_class
                if best_acc > overall_best_acc:
                    overall_best_acc = best_acc
                w = copy.deepcopy(model.state_dict())
            if val_loss < best_loss:
                early_stopping_patience = 100
                best_loss = val_loss
            else:
                early_stopping_patience -= 1
            if early_stopping_patience == 0:
            #     print("Early stopping at epoch:" + str(epoch))
            #     print("Best accuracy: " + str(best_acc))
            #     print("Best accuracy per class" + str(best_acc_per_class))
                break
            
            print("Epoch nr "+str(epoch)+":\n      Training loss:" + str(train_loss) + ", Validation loss: " +
                str(val_loss) + "\n      Training acc:" + str(train_acc) + ", Validation acc: " + str(val_acc))
        
        best_accs += best_acc
        torch.save(w, "trained_model_weights/"+participant+"_Net1D_weights"+str(fold)+".pt")
    print("Overall best accuracy",str(overall_best_acc))
    print("Average best accuracy:", str(best_accs/k))


def run_3DCNN():
    ##############################################################
     
    optim = {
        "lr_rate": 0.0001,
        "weight_decay": 0.00001,
        "n_epochs": 200
    }

    np.seterr(divide='ignore', invalid='ignore')
    device = get_default_device()
    participant = "participant_4"
    json = participant + "_prediction.json"
    frames_file = participant + "_3frames_histmatched.npy"
    frames = np.load(frames_file, allow_pickle=True)
    bvp_df = pd.read_csv(participant + "/BVP_sync_video.csv")
    hr_df = pd.read_csv(participant + "/HR_sync_video.csv")
    eda_df = pd.read_csv(participant + "/EDA_Phasic.csv")
    acc_df = pd.read_csv(participant + "/ACC_sync_video.csv",sep=';')
    mfccs = np.load(participant+'/'+participant+'_mfccs.npy', allow_pickle=True)
    trfm_acc_df = acc_df.apply(lambda r: np.sqrt((r['x']**2)+(r['y']**2)+(r['z']**2)),axis=1)
    
    transformed_df_BVP = transform_bvp(bvp_df, "BVP")
    nr_of_each_label = [32,45,45,45,45,45,45] # Number of how many samples to be extracted from each label category
    k = 3
    kf = KFold(n_splits=k,shuffle=True,random_state=1)
    

    if not os.path.exists(participant+'_toadstool.npy') or not os.path.exists(participant+'_toadstool_Physfused.npy'):
        dataset_arr,phys_dataset = create_dataset(json,frames,transformed_df_BVP,hr_df,eda_df,trfm_acc_df,mfccs,nr_of_each_label)
        dataset_arr = np.asarray(dataset_arr)
        phys_dataset = np.asarray(phys_dataset)
        np.save(participant+'_toadstool.npy', dataset_arr, allow_pickle=True)
        np.save(participant+'_toadstool_Physfused.npy', phys_dataset, allow_pickle=True)

    else:
        dataset_arr = np.load(participant+'_toadstool.npy', allow_pickle=True)
        phys_dataset = np.load(participant+'_toadstool_Physfused.npy', allow_pickle=True)
    seq_len = 10
    best_accs = 0
    dataset = Toadstool(dataset_arr,grayscale,seq_len)
    for fold,(train_ids,test_ids) in enumerate(kf.split(dataset)):

        early_stopping_patience = 8

        batch_num = 16

        model = C3D(sample_size=112,
                    sample_duration=10,
                    num_classes=7)

        model.cuda()
        crnn_params = list(model.parameters())
        optimizer = torch.optim.Adam(
            crnn_params, lr=optim["lr_rate"], weight_decay=optim["weight_decay"])

        device = get_default_device()
        criterion = nn.CrossEntropyLoss() #weight=class_weights,reduction='mean'
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_num, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_num, sampler=test_subsampler)
        

        train_dl = DeviceDataLoader(trainloader, device)
        val_dl = DeviceDataLoader(testloader, device)
        best_acc = 0
        best_loss = 10000
        for epoch in range(1, optim["n_epochs"] + 1):
            train_loss, train_acc = train_epoch(
                model, train_dl, criterion, optimizer, device)

            val_loss, val_acc, acc_per_class = val_epoch(model, val_dl, criterion, device)
            
            if val_acc>best_acc: #any(i<val_acc for i in best_accs)
                best_acc = val_acc
                best_acc_per_class = acc_per_class
            if val_loss < best_loss:
                early_stopping_patience = 8
                best_loss = val_loss
            else:
                early_stopping_patience -= 1
            if early_stopping_patience == 0:
                print("Early stopping at epoch:" + str(epoch))
                print("Best accuracy: " + str(best_acc))
                print("Best accuracy per class" + str(best_acc_per_class))
                break
            print("Epoch nr "+str(epoch)+":\n      Training loss:" + str(train_loss) + ", Validation loss: " +
                str(val_loss) + "\n      Training acc:" + str(train_acc) + ", Validation acc: " + str(val_acc))

        best_accs += best_acc
    print("Average best accuracy:", str(best_accs/k))
    
if __name__ == '__main__':
    run_multimodal()