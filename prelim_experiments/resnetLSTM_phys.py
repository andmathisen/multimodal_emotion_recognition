import torch
import torch.nn as nn
import math
from models import *
from torchvision import transforms
import numpy as np
import torch
import copy
from sklearn.model_selection import KFold 
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
from helpers import *
#transforms.Grayscale(num_output_channels=1),
train_trfm = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
val_trfm = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])



def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()





def train_epoch(model, data_loader, criterion, optimizer, epoch, device, log_interval):
    model.train()
    train_loss = 0.0
    losses = 0
    tot_datapoints = 0
    for batch_idx, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        targets = targets.to(torch.float32)

        loss = criterion(outputs.view(-1), targets)

        train_loss += loss
        losses += loss
        tot_datapoints += data.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(data_loader), 100. * (batch_idx + 1) / len(data_loader), avg_loss))
            train_loss = 0.0

    print('Train set ({:d} samples): Average loss: {:.4f}'.format(
        len(data_loader), losses/tot_datapoints))

    return losses/tot_datapoints,model


def val_epoch(model, data_loader, criterion, device):
    model.eval()

    losses = 0
    tot_batches = 0
    tot_datapoints = 0
    running_error = 0
    running_squared_error = 0
    with torch.no_grad():
        for (data, targets) in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            outputs = outputs.view(-1).cpu()
            targets = targets.to(torch.float32).cpu()
            
            running_error += mean_absolute_error(targets,outputs)
            running_squared_error += mean_squared_error(targets,outputs)
            losses += criterion(outputs, targets)
            tot_batches += 1
            tot_datapoints += data.size(0)
    
    mse = math.sqrt(running_squared_error/tot_batches)
    mae = running_error/tot_batches
    return losses/tot_datapoints,mse,mae

    
def main():
    params_model = {
        "num_classes": 1,
        "dr_rate": 0.5,
        "pretrained": True,
        "rnn_hidden_size": 256,
        "rnn_num_layers": 2,
        "features":"none"
    }

    optim = {
        "lr_rate": 0.0001,
        "weight_decay": 0.00001,
        "log_interval": 50,
        "n_epochs": 50
    }

    
    seq_len = 10
    step_size = 3
    random_seed = 30
    k = 3
    kf = KFold(n_splits=k,shuffle=True,random_state=None)
    print(kf)
    torch.manual_seed(random_seed)
    # how many epochs to wait before stopping when loss is not improving
    early_stopping_patience = 5
    # minimum difference between new loss and old loss for new loss to be considered as an improvement
    early_stopping_delta = 0
    batch_num = 1
    criterion = nn.MSELoss()
    device = get_default_device()
    
    for nr in range(10):
           
        print('--------------------------------')
        participant = "participant_"+str(nr)
        print(participant+":")
        frames = np.load(participant+'_3frames_histmatched'+'.npy', allow_pickle=True)
        df_BVP = pd.read_csv(participant+"/BVP_sync_video.csv")
        transformed_df_BVP = transform_bvp_amplitudes(df_BVP, "BVP")


        seq_arr = create_face_bvp_arr(frames,transformed_df_BVP,seq_len,step_size)
        

        dataset = Toadstool_LSTM(seq_arr, train_trfm, seq_len)
        
        rnn_rmse_tot = 0
        rnn_mae_tot = 0
        zeroR_rmse_tot = 0
        zeroR_mae_tot = 0
        for fold,(train_ids,test_ids) in enumerate(kf.split(dataset)):
            print(f'FOLD {fold}')
            print('--------------------------------')
            model = ResNet_LSTM(params_model)
            model.apply(reset_weights)
            model.cuda()
            crnn_params = list(model.parameters())
            optimizer = torch.optim.Adam(
                crnn_params, lr=optim["lr_rate"], weight_decay=optim["weight_decay"])

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            trainloader = torch.utils.data.DataLoader(
                          dataset, 
                          batch_size=30, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=30, sampler=test_subsampler)
            zeroR_rmse, zeroR_mae = zero_rule_algorithm_classification(trainloader,testloader)
            print("Zero rule classification root mean squared error: "+str(zeroR_rmse))
            print("Zero rule classification mean absolute error: "+str(zeroR_mae))
            zeroR_rmse_tot += zeroR_rmse
            zeroR_mae_tot += zeroR_mae
            train_dl = DeviceDataLoader(trainloader, device)
            test_dl = DeviceDataLoader(testloader, device)
            best_loss = None
            early_stop_counter = 0
            for epoch in range(1, optim["n_epochs"] + 1):
                train_loss,model = train_epoch(
                    model, train_dl, criterion, optimizer, epoch, device, optim["log_interval"])
                if best_loss == None or train_loss <  best_loss:
                    best_loss = train_loss
                    early_stop_counter = 0
                    bestmodel = copy.deepcopy(model)
                elif train_loss>best_loss:
                    early_stop_counter += 1
                if early_stop_counter == early_stopping_patience:
                    print("Early stopping at epoch: "+str(epoch))
                    break
            
            val_loss,rmse,mae = val_epoch(bestmodel,test_dl,criterion,device)
            print("Root mean squared error: "+ str(rmse))
            print("Mean absolute error: "+ str(mae))
            rnn_rmse_tot += rmse
            rnn_mae_tot += mae.item()

        with open("results_resnetLSTM_phys.txt","a") as f:
            f.write("----------------------\n")
            f.write(participant + "\n")
            f.write(" Zero R: \n")
            f.write("   RMSE: " + str(zeroR_rmse_tot/k) + "\n")
            f.write("   MAE: " + str(zeroR_mae_tot/k) + "\n")
            f.write(" LSTM: \n")
            f.write("   RMSE: " + str(rnn_rmse_tot/k) + "\n")
            f.write("   MAE: " + str(rnn_mae_tot/k) + "\n")

if __name__ == "__main__":
    main()
