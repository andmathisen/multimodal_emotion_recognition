
import torch
import torch.nn as nn
from models import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import torch
import copy
import matplotlib.pyplot as plt
from helpers import *
train_trfm = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
val_trfm = transforms.Compose(
    [

        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

val_trfm2 = transforms.Compose(
    [

        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ])



grayscale = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])



flip = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.75)
    ])

rotate = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomRotation((-20,20))
    ])


def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * \
        (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new



def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    for (inputs, targets) in data_loader:

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        prec1 = calculate_accuracy(outputs.data, targets.data, topk=(1,))

        top1.update(prec1[0].item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg



def val_epoch(model, data_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        confusion_matrix = torch.zeros(7, 7)
        for (inputs, targets) in data_loader:

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            prec1 = calculate_accuracy(outputs.data, targets.data, topk=(1,))
            top1.update(prec1[0].item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))
            _, preds = torch.max(outputs, 1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)
    
    return losses.avg, top1.avg, acc_per_class.tolist()

def get_video_values(data,grayscale):
    all = []
    for dp in data:
        for img in dp[0]:
            all.append(grayscale(img))

    mean = torch.mean(torch.stack(all,0))
    std = torch.std(torch.stack(all,0))

    print(mean,std)


def main():
    

    optim = {
        "lr_rate": 0.0001,
        "weight_decay": 0.00001,
        "n_epochs": 150
    }

    seq_len = 10

    device = get_default_device()
    early_stopping_patience = 8
    # minimum difference between new loss and old loss for new loss to be considered as an improvement
    early_stopping_delta = 0
    batch_num = 25

    model = C3D(sample_size=112,
                 sample_duration=10,
                 num_classes=7)

    to_device(model,device)
    crnn_params = list(model.parameters())
    optimizer = torch.optim.Adam(
        crnn_params, lr=optim["lr_rate"], weight_decay=optim["weight_decay"])


    rootdir = "CK-Dataset/CK_seq"
    if not os.path.exists('ck_data.npy'):
        dataset_arr = create_ck_arr(rootdir, seq_len, rotate, flip)
        dataset_arr = np.asarray(dataset_arr)
        np.save('ck_data.npy', dataset_arr, allow_pickle=True)
    else:
        dataset_arr = np.load('ck_data.npy', allow_pickle=True)

 


    
    criterion = nn.CrossEntropyLoss() 
    train_arr, val_arr = train_test_split(
            dataset_arr, shuffle=True, test_size=0.15)
    
    train_set = CKDataset(train_arr, train_trfm, seq_len)
    val_set = CKDataset(val_arr, val_trfm, seq_len)

    train_dl = DataLoader(train_set, batch_num,
                          shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_set, batch_num, num_workers=4, pin_memory=True)


    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    best_acc = 0
    best_loss = 10000
    train_losses = []
    val_losses = []
    best_accs = [0,0,0]
    for epoch in range(1, optim["n_epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, train_dl, criterion, optimizer, device)

        val_loss, val_acc, acc_per_class = val_epoch(model, val_dl, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if any(i<val_acc for i in best_accs): #any(i<val_acc for i in best_accs)
            best_acc = val_acc
            w = copy.deepcopy(model.state_dict())
            for i in range(3):
                if val_acc>best_accs[i]:
                    best_accs[i] = val_acc
                    best_accs.sort(reverse=True)
                    torch.save(w, "trained_model_weights/C3D_224_Adam_weights"+str(i)+".pt")
                    break
            model_acc_per_class = acc_per_class
        if val_loss < best_loss:
            early_stopping_patience = 8
            best_loss = val_loss
        else:
            early_stopping_patience -= 1
        if early_stopping_patience == 0:
            print("Early stopping at epoch:" + str(epoch))
            print("Best accuracy: " + str(best_acc))
            break
        print("Epoch nr "+str(epoch)+":\n      Training loss:" + str(train_loss) + ", Validation loss: " +
              str(val_loss) + "\n      Training acc:" + str(train_acc) + ", Validation acc: " + str(val_acc))
    
    print(model_acc_per_class)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("loss.png")

main()
