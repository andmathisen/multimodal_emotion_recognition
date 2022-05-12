
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
import os
from models import *
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from helpers import *
# asisgining the classes to numbers of the classes
classes = {
        'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5, 'neutral':6
}


train_trfm = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((48,48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(48, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5), inplace=True)
    ])
val_trfm = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((48,48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5), inplace=True)

    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.5), (0.5))

    ]),
}



def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item()/len(preds))



def evaluate(model, val_loader):
    # This function will evaluate the model and give back the val acc and loss
    model.eval()
    with torch.no_grad():
        outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# getting the current learning rate

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']




def fit(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []  # keep track of the evaluation results

    # setting upcustom optimizer including weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # setting up 1cycle lr scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    best_epoch = 0
    for epoch in range(epochs):
        # training
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # record the lr
            lrs.append(get_lr(optimizer))
            sched.step()

        # validation
        result = evaluate(model, val_loader)
        acc = result['val_acc']
        if acc > best_epoch:
            best_epoch = acc
            print("new best epoch:", best_epoch)
            bestweights = model.state_dict()
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history,bestweights

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    return best_model_wts


if __name__ == "__main__":
    rootdir_train = "CK-Dataset/train_test_224/train"
    rootdir_val = "CK-Dataset/train_test_224/test"
   
    train_data = CKDataset_2D(rootdir_train, train_trfm)
    val_data = CKDataset_2D(rootdir_val,val_trfm)

    random_seed = 30
    torch.manual_seed(random_seed)

    batch_num = 8
    
    train_dl = DataLoader(train_data, batch_num,
                          shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_data, batch_num, num_workers=4, pin_memory=True)
    

    device = get_default_device()

    
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    dataloaders = {"train":train_dl,"val":val_dl}
    dataset_sizes = {"train":len(train_data),"val":len(val_data)}
   
    model = ResNet(BasicBlock,[2,2,2,2],7)
    
    to_device(model,device)
   
    max_lr = 0.001
    grad_clip = 0.05
    weight_decay = 1e-6

    history,model_state = fit(30, max_lr, model, train_loader=train_dl, val_loader=val_dl, weight_decay=weight_decay,
                  grad_clip=grad_clip)


    torch.save(model_state, "model_parameter_resCK.pt")
    
