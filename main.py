import os
import sys
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torchvision
from sklearn import metrics
import _pickle as cPickle
import shutil

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import Cnn14, Cnn14_scatter
from data_generator import monoDataset

from losses import get_loss_func

def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory,'model_best.pth.tar'))
        
DATA_ROOT = '/home/laura/monophonic/data'

train_npz_path=DATA_ROOT+'/train.npz'
val_npz_path=DATA_ROOT+'/validation.npz'

train_dataset = monoDataset(npz_path=train_npz_path, audio_length=220500, classes_num=7)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    num_workers=0,
    shuffle=True
)

val_dataset = monoDataset(npz_path=val_npz_path, audio_length=220500, classes_num=7)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=16,
    num_workers=0,
    shuffle=True
)

# model = Cnn14(sample_rate=44100, window_size=400, hop_size=160, mel_bins=64, fmin=50, fmax=14000, classes_num=7)
J = 6
Q = 8

model = Cnn14_scatter(classes_num=7, J=J, Q=Q, audio_length = 110250)

model.cuda()

loss_func = get_loss_func('clip_bce')
optimizer = optim.Adam(model.parameters(), lr=1e-3, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

max_epoch = 10
best_acc = 0

logs = {}
logs['J'] = J
logs['Q'] = Q

num_try = '1_scatter_fixed'

for epoch in range(max_epoch):
    running_loss = 0.0
    running_corrects = 0
    
    model.train()
    for batch_data_dict in train_loader:
        
        batch_input = batch_data_dict['waveform'].cuda()
        batch_target = batch_data_dict['target'].cuda()

        batch_output_dict = model(batch_input)
        batch_target_dict = {'target': batch_target}
        
        _, preds = torch.max(batch_output_dict['clipwise_output'], 1)
        _, labels = torch.max(batch_target, 1)

        loss = loss_func(batch_output_dict, batch_target_dict)

        # Backward
        loss.backward()
        running_loss += loss.item() * train_loader.batch_size
        
        optimizer.step()
        optimizer.zero_grad()
    
    model.eval()
    for batch_data_dict in val_loader:
        
        batch_input = batch_data_dict['waveform'].cuda()
        batch_target = batch_data_dict['target'].cuda()

        batch_output_dict = model(batch_input)
        batch_target_dict = {'target': batch_target}
        
        _, preds = torch.max(batch_output_dict['clipwise_output'], 1)
        _, labels = torch.max(batch_target, 1)
        running_corrects += torch.sum(preds == labels)
    
    epoch_loss = running_loss / train_dataset.__len__()
    epoch_acc = running_corrects.float() / train_dataset.__len__()
    logs[epoch] = [epoch_loss, epoch_acc.item()]
    
    is_best = epoch_acc > best_acc
    best_acc = max(epoch_acc, best_acc)

    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, num_try)

    print(logs)

print('Best accuracy', best_acc.item())
np.save(os.path.join(num_try,'logs_'+num_try), logs)