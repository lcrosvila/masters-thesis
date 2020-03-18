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
from data_generator import medleyDataset

from losses import get_loss_func


def save_checkpoint(state, is_best, directory, filename="checkpoint.pth.tar"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, "model_best.pth.tar"))


DATA_ROOT = "/home/laura/MedleyDB/processed/debug"

debug_dataset = medleyDataset(DATA_ROOT, 517, 38, 87, 18)
debug_loader = torch.utils.data.DataLoader(
    debug_dataset, batch_size=1, num_workers=0, shuffle=True
)

model = CNN_two(classes_num=18, input_length=517)

model.cuda()

loss_func = get_loss_func("clip_bce")
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0,
    amsgrad=True,
)

max_epoch = 3
# best_acc = 0

# logs = {}
# logs['J'] = J
# logs['Q'] = Q

# num_try = '1_scatter_fixed'

for epoch in range(max_epoch):
    running_loss = 0.0
    # running_corrects = 0

    model.train()
    for batch_data_dict in train_loader:

        batch_input1 = batch_data_dict["order1"].cuda()
        batch_input2 = batch_data_dict["order2"].cuda()
        batch_target = batch_data_dict["target"].cuda()

        batch_output_dict = model(batch_input1, batch_input2)
        batch_target_dict = {"target": batch_target}

        _, preds = torch.max(batch_output_dict["clipwise_output"], 1)
        _, labels = torch.max(batch_target, 1)

        loss = loss_func(batch_output_dict, batch_target_dict)

        # Backward
        loss.backward()
        running_loss += loss.item() * train_loader.batch_size

        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / train_dataset.__len__()
    print("Epoch loss: " + str(epoch_loss))
