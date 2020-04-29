import os
import sys

import multiprocessing as mp
import time
import pylab as pl
from IPython import display

import numpy as np
import argparse
import h5py
import math
import time
import logging
import torchvision
from sklearn import metrics
import _pickle as cPickle
import shutil

from tqdm import tqdm
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import CNN_two, Cnn6
from data_generator import (
    medleyDataset,
    medleyDataset_logmel,
    medleyDataset_audio,
    get_mean_var,
)

from losses import get_loss_func
from metrics import macro_f1, instrument_f1


def save_checkpoint(state, is_best, directory, filename="checkpoint.pth.tar"):
    if not directory == None:
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(directory, "model_best.pth.tar"))


def evaluate(model, loader, scatter_type, loss_func, threshold):
    total_loss = 0.0
    batches_target = []
    batches_pred = []

    for batch_data_dict in loader:
        if "logmel" in scatter_type.split("_"):
            batch_input = batch_data_dict["logmel"].cuda()
            batch_target = batch_data_dict["target"].cuda()
            batch_output_dict = model(batch_input)
        else:
            batch_input1 = batch_data_dict["order1"].cuda()
            batch_input2 = batch_data_dict["order2"].cuda()
            batch_target = batch_data_dict["target"].cuda()
            batch_output_dict = model(batch_input1, batch_input2)

        batch_target_dict = {"target": batch_target}

        loss = loss_func(
            batch_output_dict["clipwise_output"], batch_target_dict["target"]
        )

        total_loss += loss.item() * loader.batch_size

        batches_target.append(batch_target_dict["target"].detach().cpu().numpy()[0])
        batches_pred.append(
            (batch_output_dict["clipwise_output"].detach().cpu().numpy()[0] > threshold)
            * 1.0
        )

    total_loss = total_loss / len(loader.dataset)

    return total_loss, batches_target, batches_pred


def f(split):
    print("Current split: ", split)
    standardize = True
    scatter_type = str(sys.argv[2])

    train_dir = "/home/laura/MedleyDB/processed/" + scatter_type + "/train"
    val_dir = "/home/laura/MedleyDB/processed/" + scatter_type + "/val"
    test_dir = "/home/laura/MedleyDB/processed/" + scatter_type + "/test"

    if "reduced" in scatter_type.split("_"):
        print("5 instruments")
        classes_num = 5
    else:
        print("16 instruments")
        classes_num = 16

    if "logmel" in scatter_type.split("_"):
        print("Logmel data")
        if standardize:
            print("Standardization")
            mean_logmel, var_logmel = get_mean_var(train_dir, split)
        else:
            print("No standardization")
            mean_logmel, var_logmel = [None, None]

        time_steps = 827
        freq_bins = 64

        train_dataset = medleyDataset_logmel(
            train_dir,
            time_steps,
            freq_bins,
            classes_num,
            mean_logmel,
            var_logmel,
            split,
        )
        val_dataset = medleyDataset_logmel(
            val_dir, time_steps, freq_bins, classes_num, mean_logmel, var_logmel, 1,
        )
        test_dataset = medleyDataset_logmel(
            test_dir, time_steps, freq_bins, classes_num, mean_logmel, var_logmel, 1,
        )

    else:
        print("Scatter data")
        if "9_8_132300" in scatter_type.split("_reduced"):
            input_length = 259
            order1_length = 62
            order2_length = 237
        elif "6_8_33075" in scatter_type.split("_reduced"):
            input_length = 517
            order1_length = 38
            order2_length = 87

        if standardize:
            print("Standardization")
            mean_order1, var_order1, mean_order2, var_order2 = get_mean_var(
                train_dir, split
            )
        else:
            print("No standardization")
            mean_order1, var_order1, mean_order2, var_order2 = [
                None,
                None,
                None,
                None,
            ]

        train_dataset = medleyDataset(
            train_dir,
            input_length,
            order1_length,
            order2_length,
            classes_num,
            mean_order1,
            var_order1,
            mean_order2,
            var_order2,
            split,
        )
        val_dataset = medleyDataset(
            val_dir,
            input_length,
            order1_length,
            order2_length,
            classes_num,
            mean_order1,
            var_order1,
            mean_order2,
            var_order2,
            1,
        )
        test_dataset = medleyDataset(
            test_dir,
            input_length,
            order1_length,
            order2_length,
            classes_num,
            mean_order1,
            var_order1,
            mean_order2,
            var_order2,
            1,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=os.cpu_count(),
        shuffle=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=os.cpu_count(), shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=os.cpu_count(), shuffle=True
    )

    if "logmel" in scatter_type.split("_"):
        print("Creating Cnn6 model")
        model = Cnn6(
            classes_num=classes_num,
            time_steps=time_steps,
            freq_bins=freq_bins,
            spec_aug=False,
        )
    else:
        print("Creating CNN_two model")
        model = CNN_two(
            classes_num=classes_num,
            input_length=input_length,
            order1_length=order1_length,
            order2_length=order2_length,
        )

    model.cuda()

    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    losses_train = []
    losses_val = []
    f1_val = []
    epoch = 0
    weight_updates = 0
    threshold = 0.5

    directory_save = (
        "/home/laura/thesis/two_inputs/models/"
        + str(sys.argv[1])
        + "/"
        + scatter_type
        + "/"
        + str(round(split * 100))
    )

    max_epoch = 100

    early_stop = False
    plot = False
    dropout = True

    while not early_stop:
        running_loss = 0.0
        is_best = False

        for batch_data_dict in train_loader:
            if "logmel" in scatter_type.split("_"):
                batch_input = batch_data_dict["logmel"].cuda()
                batch_target = batch_data_dict["target"].cuda()
                batch_output_dict = model(batch_input, dropout)
            else:
                batch_input1 = batch_data_dict["order1"].cuda()
                batch_input2 = batch_data_dict["order2"].cuda()
                batch_target = batch_data_dict["target"].cuda()
                batch_output_dict = model(batch_input1, batch_input2, plot, dropout)

            batch_target_dict = {"target": batch_target}

            loss = loss_func(
                batch_output_dict["clipwise_output"], batch_target_dict["target"]
            )

            # Backward
            loss.backward()
            running_loss += loss.item() * train_loader.batch_size

            optimizer.step()
            weight_updates += 1
            optimizer.zero_grad()

        epoch_loss = running_loss / len(train_dataset)
        losses_train.append(epoch_loss)

        plot = False

        model.eval()

        val_loss, batches_target, batches_pred = evaluate(
            model, val_loader, scatter_type, loss_func, threshold
        )

        losses_val.append(val_loss)

        if epoch > max_epoch and losses_val[-1] > min(losses_val):
            early_stop = True

        f1_score = macro_f1(batches_target, batches_pred)
        f1_val.append(f1_score)

        if min(losses_val) == val_loss:
            f1_instr_val = instrument_f1(batches_target, batches_pred)
            losses_test, batches_target_test, batches_pred_test = evaluate(
                model, test_loader, scatter_type, loss_func, threshold
            )
            f1_instr_test = instrument_f1(batches_target_test, batches_pred_test)
            is_best = True

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "weight_updates": weight_updates,
                "state_dict": model.state_dict(),
                "train_losses": losses_train,
                "val_losses": losses_val,
                "test_loss": losses_test,
                "f1_instr_val": f1_instr_val,
                "f1_instr_test": f1_instr_test,
                "macro_f1_val": f1_val[np.argmin(losses_val)],
                "split": split,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            directory_save,
        )

        model.train()

        epoch += 1


if __name__ == "__main__":
    splits = []
    if len(sys.argv) > 3:
        for i in range(3, len(sys.argv)):
            splits.append(float(sys.argv[i]))

    else:
        splits = [(i + 1) / 10 for i in range(10)]

    print("Splits: ", splits)
    for split in splits:
        f(split)
