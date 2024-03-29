import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score


def macro_f1(batches_target, batches_pred):
    macro_f1 = f1_score(batches_target, batches_pred, average="macro")
    return macro_f1


def instrument_f1(batches_target, batches_pred):
    instrument_f1 = []
    for i in range(len(batches_target[0])):
        target_instr = [sample[i] for sample in batches_target]
        pred_instr = [sample[i] for sample in batches_pred]
        instrument_f1.append(f1_score(target_instr, pred_instr))

    return instrument_f1


def instrument_precision(batches_target, batches_pred):
    instrument_precision = []
    for i in range(len(batches_target[0])):
        target_instr = [sample[i] for sample in batches_target]
        pred_instr = [sample[i] for sample in batches_pred]
        instrument_precision.append(precision_score(target_instr, pred_instr))

    return instrument_precision


def instrument_recall(batches_target, batches_pred):
    instrument_recall = []
    for i in range(len(batches_target[0])):
        target_instr = [sample[i] for sample in batches_target]
        pred_instr = [sample[i] for sample in batches_pred]
        instrument_recall.append(recall_score(target_instr, pred_instr))

    return instrument_recall


def auc_score():
    return None
