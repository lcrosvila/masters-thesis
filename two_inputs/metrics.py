import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


def macro_f1(batches_target, batches_pred):
    macro_f1 = f1_score(batches_target, batches_pred, average="macro")
    print("Macro F1", macro_f1)
    return macro_f1

def auc_score():
    return None