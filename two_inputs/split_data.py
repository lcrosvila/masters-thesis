import os
import shutil

import numpy as np
from skmultilearn.dataset import load_dataset
from scipy.sparse import lil_matrix
import pandas as pd
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from skmultilearn.model_selection import iterative_train_test_split
from collections import Counter

scatter_type = "9_8_132300"

input_dir = "/home/laura/MedleyDB/processed/" + scatter_type + "/input"
files = os.listdir(input_dir)
labels_dir = "/home/laura/MedleyDB/processed/" + scatter_type + "/labels"

X = lil_matrix((len(files), 1))

int_to_file = {}
i = 0

matrix_labels = []

for file in files:
    int_to_file[i] = file
    X[i] = i

    label = np.load(os.path.join(labels_dir, file), allow_pickle=True)
    matrix_labels.append(label)

    i = i + 1

y = lil_matrix(matrix_labels)

X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)
X_train, y_train, X_val, y_val = iterative_train_test_split(
    X_train, y_train, test_size=0.2
)

train_files = []
train_dir = "/home/laura/MedleyDB/processed/" + scatter_type + "/train/"

for i in range(X_train.shape[0]):
    file = int_to_file[int(X_train[i, 0])]
    train_files.append(file)
    shutil.copy2(
        os.path.join(input_dir, file), os.path.join(train_dir + "input", file),
    )
    shutil.copy2(
        os.path.join(labels_dir, file), os.path.join(train_dir + "labels", file),
    )

val_files = []
val_dir = "/home/laura/MedleyDB/processed/" + scatter_type + "/val/"

for i in range(X_val.shape[0]):
    file = int_to_file[int(X_val[i, 0])]
    val_files.append(file)
    shutil.copy2(
        os.path.join(input_dir, file), os.path.join(val_dir + "input", file),
    )
    shutil.copy2(
        os.path.join(labels_dir, file), os.path.join(val_dir + "labels", file),
    )

test_files = []
test_dir = "/home/laura/MedleyDB/processed/" + scatter_type + "/test/"

for i in range(X_test.shape[0]):
    file = int_to_file[int(X_test[i, 0])]
    test_files.append(file)
    shutil.copy2(
        os.path.join(input_dir, file), os.path.join(test_dir + "input", file),
    )
    shutil.copy2(
        os.path.join(labels_dir, file), os.path.join(test_dir + "labels", file),
    )

np.savez(
    "/home/laura/MedleyDB/processed/" + scatter_type + "/data_split.npz",
    train=train_files,
    val=val_files,
    test=test_files,
)

