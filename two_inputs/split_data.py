import os
import shutil

import random
import torch
import librosa
import numpy as np
from skmultilearn.dataset import load_dataset
from scipy.sparse import lil_matrix
import pandas as pd
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from skmultilearn.model_selection import iterative_train_test_split
from collections import Counter
import multiprocessing as mp
from tqdm import tqdm


def songwise_split(split=0.8, random_seed=None):
    random.seed(random_seed)

    audio_paths = os.listdir("/home/laura/MedleyDB/Audio")
    l = random.sample(range(1, 100), 10)

    songs = [name.split("_MIX.wav")[0] for name in audio_paths]

    indices = [i for i in range(len(songs))]
    indices = random.sample(indices, len(indices))

    train_length = round(split * len(indices))
    val_length = round((len(indices) - train_length) / 2)
    test_length = len(indices) - val_length - train_length

    train_songs = [songs[idx] for idx in indices[:train_length]]
    val_songs = [
        songs[idx] for idx in indices[train_length : train_length + val_length]
    ]
    test_songs = [songs[idx] for idx in indices[-test_length:]]

    return train_songs, val_songs, test_songs


def get_split_samples(scatter_type="9_8_132300", split=0.8, random_seed=None):
    train_songs, val_songs, test_songs = songwise_split(
        split=split, random_seed=random_seed
    )
    source_input = "/home/laura/MedleyDB/processed/" + scatter_type + "/input/"

    input_files = os.listdir(source_input)

    train_files = []
    val_files = []
    test_files = []

    for file_name in input_files:
        song_name = file_name.split("_")[0] + "_" + file_name.split("_")[1]

        if song_name in train_songs:
            train_files.append(file_name)
        elif song_name in val_songs:
            val_files.append(file_name)
        elif song_name in test_songs:
            test_files.append(file_name)

    return train_files, val_files, test_files


def is_balanced(
    train_files, val_files, test_files, high_limit=30, scatter_type="9_8_132300"
):
    labels_dir = "/home/laura/MedleyDB/processed/" + scatter_type + "/labels"

    matrix_labels = []

    for file in train_files:
        label = np.load(os.path.join(labels_dir, file), allow_pickle=True)
        matrix_labels.append(label)

    y_train = lil_matrix(matrix_labels)

    matrix_labels = []

    for file in val_files:
        label = np.load(os.path.join(labels_dir, file), allow_pickle=True)
        matrix_labels.append(label)

    y_val = lil_matrix(matrix_labels)

    matrix_labels = []

    for file in test_files:
        label = np.load(os.path.join(labels_dir, file), allow_pickle=True)
        matrix_labels.append(label)

    y_test = lil_matrix(matrix_labels)

    table = pd.DataFrame(
        {
            "train": Counter(
                str(combination)
                for row in get_combination_wise_output_matrix(y_train.A, order=2)
                for combination in row
            ),
            "validation": Counter(
                str(combination)
                for row in get_combination_wise_output_matrix(y_val.A, order=2)
                for combination in row
            ),
            "test": Counter(
                str(combination)
                for row in get_combination_wise_output_matrix(y_test.A, order=2)
                for combination in row
            ),
        }
    ).T.fillna(0.0)

    balanced = True
    count_val_0 = 0
    count_test_0 = 0

    for key in table.keys():
        if table[key]["train"] == 0.0 and (
            table[key]["validation"] != 0.0 or table[key]["test"] != 0.0
        ):
            balanced = False
            break

    for key in table.keys():
        if table[key]["validation"] == 0.0:
            count_val_0 += 1
        if table[key]["test"] == 0.0:
            count_test_0 += 1
        if count_val_0 > high_limit or count_test_0 > high_limit:
            balanced = False
            break

    if balanced:
        print("count val 0", count_val_0)
        print("count test 0", count_test_0)

    return balanced


def split_scikit(scatter_type="logmel"):
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

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if not os.path.exists(os.path.join(train_dir, "input")):
        os.mkdir(os.path.join(train_dir, "input"))

    if not os.path.exists(os.path.join(train_dir, "labels")):
        os.mkdir(os.path.join(train_dir, "labels"))

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

    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    if not os.path.exists(os.path.join(val_dir, "input")):
        os.mkdir(os.path.join(val_dir, "input"))

    if not os.path.exists(os.path.join(val_dir, "labels")):
        os.mkdir(os.path.join(val_dir, "labels"))

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

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    if not os.path.exists(os.path.join(test_dir, "input")):
        os.mkdir(os.path.join(test_dir, "input"))

    if not os.path.exists(os.path.join(test_dir, "labels")):
        os.mkdir(os.path.join(test_dir, "labels"))

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


def f(random_seeds, high_limits):
    scatter_type = "9_8_132300"
    balanced = False

    for high_limit in tqdm(high_limits, unit="lim"):
        if os.path.exists("/home/laura/MedleyDB/processed/" + scatter_type + "/data_split.npz"):
            break

        print("high_limit", high_limit)
        for random_seed in tqdm(random_seeds, unit="seed"):
            train_files, val_files, test_files = get_split_samples(
                scatter_type=scatter_type, random_seed=random_seed
            )
            balanced = is_balanced(
                train_files,
                val_files,
                test_files,
                high_limit=high_limit,
                scatter_type=scatter_type,
            )

            if balanced:
                print("balanced!")
                break
        if balanced:
            break

    print(random_seed)
    print(high_limit)
    np.savez(
        "/home/laura/MedleyDB/processed/" + scatter_type + "/data_split.npz",
        high_limit=high_limit,
        seed=random_seed - 1,
        train=train_files,
        val=val_files,
        test=test_files,
    )


if __name__ == "__main__":
    processes = []

    high_limits = [i for i in range(45, 60)]

    for i in range(mp.cpu_count()):
        random_seeds = [i for i in range(625 * i, 625 * (i + 1))]
        p = mp.Process(target=f, args=(random_seeds, high_limits,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
