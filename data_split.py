import os
import sys
import numpy as np
import random

def split_dataset(split_factor=0.8):
    directory = '/home/laura/Essid/Essid'
    subdir = [x[0] for x in os.walk(directory)]

    instrument = 0
    train_labels = []
    train_paths = []
    train_sample_key = []

    test_labels = []
    test_paths = []
    test_sample_key = []

    for dir in subdir:
        if dir != directory:
            label = np.zeros(len(subdir)-1)
            label[instrument] = 1
            files =  [f for f in os.listdir(dir) if f.endswith('.wav')]

            split = round(len(files)*split_factor)
            random.shuffle(files)

            train = files[:split]
            test = files[split:]

            for file in train:
                train_labels.append(label)
                train_sample_key.append(file)
                train_paths.append(os.path.join(dir,file))

            for file in test:
                test_labels.append(label)
                test_sample_key.append(file)
                test_paths.append(os.path.join(dir,file))

            instrument += 1

    
    print(np.unique(np.argmax(train_labels, axis=1)))
    print(np.unique(np.argmax(test_labels, axis=1)))

    np.savez('/home/laura/monophonic/data/train.npz', sample_key=np.asarray(train_sample_key), path=np.asarray(train_paths), Y=np.asarray(train_labels))
    np.savez('/home/laura/monophonic/data/test.npz', sample_key=np.asarray(test_sample_key), path=np.asarray(test_paths), Y=np.asarray(test_labels))