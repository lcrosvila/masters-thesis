import os
import sys
import numpy as np
import random

train = np.load('data/train.npz', allow_pickle=True)
indices = np.arange(len(train['sample_key']))

random.shuffle(indices)

split = round(len(indices)*0.9)

train_sample_key = train['sample_key'][indices[:split]]
val_sample_key = train['sample_key'][indices[split:]]

train_paths = train['path'][indices[:split]]
val_paths = train['path'][indices[split:]]

train_Y = train['Y'][indices[:split]]
val_Y = train['Y'][indices[split:]]

print(np.unique(np.argmax(train_Y, axis=1)))
print(np.unique(np.argmax(val_Y, axis=1)))

np.savez('train.npz', sample_key=np.asarray(train_sample_key), path=np.asarray(train_paths), Y=np.asarray(train_Y))
np.savez('validation.npz', sample_key=np.asarray(val_sample_key), path=np.asarray(val_paths), Y=np.asarray(val_Y))