import numpy as np
import os
import random
import torch
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
from stft import Spectrogram, LogmelFilterBank
import multiprocessing as mp
import math
from tqdm import tqdm

sample_rate = 22050
window_size = 400
hop_size = 160
mel_bins = 64
fmin = 50
fmax = 14000
classes_num = 18
window = "hann"
center = True
pad_mode = "reflect"
ref = 1.0
amin = 1e-10
top_db = None

# Spectrogram extractor
spectrogram_extractor_openmic = Spectrogram(
    n_fft=window_size,
    hop_length=hop_size,
    win_length=window_size,
    window=window,
    center=center,
    pad_mode=pad_mode,
    freeze_parameters=True,
)

# Logmel feature extractor
logmel_extractor_openmic = LogmelFilterBank(
    sr=sample_rate,
    n_fft=window_size,
    n_mels=mel_bins,
    fmin=fmin,
    fmax=fmax,
    ref=ref,
    amin=amin,
    top_db=top_db,
    freeze_parameters=True,
)


def preprocess(x):
    # normalize raw numpy array then convert it to torch tensor
    x = x.astype(float)
    x /= np.abs(x).max()
    x = x.reshape(1, -1)
    x = torch.from_numpy(x).float()
    return x


def get_logmel(x):
    x = preprocess(x)
    x = spectrogram_extractor_openmic(x)  # (batch_size, 1, time_steps, freq_bins)
    x = logmel_extractor_openmic(x)

    return x


scatter_type = "9_8_132300_reduced"

scatter_path = "/home/laura/MedleyDB/processed/" + scatter_type
logmel_path = "/home/laura/MedleyDB/processed/logmel_reduced"

if not os.path.exists("/home/laura/MedleyDB/processed/logmel_reduced/input/"):
    os.mkdir("/home/laura/MedleyDB/processed/logmel_reduced/input/")

if not os.path.exists("/home/laura/MedleyDB/processed/logmel_reduced/labels/"):
    os.mkdir("/home/laura/MedleyDB/processed/logmel_reduced/labels/")


def f(files):
    save_path = os.path.join(logmel_path, "input")

    for file in tqdm(files):
        sample_name = file[:-4]

        start = int(sample_name.split("_")[2]) * 22050

        wav_name = (
            sample_name.split("_")[0] + "_" + sample_name.split("_")[1] + "_MIX.wav"
        )

        audio_path = os.path.join("/home/laura/MedleyDB/Audio/", wav_name)

        y, _ = librosa.load(audio_path, sr=22050, res_type="kaiser_fast")
        waveform = y[start : start + int(scatter_type.split("_")[2])]

        logmel = get_logmel(waveform)

        np.save(os.path.join(save_path, file), logmel)


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


if __name__ == "__main__":
    processes = []
    num_cpu = mp.cpu_count()

    files = os.listdir(os.path.join(scatter_path, "input/"))
    files = list(divide_chunks(files, math.ceil(len(files) / num_cpu)))

    for i in range(num_cpu):
        p = mp.Process(target=f, args=(files[i],))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
