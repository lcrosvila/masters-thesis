import numpy as np
import os
import random
import torch
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
from stft import Spectrogram, LogmelFilterBank
from multiprocessing import Process

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


scatter_type = "9_8_132300"

scatter_path = "/home/laura/MedleyDB/processed/" + scatter_type
spec_path = "/home/laura/MedleyDB/processed/spec"


def f(directories):
    for split in directories:
        processed_path = os.path.join(scatter_path, split)
        save_path = os.path.join(spec_path, split + "/input")

        files = os.listdir(os.path.join(processed_path, "input/"))
        for file in files:
            sample_name = file[:-4]

            start = int(sample_name.split("_")[2]) * 22050

            wav_name = (
                sample_name.split("_")[0] + "_" + sample_name.split("_")[1] + "_MIX.wav"
            )

            audio_path = os.path.join("/home/laura/MedleyDB/Audio/", wav_name)
            target_path = os.path.join(processed_path, "labels/" + file)

            target = np.load(target_path, allow_pickle=True)
            target = target.astype("float32")

            y, _ = librosa.load(audio_path, sr=22050, res_type="kaiser_fast")
            waveform = y[start : start + 6 * 22050]

            logmel = get_logmel(waveform)

            print(file)
            np.save(os.path.join(save_path, file), logmel)


if __name__ == "__main__":
    directories = ["train", "val", "test"]
    p = Process(target=f, args=(directories,))
    p.start()
    p.join()
