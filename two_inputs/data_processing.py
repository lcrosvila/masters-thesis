import concurrent
import os

import IPython.display as ipd
import librosa
import librosa.display
from librosa.effects import split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kymatio import Scattering1D
from tqdm import tqdm
import multiprocessing as mp
import math

# Based off of OpenMICs taxonomy discussions and the MedleyDB taxonomy yaml
OPENMIC_TO_MEDLEY = {
    "drums": ["drum set"],
    "bass": ["electric bass", "double bass"],
    "guitar": ["distorted electric guitar", "clean electric guitar", "acoustic guitar"],
    "voice": [
        "male singer",
        "female singer",
        "male speaker",
        "female speaker",
        "male rapper",
        "female rapper",
        "beatboxing",
        "vocalists",
        "choir",
        "male screamer",
        "female screamer",
    ],
    "piano": ["piano", "tack piano", "electric piano"],
    # "synthesizer": ["synthesizer"],
    # "cello": ["cello", "cello section"],
    # "clarinet": ["clarinet", "clarinet section", "bass clarinet"],
    # "cymbals": ["cymbal"],
    # "flute": [
    #     "flute",
    #     "dizi",
    #     "flute",
    #     "flute section",
    #     "piccolo",
    #     "bamboo flute",
    #     "panpipes",
    #     "recorder",
    # ],
    # "mallet_percussion": ["xylophone", "vibraphone", "glockenspiel", "marimba"],
    # "mandolin": ["mandolin"],
    # "saxophone": [
    #     "alto saxophone",
    #     "baritone saxophone",
    #     "tenor saxophone",
    #     "soprano saxophone",
    # ],
    # "trombone": ["trombone", "trombone section"],
    # "trumpet": ["trumpet", "trumpet section"],
    # "violin": ["violin", "violin seciton"],
}

INSTRUMENTS = OPENMIC_TO_MEDLEY.keys()
INSTRUMENT_INDEX = {key: i for i, (key, _) in enumerate(OPENMIC_TO_MEDLEY.items())}
MEDLEY_TO_OPENMIC = {v: k for k, v_list in OPENMIC_TO_MEDLEY.items() for v in v_list}
MEDLEY_TO_INDEX = {k: INSTRUMENT_INDEX[v] for k, v in MEDLEY_TO_OPENMIC.items()}
REV_INSTRUMENT_INDEX = {v: k for k, v in INSTRUMENT_INDEX.items()}

homedir = os.path.expanduser("~")
audiodir = os.path.join(homedir, "MedleyDB/Audio/")
source_path = os.path.join(homedir, "MedleyDB/Source_ID/")


def preprocess(x):
    # normalize raw numpy array then convert it to torch tensor
    x = x.astype(float)
    x /= np.abs(x).max()
    x = torch.from_numpy(x).float()
    return x


def get_scattering_coefficients(x, order1_indices, order2_indices, forward):
    out_dict = {}

    Sx = forward(x)
    order1_coef = Sx[order1_indices]
    order2_coef = Sx[order2_indices]

    out_dict["order1"] = order1_coef
    out_dict["order2"] = order2_coef

    return out_dict


sr = 22050
snippet_len = 6
samples_per_snippet = int(sr * snippet_len)
T = samples_per_snippet
J = 9
Q = 8
meta = Scattering1D.compute_meta_scattering(J, Q)
order1_indices = meta["order"] == 1
order2_indices = meta["order"] == 2

scattering = Scattering1D(J, T, Q)


def preprocess_track(audio_file):
    track_id = audio_file.split("_MIX.wav")[0]
    iad_path = os.path.join(source_path, "%s_SOURCEID.lab" % track_id)

    # Top level directory will look like "J_Q_T"
    spec_path = os.path.join(
        homedir, "MedleyDB/processed/%d_%d_%d_reduced/" % (J, Q, T)
    )

    input_path = os.path.join(spec_path, "input")
    label_path = os.path.join(spec_path, "labels")

    if not (os.path.exists(input_path)):
        os.makedirs(input_path)
    if not (os.path.exists(label_path)):
        os.makedirs(label_path)

    y, _ = librosa.load(
        os.path.join(audiodir, audio_file), sr=sr, res_type="kaiser_fast"
    )

    intervals = split(
        y, top_db=10, frame_length=samples_per_snippet, hop_length=samples_per_snippet
    )

    for start_i, end_i in intervals:
        for i in tqdm(range(start_i, end_i, samples_per_snippet), unit="clip"):
            sound_bite = y[i : i + samples_per_snippet]
            if len(sound_bite) == samples_per_snippet:
                sound_bite = preprocess(sound_bite)

                S_dict = get_scattering_coefficients(
                    sound_bite, order1_indices, order2_indices, scattering.forward
                )

                label = preprocess_label(iad_path, i, i + samples_per_snippet)

                np.save(
                    os.path.join(input_path, "%s_%d.npy" % (track_id, int(i / sr))),
                    S_dict,
                )
                np.save(
                    os.path.join(label_path, "%s_%d.npy" % (track_id, int(i / sr))),
                    label,
                )

    return track_id


def preprocess_label(iad_path, start_i, end_i):
    instrument_annotations = np.zeros(len(INSTRUMENTS))
    annotations = pd.read_csv(iad_path)
    for _, row in annotations.iterrows():
        if row["instrument_label"] in MEDLEY_TO_OPENMIC.keys():
            instrument = MEDLEY_TO_OPENMIC[row["instrument_label"]]
        else:
            instrument = row["instrument_label"]

        if instrument in INSTRUMENTS:
            s_t = row["start_time"] * sr
            e_t = row["end_time"] * sr

            if e_t > end_i:
                e_t = end_i

            if s_t < start_i:
                s_t = start_i

            d_t = e_t - s_t

            if d_t > 0:
                instrument_annotations[INSTRUMENT_INDEX[instrument]] = 1

    return instrument_annotations


def f(files):
    for t in tqdm(files, unit="track"):
        track_id = t.split("_MIX.wav")[0]
        iad_path = os.path.join(source_path, "%s_SOURCEID.lab" % track_id)

        if os.path.exists(iad_path):
            id_track = preprocess_track(t)


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


if __name__ == "__main__":
    processes = []
    # num_cpu = mp.cpu_count()

    all_files = os.listdir(audiodir)
    f(all_files)
    # all_files = list(divide_chunks(all_files, math.ceil(len(all_files) / num_cpu)))

    """
    for i in range(num_cpu):
        p = mp.Process(target=f, args=(all_files[i],))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    """

