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
import yaml

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
MEDLEYDB_INSTRUMENTS = [
    item for sublist in OPENMIC_TO_MEDLEY.values() for item in sublist
]
INSTRUMENT_INDEX = {key: i for i, (key, _) in enumerate(OPENMIC_TO_MEDLEY.items())}
MEDLEY_TO_OPENMIC = {v: k for k, v_list in OPENMIC_TO_MEDLEY.items() for v in v_list}
MEDLEY_TO_INDEX = {k: INSTRUMENT_INDEX[v] for k, v in MEDLEY_TO_OPENMIC.items()}
REV_INSTRUMENT_INDEX = {v: k for k, v in INSTRUMENT_INDEX.items()}

homedir = os.path.expanduser("~")
medleydir = os.path.join(homedir, "V1")

audiodir = os.path.join(homedir, "MedleyDB/Audio/")
source_path = os.path.join(homedir, "MedleyDB/Source_ID/")

sr = 22050  # sampling rate
snippet_len = 6  # 6s long snippets
samples_per_snippet = int(sr * snippet_len)
T = samples_per_snippet
J = 9
Q = 8
meta = Scattering1D.compute_meta_scattering(J, Q)
order1_indices = meta["order"] == 1
order2_indices = meta["order"] == 2

scattering = Scattering1D(J, T, Q)


def preprocess(x):
    # normalize raw numpy array then convert it to torch tensor
    x = x.astype(float)
    x /= np.abs(x).max()
    x = torch.from_numpy(x).float()
    return x


def get_scattering_coefficients(x, order1_indices, order2_indices, forward):
    # get first and second order coefficients in a dictionary
    out_dict = {}

    Sx = forward(x)
    order1_coef = Sx[order1_indices]
    order2_coef = Sx[order2_indices]

    out_dict["order1"] = order1_coef
    out_dict["order2"] = order2_coef

    return out_dict


def preprocess_track(track_id):
    iad_path = os.path.join(source_path, "%s_SOURCEID.lab" % track_id)

    # the directory looks like "J_Q_T_reduced"
    spec_path = os.path.join(
        homedir, "MedleyDB/processed/%d_%d_%d_reduced_new/" % (J, Q, T)
    )

    input_path = os.path.join(spec_path, "input")
    label_path = os.path.join(spec_path, "labels")

    if not (os.path.exists(input_path)):
        os.makedirs(input_path)
    if not (os.path.exists(label_path)):
        os.makedirs(label_path)

    # load audio list

    # y, _ = librosa.load(
    #     os.path.join(audiodir, audio_file), sr=sr, res_type="kaiser_fast"
    # )

    meta_file = (
        "/home/laura/medleydb/medleydb/data/Metadata/" + track_id + "_METADATA.yaml"
    )

    with open(meta_file) as f:
        data = yaml.load(f)

    stems_files = [
        stem["filename"]
        for stem in data["stems"].values()
        if stem["instrument"] in MEDLEYDB_INSTRUMENTS
    ]

    if stems_files == []:
        return track_id

    y = np.mean(
        np.stack(
            [
                librosa.load(
                    os.path.join(medleydir, track_id, track_id + "_STEMS", audio_file),
                    sr=sr,
                    res_type="kaiser_fast",
                )[0]
                for audio_file in stems_files
            ],
        ),
        axis=0,
    )

    # get the non-silent intervals
    intervals = split(
        y, top_db=10, frame_length=samples_per_snippet, hop_length=samples_per_snippet
    )

    for start_i, end_i in intervals:
        for i in tqdm(range(start_i, end_i, samples_per_snippet), unit="clip"):
            sound_bite = y[i : i + samples_per_snippet]
            # get all sound_bites except the last one (which is not 6s long)
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
        # check which instrumetn it is in the openmic setting (instead of medley)
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
    for track_id in tqdm(files, unit="track"):
        iad_path = os.path.join(source_path, "%s_SOURCEID.lab" % track_id)

        if os.path.exists(iad_path):
            id_track = preprocess_track(track_id)


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


if __name__ == "__main__":
    processes = []
    # num_cpu = mp.cpu_count()

    ids = os.listdir(medleydir)
    f(ids)

    # all_files = os.listdir(audiodir)
    # f(all_files)
    # all_files = list(divide_chunks(all_files, math.ceil(len(all_files) / num_cpu)))

    """
    for i in range(num_cpu):
        p = mp.Process(target=f, args=(all_files[i],))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    """
