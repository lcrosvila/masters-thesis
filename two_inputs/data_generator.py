import os
import sys

import numpy as np
import h5py
import csv
import time
import logging
import librosa
import torch
import matplotlib.pyplot as plt

eps = sys.float_info.epsilon


def get_mean_var(processed_path):
    scatter_type = processed_path.split("/")[5]
    files = os.listdir(os.path.join(processed_path, "input/"))

    if scatter_type == "logmel" or scatter_type == 'logmel_reduced':
        logmel_list = []

        for file in files:
            input_path = os.path.join(processed_path, "input/" + file)
            logmel = np.load(input_path, allow_pickle=True)

            logmel_list.append(logmel)

        mean_logmel = sum(logmel_list) / len(logmel_list)
        var_logmel = sum((xi - mean_logmel) ** 2 for xi in logmel_list) / len(
            logmel_list
        )

        var_logmel = var_logmel + eps

        return mean_logmel, var_logmel

    else:
        order1_list = []
        order2_list = []

        for file in files:
            input_path = os.path.join(processed_path, "input/" + file)
            scatter = np.load(input_path, allow_pickle=True)

            order1 = scatter.item().get("order1")
            order2 = scatter.item().get("order2")

            order1_list.append(order1)
            order2_list.append(order2)

        mean_order1 = sum(order1_list) / len(order1_list)
        var_order1 = sum((xi - mean_order1) ** 2 for xi in order1_list) / len(
            order1_list
        )
        var_order1 = var_order1 + eps

        mean_order2 = sum(order2_list) / len(order2_list)
        var_order2 = sum((xi - mean_order2) ** 2 for xi in order2_list) / len(
            order2_list
        )
        var_order2 = var_order2 + eps

        return mean_order1, var_order1, mean_order2, var_order2


class medleyDataset(object):
    def __init__(
        self,
        processed_path,
        input_length,
        order1_length,
        order2_length,
        classes_num,
        mean_order1,
        var_order1,
        mean_order2,
        var_order2,
    ):
        """Medley dataset for later used by DataLoader.  
        """
        self.classes_num = classes_num
        self.input_length = input_length
        self.order1_length = order1_length
        self.order2_length = order2_length

        self.mean_order1 = mean_order1
        self.var_order1 = var_order1
        self.mean_order2 = mean_order2
        self.var_order2 = var_order2

        self.path = processed_path
        self.files = os.listdir(os.path.join(processed_path, "input/"))

        logging.info("Scattering samples: {}".format(len(self.files)))

    def __getitem__(self, index):
        """Load scattering and target of the audio index. If index is -1 then 
            return None. 
        
        Returns: {'sample_name': str, 'order1': (order1_length, input_length), 'order2': (order2_length, input_length), 'target': (classes_num,)}
        """
        if index == -1:
            sample_name = None
            order1 = torch.zeros([self.order1_length, self.input_length])
            order2 = torch.zeros([self.order2_length, self.input_length])
            target = np.zeros((self.classes_num,), dtype=np.float32)
        else:
            file = self.files[index]
            sample_name = file[:-4]

            input_path = os.path.join(self.path, "input/" + file)
            target_path = os.path.join(self.path, "labels/" + file)

            target = np.load(target_path, allow_pickle=True)
            target = target.astype("float32")

            scatter = np.load(input_path, allow_pickle=True)

            order1 = scatter.item().get("order1")
            if self.mean_order1 != None:
                order1 = (order1 - self.mean_order1) / self.var_order1

            order2 = scatter.item().get("order2")
            if self.mean_order2 != None:
                order2 = (order2 - self.mean_order2) / self.var_order2

        data_dict = {
            "sample_name": sample_name,
            "order1": order1,
            "order2": order2,
            "target": target,
        }

        return data_dict

    def __len__(self):
        return len(self.files)


class medleyDataset_logmel(object):
    def __init__(
        self,
        processed_path,
        time_steps,
        freq_bins,
        classes_num,
        mean_logmel,
        var_logmel,
    ):
        """Medley dataset for later used by DataLoader.  
        """
        self.classes_num = classes_num
        self.time_steps = time_steps
        self.freq_bins = freq_bins

        self.mean_logmel = mean_logmel
        self.var_logmel = var_logmel

        self.path = processed_path
        self.files = os.listdir(os.path.join(processed_path, "input/"))

        logging.info("Scattering samples: {}".format(len(self.files)))

    def __getitem__(self, index):
        """Load scattering and target of the audio index. If index is -1 then 
            return None. 
        
        Returns: {'sample_name': str, 'logmel': (freq_bins, time_steps), 'target': (classes_num,)}
        """
        if index == -1:
            sample_name = None
            logmel = torch.zeros([self.freq_bins, self.time_steps])
            target = np.zeros((self.classes_num,), dtype=np.float32)
        else:
            file = self.files[index]
            sample_name = file[:-4]

            logmel_path = os.path.join(self.path, "input/" + file)
            target_path = os.path.join(self.path, "labels/" + file)

            logmel = np.load(logmel_path, allow_pickle=True)
            if self.mean_logmel != None:
                logmel = (logmel - self.mean_logmel) / self.var_logmel

            target = np.load(target_path, allow_pickle=True)
            target = target.astype("float32")

        data_dict = {"audio_name": sample_name, "logmel": logmel, "target": target}

        return data_dict

    def __len__(self):
        return len(self.files)


class medleyDataset_audio(object):
    def __init__(
        self, processed_path, input_length, order1_length, order2_length, classes_num
    ):
        """Medley dataset for later used by DataLoader.  
        """
        self.classes_num = classes_num
        self.input_length = input_length
        self.order1_length = order1_length
        self.order2_length = order2_length

        self.path = processed_path
        self.files = os.listdir(os.path.join(processed_path, "input/"))

        logging.info("Scattering samples: {}".format(len(self.files)))

    def __getitem__(self, index):
        """Load scattering and target of the audio index. If index is -1 then 
            return None. 
        
        Returns: {'sample_name': str, 'order1': (order1_length, input_length), 'order2': (order2_length, input_length), 'target': (classes_num,)}
        """
        if index == -1:
            audio_name = None
            waveform = np.zeros((28,), dtype=np.float32)
            target = np.zeros((self.classes_num,), dtype=np.float32)
        else:
            file = self.files[index]
            sample_name = file[:-4]

            start = int(sample_name.split("_")[2]) * 22050

            wav_name = (
                sample_name.split("_")[0] + "_" + sample_name.split("_")[1] + "_MIX.wav"
            )

            audio_path = os.path.join("/home/laura/MedleyDB/Audio/", wav_name)
            target_path = os.path.join(self.path, "labels/" + file)

            target = np.load(target_path, allow_pickle=True)
            target = target.astype("float32")

            y, _ = librosa.load(audio_path, sr=22050, res_type="kaiser_fast")
            waveform = y[start : start + 6 * 22050]

        data_dict = {"audio_name": sample_name, "waveform": waveform, "target": target}

        return data_dict

    def __len__(self):
        return len(self.files)
