import os
import sys
import numpy as np
import h5py
import csv
import time
import logging
import librosa
import soundfile as sf

class monoDataset(object):
    def __init__(self, npz_path, audio_length, classes_num):
        """Monophonic dataset for later used by DataLoader. This class takes an 
        audio index as input and output corresponding waveform and target. 
        """
        self.audio_length = audio_length
        self.classes_num = classes_num

        monophonic = np.load(npz_path, allow_pickle=True)
        self.paths = monophonic['path']
        self.audio_names = monophonic['sample_key']
        self.target = monophonic['Y']
        self.indexes = range(len(self.audio_names))

        logging.info('Audio samples: {}'.format(len(self.audio_names)))
    
    def __getitem__(self, index):
        """Load waveform and target of the audio index. If index is -1 then 
            return None. 
        
        Returns: {'audio_name': str, 'waveform': (audio_length,), 'target': (classes_num,)}
        """
        if index == -1:
            audio_name = None
            waveform = np.zeros((self.audio_length,), dtype=np.float32)
            target = np.zeros((self.classes_num,), dtype=np.float32)
        else:
            audio_path = self.paths[index]
            audio_name = self.audio_names[index]
            target = self.target[index]
            target = target.astype('float32')

            waveform, _ = librosa.load(audio_path)
                
        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
        
        return data_dict
    
    def __len__(self):
        return len(self.audio_names)