import os
import torch
import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data.dataset import Dataset
class Wav2Vec2Dataset(Dataset):
    def __init__(self, data_dir, subdir, sample_rate=16000):
        """
        data_dir (str): Path to directory containing audio files.
        sample_rate (int): Sampling rate to resample audio files to (default: 16000).
        """
        self.data_dir = os.path.join(data_dir, subdir)
        self.sample_rate = sample_rate
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.filepaths, self.labels = self.load_filepaths_and_labels()
        self.max_length=16000
    def load_filepaths_and_labels(self):
        filepaths = []
        labels = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for filename in os.listdir(class_dir):
                filepaths.append(os.path.join(class_dir, filename))
                labels.append(self.class_to_idx[class_name])
        return filepaths, labels
    
        
    def __getitem__(self, idex):
        filepath = self.filepaths[idex]
        label = self.labels[idex]
        
        # Load audio waveform and resample if sample rate does not matches
        waveform, sr = torchaudio.load(filepath, normalize=True)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        # Normalize amplitude values to be within [-1, 1]   
        waveform = waveform / torch.max(torch.abs(waveform))
            
        # Truncate waveform to make sure the same length
        if waveform.size(1) > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.size(1) < self.max_length:
            padding = self.max_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform, torch.tensor(label)

    def __len__(self):
        return len(self.filepaths)


