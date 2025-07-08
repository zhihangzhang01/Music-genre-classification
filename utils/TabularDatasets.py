import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import normalize, LabelEncoder
'''Some code refered from the sample project provided by lecturer Cunningham Eoghan from moodle'''
class TabularGTZANDataset(Dataset):
    def __init__(self, data_dir, file_name, apply_normalization=True):
        """
        data_dir (str): Path to data containing data and labels.
        file_name: Csv file name containing data and labels
        """

        df = pd.read_csv(os.path.join(data_dir, file_name))
        # Convert to numpy
        data = df.loc[:, (df.columns != 'label') & (df.columns != 'filename') & (df.columns != 'length')].to_numpy()
        labels = df.label.to_numpy()
        
        # One-hot Encoding to convert nominal feature
        if labels.dtype == 'object':
            encoder = LabelEncoder()
            labels = encoder.fit_transform(labels)
            
        # Apply normalization helping reduce outlier and better converge if true
        if apply_normalization:
            data = normalize(data, axis=1, norm='l2')
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        X = self.data[index].float()
        y = self.labels[index]
        return X, y

    def __len__(self):
        return len(self.data)


