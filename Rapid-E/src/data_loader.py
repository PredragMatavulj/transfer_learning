# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 02:09:46 2020

@author: Korisnik
"""

import os
#import pandas as pd
import numpy as np
import pickle
import torch
#from torch import nn
#import matplotlib.pyplot as plt
#import functools as fnc
from torch.utils.data import Dataset, DataLoader
from sampler import StratifiedSampler
#from skimage import io, transform




class RapidEDataset(Dataset):
    """RapidE dataset."""

    def __init__(self, df, dir_path):
        """
        Args:
            pollen_type - one of 26 possible pollen types
            df (string): pandas frame with metada and .
            dir_path (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.dir_path = dir_path
        #self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(os.path.join(self.dir_path, self.df.loc[idx,'FILENAME']), 'rb') as file:
            X = pickle.load(file)
            X[0] = torch.Tensor(X[0]).unsqueeze_(0).permute(1, 0, 2, 3)
            X[1] = torch.Tensor(X[1]).unsqueeze_(0).permute(1, 0, 2, 3)
            X[2] = torch.Tensor(X[2]).unsqueeze_(0).permute(1, 0, 2, 3)
            X[3] = torch.Tensor(X[3])
            X[4] = torch.Tensor(X[4]).unsqueeze_(0).permute(1, 0)
            y = np.array(list(self.df.iloc[idx,3:]))
        
        
        

        return X, y
    
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]
