# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 02:09:46 2020

@author: Korisnik
"""

import os
import pandas as pd
#import numpy as np
import pickle
import torch
#from torch import nn
#import matplotlib.pyplot as plt
#import functools as fnc
from torch.utils.data import Dataset
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
        
        X = pickle.load(os.path.join(self.dir_path, self.df.loc[idx,'FILENAME']))
        y = self.df.iloc[idx,2:]
        

        return X, y
