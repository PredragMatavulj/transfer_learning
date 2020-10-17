# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 02:09:46 2020

@author: Korisnik
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import torch
from torch import nn
import matplotlib.pyplot as plt
import functools as fnc
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform


def numpar_per_hour(dir_path, hirst_data_path):
    hirst = pd.read_excel(hirst_data_path)
    lista = list(hirst["Unnamed: 0"])
    #print(lista)
    #cnt = 0;
    dt = [];
    calibs = ["2019-02-18 16", "2019-02-25 12", "2019-03-11 12", "2019-03-15 12", "2019-03-15 13", 
          "2019-03-19 07", "2019-03-19 08", "2019-03-25 08","2019-03-25 09", "2019-03-26 08", 
          "2019-03-29 08", "2019-04-04 10", "2019-04-04 11", "2019-04-09 07", "2019-04-15 07", 
          "2019-04-18 16", "2019-04-22 08", "2019-04-22 14", "2019-04-29 10", "2019-05-03 08", 
          "2019-05-07 16", "2019-05-10 16", "2019-05-24 13", "2019-06-03 11", "2019-06-13 15", 
          '2019-05-15 03', '2019-05-15 04', '2019-08-03 03', '2019-09-23 04', '2019-09-24 21', 
          '2019-09-27 13', '2019-10-03 02', '2019-10-03 06', '2019-10-06 00', '2019-10-07 06', 
          '2019-10-11 07', '2019-10-11 08', '2019-10-12 02', '2019-10-13 03']
    for x in lista:
        if (x[:-12] in calibs):
            #print("Calibration hour. SKIPPED.")
            continue
        fpath = os.path.join(dir_path,x[:-12] + '.pkl')
        #print(fpath)
        if os.path.exists(fpath):
             with open(fpath, 'rb') as fp:
                 data = pickle.load(fp)
                 #print([x[:-12], len(data[0])])
                 dt.append([x[:-12], len(data[0])])
                 #df.loc[cnt] = [x[:-12], len(data[0])]
                
                 #print(x, tt)
                 #cnt += 1
                 #totals.append(tt)
    #print(dt)
    df = pd.DataFrame(dt, columns = ['Hour', 'Total'])
    df.to_excel('particle_counts_hour.xlsx')
    return df


class RapidEDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pollen_type="AMBR"):
        """
        Args:
            pollen_type - one of 26 possible pollen types
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
