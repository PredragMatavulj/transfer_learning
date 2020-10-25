# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:42:33 2020

@author: sjelic
"""

import os
#import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import pandas as pd




class RapidEDataset(Dataset):
    """RapidE dataset."""

    def __init__(self, df, dir_path, df_pollen_info, name = 'dataset'):
        """
        Args:
            pollen_type - one of 26 possible pollen types
            df (string): pandas frame with metada and .
            dir_path (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df.copy()
        self.dir_path = dir_path
        self.name = name
        self.df_pollen_info = df_pollen_info
        self.pollen_types = list(df.columns[4:])
        self.num_of_classes = len(self.pollen_types)
        #print(self.df_pollen_info)
        #print(self.pollen_types)
        self.monts_of_types = df_pollen_info[['START', 'END']][df_pollen_info['CODE'].isin(self.pollen_types)]
        
        self.monts_of_types = self.monts_of_types.set_index(pd.Index(list(range(len(self.monts_of_types)))))
        for i in range(len(self.pollen_types)):
            inseas = sum(list(map(lambda x: 1 if (x>= self.monts_of_types.loc[i,'START'] and x<=self.monts_of_types.loc[i,'START']) else 0, list(self.df['MONTH']))))
            outseas = len(self.df) - inseas
            weights = list(map(lambda x: 1.0/(2*inseas) if (x>= self.monts_of_types.loc[i,'START'] and x<=self.monts_of_types.loc[i,'START']) else 1.0/(2*outseas), list(self.df['MONTH'])))
            self.df[self.pollen_types[i]+'_W'] = weights
        
        
        
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
            y = torch.tensor(np.array(list(self.df.iloc[idx,4:(4+self.num_of_classes)])),dtype=torch.float32)
            w = torch.tensor(np.array(list(self.df.iloc[idx,(4+self.num_of_classes):])),dtype=torch.float32)
            #print(y)
            #print(y)
            #w = np.array(list(self.df.iloc[idx,(4+len(self.pollen_types)):]))
            
            
            
        return X, y, w
