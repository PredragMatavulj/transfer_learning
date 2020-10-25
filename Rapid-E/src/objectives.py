# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:44:30 2020

@author: sjelic
"""

import torch




class  WeightedSELoss():
    
    def __init__(self, name = 'WeightedSELoss', sense= 'min', reduction='sum', selection = False):
        self.reduction = reduction
        self.sense = sense
        self.name = name
        self.selection = selection

    
    def __call__(self,output, target, weights):
        # print(weights.type())
        # print(target.type())
        # print(output.type())
        # print(((output - target)**2).type())
        return torch.dot(weights,(output - target)**2)



class PearsonCorrelationLoss():
    def __init__(self, name='PearsonCorrelationLoss', sense= 'max', reduction='mean', selection = False):
        self.reduction = reduction
        self.sense = sense
        self.name = name
        self.selection = selection
    
    def __call__(self, x, y, weights = None):
        x = x - torch.mean(x)
        y = y - torch.mean(y)
        return torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))
    
