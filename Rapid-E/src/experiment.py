# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:46:06 2020

@author: sjelic
"""

import os
import copy
#import pandas as pd
import numpy as np
import torch
import logging
import datetime
import pandas as pd
#from torch import nn
#import matplotlib.pyplot as plt
#import functools as fnc
from torch.utils.data import DataLoader
from sampler import StratifiedSampler
from utils import train_test_split, open_excel_file_in_pandas, gridsearch_hparam_from_json, my_collate
import torch.optim as optim
from objectives import WeightedSELoss, PearsonCorrelationLoss
from dataset import RapidEDataset
from model import RapidENet

args = {
'experiment_name': 'Experiment',
'data_dir_path': '../data/novi_sad_2019_/',
'model_dir_path': './models',
'metadata_path': './Libraries/data_pandas_frame.xlsx',
'pollen_info_path': './Libraries/pollen_types.xlsx',
'hparam_path': 'hyper_params.json',
'objective_criteria': 'WeightedSELoss',
'additional_criteria': ['PearsonCorrelationLoss'],
'selection_criteria': 'WeightedSELoss',
'cross_valid_type': 'cross_seasonal',
'hparam_search_strategy': 'gridsearch',
'num_of_valid_splits': 2,
'num_of_test_splits': 2,
'train_batch_size': 500,
'valid_batch_size': 100,
'test_batch_size': 100,
'model': 'RapidENet',
'pretrained_model_state_path': './models/novi_sad/model_pollen_types_ver0/Ambrosia_vs_all.pth',
'number_of_classes': 2,
'logging_per_batch': True,
'logging': True
}



        
class Experiment:
    def __init__ (self, args):
        self.args = args
        self.df = open_excel_file_in_pandas(args['metadata_path'])
        self.df_pollen_types = open_excel_file_in_pandas(args['pollen_info_path'])
        if args['hparam_search_strategy'] == 'gridsearch':
            self.hyperparameters = gridsearch_hparam_from_json(args['hparam_path'])
        else:
            raise RuntimeError('Only gridsearch is implemented.')
        
        self.criteria = {'objective_criteria': self.set_metric_by_name(self.args['objective_criteria']),
                         'additional_criteria': [self.set_metric_by_name(name) for name in self.args['additional_criteria']],
                         'selection_criteria': self.set_metric_by_name(self.args['selection_criteria'])
                         }
        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.train_dict = {}
        self.model_dir_path = os.path.join(self.args['model_dir_path'],self.args['experiment_name'])
        os.mkdir(self.model_dir_path + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.train_dict = None;
        self.logging = self.args['logging']
        self.logging_per_batch = self.args['logging_per_batch']
    
    
    
    def set_model(self, hp, model_path = None):
        
        if self.args['model'] == 'RapidENet':
            self.model = RapidENet(dropout_rate = hp['drop_out'], number_of_classes = self.args['number_of_classes']).float()
            if model_path:
                self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
            else:
                self.model.load_state_dict(torch.load(self.args['pretrained_model_state_path'], map_location=lambda storage, loc: storage), strict=False)
        else:
            raise RuntimeError('Only RapidENet is implemented.')
        
        
        
    def set_optimizer(self, hp):
        if hp['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr = hp['lr'], weight_decay=hp['weight_decay'])
            #self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=hp['lr']/10, max_lr=hp['lr'])
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=4, factor=0.5, verbose=False)
            
        elif hp['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr = hp['lr'], momentum=hp['momentum'], weight_decay=hp['weight_decay'], nesterov=True)
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=hp['lr']/10, max_lr=hp['lr'])
        
        elif hp['optimizer'] == 'lbfgs':
            self.optimizer = optim.LBFGS(self.model.parameters(), lr =  hp['lr'], max_iter=hp['max_iter'], line_search_fn='strong_wolfe')
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=hp['lr']/10, max_lr=hp['lr'])
        
        else:
            logging.warning('This optimizer is not implemented. Adam will be used instead.')
            self.optimizer = optim.AdamW(self.model.parameters(), lr = hp['lr'], weight_decay=hp['weight_decay'])
        
    
    def set_metric_by_name(self, name):
        if name ==  'WeightedSELoss':
            criteria = WeightedSELoss(selection=(True if name == self.args['selection_criteria'] else False))
        if name ==  'PearsonCorrelationLoss':
            criteria = PearsonCorrelationLoss(selection=(True if name == self.args['selection_criteria'] else False))
        return criteria
    
    def set_train_dict(self, num_of_epochs):
        self.train_dict = {'train': {},
                            'valid': {}}
        
        for dset in ['train', 'valid']:
            for criteria in [self.criteria['objective_criteria']] + self.criteria['additional_criteria']:
                self.train_dict[dset][criteria.name] = {'epochs_sum': torch.zeros(num_of_epochs), 
                                                            'epoch_mean': torch.zeros(num_of_epochs),
                                                            'best_value': float('inf') if criteria.sense == 'min' else float('-inf'),
                                                             }
    
    def prepare_data_loader(self, dframe, batch_size, dataset_name):
        dataset = RapidEDataset(dframe, self.args['data_dir_path'], self.df_pollen_types, name = dataset_name)
        stratified_train_sampler = StratifiedSampler(torch.from_numpy(np.array(list(dframe['CLUSTER']))), batch_size)
        return DataLoader(dataset, batch_size=batch_size, sampler = stratified_train_sampler, collate_fn=my_collate)

    
    def tune_hparam(self, inner):
        means = torch.mean(inner,1)
        #stdevs = torch.std(inner,1)
        if (self.criteria['selection_criteria'].sense == 'max'):
            jopt = torch.argmax(means).item()
        else:
            jopt = torch.argmin(means).item()
        return jopt
    
    

            
    def update_batch_info(self, dataset_type, output, target, weights, batch_idx, epoch_idx, num_of_batches):
        
        batch_loss = self.criteria['objective_criteria'](output, target, weights)
        self.train_dict[dataset_type][self.args['objective_criteria']]['epochs_sum'][epoch_idx] += batch_loss
        if (batch_idx == num_of_batches - 1):
               self.train_dict[dataset_type][self.args['objective_criteria']]['epochs_mean'][epoch_idx] = self.train_dict[dataset_type][self.args['objective_criteria']]['epochs_sum'][epoch_idx] / num_of_batches      
        for criteria in self.criteria['additional_criteria']:
            batch_loss = criteria(output,target,weights)
            self.train_dict[dataset_type][criteria.name]['epochs_sum'][epoch_idx] += batch_loss
            if (batch_idx == num_of_batches - 1):
                self.train_dict[dataset_type][criteria.name]['epochs_mean'][epoch_idx] = self.train_dict[dataset_type][self.criteria.name]['epochs_sum'][epoch_idx] / num_of_batches

    def nested_crossvalidation(self):
        test_split_groups = np.array(list(self.df['CLUSTER']))
        train_valid_test = train_test_split(test_split_groups, num_splits = self.args['num_of_test_splits'])
        outer = torch.zeros(self.args['num_of_test_splits'])
        for i, (train_valid_data, test_data) in enumerate(train_valid_test):
            
            # train model
            
            df_train_valid = self.df.iloc[sorted(train_valid_data)]
            #print(df_train_valid.index.tolist())
            df_train_valid = df_train_valid.set_index(pd.Index(list(range(len(df_train_valid)))))
            #print(df_train_valid.index.tolist())
            #break
            valid_split_groups = np.array(list(df_train_valid['CLUSTER']))
            train_valid = train_test_split(valid_split_groups, num_splits = self.args['num_of_valid_splits'])
            inner = torch.zeros(len(self.hyperparameters), self.args['num_of_valid_splits'])
            for j, (train_data, valid_data) in enumerate(train_valid):
                # print(j)
                # print(len(train_data))
                # print(len(valid_data))
                
                df_train = df_train_valid.iloc[sorted(train_data)]
                df_train = df_train.set_index(pd.Index(list(range(len(df_train)))))
                train_loader = self.prepare_data_loader(df_train, self.args['train_batch_size'], str(i+1) + '_' + str(j+1)+ '_' +'train')
                df_valid = df_train_valid.iloc[sorted(valid_data)]
                df_valid = df_valid.set_index(pd.Index(list(range(len(df_valid)))))
                valid_loader = self.prepare_data_loader(df_valid, self.args['valid_batch_size'], str(i+1) + '_' + str(j+1)+ '_' +'valid')
                
                for k, hp in enumerate(self.hyperparameters):
                    
                    inner[k][j] = self.train(train_loader, hp, save_model = False, valid_loader=valid_loader)
                    
            hp_best = self.hyperparameters[self.tune_hparam(inner)]
            df_test = self.df.iloc[sorted(test_data)]
            trainvalid_loader = self.prepare_data_loader(train_valid, self.args['train_batch_size'], str(i+1) + '_' +'trainvalid')
            test_loader = self.prepare_data_loader(df_test, self.args['test_batch_size'], str(i+1) + '_' +'test')
            outer[i] = self.train(trainvalid_loader, hp_best, save_model=True, valid_loader=test_loader)
            
        return {'mean_objective_loss': torch.mean(outer), 'std_objective_loss': torch.std(outer)}
    
    
    
    def hparam2str(self,hp):
        hpstr = ''
        for key in hp:
            hpstr += ( '\t\t' + key + ' = ' + str(hp[key]) + '\n')
        return hpstr
            

            

    def train(self, train_loader, hp, save_model = False, valid_loader = None):
        
        
        self.set_model(hp)
        self.set_optimizer(hp)
        
        if self.logging:
            print('Training started.')
            print('\tModel: '+ 'RapidENet')
            print('\tOptimizer: ' + hp['optimizer'])
            print('\tHyperparameter:\n' + self.hparam2str(hp))
        
        
        
        

        self.set_train_dict(hp['num_of_epochs'])
        
        
        
        for epoch in range(hp['num_of_epochs']):
            
            if logging:
                print('Epoch '+ str(epoch+1) + ' started.')
            self.model.train()
            # iterating on train batches and update model weights
            #print(train_loader.dataset.df)
            for i, train_batch in enumerate(train_loader):
                #print(i)
                #print(train_batch[0][0][1].shape)
                
                train_batch_data, train_batch_target, train_batch_weights = train_batch
                #print(train_batch_target)
                #print(train_batch_weights)
                
                
                self.optimizer.zero_grad()
                train_batch_output = self.model(train_batch_data)
                #print(train_batch_output)
                objective_batch_loss = self.criteria['objective_criteria'](train_batch_output, train_batch_target, train_batch_weights)
                objective_batch_loss.backward()
                
                self.update_batch_info('train', train_batch_output, train_batch_target, train_batch_weights, i, epoch, len(train_loader))
                
                self.optimizer.step(lambda: objective_batch_loss)
                self.scheduler.step(objective_batch_loss)

            # iterating on valid batches
            self.model.eval()
            if valid_loader:
                for j, valid_batch in enumerate(valid_loader):
                    valid_batch_data, valid_batch_target, valid_batch_weights = valid_batch
                    valid_batch_output = self.model(valid_batch_data)
                    self.update_batch_info('valid', valid_batch_output, valid_batch_target, valid_batch_weights, j, epoch, len(valid_loader))
            
            
            self.update_best_model_for_each_criteria(train_loader.dataset.name, valid_loader.dataset.name if valid_loader != None else None, hp, epoch, save_model)
            
            #print epoch results from statedict
            
            for ds in ['train', 'valid']:
                print("\t\t" + ds + ":")
                for cn in [self.args['objective_criteria']] + self.args['additional_criteria']:
                    print('\t\t\t' + cn + ': ' + str(self.train_dict[ds][cn]['epochs_mean']))
            
            
            
            
            
        
        if valid_loader:
            return self.train_dict['valid'][self.args['selection_criteria']]['best_value']
        else:
            return self.train_dict['train'][self.args['selection_criteria']]['best_value']
    
    
    def create_file_name(self, hp, dataloader, criteria_name, dataset_type = 'valid'):
        fname = dataloader.dataset.name +'_' + dataset_type + '_' + criteria_name
        for key in hp:
            fname + '_' + key + '=' + str(hp[key])
        
        return fname + '.pt'
    
    def save_model_state(self, hp, epoch_idx, dataset_name, criteria_name, dataset_type):
        state = copy.deepcopy(hp)
        state['epoch'] = epoch_idx
        state['model_state_dict'] = self.args['model'].state_dict()
        state['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save( state, os.path.join(self.model_dir_path,self.create_file_name(hp,dataset_name,criteria_name, dataset_type)))
    
    def update_best_model_for_each_criteria(self, traindataset_name, valid_dataset_name, hp, epoch_idx, save_model):
        for criteria in [self.args['objective_criteria']] + self.args['additional_criteria']:
            if criteria.sense == 'min':
                if  self.train_dict['train'][criteria.name]['epoch_'+criteria.reduction][epoch_idx] <  self.train_dict['train'][criteria.name]['best_value']:
                    self.train_dict['train'][criteria.name]['best_value'] =  self.train_dict['train'][criteria.name]['epoch_'+criteria.reduction][epoch_idx]
                    if save_model:
                            self.save_model_state(hp, epoch_idx, traindataset_name, 'train')
                if valid_dataset_name:
                    if  self.train_dict['valid'][criteria.name]['epoch_'+criteria.reduction][epoch_idx] <  self.train_dict['valid'][criteria.name]['best_value']:
                        self.train_dict['valid'][criteria.name]['best_value'] =  self.train_dict['valid'][criteria.name]['epoch_'+criteria.reduction][epoch_idx]
                        if save_model:
                                self.save_model_state(hp, epoch_idx, valid_dataset_name, 'valid')      
            else:
                if  self.train_dict['train'][criteria.name]['epoch_'+criteria.reduction][epoch_idx] >  self.train_dict['train'][criteria.name]['best_value']:
                    self.train_dict['train'][criteria.name]['best_value'] =  self.train_dict['train'][criteria.name]['epoch_'+criteria.reduction][epoch_idx]
                    if save_model:
                            self.save_model_state(hp, epoch_idx, traindataset_name, 'train')
                
                if valid_dataset_name:
                
                    if  self.train_dict['valid'][criteria.name]['epoch_'+criteria.reduction][epoch_idx] >  self.train_dict['valid'][criteria.name]['best_value']:
                        self.train_dict['valid'][criteria.name]['best_value'] =  self.train_dict['valid'][criteria.name]['epoch_'+criteria.reduction][epoch_idx]
                        if save_model:
                                self.save_model_state(hp, epoch_idx, valid_dataset_name, 'valid')
                            
    def validate(self, valid_loader, hp,  model_path = None):
        self.set_model(hp, model_path)
        self.model.eval()
        self.set_trainvalid_dict(1)
        for j, valid_batch in enumerate(valid_loader):
                valid_batch_data, valid_batch_target, valid_batch_weights = valid_batch
                valid_batch_output = self.model(valid_batch_data)
                self.update_batch_info('valid', valid_batch_output, valid_batch_target, valid_batch_weights, j, 0, len(valid_loader))
                
    def deploy(self, save_model = True):
        valid_split_groups = np.array(list(self.df['CLUSTER']))
        train_valid = train_test_split(valid_split_groups, num_splits = self.args['num_of_test_splits'])
        inner = torch.zeros(len(self.hyperparameters), self.args['num_of_test_splits'])
        for j, train_data, valid_data in enumerate(train_valid):
            
            df_train = self.df.iloc[sorted(train_data)]
            train_loader = self.prepare_data_loader(df_train, self.args['train_batch_size'], str(j+1) + '_' +'train')
            df_valid = self.df.iloc[sorted(valid_data)]
            valid_loader = self.prepare_data_loader(df_valid, self.args['valid_batch_size'], str(j+1) + '_' +'valid')
            
            for k, hp in enumerate(self.hyperparameters):
                
                inner[k][j] = self.train(train_loader, valid_loader, hp)
                
        hp_best = self.hyperparameters[self.tune_hparam(inner)]
        data_loader = self.prepare_data_loader(self.df, self.args['train_batch_size'], 'deploy')
        result = self.train(data_loader, hp_best, save_model=True)
        return result
        
        
        
        
exp1 = Experiment(args)
exp1.nested_crossvalidation() 

