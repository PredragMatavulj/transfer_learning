import torch
import numpy as np
from torch import nn
from scipy import stats
import json
import logging

from Sampler import StratifiedSampler
from RapidECalibrationDataset import RapidECalibrationDataset
from Standardize import RapidECalibrationStandardize
from RapidEClassifier import RapidEClassifier
from torch.utils.data import DataLoader
from CrossValidator import CrossValidator
from StratifiedSplitter import StratifiedSplitterForRapidECalibDataset
from Trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from NestedCrossValidator import NestedCrossValidator

with open('/home/guest/coderepos/transfer_learning/Rapid-E/src/hyperparameters.json','r') as f:
    hyperparameters = json.load(f)

exp_name = 'all_pollen_types'
base_dir = '/home/guest/coderepos/transfer_learning/Rapid-E'
tensor_subdir = 'data/calib_data_ns_tensors'
meta_subdir = 'meta_jsons'
dataset_meta_json_path = '/home/guest/coderepos/transfer_learning/Rapid-E/meta_jsons/00000_train.pt'

writer = SummaryWriter(log_dir='../runs/' + exp_name, flush_secs=10)




standardization = RapidECalibrationStandardize(dataset_meta_json_path)
dataset = RapidECalibrationDataset(dataset_meta_json_path, transform=standardization)

model = RapidEClassifier(number_of_classes=26, dropout_rate=0.5, name='All Pollen Types')



if hyperparameters['gpu']:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device("cpu")
        model = model
        logging.warning('CUDA is not availible on this machine')


else:
    device = torch.device("cpu")
    model = model
model.to(device)

labels = dataset.gettargets()
set_l = set(labels)
freq = stats.relfreq(np.array(labels), numbins=len(set_l)).frequency
weights = torch.tensor((1/freq)/np.sum(1/freq), dtype=torch.float32)
weights = weights.to(device)
loss = nn.CrossEntropyLoss(weight=weights, reduction='sum')

sampler = StratifiedSampler(torch.tensor(labels), hyperparameters['train_batch_size'])
train_loader = DataLoader(dataset, sampler=sampler, batch_size=hyperparameters['train_batch_size'], num_workers=hyperparameters['num_workers'])


splitter = StratifiedSplitterForRapidECalibDataset(hyperparameters['num_of_folds'],dataset)

#trainer = Trainer(model=model, objectiveloss=loss, trainloader=train_loader, validloader=None, hyperparams=hyperparameters)
#trainer()

#crossvalidator = CrossValidator(model=model,device=device, objectiveloss=loss, splitter=splitter,hyperparams=hyperparameters, tbwriter=writer)
#crossvalidator()
nestedcrossvalidator =NestedCrossValidator(model=model, device=device, objectiveloss=loss,splitter=splitter, hyperparams=hyperparameters, tbwriter=writer)
nestedcrossvalidator()
writer.close()
# TRAINER test
