from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
'''Some code refered from the sample project provided by lecturer Cunningham Eoghan from moodle'''
'''
Part of format from pytorch examples repo: 
https://github.com/pytorch/examples/blob/master/mnist/main.py
https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
'''

class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()
        self.l1 = nn.Linear(57, 512)  # input feature size is 58
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 10) # 10 output classes
        # A random subset of units are set to zero with a probability of 0.5
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(num_features=512)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128)
        
    def forward(self, x):
        nn = self.l1(x)
        nn = self.batchnorm1(nn)
        nn = F.relu(nn)
        nn = self.l2(nn)
        nn = self.batchnorm2(nn)
        nn = F.relu(nn)
        nn = self.dropout(nn)
        nn = self.l3(nn)
        return nn


def train(model, device, train_loader, optimizer):
    model.train()
    cost = nn.CrossEntropyLoss()
    with tqdm(total=len(train_loader)) as progress_bar:
        for batch_idx, (data, label) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = cost(output, label)        
            loss.backward()
            optimizer.step()
            progress_bar.update(1)


def val(model, device, val_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)     
    # Need this line for things like dropout etc.  
    model.eval()
    preds = []
    targets = []
    cost = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to(device)
            label = label.to(device)
            target = label.clone()
            output = model(data)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            losses.append(cost(output, label).item())
    loss = np.mean(losses)
    preds = np.argmax(np.concatenate(preds), axis=1)
    targets  = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return loss, acc

def test(model, device, test_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)    
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            target = label.clone()
            output = model(data)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
        
    preds = np.argmax(np.concatenate(preds), axis=1)
    targets  = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return acc

