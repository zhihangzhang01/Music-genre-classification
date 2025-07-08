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
Part of format and full model from pytorch examples repo: https://github.com/pytorch/examples/blob/master/mnist/main.py

https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)  # Layer 1 with 3 input channel, 8 output channels, 3*3 kernel size and 1 stride, 1 padding
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # Layer 2 with 8 input channel, 16 output channels
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # Layer 3 with 16 input channel, 32 output channels
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)# Layer 4 with 32 input channel, 64 output channels
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)# Layer 4 with 64 input channel, 128 output channels
        self.dropout = nn.Dropout2d(0.6) # Dropout to reduce overfitting
        self.fc1 = nn.Linear(128 * 7 * 7, 128)  # Fully connected layer with 128 channels × 7 width × 7 height,  after pooling and stride to reduce feature map, and 128 neurons in the fully connected layer.
        self.fc2 = nn.Linear(128, 10)  #FINAL forward network layer 10 outputs for 10 classes
        
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)
        self.batchnorm3 = nn.BatchNorm2d(num_features=32)
        self.batchnorm4 = nn.BatchNorm2d(num_features=64)
        self.batchnorm5 = nn.BatchNorm2d(num_features=128)
    def forward(self, x):
        nn = self.conv1(x) # apply conv1 to edtract features from input data
        nn = self.batchnorm1(nn)
        nn = F.relu(nn) # apply relu activation function
        nn = F.max_pool2d(nn, 2) # apply pooling to reduce feature map by taking the maximum value in each 2x2 region of the feature maps.
        
        nn = self.conv2(nn) # apply conv2 to on output of previous layer
        nn = self.batchnorm2(nn)
        nn = F.relu(nn) 
        nn = F.max_pool2d(nn, 2) 
        
        nn = self.conv3(nn) # apply conv3 to on output of previous layer
        nn = self.batchnorm3(nn)
        nn = F.relu(nn) 
        nn = F.max_pool2d(nn, 2) 
        
        nn = self.conv4(nn) # apply conv4 to on output of previous layer
        nn = self.batchnorm4(nn)
        nn = F.relu(nn) 
        nn = F.max_pool2d(nn, 2) 
        
        nn = self.conv5(nn) # apply conv4 to on output of previous layer
        nn = self.batchnorm5(nn)
        nn = F.relu(nn) 
        nn = F.max_pool2d(nn, 2)
        nn = self.dropout(nn)
       
        nn = torch.flatten(nn, 1)
        nn = self.fc1(nn)
        nn = F.relu(nn)
        output = self.fc2(nn)  # Output layer
        return output

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

