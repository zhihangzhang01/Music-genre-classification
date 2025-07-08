from __future__ import print_function
import os
import argparse
from torchaudio.models import wav2vec2_base
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
'''Part of code refered from
https://pytorch.org/audio/0.9.0/_modules/torchaudio/models/wav2vec2/model.html#Wav2Vec2Model
https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/model.html#wav2vec2_base
https://pytorch.org/audio/main/generated/torchaudio.models.Wav2Vec2Model.html
https://pytorch.org/audio/stable/generated/torchaudio.models.wav2vec2_model.html#torchaudio.models.wav2vec2_model
'''
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.model = wav2vec2_base()
        # A linear layer to output the probabilities for each genre
        self.fc1 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(768, 10)
        self.batchnorm1 = nn.BatchNorm1d(num_features=768)
        
    def forward(self, input_values):
    # Check and remove unnecessary channel dimension
        if input_values.dim() == 3 and input_values.size(1) == 1:
            input_values = input_values.squeeze(1)  # Removes the second dimension
        
        # Pass the input through Wav2Vec2base
        outputs = self.model(input_values)
        
        # If the model returns a tuple, take the first element (the output tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Pooling method using mean taking mean value from each feature dimension, the size of hidden state is 768
        outputs = torch.mean(outputs, dim=1)
        outputs = self.dropout(outputs)
        outputs = self.fc1(outputs)
        outputs = self.batchnorm1(outputs)
        outputs = torch.relu(outputs)
        outputs = self.dropout(outputs)
        # Pass through the classifier to get genre predictions (fine-tune)
        outputs = self.fc2(outputs)
        return outputs
        
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

