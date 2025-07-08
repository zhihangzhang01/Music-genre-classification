from __future__ import print_function
import os
import time
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

import models 
from utils import TabularDatasets
from utils import ImageDatasets
from utils import AudioDatasets
from utils.params import Params
'''Code refered from the sample project provided by lecturer Cunningham Eoghan from moodle'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "val_json", type=str, help="Directory of validation json file which indictates the best epoch.")
    parser.add_argument(
        "eval_iter", type=int, default=5, help="Number of times to train and evaluate model")
    args = parser.parse_args()

    with open(args.val_json) as f:  
        model_params  = json.load(f)  

    params = Params("hparams.yaml", model_params["model"])
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu_vis_dev
    use_gpu= torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    log_dir = os.path.join(params.log_dir, "eval_logs")
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    model_module = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    
    if params.model_name == 'mlp':
        train_data = TabularDatasets.TabularGTZANDataset(params.data_dir,"train.csv", apply_normalization=params.apply_normalization)
        val_data = TabularDatasets.TabularGTZANDataset(params.data_dir,"val.csv", apply_normalization=params.apply_normalization)
    elif params.model_name == 'cnn':
        train_data = ImageDatasets.ImageGTZANDataset(params.data_dir,"train", transform=params.transform)
        val_data = ImageDatasets.ImageGTZANDataset(params.data_dir,"val", transform=params.transform)
    else:
        train_data = AudioDatasets.Wav2Vec2Dataset(params.data_dir,"train", sample_rate=params.sample_rate)
        val_data = AudioDatasets.Wav2Vec2Dataset(params.data_dir,"val", sample_rate=params.sample_rate)
    
    
    
    acc_scores = []
    for iter_i in range(args.eval_iter):
        print("Training model for iteration {}...".format(iter_i))
        model = model_module.net().to(device)
        train = model_module.train
        test = model_module.test

        optimizer = optim.Adam(model.parameters(), lr=model_params['lr'])
        
        
        if params.model_name == 'mlp':
            train_data = TabularDatasets.TabularGTZANDataset(params.data_dir,"train.csv", apply_normalization=params.apply_normalization)
            val_data = TabularDatasets.TabularGTZANDataset(params.data_dir,"val.csv", apply_normalization=params.apply_normalization)
        elif params.model_name == 'cnn':
            train_data = ImageDatasets.ImageGTZANDataset(params.data_dir,"train", transform=params.transform)
            val_data = ImageDatasets.ImageGTZANDataset(params.data_dir,"val", transform=params.transform)
        else:
            train_data = AudioDatasets.Wav2Vec2Dataset(params.data_dir,"train", sample_rate=params.sample_rate)
            val_data = AudioDatasets.Wav2Vec2Dataset(params.data_dir,"val", sample_rate=params.sample_rate)
        
        
        train_loader = DataLoader(
            train_data, 
            batch_size=params.batch_size,
            shuffle=True
            )
        if not os.path.exists(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)
        for epoch in range(1, model_params["best_val_epoch"] + 1):
            train(model, device, train_loader, optimizer)
        # Just save the last epoch of each iteration.
        torch.save(
            model.state_dict(), os.path.join(
                params.checkpoint_dir,
                "checkpoint_{}_epoch_{}_iter_{}".format(
                model_params["model"], 
                epoch,
                iter_i
                )
            )
        )
        print("Evaluating model for iteration {}...".format(iter_i))

        if params.model_name == 'mlp':
            test_data = TabularDatasets.TabularGTZANDataset(params.data_dir,"test.csv", apply_normalization=params.apply_normalization)
        elif params.model_name == 'cnn':
            test_data = ImageDatasets.ImageGTZANDataset(params.data_dir,"test", transform=params.transform)
        else:
            test_data = AudioDatasets.Wav2Vec2Dataset(params.data_dir,"test", sample_rate=params.sample_rate)
        
        test_loader = DataLoader(
            test_data, 
            batch_size=params.batch_size,
            shuffle=False
            )
        
        acc_score = test(model, device, test_loader)
        print("Accuracy for iteration {}\t {}".format(iter_i, acc_score))

        acc_scores.append(float(acc_score))
    logs ={
            "model": model_params["model"], 
            "num_epochs": model_params["best_val_epoch"],
            "lr": model_params['lr'], 
            "batch_size": model_params["batch_size"],
            "eval_iterations": args.eval_iter,
            "acc_scores": acc_scores,
            "mean_acc": float(np.mean(acc_scores)),
            "var_acc": float(np.var(acc_scores)),
            }

    with open(
        os.path.join(log_dir, "{}_{}.json".format(model_params["model"], args.eval_iter)), 'w') as f:
        json.dump(logs, f)

if __name__ == '__main__':

    main()