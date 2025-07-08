from __future__ import print_function
import os
import time
import json
import argparse
import shutil
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import soundfile as sf

import transformers
from transformers import AutoModelForAudioClassification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import models
from utils import TabularDatasets
from utils import ImageDatasets
from utils import AudioDatasets
from utils.params import Params
from utils.plotting import plot_training

'''Some code refered from the sample project provided by lecturer Cunningham Eoghan from moodle'''
def main():
    start_time = time.strftime("%d%m%y_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", 
        type=str, 
        help="Pass name of model as defined in hparams.yaml."
        )
    parser.add_argument(
        "--write_data",
        required = False,
        default=False,
                help="Set to true to write_data."
        )
    args = parser.parse_args()
    # Parse our YAML file which has our model parameters. 
    params = Params("hparams.yaml", args.model_name)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu_vis_dev
    # Check if a GPU is available and use it if so. 
    use_gpu= torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # Load model that has been chosen via the command line arguments. 
    model_module = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    model = model_module.net()
    # Send the model to the chosen device. 
    # To use multiple GPUs
    # model = nn.DataParallel(model)
    model.to(device)
    # Grap your training and validation functions for your network.
    train = model_module.train
    val = model_module.val
    
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    
    # Write data if specified in command line arguments. 
    if args.write_data:
        # Split tabular data into train and test
        data = pd.read_csv('Data\\features_30_sec.csv')
        random_seed = 2023
        shuffled_data = shuffle(data, random_state=random_seed)
        split_index = int(0.8 * len(shuffled_data))
        train_data = shuffled_data[:split_index]
        test_data = shuffled_data[split_index:]
        train_data.to_csv('Data\\features_30_sec_train.csv', index=False)
        test_data.to_csv('Data\\features_30_sec_test.csv', index=False)
        
        data = pd.read_csv('Data\\features_30_sec_train.csv')
        test_data = pd.read_csv('Data\\features_30_sec_test.csv')
        val_split = round(data.shape[0]*0.2)
        data = shuffle(data)
        train_data = data.iloc[val_split:]
        val_data = data.iloc[:val_split]
        train_data.to_csv(os.path.join(params.data_dir, "train.csv"), index=False)
        val_data.to_csv(os.path.join(params.data_dir, "val.csv"), index=False)
        test_data.to_csv(os.path.join(params.data_dir, "test.csv"), index=False)
        
        
        
        
        # Walk through the directory to remove the white board from each of image
        target_width = 315
        target_height = 217
        for subdir, dirs, files in os.walk('Data\\images_original'):
            for file in files:
                if file.endswith('.png'): 
                    filepath = os.path.join(subdir, file)
                    img = Image.open(filepath)
                    width, height = img.size
                    left = (width - target_width) // 2
                    top = (height - target_height) // 2
                    right = (width + target_width) // 2
                    bottom = (height + target_height) // 2
                    cropped_img = img.crop((left, top, right, bottom))
                    directory_path = os.path.dirname(filepath)
                    last_directory_name = os.path.basename(directory_path)
                    save_dir = os.path.join('Data\\images_cropped', last_directory_name)
                    print(save_dir)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, file)    
                    cropped_img.save(save_path)
        

        # Split image data into train and test
        np.random.seed(2023)
        images = [] # To store tuples of (image_path, class)
        
        # Collect all images and their labels
        for cls in os.listdir('Data\\images_cropped'):
            class_dir = os.path.join('Data\\images_cropped', cls)
            for img in os.listdir(class_dir):
                if img.endswith('.png'):
                    images.append((os.path.join(class_dir, img), cls))
                        
        # Shuffle all images together,
        images = shuffle(images)
        # Calculate split index
        split_index = int(len(images) * (1 - 0.2))
        
        # Create train and test directories
        train_dir = os.path.join('Data\\images_cropped', 'train')
        test_dir = os.path.join('Data\\images_cropped', 'test')
        val_dir = os.path.join('Data\\images_cropped', 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Create class subdirectories in train and test directories
        classes = set([cls for _, cls in images])
        for cls in classes:
            os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
            os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        
        # Distribute image files into train and test directories
        train_dt = images[:split_index]
        test_dt = images[split_index:]
        for img, cls in test_dt:
            shutil.copy(img, os.path.join(test_dir, cls))

        # Distribute image files into validation and train directories
        train_dt = shuffle(train_dt)
        split_index = int(len(train_dt) * (1 - 0.2))
        val_dt = train_dt[split_index:]
        train_dt = train_dt[:split_index]
        for img, cls in train_dt:
            shutil.copy(img, os.path.join(train_dir, cls))
        for img, cls in val_dt:
            shutil.copy(img, os.path.join(val_dir, cls))
            
        
        
        for subdir, dirs, files in os.walk('Data\\genres_original'):
            for file in files:
                if file.endswith('.wav'):  # Check for WAV files
                    filepath = os.path.join(subdir, file)
                    try:
                        data, samplerate = sf.read(filepath)  # Attempt to read the file
                    except RuntimeError as e:  # Adjust the exception type based on the library's specifics
                        print(f"Corrupted file detected: {filepath}. Error: {e}")
                        os.remove(filepath)  # Delete the problematic file
                        print(f"Deleted {filepath}")
        
        
        
        
        # Split audio data into train and test
        np.random.seed(2023)
        audios = [] # To store tuples of (audio_path, class)
        
        # Collect all audios and their labels
        for cls in os.listdir('Data\\genres_original'):
            class_dir = os.path.join('Data\\genres_original', cls)
            for audio in os.listdir(class_dir):
                if audio.endswith('.wav'):
                    audios.append((os.path.join(class_dir, audio), cls))
                        
        # Shuffle all audios together,
        audios = shuffle(audios)
        # Calculate split index
        split_index = int(len(audios) * (1 - 0.2))
        
        # Create train and test directories
        train_dir = os.path.join('Data\\genres_original', 'train')
        test_dir = os.path.join('Data\\genres_original', 'test')
        val_dir = os.path.join('Data\\genres_original', 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Create class subdirectories in train and test directories
        classes = set([cls for _, cls in audios])
        for cls in classes:
            os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
            os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        
        # Distribute audio files into train and test directories
        train_dt = audios[:split_index]
        test_dt = audios[split_index:]
        for audio, cls in test_dt:
            shutil.copy(audio, os.path.join(test_dir, cls))

        # Distribute audio files into validation and train directories
        train_dt = shuffle(train_dt)
        split_index = int(len(train_dt) * (1 - 0.2))
        val_dt = train_dt[split_index:]
        train_dt = train_dt[:split_index]
        for audio, cls in train_dt:
            shutil.copy(audio, os.path.join(train_dir, cls))
        for audio, cls in val_dt:
            shutil.copy(audio, os.path.join(val_dir, cls))
        

    # This is useful if you have multiple custom datasets defined. 
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
    val_loader = DataLoader(
        val_data,
        batch_size=params.batch_size,
        shuffle=False
    )
    if not os.path.exists(params.log_dir): os.makedirs(params.log_dir)
    if not os.path.exists(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)
    if not os.path.exists("figs"): os.makedirs("figs")

    val_accs = []
    val_losses = []
    train_losses = []
    train_accs = []
    for epoch in range(1, params.num_epochs + 1):
        print("Epoch: {}".format(epoch))
        # Call training function. 
        train(model, device, train_loader, optimizer)
        # Evaluate on both the training and validation set. 
        train_loss, train_acc = val(model, device, train_loader)
        val_loss, val_acc = val(model, device, val_loader)
        # Collect some data for logging purposes. 
        train_losses.append(float(train_loss))
        train_accs.append(train_acc)
        val_losses.append(float(val_loss))
        val_accs.append(val_acc)

        print('\n\ttrain Loss: {:.6f}\ttrain acc: {:.6f} \n\tval Loss: {:.6f}\tval acc: {:.6f}'.format(train_loss, train_acc, val_loss, val_acc))
        # Here is a simply plot for monitoring training. 
        # Clear plot each epoch 
        fig = plot_training(train_losses, train_accs,val_losses, val_accs)
        fig.savefig(os.path.join("figs", "{}_training_vis".format(args.model_name)))
        # Save model every few epochs (or even more often if you have the disk space).
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(params.checkpoint_dir,"checkpoint_{}_epoch_{}".format(args.model_name,epoch)))
    # Some log information to help you keep track of your model information. 
    logs ={
        "model": args.model_name,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "best_val_epoch": int(np.argmax(val_accs)+1),
        "model": args.model_name,
        "lr": params.lr,
        "batch_size":params.batch_size,
        "weight_decay":params.weight_decay
    }

    with open(os.path.join(params.log_dir,"{}_{}.json".format(args.model_name,  start_time)), 'w') as f:
        json.dump(logs, f)



if __name__ == '__main__':
    main()
