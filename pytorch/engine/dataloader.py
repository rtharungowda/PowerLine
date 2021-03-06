import pandas as pd
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

class powerline(Dataset):
    def __init__ (self, dataframe, transform):
        self.df = dataframe
        self.tf = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,ind):
        x = Image.open(self.df['path'].iloc[ind])
        # x = self.tf(x)
        x = np.array(x)
        x = self.tf(image=x)['image']
        y = self.df['label'].iloc[ind]
        return x, y
    
def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)*255
    inp = inp.astype(np.uint8)
    # print(inp)
    img = Image.fromarray(inp)
    img.save('vis.png')

def albu():
    transform ={"train":A.Compose([
                                    A.Resize(width=224, height=224),
                                    A.Rotate(always_apply=False, p=1.0, limit=(-24, 24), interpolation=0,
                                            border_mode=0, value=(0, 0, 0), mask_value=None),
                                    A.HorizontalFlip(always_apply=False, p=0.5),
                                    A.VerticalFlip(always_apply=False, p=0.5),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ToTensorV2(),
                                ]),
                "val":A.Compose([
                                    A.Resize(width=224, height=224),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ToTensorV2()
                                ])
                } 
    return transform

def transformer():
    transform = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        'val' : transforms.Compose([
            transforms.Resize(224), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    }
    return transform

def loader(path,ratio):
    df = pd.read_csv(path)

    train_df,valid_df = train_test_split(df,stratify=df['label'],test_size=ratio,random_state=4)

    dfs = {
        'train':train_df,
        'val' :valid_df
    }

    transform = albu()

    img_datasets = {
        x:powerline(dfs[x],transform[x])
        for x in ['train','val']
    }

    dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}

    print(dataset_sizes)

    dataloaders = {
        x: DataLoader(img_datasets[x], batch_size=16,shuffle=True, num_workers=2)
        for x in ['train', 'val']}
    
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out)

    return dataloaders,dataset_sizes

if __name__=='__main__':
    _,_ =loader("/content/drive/MyDrive/competitions/recog-r2/train.csv",0.2)
     