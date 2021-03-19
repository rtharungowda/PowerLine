import sys
sys.path.insert(1,'/content/PowerLine/utils')
from tools import load_ckp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import pandas as pd
import glob
from PIL import Image
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_pred(model,path):
    files = glob.glob(path+'/*.bmp')
    model.eval()
    data = {
        "image file name":[],
        "Powerline":[], 
    }
    for i,f in enumerate(files):
        img = Image.open(f)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        new_img = transform(img).unsqueeze(0)
        mew_img = new_img.to(DEVICE)
        with torch.no_grad():
            outputs = model(new_img)
            _, preds = torch.max(outputs,1)
        file_name = os.path.basename(f)
        print(i,file_name)
        data["image file name"].append(file_name)
        if preds[0] == 0:
            data["Powerline"].append("NO")
        else:
            data["Powerline"].append("YES")

    df = pd.DataFrame(data,index=False)
    df.reset_index(drop=True, inplace=True)
    df = df.drop("Unnamed: 0",axis=1)
    df.to_csv("/content/drive/MyDrive/competitions/recog-r2/submit_2.csv")    

if __name__=="__main__":
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    
    model_ft.fc = nn.Linear(num_ftrs, 2)

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    model, _, epoch, val_acc = load_ckp("/content/drive/MyDrive/competitions/recog-r2/resnet18_tts_2.pt", model_ft, optimizer_ft, DEVICE)
    make_pred(model,"/content/drive/MyDrive/competitions/recog-r2/Data/test")