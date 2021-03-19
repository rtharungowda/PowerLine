import pandas as pd
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch

def plot(loss_p,acc_p,epochs):
    x = [i for i in range(epochs)]
    plt.plot(x,loss_p['train'],color='red', marker='o')
    plt.title('Train loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True) 
    plt.savefig('/content/PowerLine/utils/train_loss.png')
    plt.clf()

    plt.plot(x, loss_p['val'],color='red', marker='o')
    plt.title('Val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True) 
    plt.savefig('/content/PowerLine/utils/val_loss.png')
    plt.clf()
    
    #acc
    plt.plot(x, acc_p['train'],color='red', marker='o')
    plt.title('Train acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.grid(True) 
    plt.savefig('/content/PowerLine/utils/train_acc.png')
    plt.clf()

    plt.plot(x, acc_p['val'],color='red', marker='o')
    plt.title('Val acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.grid(True) 
    plt.savefig('/content/PowerLine/utils/val_acc.png')
    plt.clf()

def save_ckp(state, checkpoint_path):
    f_path = checkpoint_path
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer, device):

    checkpoint = torch.load(checkpoint_fpath,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_acc = checkpoint['valid_acc'] 
    return model, optimizer, checkpoint['epoch'], valid_acc

def vis(path):
    img = Image.open(path)
    print(img.size)
    img.save('image1.png')

def create_csv(path):
    no_pl = os.path.join(path,"train/No_powerline")
    n_files = glob.glob(no_pl+'/*.bmp')
    pl = os.path.join(path,"train/Powerline")
    p_files = glob.glob(pl+'/*.bmp')
    tst = os.path.join(path,"test")
    t_files = glob.glob(tst+'/*.bmp')
    print(no_pl,len(n_files))
    print(pl,len(p_files))
    print(tst,len(t_files))

    path=[]
    label=[]

    for f in n_files:
        path.append(f)
        label.append(0)

    for f in p_files:
        path.append(f)
        label.append(1)

    data = {
        'path':path,
        'label':label
    }

    df = pd.DataFrame(data)
    df.to_csv('/content/drive/MyDrive/competitions/recog-r2/train.csv')

if __name__=="__main__":
    vis("/content/drive/MyDrive/competitions/recog-r2/Data/test/Check   (1).bmp")