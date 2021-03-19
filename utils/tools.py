import pandas as pd
import os
import glob
from PIL import Image


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
    vis("/content/drive/MyDrive/competitions/recog-2/Data/train/Powerline/Powerline (1).bmp")