import sys
sys.path.insert(1,'/content/PowerLine/utils')
from tools import load_ckp,save_ckp

import sys
sys.path.insert(1,'/content/PowerLine/engine')
from dataloader import loader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from efficientnet_pytorch import EfficientNet
import pretrainedmodels

from PIL import Image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 25

class Ens(nn.Module):
    def __init__(self,input):
        super(Ens,self).__init__()
        self.l1 = nn.Linear(2*input,16)
        self.l2 = nn.Linear(16,2)
    
    def forward(self,x):
        return self.l2(self.l1(x))

def train_model(model, model1,model2,model3, criterion, optimizer, scheduler, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_p = {'train':[],'val':[]}
    acc_p = {'train':[],'val':[]}

    model1.eval()
    model2.eval()
    model3.eval()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders["train"]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
            
                with torch.no_grad():
                    output1 = model1(inputs)
                    output2 = model2(inputs)
                    output3 = model3(inputs)
                ot = torch.cat([output1,output2,output3],dim=1)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(ot)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scaler.scale(loss).backward()
                        # scaler.step(optimizer)
                        # scaler.update()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                print("running loss ",running_loss)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            loss_p[phase].append(epoch_loss)
            acc_p[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                checkpoint = {
                    'epoch': epoch,
                    'valid_acc': best_acc,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                checkpoint_path = "/content/drive/MyDrive/competitions/recog-r2/ens_1.pt"
                save_ckp(checkpoint, checkpoint_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    plot(loss_p,acc_p,num_epochs)

    return model, best_acc

def make_pred(model,path):
    files = glob.glob(path+'/*.bmp')
    model.eval()
    data = {
        "image file name":[],
        "Powerline":[], 
    }
    # files = files[:3]
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

    df = pd.DataFrame(data)
    df.reset_index(drop=True, inplace=True)
    # df = df.drop("Unnamed: 0",axis=1)
    df.to_csv("/content/drive/MyDrive/competitions/recog-r2/submit_4.csv")    

def mdl(type):
    if type == "res18":
        model_ft = models.resnet18(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)

        return model_ft

    elif type == "res50":
        model_ft = models.resnet50(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)

    elif type == "eff-b6":
        model = EfficientNet.from_name('efficientnet-b6', num_classes=2)
        return model

    elif type == "eff-b3":
        model = EfficientNet.from_name('efficientnet-b3', num_classes=2)
        return model

    elif type == "dns201":
        model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet201', pretrained=False)
        return model

    elif type == "rsnxt-50":
        model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](num_classes=2,pretrained=None)
        return model

if __name__=="__main__":
    dataloaders,dataset_sizes = loader("/content/drive/MyDrive/competitions/recog-r2/train.csv",0.2)
    model1 = mdl('eff-b6')
    model2 = mdl('eff-b3')
    model3 = mdl('res18')

    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    optimizer3 = optim.Adam(model3.parameters(), lr=0.001)

    model1, _, epoch, val_acc = load_ckp("/content/drive/MyDrive/competitions/recog-r2/eff_b6_tts_2_albu.pt", model1, optimizer1, DEVICE)
    model2, _, epoch, val_acc = load_ckp("/content/drive/MyDrive/competitions/recog-r2/eff_b3_tts_2_albu.pt", model2, optimizer2, DEVICE)
    model3, _, epoch, val_acc = load_ckp("/content/drive/MyDrive/competitions/recog-r2/resnet18_ttsplit.pt", model3, optimizer3, DEVICE)
    # make_pred(model,"/content/drive/MyDrive/competitions/recog-r2/Data/test")
    model1.to(DEVICE)
    model2.to(DEVICE)
    model3.to(DEVICE)

    mdl = Ens(3).to(DEVICE)
    optimizer = optim.Adam(mdl.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model_ft, best_acc = train_model(mdl, model1,model2,model3, criterion, optimizer, exp_lr_scheduler,dataset_sizes,num_epochs=EPOCHS)
    # mdl = Ens(3)
    