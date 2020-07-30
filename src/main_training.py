#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import os
from random import randint
from skimage import io, transform
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def save_checkpoint(state, filename='/home/jamal/Downloads/GTSRB_Final_Training_Images/GTSRB/Final_Training/checkpoint.pth.tar'):
    torch.save(state, filename)



file_path ='/home/jamal/Downloads/GTSRB_Final_Training_Images/GTSRB/Final_Training/path_labels_all_train.txt'


load_net = 0
lr = 0.01

import torch
import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.conv11 = nn.Conv2d(3,64,3)
        self.n11 = nn.BatchNorm2d(64)
        self.conv21 = nn.Conv2d(64,96,3)
        self.n21 = nn.BatchNorm2d(96)
        self.conv31 = nn.Conv2d(96,128,4)

        self.ip1 = nn.Linear(128,128)
        self.ip2 = nn.Linear(128,43)


    def forward(self,x):
        x = self.conv11(x)
        x = self.n11(x)
        x = F.relu(x)

        x = F.max_pool2d(x,3,stride = 3)


        x = self.conv21(x)

        x = self.n21(x)
        x = F.relu(x)
        x = F.max_pool2d(x,3,stride = 3)
        x = self.conv31(x)
        #
        # x = self.n31(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,3)
        x = x.view(-1,128)
        x = self.ip1(x)
        x = F.relu(x)
        x = self.ip2(x)
        return x



class trafic_sign_dataset(Dataset):

    def __init__(self,images_labels_txt,shape,transform=None):
        self.txt_file = open(images_labels_txt,'r')
        # self.root_dir = root_dir
        self.transform = transform
        self.shape = shape
        lst = self.txt_file.readlines()
        self.txt_file.close()
        paths = [item.rstrip().split('\t')[0] for item in lst]
        labels = [int(item.rstrip().split('\t')[1])  for item in lst]
        self.paths = paths
        self.labels = labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self,idx):
        img1_name = self.paths[idx]
        image1 = io.imread(img1_name)
        image1 = transform.resize(image1,shape)
        image1 = torch.Tensor.float(torch.from_numpy(image1))
        image1 = (torch.Tensor.permute(image1,(2,0,1)))
        image1 = image1/256.0

        annot = self.labels[idx]
        # annot = annot.astype('float').reshape(-1,2)
        sample = {'image1':image1,'lab' : annot}
        return sample

use_cuda = torch.cuda.is_available()
device = torch.device( "cpu")
shape = (48,48)

dataset = trafic_sign_dataset(file_path,shape)
if load_net:
    net = Net().to(device)
    checkpoint = torch.load('/home/jamal/Downloads/GTSRB_Final_Training_Images/GTSRB/Final_Training\checkpoint.pth.tar')
    net.load_state_dict ( checkpoint['state_dict'])
    optimizer = optim.Adam(net.parameters(),lr = lr, weight_decay = 0.0005)
    optimizer.load_state_dict = checkpoint['optimizer']
else:

    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(),lr = lr, weight_decay = 0.0005)
    checkpoint= {'epoch':0}




optimizer = optim.Adam(net.parameters(),lr = 0.01, weight_decay = 0.0005)


train_dataloader = DataLoader(dataset,batch_size=32,shuffle = True)

err = []
acc = []

criterion = nn.CrossEntropyLoss()
for ep in range(checkpoint['epoch'],10):
    for i,data in enumerate(train_dataloader):
        optimizer.zero_grad()
        input1 , label = data.items()
        input1 = input1[1].to(device)
        label1 = label[1].to(device)
        label1 = torch.Tensor.long(label1)
        output = net(input1)
        loss = criterion (output,label1)
        loss.backward()
        optimizer.step()
        err.append(loss.item())
        if (i%5==0):
            print (loss.item(),i,ep)

    save_checkpoint({
            'epoch': ep + 1,
            'state_dict': net.state_dict(),

            'optimizer' : optimizer.state_dict(),
        })
