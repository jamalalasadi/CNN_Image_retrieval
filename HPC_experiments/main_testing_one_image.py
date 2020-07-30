#import pandas as pd
import numpy as np
from random import randint
#import matplotlib.pyplot as plt
import os
import csv
from skimage import io, transform
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


file_path ='/home/usr/CNN_Image_retrieval/labels/path_labels_one_test.txt'


load_net = 1

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

        x1 = x.view(-1,128)
        x=x1


        x = self.ip1(x)
        x = F.relu(x)
        x = self.ip2(x)
        x1_list = x1.tolist()[0]
        #file_name = '/home/jamal/Desktop/'+'x1'+str(randint(1000, 9999))+'list.csv'
        #with open(file_name, 'w', newline='') as myfile:
        #    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        #    wr.writerow(x1_list)
        return x,x1



class trafic_sign_dataset(Dataset):

    def __init__(self,images_labels_txt,shape,transform=None):
        self.txt_file = open(images_labels_txt,'r')
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
        sample = {'image1':image1,'lab' : annot}
        return sample

use_cuda = torch.cuda.is_available()
device = torch.device( "cpu")
shape = (48,48)

dataset = trafic_sign_dataset(file_path,shape)


net = Net().to(device)

checkpoint = torch.load('/home/usr/CNN_Image_retrieval/checkpoint/checkpoint.pth.tar')
net.load_state_dict ( checkpoint['state_dict'])
optimizer = optim.Adam(net.parameters(),lr = 0.001, weight_decay = 0.0005)
checkpoint= {'epoch':0}

train_dataloader = DataLoader(dataset,batch_size=1,shuffle = True)

err = []
acc = []
total = 0
correct = 0
criterion = nn.CrossEntropyLoss()
for i,data in enumerate(train_dataloader):
    input1 , label = data.items()
    input1 = input1[1].to(device)
    label1 = label[1].to(device)
    label1 = torch.Tensor.long(label1)
    output,x1 = net(input1)
    _,predicted = torch.max(output.data,1)
    total +=label1.size(0)
    correct += (predicted == label1).sum().item()
    if (i%50==0):
        print('Accuracy: %d %%'%(100*correct/total))
    acc.append(100*correct/total)
