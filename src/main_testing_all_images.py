#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import os
from skimage import io, transform
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
file_path ='/home/aymenlinux/CNN_Image_retrieval/labels/path_labels_all_test.txt'
#file_path ='/home/jamal/Downloads/GTSRB_Final_Test_Images/GTSRB/Final_Test/path_labels_one_test.txt'


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
        #
        # x = self.n31(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,3)
        x = x.view(-1,128)


        # y = self.conv12(y)
        #
        # y = self.n12(y)
        # y = F.relu(y)
        # # y = F.max_pool2d(y,5,stride = 1)
        # y = self.conv22(y)
        # y = self.n22(y)
        # y = F.relu(y)
        # # y = F.max_pool2d(y,5,stride = 2)
        #
        # #y = self.conv32(y)
        # #y = F.max_pool2d(y,3)
        # #y = self.n32(y)
        # #y = F.relu(y)
        # y = y.view(-1,1200)
        #
        # x = torch.cat((x,y),1)
        x = self.ip1(x)
        x = F.relu(x)
        x = self.ip2(x)
        # x = F.relu(x)
        # x = F.softmax(x,1)
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


net = Net().to(device)

checkpoint = torch.load('/home/jamal/Downloads/GTSRB_Final_Training_Images/GTSRB/Final_Training/checkpoint.pth.tar')
net.load_state_dict ( checkpoint['state_dict'])
optimizer = optim.Adam(net.parameters(),lr = 0.001, weight_decay = 0.0005)
checkpoint= {'epoch':0}

# optimizer = optim.Adam(net.parameters(),lr = 0.01, weight_decay = 0.0005)


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
    output = net(input1)
    _,predicted = torch.max(output.data,1)
    total +=label1.size(0)
    correct += (predicted == label1).sum().item()
    if (i%50==0):
        print('Accuracy: %d %%'%(100*correct/total))
    acc.append(100*correct/total)

    # if (i%50==0):
    #     val_index1 = np.random.permutation(val_index)[:100]
    #     val_dataloader = DataLoader(dataset,batch_size=30,sampler = SubsetRandomSampler(val_index1))
    #     val_iter = iter(val_dataloader)
    #
    #     total = 0
    #     correct = 0
    #     for j,dataj in enumerate(val_dataloader):
    #         input1j, input2j, labelj = dataj.items()
    #         input1j, input2j = input1j[1].to(device), input2j[1].to(device)
    #
    #         labelj = labelj[1].to(device)
    #         output = net(input1j,input2j)
    #         _,predicted = torch.max(output.data,1)
    #         total +=labelj.size(0)
    #         correct += (predicted == labelj).sum().item()
    #     print('Accuracy: %d %%'%(100*correct/total))
    #     acc.append(100*correct/total)








# sample = dataset[1]
# img = plt.imread(images_path+'000001.png')
# img2 = np.expand_dims(img,0)
# img3 = np.rollaxis(img2,3,1)
# output = Net(torch.from_numpy(img3))
