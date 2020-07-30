import os
import csv
import time
import flask
import pathlib
import werkzeug
import numpy as np
from random import randint
from skimage import io, transform
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


load_net = 1

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
  
        return x,x1



class trafic_sign_dataset(Dataset):

    def __init__(self,image_path,shape,transform=None):

        self.transform = transform
        self.shape = shape
        paths = []
        paths.append(image_path)
        print (paths)
        labels = [16]
        self.paths = paths
        self.labels = labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self,idx):
        
        img1_name = self.paths[0]
        print (img1_name,'from getitem')
        image1 = io.imread(img1_name)
        image1 = transform.resize(image1,(48,48))
        image1 = torch.Tensor.float(torch.from_numpy(image1))
        image1 = (torch.Tensor.permute(image1,(2,0,1)))
        image1 = image1/256.0

        annot = self.labels[0]
        # annot = annot.astype('float').reshape(-1,2)
        sample = {'image1':image1,'lab' : annot}
        return sample


    app = flask.Flask(__name__)
    @app.route('/', methods = ['GET', 'POST'])
    def handle_request():
            files_ids = list(flask.request.files)
            print("\nNumber of Received Images : ", len(files_ids))
            image_num = 1
            for file_id in files_ids:
                print("\nSaving Image ", str(image_num), "/", len(files_ids))
                imagefile = flask.request.files[file_id]
                filename = werkzeug.utils.secure_filename(imagefile.filename)
                print("Image Filename : " + imagefile.filename)
                timestr = time.strftime("%Y%m%d-%H%M%S")
                imagefile.save(timestr+'_'+filename)
                image_num = image_num + 1
            print("\n")
            use_cuda = torch.cuda.is_available()
            device = torch.device( "cpu")
            shape = (48,48)
            file_path = str((os.path.join(str(pathlib.Path().absolute()), timestr+'_'+filename)))
            print (file_path,'from flask')
            dataset = trafic_sign_dataset(file_path,shape)
            print('processing the image')
            net = Net().to(device)
            checkpoint = torch.load('/home/aymenlinux/CNN_Image_retrieval/checkpoint/checkpoint.pth.tar')
            net.load_state_dict ( checkpoint['state_dict'])
            optimizer = optim.Adam(net.parameters(),lr = 0.001, weight_decay = 0.0005)
            checkpoint= {'epoch':0}
            train_dataloader = DataLoader(dataset,batch_size=1,shuffle = True)

            err = []
            acc = []
            total = 0
            correct = 0
            x11 = []
            criterion = nn.CrossEntropyLoss()
            for i,data in enumerate(train_dataloader):
                input1 , label = data.items()
                input1 = input1[1].to(device)
                label1 = label[1].to(device)
                label1 = torch.Tensor.long(label1)
                output,x1 = net(input1)
                x11.append(x1)
                _,predicted = torch.max(output.data,1)
                total +=label1.size(0)
                correct += (predicted == label1).sum().item()
                if (i%50==0):
                    print('Accuracy: %d %%'%(100*correct/total))
                acc.append(100*correct/total)
            return str('Accuracy: %d %%'%(acc[0]))
 


if __name__ == "__main__":

    
    dd = trafic_sign_dataset
    dd.app.run(host='192.168.1.48', port=33, debug=True) #Start the server
    
    

    

    
