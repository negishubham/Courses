#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:14:15 2020

@author: snegi
"""


import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.datasets as datasets

from torchvision import transforms as tvt
from torch.utils.data import DataLoader
import pdb

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

#Parameters
lr 	 = 1e-3
momentum = 0.9
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 20

transform_train = tvt.Compose([
            tvt.RandomResizedCrop(224),
            tvt.RandomHorizontalFlip(),
            tvt.ToTensor(),
            tvt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

transform_test = tvt.Compose([
            tvt.RandomResizedCrop(224),
            tvt.ToTensor(),
            tvt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])


#Reference[1] The below imagenet images were downloaded using the below code:
#https://github.com/mf1024/ImageNet-Datasets-Downloader
path_train = '//home/min/a/snegi/Lab_Assignment/data_root_folder/imagenet/imagenet_images_train/'
path_test  =  '/home/min/a/snegi/Lab_Assignment/data_root_folder/imagenet/imagenet_images_test/'
trainset = torchvision.datasets.ImageFolder(path_train, transform_train)
testset = torchvision.datasets.ImageFolder(path_test, transform_test)



trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

'''
variantn is shown with comment '#variantn' on the line
lines with no comment means they are present in all the variants
'''
'''
variant1 has one skip connection and the skip conneciton is added before the relu operation
Train Accuracy: 60.73648971783835 % Train Loss: 0.031183738317905052
Test Accuracy: 60.2015113350126  
'''

'''
variant2 has two skip connections and the skip connection is added before the relu operation
Train Accuracy: 62.17120994739359 % Train Loss: 0.03049152875036904  
Test Accuracy: 65.57514693534844
'''         

'''
variant3
same as variant1 but adding the skip connection after relu
Train Accuracy: 62.69727403156384 Train Loss: 0.030546550610751816
Test Accuracy: 70.36104114189756

'''

'''
variant4
same as variant2 but adding the skip connection after relu
Train Accuracy: 65.23194643711143 % Train Loss: 0.028495787360677212  
Test Accuracy: 60.53736356003358

'''

'''
variant5
Adding bn2 of after adding the output of conv2 and shortcut connection
loss: 0.03314016990815148 Accuracy: 58.20181731229077 
Test Acc: 60.957178841309826
'''

'''
So from above results it can be concluded that variant3 gives the best accuracy hence the below code is implemented
for variant3
'''

f = open('output.txt', 'w')
class SkipBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(SkipBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2  = nn.BatchNorm2d(out_ch)

      
        self.shortcut1 = nn.Sequential(                                         #variant1 and varian2 and variant3 and variant4
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        

#        self.shortcut2 = nn.Sequential(                                        #variant2 and variant4
#                nn.Conv2d(out_ch, out_ch, 1, stride=1, bias=False),
#                nn.BatchNorm2d(out_ch))     
        


    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out1 = out                          #variant2  #variant4
#        out = self.conv2(out) + self.shortcut1(x)    #variant5 
#        out = self.bn2(out)  #variant5
        
        out = self.bn2(self.conv2(out))
       
#        out = out + self.shortcut1(x)    #variant1 

#        out = out + self.shortcut1(x) + self.shortcut2(out1)     #variant2
        out = F.relu(out)
        out = out + self.shortcut1(x)    #variant3 
#        out = out + self.shortcut1(x) + self.shortcut2(out1)     #variant4
       
        return out

'''
A 26 layered network is made using the skip block class shown above. 
The way the image height and width are decreased as we go deeper is similar to ResNet20 paper. 
https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
Basically having conv layers with stride=2
Different variants in skip block are tried.
'''
         
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.in_planes = 16
        
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn  = nn.BatchNorm2d(16)
            
        self.skip161 = SkipBlock(16, 16, stride=1)
        self.skip162 = SkipBlock(16, 16, stride=1)
        self.skip163 = SkipBlock(16, 16, stride=1)

        self.skip321 = SkipBlock(16, 32, stride=2)      # stride of 2 taken to reduce the size of image
        self.skip322 = SkipBlock(32, 32, stride=1)
        self.skip323 = SkipBlock(32, 32, stride=1)
        
        self.skip641 = SkipBlock(32, 64, stride=2)
        self.skip642 = SkipBlock(64, 64, stride=1)
        self.skip643 = SkipBlock(64, 64, stride=1)        

        self.skip1281 = SkipBlock(64, 128, stride=2)
        self.skip1282 = SkipBlock(128, 128, stride=1)
        self.skip1283 = SkipBlock(128, 128, stride=1)
        
        self.fc2 =  nn.Linear(128, 5)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.skip161(out)
        out = self.skip162(out)
        out = self.skip163(out)
        
        out = self.skip321(out)
        out = self.skip322(out)
        out = self.skip323(out)
        
        out = self.skip641(out)
        out = self.skip642(out)
        out = self.skip643(out)        
        
        out = self.skip1281(out)
        out = self.skip1282(out)
        out = self.skip1283(out)        
        
#        print('outsize',out.size())
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc2(out)
        return out           



def training(net):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = momentum)
    for epoch in range(epochs):        
        running_loss = 0	
        train_correct = 0
        total =0
        for i, data in enumerate(trainloader):
            inputs, labels = data	
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
            prediction = torch.max(outputs, 1)  # second param "1" represents the dimension to be reduced
            total += labels.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == labels.cpu().numpy())            
                

        print('Epoch {}: loss: {} Accuracy: {}' .format(epoch+1, running_loss/((i+1)*32), 100.*train_correct / total))
        f.write('Epoch {}: {}\n'.format(epoch+1, running_loss/((i+1)*32)))

def testing(net):
    net = net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1):
        test_loss=0 
        test_correct = 0
        total =0      
        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss   = criterion(outputs, labels)
            test_loss+=loss.item()
            _,predicted = outputs.max(1) 
            total += labels.size(0)

            test_correct += np.sum(predicted[1].cpu().numpy() == labels.cpu().numpy())            

        print('Classification Accuracy: {}' .format(100.*test_correct / total))
        f.write('Classification Accuracy: {}' .format(100.*test_correct / total))

               
net = Mynet()
training(net)

testing(net) 
               
f.close()


