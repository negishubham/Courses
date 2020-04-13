#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:31:14 2020

@author: snegi
"""

"""
Noise Classification Accuracy: 91.1
Noise Confusion Matrix:
                             0             20             50             80

           0:            99.90           0.10           0.00           0.00
          20:             0.10          90.60           7.30           2.00
          50:             0.00           1.30          72.50          26.20
          80:             0.00           0.00           0.50          99.50

Dataset0 Classification Accuracy: 89.2%
Dataset0 Confusion Matrix:
                      rectangle       triangle           disk           oval           star

   rectangle:            78.00           0.00           0.00          20.00           2.00
    triangle:             0.00          98.01           0.00           0.00           1.99
        disk:             0.00           0.00          94.00           2.00           4.00
        oval:            24.00           0.00           0.00          76.00           0.00
        star:             0.00           0.00           0.00           0.00         100.00

Dataset20 Classification Accuracy: 77.5%
Dataset20 Confusion Matrix:
                      rectangle       triangle           disk           oval           star

   rectangle:            70.65           3.26           8.70          11.41           5.98
    triangle:             0.56          92.66           6.78           0.00           0.00
        disk:            15.51           2.67          77.54           0.00           4.28
        oval:            39.67           2.17           0.00          52.72           5.43
        star:             3.72           0.53           0.00           1.60          94.15

Dataset50 Classification Accuracy: 69.24034869240349%
Dataset50 Confusion Matrix:
                      rectangle       triangle           disk           oval           star

   rectangle:            35.00           8.00           5.50           8.00          22.00
    triangle:             4.48          63.68           8.96           0.00           1.00
        disk:             1.00           6.50          69.00           1.50           7.50
        oval:            20.50           0.00          10.50          41.50           6.00
        star:             4.52           4.52           3.02           0.00          68.84

Dataset80 Classification Accuracy: 63.50822239624119%
Dataset80 Confusion Matrix:
                      rectangle       triangle           disk           oval           star

   rectangle:            67.45           0.39           9.80          16.08           6.27
    triangle:            14.12          70.20           6.67           0.00           9.02
        disk:            16.92           4.62          74.62           1.92           1.92
        oval:            51.19           0.79           7.14          35.71           5.16
        star:            22.35           1.18           3.14           4.31          69.02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gzip
import pickle
import numpy as np 
import os, sys
import pdb
import matplotlib.pyplot as plt
import torchvision
import copy
import math
import numbers
import random
#from hw5 import LOADnet2

seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


lr = 1e-5
momentum = 0.9
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 2
dataroot = "/local/scratch/a/snegi/PhD/Courses/Deep_Learning/Lab_Assignment/hw5/data/"
batch_size =4 
debug_train =1
debug_test =1
path_saved_model = "./saved_model"
classes = ('rectangle','triangle','disk','oval','star')


#Custom Dataloader
f = open('output.txt', 'a+')
f.write("+Task 3:\n")
# Below code is for loading all the datasets into single dataloader
# It is edited version of PurdueShape5Dataset to load all the noisy and zero noise images
class PurdueShapes5DatasetTotal(torch.utils.data.Dataset):
    def __init__(self, train_or_test, dataroot, transform=None):
        super(PurdueShapes5DatasetTotal, self).__init__()
        if train_or_test == 'train' :
            if os.path.exists("torch-saved-PurdueShapes5-40000-dataset.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map-40000.pt")and \
                      os.path.exists("torch-saved-PurdueShapes5-noiselabel.pt") :
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-40000-dataset.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map-40000.pt")
                self.noiselabel = torch.load("torch-saved-PurdueShapes5-noiselabel.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.class_noiselabels = ['0', '20', '50', '80']
                self.transform = transform
            else: 
                print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                      """the dataset for this script. First time loading could take\n"""
                      """a minute or so.  Any subsequent attempts will only take\n"""
                      """a few seconds.\n\n\n""")
                root_dir = dataroot
                f1 = gzip.open(root_dir + "PurdueShapes5-10000-train.gz", 'rb')
                f2 = gzip.open(root_dir + "PurdueShapes5-10000-train-noise-20.gz", 'rb')
                f3 = gzip.open(root_dir + "PurdueShapes5-10000-train-noise-50.gz", 'rb')
                f4 = gzip.open(root_dir + "PurdueShapes5-10000-train-noise-80.gz", 'rb')
                
                dataset1 = f1.read()
                if sys.version_info[0] == 3:
                    self.dataset1, self.label_map = pickle.loads(dataset1, encoding='latin1')
                else:
                    self.dataset1, self.label_map = pickle.loads(dataset1)

                dataset2 = f2.read()
                if sys.version_info[0] == 3:
                    self.dataset2, self.label_map = pickle.loads(dataset2, encoding='latin1')
                else:
                    self.dataset2, self.label_map = pickle.loads(dataset2)

                dataset3 = f3.read()
                if sys.version_info[0] == 3:
                    self.dataset3, self.label_map = pickle.loads(dataset3, encoding='latin1')
                else:
                    self.dataset3, self.label_map = pickle.loads(dataset3)

                dataset4 = f4.read()
                if sys.version_info[0] == 3:
                    self.dataset4, self.label_map = pickle.loads(dataset4, encoding='latin1')
                else:
                    self.dataset4, self.label_map = pickle.loads(dataset4)
                
                self.dataset = dict.fromkeys(range(40000))
                for i in range(len(self.dataset)):
                    if i>=0 and i<=9999:
                        self.dataset[i] = self.dataset1[i%len(self.dataset1)]
                    if i>=10000 and i<=19999:
                        self.dataset[i] = self.dataset2[i%len(self.dataset2)]                
                    if i>=20000 and i<=29999:
                        self.dataset[i] = self.dataset3[i%len(self.dataset3)]
                    if i>=30000 and i<=39999:
                        self.dataset[i] = self.dataset4[i%len(self.dataset4)]                        
                self.class_noiselabels = ['0', '20', '50', '80']        
                self.noiselabel = []
                self.noiselabel[0:9999] = [0]*10000
                self.noiselabel[10000:19999] = [1]*10000
                self.noiselabel[20000:29999] = [2]*10000
                self.noiselabel[30000:39999] = [3]*10000                
                torch.save(self.dataset, "torch-saved-PurdueShapes5-40000-dataset.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map-40000.pt")
                torch.save(self.noiselabel, "torch-saved-PurdueShapes5-noiselabel.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
        else:
            root_dir = dataroot
            f1 = gzip.open(root_dir + "PurdueShapes5-1000-test.gz", 'rb')
            f2 = gzip.open(root_dir + "PurdueShapes5-1000-test-noise-20.gz", 'rb')
            f3 = gzip.open(root_dir + "PurdueShapes5-1000-test-noise-50.gz", 'rb')
            f4 = gzip.open(root_dir + "PurdueShapes5-1000-test-noise-80.gz", 'rb')
            
            dataset1 = f1.read()
            if sys.version_info[0] == 3:
                self.dataset1, self.label_map = pickle.loads(dataset1, encoding='latin1')
            else:
                self.dataset1, self.label_map = pickle.loads(dataset1)

            dataset2 = f2.read()
            if sys.version_info[0] == 3:
                self.dataset2, self.label_map = pickle.loads(dataset2, encoding='latin1')
            else:
                self.dataset2, self.label_map = pickle.loads(dataset2)

            dataset3 = f3.read()
            if sys.version_info[0] == 3:
                self.dataset3, self.label_map = pickle.loads(dataset3, encoding='latin1')
            else:
                self.dataset3, self.label_map = pickle.loads(dataset3)

            dataset4 = f4.read()
            if sys.version_info[0] == 3:
                self.dataset4, self.label_map = pickle.loads(dataset4, encoding='latin1')
            else:
                self.dataset4, self.label_map = pickle.loads(dataset4)
            
            self.dataset = dict.fromkeys(range(4000))
            for i in range(len(self.dataset)):
                if i>=0 and i<=999:
                    self.dataset[i] = self.dataset1[i%len(self.dataset1)]
                if i>=1000 and i<=1999:
                    self.dataset[i] = self.dataset2[i%len(self.dataset2)]                
                if i>=2000 and i<=2999:
                    self.dataset[i] = self.dataset3[i%len(self.dataset3)]
                if i>=3000 and i<=3999:
                    self.dataset[i] = self.dataset4[i%len(self.dataset4)] 
            
                
            self.noiselabel = []
            self.noiselabel[0:999] = [0]*1000
            self.noiselabel[1000:1999] = [1]*1000
            self.noiselabel[2000:2999] = [2]*1000
            self.noiselabel[3000:3999] = [3]*1000               
            # reverse the key-value pairs in the label dictionary:
            self.class_labels = dict(map(reversed, self.label_map.items()))
            self.class_noiselabels = ['0', '20', '50', '80']
            self.transform = transform

    def __len__(self):
#        pdb.set_trace()
        return len(self.dataset)

        
    def __getitem__(self, idx):
        r = np.array(self.dataset[idx][0])
        g = np.array(self.dataset[idx][1])
        b = np.array(self.dataset[idx][2])
        R, G, B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
        
        im_tensor = torch.zeros(3,32,32, dtype = torch.float)
        im_tensor[0,:,:] = torch.from_numpy(R)
        im_tensor[1,:,:] = torch.from_numpy(G)
        im_tensor[2,:,:] = torch.from_numpy(B)
        
        bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
        sample = {'image': im_tensor,
                  'bbox': bb_tensor,
                  'label': self.dataset[idx][4],
                  'noiselabel':self.noiselabel[idx]
                }
        
        if self.transform:
            sample = self.transform(sample)
        return sample 


#This code is taken from dlstudio class: https://engineering.purdue.edu/kak/distDLS/DLStudio-1.0.7.html
class PurdueShapes5Dataset(torch.utils.data.Dataset):
    def __init__(self, train_or_test, dataset_file, dataroot, transform=None):
        super(PurdueShapes5Dataset, self).__init__()
        if train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train.gz":
            if os.path.exists("torch-saved-PurdueShapes5-10000-dataset.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
            else: 
                print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                      """the dataset for this script. First time loading could take\n"""
                      """a minute or so.  Any subsequent attempts will only take\n"""
                      """a few seconds.\n\n\n""")
                root_dir = dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
        elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-20.gz":
            if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
            else: 
                print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                      """the dataset for this script. First time loading could take\n"""
                      """a minute or so.  Any subsequent attempts will only take\n"""
                      """a few seconds.\n\n\n""")
                root_dir = dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
        elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-50.gz":
            if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
            else: 
                print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                      """the dataset for this script. First time loading could take\n"""
                      """a minute or so.  Any subsequent attempts will only take\n"""
                      """a few seconds.\n\n\n""")
                root_dir = dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
        elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-80.gz":
            if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
            else: 
                print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                      """the dataset for this script. First time loading could take\n"""
                      """a minute or so.  Any subsequent attempts will only take\n"""
                      """a few seconds.\n\n\n""")
                root_dir = dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
        else:
            root_dir = dataroot
            f = gzip.open(root_dir + dataset_file, 'rb')
            dataset = f.read()
            if sys.version_info[0] == 3:
                self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
            else:
                self.dataset, self.label_map = pickle.loads(dataset)
            # reverse the key-value pairs in the label dictionary:
            self.class_labels = dict(map(reversed, self.label_map.items()))
            self.transform = transform

    def __len__(self):
        return len(self.dataset)

        
    def __getitem__(self, idx):
        
        r = np.array(self.dataset[idx][0])
        g = np.array(self.dataset[idx][1])
        b = np.array(self.dataset[idx][2])
        R, G, B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
        
        im_tensor = torch.zeros(3,32,32, dtype = torch.float)
        im_tensor[0,:,:] = torch.from_numpy(R)
        im_tensor[1,:,:] = torch.from_numpy(G)
        im_tensor[2,:,:] = torch.from_numpy(B)
        
        bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
        sample = {'image': im_tensor,
                  'bbox': bb_tensor,
                  'label': self.dataset[idx][4]
                }
        
        if self.transform:
            sample = self.transform(sample)
        return sample 


dataset_train_noise = PurdueShapes5DatasetTotal(train_or_test='train', dataroot = dataroot)       #no noise
dataset_test_noise = PurdueShapes5DatasetTotal(train_or_test='test', dataroot = dataroot)   

train_dataloader_noise = torch.utils.data.DataLoader(dataset_train_noise, batch_size = batch_size, shuffle = True, num_workers = 4)
test_dataloader_noise = torch.utils.data.DataLoader(dataset_test_noise, batch_size = batch_size, shuffle = False, num_workers = 4)


class SkipBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
        super(SkipBlock, self).__init__()
        self.downsample = downsample
        self.skip_connections = skip_connections
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        norm_layer = nn.BatchNorm2d
        self.bn = norm_layer(out_ch)
        if downsample:
            self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

    def forward(self, x):
        identity = x                                     
        out = self.convo(x)                              
        out = self.bn(out)                              
        out = torch.nn.functional.relu(out)
        if self.in_ch == self.out_ch:
            out = self.convo(out)                              
            out = self.bn(out)                              
            out = torch.nn.functional.relu(out)
        if self.downsample:
            out = self.downsampler(out)
            identity = self.downsampler(identity)
        if self.skip_connections:
            if self.in_ch == self.out_ch:
                out += identity                              
            else:
                out[:,:self.in_ch,:,:] += identity
                out[:,self.in_ch:,:,:] += identity
        return out
#https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/9
class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        
        self.kernel_size=kernel_size
        
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        return self.conv(input, weight=self.weight,stride=1,padding=self.kernel_size[0]//2, groups=self.groups)
    
class LOADnet2(nn.Module):
    """
    The acronym 'LOAD' stands for 'LOcalization And Detection'.
    LOADnet2 uses both convo and linear layers for regression
    """ 
    def __init__(self, skip_connections=True, depth=32, sigma=0, kernel_size=3):
        super(LOADnet2, self).__init__()
        self.pool_count = 3
        self.depth = depth // 2
        self.sigma = sigma
        if self.sigma:
            self.filter = GaussianSmoothing(3, kernel_size, self.sigma, 2)
        
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.skip64 = SkipBlock(64, 64, 
                                                   skip_connections=skip_connections)
        self.skip64ds = SkipBlock(64, 64, 
                                   downsample=True, skip_connections=skip_connections)
        self.skip64to128 = SkipBlock(64, 128, 
                                                    skip_connections=skip_connections )
        self.skip128 = SkipBlock(128, 128, 
                                                     skip_connections=skip_connections)
        self.skip128ds = SkipBlock(128,128,
                                    downsample=True, skip_connections=skip_connections)
        self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
        self.fc2 =  nn.Linear(1000, 5)
        ##  for regression
        self.conv_seqn = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )   
        self.fc_seqn = nn.Sequential(
            nn.Linear(16384, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4)
        )

    def forward(self, x): 
#        pdb.set_trace()
        if self.sigma:
            x = self.filter(x)
        x = self.pool(torch.nn.functional.relu(self.conv(x)))          
        ## The labeling section:
        x1 = x.clone()
        for _ in range(self.depth // 4):
            x1 = self.skip64(x1)                                               
        x1 = self.skip64ds(x1)
        for _ in range(self.depth // 4):
            x1 = self.skip64(x1)                                               
        x1 = self.skip64to128(x1)
        for _ in range(self.depth // 4):
            x1 = self.skip128(x1)                                               
        x1 = self.skip128ds(x1)                                               
        for _ in range(self.depth // 4):
            x1 = self.skip128(x1)                                               
        x1 = x1.view(-1, 128 * (32 // 2**self.pool_count)**2 )
        x1 = torch.nn.functional.relu(self.fc1(x1))
        x1 = self.fc2(x1)
        ## The Bounding Box regression:
        x2 = self.conv_seqn(x)
        x2 = self.conv_seqn(x2)
        # flatten
        x2 = x2.view(x.size(0), -1)
        x2 = self.fc_seqn(x2)
        return x1,x2         


class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__() 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
            )

        self.fc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),      
        )           
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
    
def train(net):
#    pdb.set_trace()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    for epoch in range(epochs):
        running_loss=0
        train_correct=0
        total=0
        for i,data in enumerate(train_dataloader_noise):
#            pdb.set_trace()
            inputs, bbox_gt, labels, noiselabel = data['image'], data['bbox'], data['label'], data['noiselabel']
            
            inputs = inputs.to(device)
            noiselabel = noiselabel.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
#            pdb.set_trace()
            loss = criterion(outputs, noiselabel)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
            prediction = torch.max(outputs, 1)
            total += noiselabel.size(0)
            
            train_correct += np.sum(prediction[1].cpu().numpy() == noiselabel.cpu().numpy())            
                

        print('Epoch {}: loss: {} Accuracy: {}' .format(epoch+1, running_loss/((i+1)*batch_size), 100.*train_correct / total))
    torch.save(net.state_dict(), "./save_MLP")
            
def testing(net, pretrained_0, pretrained_20, pretrained_50, pretrained_80):
    net.load_state_dict(torch.load("./save_MLP"))
    pretrained_0.load_state_dict(torch.load("./saved_model_Dataset0"))
    pretrained_20.load_state_dict(torch.load("./saved_model_Dataset20"))
    pretrained_50.load_state_dict(torch.load("./saved_model_Dataset50"))
    pretrained_80.load_state_dict(torch.load("./saved_model_Dataset80"))
    total0 , correct0 = 0, 0
    total20, correct20 = 0, 0
    total50, correct50 = 0, 0
    total80, correct80 = 0, 0
    net = net.to(device)
    pretrained_0 = pretrained_0.to(device)
    pretrained_20 = pretrained_20.to(device)
    pretrained_50 = pretrained_50.to(device)
    pretrained_80 = pretrained_80.to(device)
    confusion_matrix_noise = torch.zeros(4, 4)
    confusion_matrix0 = torch.zeros(5, 5)
    class_total0 = [0]*len(dataset_train.class_labels)
    confusion_matrix20 = torch.zeros(5, 5)
    class_total20 = [0]*len(dataset_train.class_labels)
    confusion_matrix50 = torch.zeros(5, 5)
    class_total50 = [0]*len(dataset_train.class_labels)
    confusion_matrix80 = torch.zeros(5, 5)
    class_total80 = [0]*len(dataset_train.class_labels)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1):
        test_loss=0 
        test_correct = 0
        total =0      
        for i, data in enumerate(test_dataloader_noise):
            inputs, bbox_gt, labels, noiselabel = data['image'], data['bbox'], data['label'], data['noiselabel']
            inputs = inputs.to(device)
            noiselabel = noiselabel.to(device)
            outputs = net(inputs)
            loss   = criterion(outputs, noiselabel)
            test_loss+=loss.item()
            _,predicted = outputs.max(1) 
            
            for index in predicted:
                if index == 0:
                    total0 +=1
                    out0, l, p=run_code_for_testing_detection_and_localization(pretrained_0, inputs[index].view(1,3,32,32), bbox_gt[index], labels[index], i)
                    correct0+=out0
                    confusion_matrix0[l,p] +=1
                    class_total0[l] +=1
                    
                if index == 1:
                    total20 +=1
                    out20, l, p=run_code_for_testing_detection_and_localization(pretrained_20, inputs[index].view(1,3,32,32), bbox_gt[index], labels[index], i)
                    correct20+=out20
                    confusion_matrix20[l,p] +=1
                    class_total20[l] +=1
                if index == 2:
                    total50 +=1
                    out50, l, p=run_code_for_testing_detection_and_localization(pretrained_50, inputs[index].view(1,3,32,32), bbox_gt[index], labels[index], i)
                    correct50+=out50
                    confusion_matrix50[l,p] +=1
                    class_total50[l] +=1
                if index == 3:
                    total80 +=1
                    out80, l, p=run_code_for_testing_detection_and_localization(pretrained_80, inputs[index].view(1,3,32,32), bbox_gt[index], labels[index], i)
                    correct80+=out80
                    confusion_matrix80[l,p] +=1
                    class_total80[l] +=1
                    
            total += noiselabel.size(0)

            test_correct += np.sum(predicted[1].cpu().numpy() == noiselabel.cpu().numpy())   
            for label,prediction in zip(noiselabel, predicted):
                confusion_matrix_noise[label][prediction] += 1         
        
        f.write('Noise Classification Accuracy: {}%' .format(100.*test_correct / total))
        f.write('\nNoise Confusion Matrix:\n')
        print('Noise Classification Accuracy: {}' .format(100.*test_correct / total))
        print('Noise Confusion Matrix:')
        out_str = "               "
        for j in range(len(dataset_train_noise.class_noiselabels)):  
                             out_str +=  "%15s" % dataset_train_noise.class_noiselabels[j]   
        print(out_str + "\n")
        f.write('%s\n'%(out_str))
        for i,label in enumerate(dataset_train_noise.class_noiselabels):
            out_percents = [100.0 * confusion_matrix_noise[i,j] / float(1000) 
                             for j in range(len(dataset_train_noise.class_noiselabels))]
            out_percents = ["%.2f" % item.item() for item in out_percents]
            out_str = "%12s:  " % dataset_train_noise.class_noiselabels[i]
            for j in range(len(dataset_train_noise.class_noiselabels)): 
                                                   out_str +=  "%15s" % out_percents[j]
            print(out_str)
            f.write('%s\n'%(out_str))
        
        
        print('\nDataset0 Classification Accuracy: {}%'.format(100.*correct0/total0))
        f.write('\nDataset0 Classification Accuracy: {}%'.format(100.*correct0/total0))
        print_confusionmatrix(confusion_matrix0, 'Dataset0', class_total0)

        print('\nDataset20 Classification Accuracy: {}%'.format(100.*correct20/total20))
        f.write('\nDataset20 Classification Accuracy: {}%'.format(100.*correct20/total20))
        print_confusionmatrix(confusion_matrix20, 'Dataset20', class_total20)
        
        print('\nDataset50 Classification Accuracy: {}%'.format(100.*correct50/total50))
        f.write('\nDataset50 Classification Accuracy: {}%'.format(100.*correct50/total50))        
        print_confusionmatrix(confusion_matrix50, 'Dataset50', class_total0)
        
        print('\nDataset80 Classification Accuracy: {}%'.format(100.*correct80/total80))
        f.write('\nDataset80 Classification Accuracy: {}%'.format(100.*correct80/total80))        
        print_confusionmatrix(confusion_matrix80, 'Dataset80', class_total80)        

def print_confusionmatrix(confusionmatrix, dataset, class_total):
    print(dataset + ' Confusion Matrix:')
    f.write('\n'+dataset + ' Confusion Matrix:\n')
   
    out_str = "                "
    for j in range(len(dataset_train.class_labels)):  
                         out_str +=  "%15s" % dataset_train.class_labels[j]   
    print(out_str + "\n")
    f.write(out_str + "\n")
    for i,label in enumerate(dataset_train.class_labels):
        out_percents = [100.0 * confusionmatrix[i,j] / float(class_total[i]) 
                         for j in range(len(dataset_train.class_labels))]
        out_percents = ["%.2f" % item.item() for item in out_percents]
        out_str = "%12s:  " % dataset_train.class_labels[i]
        for j in range(len(dataset_train.class_labels)): 
                                               out_str +=  "%15s" % out_percents[j]
        print(out_str)        
        f.write(out_str + "\n")

#This code is taken from dlstudio class: https://engineering.purdue.edu/kak/distDLS/DLStudio-1.0.7.html
def run_code_for_training_with_CrossEntropy_and_MSE_Losses(net, path_saved_model, lr):        
    filename_for_out1 = "performance_numbers_" + str(epochs) + "label.txt"
    filename_for_out2 = "performance_numbers_" + str(epochs) + "regres.txt"
    FILE1 = open(filename_for_out1, 'w')
    FILE2 = open(filename_for_out2, 'w')
    net = copy.deepcopy(net)
    net = net.to(device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), 
                 lr=lr, momentum=momentum)
    for epoch in range(epochs):  
        running_loss_labeling = 0.0
        running_loss_regression = 0.0       
        for i, data in enumerate(train_dataloader):
            gt_too_small = False
            inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
            if debug_train and i % 500 == 499:
#                    if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
                print("\n\n[epoch=%d iter=%d:] Ground Truth:     " % (epoch+1, i+1) + 
                ' '.join('%10s' % dataset_train.class_labels[labels[j].item()] for j in range(batch_size)))
            inputs = inputs.to(device)
            labels = labels.to(device)
            bbox_gt = bbox_gt.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs_label = outputs[0]
            bbox_pred = outputs[1]
            if debug_train and i % 500 == 499:
#                  if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
                inputs_copy = inputs.detach().clone()
                inputs_copy = inputs_copy.cpu()
                bbox_pc = bbox_pred.detach().clone()
                bbox_pc[bbox_pc<0] = 0
                bbox_pc[bbox_pc>31] = 31
                bbox_pc[torch.isnan(bbox_pc)] = 0
                _, predicted = torch.max(outputs_label.data, 1)
                print("[epoch=%d iter=%d:] Predicted Labels: " % (epoch+1, i+1) + 
                 ' '.join('%10s' % dataset_train.class_labels[predicted[j].item()] 
                                   for j in range(batch_size)))
                for idx in range(batch_size):
                    i1 = int(bbox_gt[idx][1])
                    i2 = int(bbox_gt[idx][3])
                    j1 = int(bbox_gt[idx][0])
                    j2 = int(bbox_gt[idx][2])
                    k1 = int(bbox_pc[idx][1])
                    k2 = int(bbox_pc[idx][3])
                    l1 = int(bbox_pc[idx][0])
                    l2 = int(bbox_pc[idx][2])
                    print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                    print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                    inputs_copy[idx,0,i1:i2,j1] = 255
                    inputs_copy[idx,0,i1:i2,j2] = 255
                    inputs_copy[idx,0,i1,j1:j2] = 255
                    inputs_copy[idx,0,i2,j1:j2] = 255
                    inputs_copy[idx,2,k1:k2,l1] = 255                      
                    inputs_copy[idx,2,k1:k2,l2] = 255
                    inputs_copy[idx,2,k1,l1:l2] = 255
                    inputs_copy[idx,2,k2,l1:l2] = 255
#                        self.dl_studio.display_tensor_as_image(
#                              torchvision.utils.make_grid(inputs_copy, normalize=True),
#                             "see terminal for TRAINING results at iter=%d" % (i+1))
            loss_labeling = criterion1(outputs_label, labels)
            loss_labeling.backward(retain_graph=True)        
            loss_regression = criterion2(bbox_pred, bbox_gt)
            loss_regression.backward()
            optimizer.step()
            running_loss_labeling += loss_labeling.item()    
            running_loss_regression += loss_regression.item()                
            if i % 500 == 499:    
                avg_loss_labeling = running_loss_labeling / float(500)
                avg_loss_regression = running_loss_regression / float(500)
                print("\n[epoch:%d, iteration:%5d]  loss_labeling: %.3f  loss_regression: %.3f  " % (epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression))
                FILE1.write("%.3f\n" % avg_loss_labeling)
                FILE1.flush()
                FILE2.write("%.3f\n" % avg_loss_regression)
                FILE2.flush()
                running_loss_labeling = 0.0
                running_loss_regression = 0.0


    print("\nFinished Training\n")
    save_model(net, path_saved_model)
#This code is taken from dlstudio class: https://engineering.purdue.edu/kak/distDLS/DLStudio-1.0.7.html
def save_model(model, path_saved_model):
    '''
    Save the trained model to a disk file
    '''
    torch.save(model.state_dict(), path_saved_model)


def run_code_for_testing_detection_and_localization(net, images, bounding_box, labels, i):
    correct = 0
    total = 0
    labels = labels.tolist()
    if debug_test and i % 50 == 0:
        print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%s' % 
dataset_train.class_labels[labels] ))
    outputs = net(images)
    outputs_label = outputs[0]
    outputs_regression = outputs[1]
    outputs_regression[outputs_regression < 0] = 0
    outputs_regression[outputs_regression > 31] = 31
    outputs_regression[torch.isnan(outputs_regression)] = 0
    output_bb = outputs_regression.tolist()
    _, predicted = torch.max(outputs_label.data, 1)
    predicted = predicted.tolist()
    if debug_test and i % 50 == 0:
        print("[i=%d:] Predicted Labels: " %i + ' '.join('%s' % 
 dataset_train.class_labels[predicted[0]] ))
        for idx in range(1):
            i1 = int(bounding_box[1])
            i2 = int(bounding_box[3])
            j1 = int(bounding_box[0])
            j2 = int(bounding_box[2])
            k1 = int(output_bb[idx][1])
            k2 = int(output_bb[idx][3])
            l1 = int(output_bb[idx][0])
            l2 = int(output_bb[idx][2])
            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
            images[idx,0,i1:i2,j1] = 255
            images[idx,0,i1:i2,j2] = 255
            images[idx,0,i1,j1:j2] = 255
            images[idx,0,i2,j1:j2] = 255
            images[idx,2,k1:k2,l1] = 255                      
            images[idx,2,k1:k2,l2] = 255
            images[idx,2,k1,l1:l2] = 255
            images[idx,2,k2,l1:l2] = 255
    if predicted[0] == labels:
        return 1, labels, predicted[0]
    else:
        return 0, labels, predicted[0]




print("Training 4 LOADnet2s for different noise levels:\n")
net  = LOADnet2(skip_connections=True, depth=32, sigma=0, kernel_size=5)
print('Dataset0 Result:\n')
dataset_train = PurdueShapes5Dataset(train_or_test='train', dataset_file = "PurdueShapes5-10000-train.gz", dataroot = dataroot)       
train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4)
run_code_for_training_with_CrossEntropy_and_MSE_Losses(net, "./saved_model_Dataset0", 1e-4)


net = LOADnet2(skip_connections=True, depth=32, sigma=5, kernel_size=3)
print('Dataset20 Result:\n')
dataset_train = PurdueShapes5Dataset(train_or_test='train', dataset_file = "PurdueShapes5-10000-train-noise-20.gz", dataroot = dataroot)      
train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4)
run_code_for_training_with_CrossEntropy_and_MSE_Losses(net, "./saved_model_Dataset20", 1e-5)


net = LOADnet2(skip_connections=True, depth=32, sigma=1, kernel_size=3)
print('Dataset50 Result:\n')
dataset_train = PurdueShapes5Dataset(train_or_test='train', dataset_file = "PurdueShapes5-10000-train-noise-50.gz", dataroot = dataroot)       
train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4)
run_code_for_training_with_CrossEntropy_and_MSE_Losses(net, "./saved_model_Dataset50", 1e-5)


net = LOADnet2(skip_connections=True, depth=32, sigma=1, kernel_size=5)
print('Dataset80 Result:\n')
dataset_train = PurdueShapes5Dataset(train_or_test='train', dataset_file = "PurdueShapes5-10000-train-noise-80.gz", dataroot = dataroot)       
train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4)
run_code_for_training_with_CrossEntropy_and_MSE_Losses(net, "./saved_model_Dataset80", 1e-5)


print('Training the network for noise detection:')
net = MLP()
train(net)
print('Testing the noise detection network and checking accuracy for 4 networks:')
net_0  = LOADnet2(skip_connections=True, depth=32, sigma=0, kernel_size=5)
net_20 = LOADnet2(skip_connections=True, depth=32, sigma=5, kernel_size=3)
net_50 = LOADnet2(skip_connections=True, depth=32, sigma=1, kernel_size=3)
net_80 = LOADnet2(skip_connections=True, depth=32, sigma=1, kernel_size=5)
testing(net, net_0, net_20, net_50, net_80)
f.write('\n\n')

            
    
    
    
    
    
    
    
    
    
    
    
    
    
    