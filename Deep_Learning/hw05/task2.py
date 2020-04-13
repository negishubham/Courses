#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:45:08 2020

@author: snegi
"""
# The below comment box shows the result for different dataset with and witout filterring
# It also shows different smoothing parameters for all the noisy datasets
'''
lr=1e-4
no noise:  loss_labeling: 0.242  loss_regression: 0.840
Prediction accuracy for rectangle : 68 %
Prediction accuracy for triangle : 99 %
Prediction accuracy for  disk : 100 %
Prediction accuracy for  oval : 84 %
Prediction accuracy for  star : 99 %
Overall accuracy of the network on the 1000 test images: 90 %
'''

'''
20% noise: 
lr = 1e-5    
No filter: 
loss_labeling: 0.608  loss_regression: 3.434     
Prediction accuracy for rectangle : 54 %
Prediction accuracy for triangle : 88 %
Prediction accuracy for  disk : 86 %
Prediction accuracy for  oval : 58 %
Prediction accuracy for  star : 81 %
Overall accuracy of the network on the 1000 test images: 73 %
    
Filter: sigma=1, kernel_size=1
loss_labeling: 0.608  loss_regression: 3.434
Prediction accuracy for rectangle : 54 %
Prediction accuracy for triangle : 88 %
Prediction accuracy for  disk : 86 %
Prediction accuracy for  oval : 58 %
Prediction accuracy for  star : 81 %
Overall accuracy of the network on the 1000 test images: 73 %


Filter: sigma=0.1, kernel_size=3
loss_labeling: 0.606  loss_regression: 3.333
Prediction accuracy for rectangle : 52 %
Prediction accuracy for triangle : 87 %
Prediction accuracy for  disk : 88 %
Prediction accuracy for  oval : 60 %
Prediction accuracy for  star : 81 %
Overall accuracy of the network on the 1000 test images: 73 %


Filter: sigma=5, kernel_size=3
loss_labeling: 0.582  loss_regression: 4.345 

Prediction accuracy for rectangle : 42 %
Prediction accuracy for triangle : 93 %
Prediction accuracy for  disk : 97 %
Prediction accuracy for  oval : 73 %
Prediction accuracy for  star : 94 %
Overall accuracy of the network on the 1000 test images: 79 %


Filter: sigma=1, kernel_size=3
loss_labeling: 0.573  loss_regression: 4.420
Prediction accuracy for rectangle : 44 %
Prediction accuracy for triangle : 93 %
Prediction accuracy for  disk : 97 %
Prediction accuracy for  oval : 74 %
Prediction accuracy for  star : 91 %
Overall accuracy of the network on the 1000 test images: 79 %


Filter: sigma=1, kernel_size=5
loss_labeling: 0.574  loss_regression: 4.867  
Prediction accuracy for rectangle : 44 %
Prediction accuracy for triangle : 88 %
Prediction accuracy for  disk : 98 %
Prediction accuracy for  oval : 76 %
Prediction accuracy for  star : 88 %
Overall accuracy of the network on the 1000 test images: 79 %


'''


'''
50% noise: 
lr = 1e-5    
No filter: 
loss_labeling: 0.870  loss_regression: 2.858     
Prediction accuracy for rectangle : 46 %
Prediction accuracy for triangle : 77 %
Prediction accuracy for  disk : 72 %
Prediction accuracy for  oval : 52 %
Prediction accuracy for  star : 71 %
Overall accuracy of the network on the 1000 test images: 63 %

    
Filter: sigma=1, kernel_size=1
loss_labeling: 0.870  loss_regression: 2.858
Prediction accuracy for rectangle : 46 %
Prediction accuracy for triangle : 77 %
Prediction accuracy for  disk : 72 %
Prediction accuracy for  oval : 52 %
Prediction accuracy for  star : 71 %
Overall accuracy of the network on the 1000 test images: 63 %

Filter: sigma=1, kernel_size=3
loss_labeling: 0.721  loss_regression: 3.816 
Prediction accuracy for rectangle : 38 %
Prediction accuracy for triangle : 89 %
Prediction accuracy for  disk : 83 %
Prediction accuracy for  oval : 73 %
Prediction accuracy for  star : 84 %
Overall accuracy of the network on the 1000 test images: 73 %


Filter: sigma=1, kernel_size=5
loss_labeling: 0.694  loss_regression: 3.906  
Prediction accuracy for rectangle : 43 %
Prediction accuracy for triangle : 88 %
Prediction accuracy for  disk : 81 %
Prediction accuracy for  oval : 69 %
Prediction accuracy for  star : 83 %
Overall accuracy of the network on the 1000 test images: 73 %
    
    
'''

'''
80% noise: 
lr=1e-5
No filter: 
loss_labeling: 1.011  loss_regression: 5.384    
Prediction accuracy for rectangle : 30 %
Prediction accuracy for triangle : 66 %
Prediction accuracy for  disk : 46 %
Prediction accuracy for  oval : 55 %
Prediction accuracy for  star : 49 %
Overall accuracy of the network on the 1000 test images: 49 %

Filter: sigma=1, kernel_size=1
loss_labeling: 1.011  loss_regression: 5.384 
Prediction accuracy for rectangle : 30 %
Prediction accuracy for triangle : 66 %
Prediction accuracy for  disk : 46 %
Prediction accuracy for  oval : 55 %
Prediction accuracy for  star : 49 %
Overall accuracy of the network on the 1000 test images: 49 %


Filter: sigma=1, kernel_size=3
loss_labeling: 0.842  loss_regression: 5.766  
Prediction accuracy for rectangle : 62 %
Prediction accuracy for triangle : 73 %
Prediction accuracy for  disk : 64 %
Prediction accuracy for  oval : 38 %
Prediction accuracy for  star : 60 %
Overall accuracy of the network on the 1000 test images: 59 %

Filter: sigma=5, kernel_size=3
loss_labeling: 0.825  loss_regression: 5.524  
Prediction accuracy for rectangle : 61 %
Prediction accuracy for triangle : 74 %
Prediction accuracy for  disk : 68 %
Prediction accuracy for  oval : 43 %
Prediction accuracy for  star : 59 %
Overall accuracy of the network on the 1000 test images: 61 %

Filter: sigma=10, kernel_size=3
loss_labeling: 0.842  loss_regression: 5.766  
Prediction accuracy for rectangle : 62 %
Prediction accuracy for triangle : 74 %
Prediction accuracy for  disk : 67 %
Prediction accuracy for  oval : 43 %
Prediction accuracy for  star : 60 %
Overall accuracy of the network on the 1000 test images: 61 %

Filter: sigma=0.1, kernel_size=3
loss_labeling: 1.012  loss_regression: 5.503 
Prediction accuracy for rectangle : 31 %
Prediction accuracy for triangle : 67 %
Prediction accuracy for  disk : 44 %
Prediction accuracy for  oval : 60 %
Prediction accuracy for  star : 50 %
Overall accuracy of the network on the 1000 test images: 50 %

Filter: sigma=1, kernel_size=5
loss_labeling: 0.809  loss_regression: 5.432  
Prediction accuracy for rectangle : 62 %
Prediction accuracy for triangle : 79 %
Prediction accuracy for  disk : 64 %
Prediction accuracy for  oval : 40 %
Prediction accuracy for  star : 65 %

Overall accuracy of the network on the 1000 test images: 62 %

'''

'''
Parameters summarized for different noise level and what is the accuracy improvement compared to when noise is present

noise  lr       Gaussian Kernel Params      Accuracy Improvement(without smoothing --> with smoothing)
0      lr=1e-4  sigma=0                     90% (Baseline without noise)
20     lr=1e-5  sigma=5 and k=3             73 --> 79 
50     lr=1e-5  sigma=1 and k=3             63 --> 73
80     lr=1e-5  sigma=1 and k=5             49 -->62
'''

'''
Results for task2:

Dataset0 Classification Accuracy: 90%
Dataset0 Confusion Matrix:
                      rectangle       triangle           disk           oval           star
   rectangle:            68.50           0.00           2.50          27.00           2.00
    triangle:             0.50          99.00           0.50           0.00           0.00
        disk:             0.00           0.00         100.00           0.00           0.00
        oval:            15.00           0.00           1.00          84.00           0.00
        star:             0.00           1.00           0.00           0.00          99.00

Dataset20 Classification Accuracy: 76%
Dataset20 Confusion Matrix:
                      rectangle       triangle           disk           oval           star
   rectangle:            47.50           3.00          17.50          23.50           8.50
    triangle:             2.00          85.00          12.50           0.50           0.00
        disk:             0.50           0.00          99.50           0.00           0.00
        oval:            25.50           0.50           8.50          63.00           2.50
        star:             4.00           1.00           4.50           1.00          89.50

Dataset50 Classification Accuracy: 70%
Dataset50 Confusion Matrix:
                      rectangle       triangle           disk           oval           star
   rectangle:            50.00          11.50          10.50          13.50          14.50
    triangle:             1.50          89.50           8.00           0.00           1.00
        disk:             2.50           2.50          89.50           1.50           4.00
        oval:            41.00           0.00          11.00          39.00           9.00
        star:             6.50           2.00           5.50           2.00          84.00

Dataset80 Classification Accuracy: 62%
Dataset80 Confusion Matrix:
                      rectangle       triangle           disk           oval           star
   rectangle:            69.00           3.00          10.50           9.50           8.00
    triangle:            11.50          81.00           6.00           0.00           1.50
        disk:            13.00           2.00          82.00           1.00           2.00
        oval:            67.00           1.00          10.50          16.00           5.50
        star:            25.00           1.50           7.00           1.00          65.50    
'''

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


seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

#lr = 1e-5
momentum = 0.9
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 2
dataroot = "/local/scratch/a/snegi/PhD/Courses/Deep_Learning/Lab_Assignment/hw5/data/"
batch_size =4 
debug_train =1
debug_test =1
#path_saved_model = "saved_model_bestmodel_0noise" #"./saved_model_Dataset0"
classes = ('rectangle','triangle','disk','oval','star')


#Custom Dataloader
f = open('output.txt', 'w')
f.write("+Task 2:\n")

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
#This code is taken from dlstudio class: https://engineering.purdue.edu/kak/distDLS/DLStudio-1.0.7.html
def display_tensor_as_image(tensor, title=""):
    '''
    This method converts the argument tensor into a photo image that you can display
    in your terminal screen. It can convert tensors of three different shapes
    into images: (3,H,W), (1,H,W), and (H,W), where H, for height, stands for the
    number of pixels in the vertical direction and W, for width, for the same
    along the horizontal direction.  When the first element of the shape is 3,
    that means that the tensor represents a color image in which each pixel in
    the (H,W) plane has three values for the three color channels.  On the other
    hand, when the first element is 1, that stands for a tensor that will be
    shown as a grayscale image.  And when the shape is just (H,W), that is
    automatically taken to be for a grayscale image.
    '''
    tensor_range = (torch.min(tensor).item(), torch.max(tensor).item())
    if tensor_range == (-1.0,1.0):
        ##  The tensors must be between 0.0 and 1.0 for the display:
        print("\n\n\nimage un-normalization called")
        tensor = tensor/2.0 + 0.5     # unnormalize
    plt.figure(title)
    ###  The call to plt.imshow() shown below needs a numpy array. We must also
    ###  transpose the array so that the number of channels (the same thing as the
    ###  number of color planes) is in the last element.  For a tensor, it would be in
    ###  the first element.
    if tensor.shape[0] == 3 and len(tensor.shape) == 3:
#            plt.imshow( tensor.numpy().transpose(1,2,0) )
        plt.imshow( tensor.numpy().transpose(1,2,0) )
    ###  If the grayscale image was produced by calling torchvision.transform's
    ###  ".ToPILImage()", and the result converted to a tensor, the tensor shape will
    ###  again have three elements in it, however the first element that stands for
    ###  the number of channels will now be 1
    elif tensor.shape[0] == 1 and len(tensor.shape) == 3:
        tensor = tensor[0,:,:]
        plt.imshow( tensor.numpy(), cmap = 'gray' )
    ###  For any one color channel extracted from the tensor representation of a color
    ###  image, the shape of the tensor will be (W,H):
    elif len(tensor.shape) == 2:
        plt.imshow( tensor.numpy(), cmap = 'gray' )
    else:
        sys.exit("\n\n\nfrom 'display_tensor_as_image()': tensor for image is ill formed -- aborting")
    plt.show()





#This code is taken from dlstudio class: https://engineering.purdue.edu/kak/distDLS/DLStudio-1.0.7.html
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

#This code is taken from dlstudio class: https://engineering.purdue.edu/kak/distDLS/DLStudio-1.0.7.html
#And it is edited to have gaussian smoothing parameters
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
#            if debug_train and i%500==499:
#                    if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
#        if epoch == epochs-1:
#            display_tensor_as_image(
#                  torchvision.utils.make_grid(inputs_copy, normalize=True),
#                 "see terminal for TRAINING results at iter=%d" % (i+1))


    print("\nFinished Training\n")
    save_model(net, path_saved_model)
#This code is taken from dlstudio class: https://engineering.purdue.edu/kak/distDLS/DLStudio-1.0.7.html
def save_model(model, path_saved_model):
    '''
    Save the trained model to a disk file
    '''
    torch.save(model.state_dict(), path_saved_model)
#This code is taken from dlstudio class: https://engineering.purdue.edu/kak/distDLS/DLStudio-1.0.7.html
def run_code_for_testing_detection_and_localization(net, dataset, path_saved_model):
    net.load_state_dict(torch.load(path_saved_model))
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(len(dataset_train.class_labels), 
                                   len(dataset_train.class_labels))
    class_correct = [0] * len(dataset_train.class_labels)
    class_total = [0] * len(dataset_train.class_labels)
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            images, bounding_box, labels = data['image'], data['bbox'], data['label']
            labels = labels.tolist()
            if debug_test and i % 50 == 0:
                print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%10s' % 
dataset_train.class_labels[labels[j]] for j in range(batch_size)))
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
                print("[i=%d:] Predicted Labels: " %i + ' '.join('%10s' % 
 dataset_train.class_labels[predicted[j]] for j in range(batch_size)))
                for idx in range(batch_size):
                    i1 = int(bounding_box[idx][1])
                    i2 = int(bounding_box[idx][3])
                    j1 = int(bounding_box[idx][0])
                    j2 = int(bounding_box[idx][2])
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
#                display_tensor_as_image(
#                      torchvision.utils.make_grid(images, normalize=True), 
#                      "see terminal for test results at i=%d" % i)
            for label,prediction in zip(labels,predicted):
                confusion_matrix[label][prediction] += 1
            total += len(labels)
            correct +=  [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
            comp = [predicted[ele] == labels[ele] for ele in range(len(predicted))]
            for j in range(batch_size):
                label = labels[j]
                class_correct[label] += comp[j]
                class_total[label] += 1
    print("\n")
    for j in range(len(dataset_train.class_labels)):
        print('Prediction accuracy for %5s : %2d %%' % (
      dataset_train.class_labels[j], 100 * class_correct[j] / class_total[j]))
    print("\n\n\nOverall accuracy of the network on the 1000 test images: %d %%" % 
                                                           (100 * correct / float(total)))

    f.write(dataset + ' Classification Accuracy: %d%%\n'%(100 * correct / float(total)))
    f.write(dataset + ' Confusion Matrix:\n')


    print("\n\nDisplaying the confusion matrix:\n")
    out_str = "                "
    for j in range(len(dataset_train.class_labels)):  
                         out_str +=  "%15s" % dataset_train.class_labels[j]   
    print(out_str + "\n")
    f.write('%s\n'%(out_str))
    for i,label in enumerate(dataset_train.class_labels):
        out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                         for j in range(len(dataset_train.class_labels))]
        out_percents = ["%.2f" % item.item() for item in out_percents]
        out_str = "%12s:  " % dataset_train.class_labels[i]
        for j in range(len(dataset_train.class_labels)): 
                                               out_str +=  "%15s" % out_percents[j]
        print(out_str)
        f.write('%s\n'%(out_str))
    f.write('\n')


net  = LOADnet2(skip_connections=True, depth=32, sigma=0, kernel_size=5)
print('Dataset0 Result:\n')
dataset_train = PurdueShapes5Dataset(train_or_test='train', dataset_file = "PurdueShapes5-10000-train.gz", dataroot = dataroot)       
dataset_test = PurdueShapes5Dataset(train_or_test='test', dataset_file = "PurdueShapes5-1000-test.gz", dataroot = dataroot)   

train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4)
test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 4)

run_code_for_training_with_CrossEntropy_and_MSE_Losses(net, "./saved_model_Dataset0", 1e-4)
run_code_for_testing_detection_and_localization(net, "Dataset0", "./saved_model_Dataset0")


net = LOADnet2(skip_connections=True, depth=32, sigma=5, kernel_size=3)
print('Dataset20 Result:\n')
dataset_train = PurdueShapes5Dataset(train_or_test='train', dataset_file = "PurdueShapes5-10000-train-noise-20.gz", dataroot = dataroot)      
dataset_test = PurdueShapes5Dataset(train_or_test='test', dataset_file = "PurdueShapes5-1000-test-noise-20.gz", dataroot = dataroot)   


train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4)
test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 4)

run_code_for_training_with_CrossEntropy_and_MSE_Losses(net, "./saved_model_Dataset20", 1e-5)
run_code_for_testing_detection_and_localization(net, "Dataset20", "./saved_model_Dataset20")


net = LOADnet2(skip_connections=True, depth=32, sigma=1, kernel_size=3)
print('Dataset50 Result:\n')
dataset_train = PurdueShapes5Dataset(train_or_test='train', dataset_file = "PurdueShapes5-10000-train-noise-50.gz", dataroot = dataroot)       
dataset_test = PurdueShapes5Dataset(train_or_test='test', dataset_file = "PurdueShapes5-1000-test-noise-50.gz", dataroot = dataroot)   

train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4)
test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 4)

run_code_for_training_with_CrossEntropy_and_MSE_Losses(net, "./saved_model_Dataset50", 1e-5)
run_code_for_testing_detection_and_localization(net, "Dataset50", "./saved_model_Dataset50")


net = LOADnet2(skip_connections=True, depth=32, sigma=1, kernel_size=5)
print('Dataset80 Result:\n')
dataset_train = PurdueShapes5Dataset(train_or_test='train', dataset_file = "PurdueShapes5-10000-train-noise-80.gz", dataroot = dataroot)       
dataset_test = PurdueShapes5Dataset(train_or_test='test', dataset_file = "PurdueShapes5-1000-test-noise-80.gz", dataroot = dataroot)   

train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4)
test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 4)

run_code_for_training_with_CrossEntropy_and_MSE_Losses(net, "./saved_model_Dataset80", 1e-5)
run_code_for_testing_detection_and_localization(net, "Dataset80", "./saved_model_Dataset80")











     

