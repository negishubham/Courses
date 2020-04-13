import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from   torchvision import transforms as tvt
from   torch.utils.data import DataLoader
import pdb


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

#Parameters
lr 	 = 1e-3
momentum = 0.9
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 1



f = open('output.txt', 'w')

transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
trainset  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader   = DataLoader(trainset, batch_size=4, shuffle = True, num_workers=2)
testloader    = DataLoader(testset,  batch_size=4,  shuffle = False,  num_workers=2)




class TemplateNet(nn.Module):
    def __init__(self, task, padding):
        super(TemplateNet, self).__init__()
        self.task = task
        self.conv1 = nn.Conv2d(3, 128, 3, padding = padding)
        if self.task == 'task2' or 'task3':
            self.conv2 = nn.Conv2d(128, 128, 3)
        self.pool  = nn.MaxPool2d(2,2)
        if self.task == 'task1':
            self.fc1   = nn.Linear(28800, 1000)
        elif self.task == 'task2':
            self.fc1   = nn.Linear(4608, 1000)
        elif self.task == 'task3':
            self.fc1   = nn.Linear(6272, 1000)
            
        self.fc2   = nn.Linear(1000, 10) 

    def forward(self, x, task):
        self.task = task
        x = self.pool(F.relu(self.conv1(x)))
        if task == 'task1':
           x = x.view(-1, 28800)
        elif task == 'task2':
            x = self.pool(F.relu(self.conv2(x)))
            x=x.view(-1,4608)
        elif task == 'task3':
            x = self.pool(F.relu(self.conv2(x)))
            x=x.view(-1,6272)            
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




def training(net, task):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = momentum)
    for epoch in range(epochs):        
        running_loss = 0	
        for i, data in enumerate(trainloader):
            inputs, labels = data	
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs, task)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
                
            if (i+1)%12000 == 0:
                f.write('[epoch:{}, batch:{}] loss: {} \n' .format(epoch+1, i+1, running_loss/2000))                
            if (i+1)%2000 == 0:
#                print('[epoch:{}, batch:{}] loss: {}' .format(epoch+1, i+1, running_loss/2000))
                running_loss=0
            
                
                
                
def testing(net, task):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        test_loss=0
        confusionmatrix=torch.zeros((10,10))        
        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs, task)
            loss   = criterion(outputs, labels)
            test_loss+=loss.item()
            _,predicted = outputs.max(1)
            for y in range(4):
                confusionmatrix[labels[y],predicted[y]] += 1
#        print(confusionmatrix)
#        f.write('{}\n'.format(str(torch.from_numpy(confusionmatrix))))
        f.write('{}\n'.format(str(confusionmatrix)))
            
            
            
                
net1 = TemplateNet('task1', padding=0)               
training(net1, 'task1')

net2 = TemplateNet('task2', padding=0)
training(net2, 'task2')
#
net3 = TemplateNet('task3', padding=1)
training(net3, 'task3')

testing(net3, 'task3')
f.close()














