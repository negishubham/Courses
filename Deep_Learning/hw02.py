#Shubham Negi
# HW 02
# Date 02/06/20
import torch
import torch.nn as nn
import numpy as np
import torchvision
from   torchvision import transforms as tvt
from   torch.utils.data import DataLoader
import matplotlib.pyplot as plt



torch.manual_seed(0)

transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   #Composing several transforms together
trainset  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

indicestrain=np.asarray(trainset.targets)
indicestrain_cat_dog=[]

for i in range(len(indicestrain)):
    if indicestrain[i] == 3 or indicestrain[i] == 5:
        indicestrain_cat_dog.append(i)
        
indicestest=np.asarray(testset.targets)
indicestest_cat_dog=[]

for i in range(len(indicestest)):
    if indicestest[i] == 3 or indicestest[i] == 5:
        indicestest_cat_dog.append(i)


samplertrain  = torch.utils.data.sampler.SubsetRandomSampler(indicestrain_cat_dog)
samplertest   = torch.utils.data.sampler.SubsetRandomSampler(indicestest_cat_dog)
trainloader   = torch.utils.data.DataLoader(trainset, batch_size=1000, sampler=samplertrain, num_workers=2)
testloader    = torch.utils.data.DataLoader(testset,  batch_size=100,  sampler=samplertest,  num_workers=2)


#def imshow(img):
#    img = img/2 + 0.5
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1,2,0)))
#    plt.show()
#    
#
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#
#imshow(torchvision.utils.make_grid(images))

dtype   = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Din, H1, H2, Dout = 3*32*32, 1000, 256, 2

w1 = torch.randn(Din, H1,   device=device, dtype=dtype)
w2 = torch.randn(H1,  H2,   device=device, dtype=dtype)
w3 = torch.randn(H2,  Dout, device=device, dtype=dtype)
w1 = w1.to(device)
w2 = w2.to(device)
w3 = w3.to(device)


Loss_train = []
Loss_test  = []
lr = 1e-12
epoch = 20

f = open('output.txt', 'w')

for epoch in range(epoch):
    train_loss=0
    test_loss=0
    total=0
    total_correct=0
    test_correct=0
#Training
    for i, data in enumerate(trainloader):
        inputs, labels = data

        labels_onehot = (labels.reshape(1000,1) == torch.tensor([5,3])).float()
        inputs        = inputs.to(device)
        labels_onehot = labels_onehot.to(device)

        x = inputs.view(inputs.size(0),-1)

        h1     = x.mm(w1)
        h1relu = h1.clamp(min=0)        
        h2     = h1relu.mm(w2)
        h2relu = h2.clamp(min=0)
        y_pred = h2relu.mm(w3)
        
        train_loss += (y_pred - labels_onehot).pow(2).mean().item()        #Mean Squared Loss
        i+=1

        y_error        = y_pred - labels_onehot
        grad_w3        = h2relu.t().mm(2.0*y_error)
        h2_error       = 2.0*y_error.mm(w3.t())
        h2_error[h2<0] = 0
        
        grad_w2        = h1relu.t().mm(2*h2_error)
        h1_error       = 2.0*h2_error.mm(w2.t())
        h1_error[h1<0] = 0
        
        grad_w1 = x.t().mm(2*h1_error)

        w1 -= lr*grad_w1
        w2 -= lr*grad_w2
        w3 -= lr*grad_w3
    Loss_train.append(train_loss/i)         # Taking average over the batches
        
# Testing
    for j, data in enumerate(testloader):
        inputs, labels = data
        labels_onehot  = (labels.reshape(100,1)==torch.tensor([5,3])).float()     ## Cat(3)= 0 , Dog(5)= 1
        
        inputs        = inputs.to(device)
        labels_onehot = labels_onehot.to(device)
        
        x = inputs.view(inputs.size(0), -1)
        
        h1     = x.mm(w1)
        h1relu = h1.clamp(min=0)
        h2     = h1.mm(w2)
        h2relu = h2.clamp(min=0)
        y_pred = h2relu.mm(w3) 
        
        test_loss += (y_pred - labels_onehot).pow(2).mean().item()          #Mean Squared Loss
        j+=1
        
        prediction    =torch.max(y_pred, 1)
        total        +=labels_onehot.size(0)
        test_correct +=np.sum(prediction[1].cpu().numpy() == (labels==torch.tensor([5])).float().cpu().numpy())
        
#        print('Test Accuracy=',100.*test_correct/total)
        
    Loss_test.append(test_loss/j)
#    print('Epoch {}: {}' .format(epoch, train_loss/i))
    f.write('Epoch {}: {}' .format(epoch, train_loss/i))	
    f.write('\n')

#print('')
#print('Testing Accuracy: {}%'.format(100.*test_correct/total))
f.write('\n')
f.write('Testing Accuracy: {}%'.format(100.*test_correct/total))
f.close()

#f = plt.figure()
#plt.title('Loss vs Epoch')
#plt.plot(Loss_train, 'r',  label='Training Loss')
#plt.plot(Loss_test,  'g',  label='Testing Loss')
#plt.legend()
#plt.show(block=False)        
#plt.pause(10)
#plt.close()

  
        
        


