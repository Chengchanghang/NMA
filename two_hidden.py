import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from time import time
from matplotlib.pyplot  import *

# Hyper Parameters 
input_size = 1201
hidden_size1 = 1000
hidden_size2 = 1000
hidden_size3 = 1000
num_classes = 51
num_epochs = 2
batch_size1 = 200
batch_size2 = 1
learning_rate = 0.0001

def ver(y):
    z = np.zeros((10,1))
    z[y] = 1.0
    return z

#data loader
IR = np.load('/home/chengch/NMA/NMA_data/IR_spectra_G_15.npy')[:10000]
Structure = np.load('/home/chengch/NMA/NMA_data/structs.npy')[:10000]

total_data = []
for x,y in zip(IR,Structure):
    total_data.append((x,y))
np.random.shuffle(total_data)

train_data = total_data[:9000]
test_data = total_data[9000:]
training_data =[]
testing_data = []

for x1,y1 in train_data:
    training_data.append((torch.FloatTensor(x1),y1)) 
for x2,y2 in test_data:
    testing_data.append((torch.FloatTensor(x2),y2))
    
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=training_data, 
                                           batch_size=batch_size1, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testing_data, 
                                          batch_size=batch_size2, 
                                          shuffle=False)


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2,hidden_size3, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)  
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, num_classes)  
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out
    
net = Net(input_size, hidden_size1, hidden_size2,hidden_size3, num_classes)


# Loss and Optimizer
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
plt.figure(figsize=(9,7))
y_set = []
# Train the Model
start = time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): 
        images = Variable(images)
        labels = Variable(torch.FloatTensor(labels.numpy()))
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    y_set.append(loss.data[0])
    print (loss.data[0])
#x_set = [x for x in range(len(y_set))]
#plt.plot(x_set,y_set)
#savefig('100.png',dpi=200)
#plt.show()
#test_data
RMSDs =[]
loss_set = []
for _,(images,labels) in enumerate(test_loader):
    images = Variable(images)
    labels = Variable(torch.FloatTensor(labels.numpy()))
    prediction = net(images)
    R = ((prediction - labels) ** 2)
    RMSD = (R.data.numpy().sum() / 17) ** 0.5    
    RMSDs.append(RMSD)
