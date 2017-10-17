import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from time import time
f1 = open('loss_5H.txt','w')
f2 = open('RMSD_5H.txt','w')

# Hyper Parameters 
hidden_size1 = 1000
hidden_size2 = 1000
hidden_size3 = 1000
num_epochs = 100  
batch_size1 = 1000
batch_size2 = 400
learning_rate = 0.0001

#data loader

IR = np.load('IR.npy')
num = len(IR)
split_point = int(num / 10 * 9) # the split_point for test_data and train_data

IR = np.load('IR.npy')
Structure = np.load('/home/chengch/NMA/data/10W/struct_params.npy')

total_data = []
for x,y in zip(IR,Structure):
    total_data.append((x,y))
np.random.shuffle(total_data)

train_data = total_data[:split_point]
test_data = total_data[split_point:]

training_data =[]
testing_data = []

for x1,y1 in train_data:
   # x1 = x1[0:len(x1):3]
    y111 = y1[:5]
    if y1[16] < 0:
        y1[16] = y[16] + 360
    training_data.append((torch.FloatTensor(x1),y111)) 
for x2,y2 in test_data:
   # x2 = x2[0:len(x2):3]
    y222 = y2[:5]
    if y2[16] < 0:
        y2[16] = y[16] + 360
    testing_data.append((torch.FloatTensor(x2),y222))
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=training_data, 
                                           batch_size=batch_size1, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testing_data, 
                                          batch_size=batch_size2, 
                                          shuffle=True)

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
net = Net(len(x1), hidden_size1, hidden_size2,hidden_size3, len(y111))
net = net.cuda()


# Loss and Optimizer
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
# Train the Model
for epoch in range(num_epochs):
    time0 = time()
    for i, (images, labels) in enumerate(train_loader): 
        images = Variable(images).cuda()
        labels = Variable(torch.FloatTensor(labels.numpy())).cuda()
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
       # net.eval()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
      #  net.train()
    f1.write(str(loss.data[0]))
    f1.write('\n')
    RMSDs =[]
    loss_set = []
    #net = net.cpu()
    time1 = time()
    for _,(images,labels) in enumerate(test_loader):
        images = Variable(images).cuda()
        labels = Variable(torch.FloatTensor(labels.numpy())).cuda()
        prediction = net(images)
        loss = criterion(prediction,labels)
    f2.write(str(loss.data[0]))
    f2.write('\n')
    time2 = time()

    print (epoch)
    print (str(time1 - time0))
    print (str(time2 - time1))
