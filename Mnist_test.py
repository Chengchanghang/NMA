import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# Hyper Parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 1
batch_size = 50
learning_rate = 0.001

def ver(y):
    z = np.zeros((10,1))
    z[y] = 1.0
    return z
# MNIST Dataset 
train_dataset = dsets.MNIST(root='/home/chengch/data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=False)

test_dataset = dsets.MNIST(root='/home/chengch/data', 
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
net = Net(input_size, hidden_size, num_classes)
  
    
# Loss and Optimizer
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
y_set = []
# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): 
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        y = labels.numpy()
        y = y.reshape(batch_size,1)
        ys = []
        for _ in y:
            ys.append(ver(_))
        labels = Variable(torch.FloatTensor(ys))
#        labels = Variable(labels)
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        y_set.append(loss.data[0])
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
           # y_set.append(loss.data[0])
x_set = [x for x in range(len(y_set))]
plt.plot(x_set,y_set)
plt.show()
'''
# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model.pkl')