#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
#from torchvision import torchvision.datasets.MNIST
import matplotlib.pyplot as plt
from PIL import Image

# Variable to check which device ( CPU or GPU - if available ) to use for processing
device_to_use = ('cuda' if torch.cuda.is_available() else 'cpu')

#torchvision.datasets.MNIST('/notebooks/',train = True,download = True)
# Transformations to be applied to images before any processing
Image_transformation_function = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()]) #,transforms.Normalize((0.1307,), (0.3081,))])
# Creating training dataset from MNIST dataset
MNIST_dataset_training = torchvision.datasets.ImageFolder('/storage/Suduko_solver_data/data/mnist_png/training',transform=Image_transformation_function)
#MNIST_dataset_training.to(device_to_use)
# Loading training data in dataloader into a batch size of 64
training_data_loader = torch.utils.data.DataLoader(MNIST_dataset_training,
                                          batch_size=64,
                                          shuffle=True)
#training_data_loader.to(device_to_use)
# Creating testing dataset from MNIST dataset
MNIST_dataset_testing = torchvision.datasets.ImageFolder('/storage/Suduko_solver_data/data/mnist_png/testing',transform=Image_transformation_function)
#MNIST_dataset_training.to(device_to_use)
# Loading test data in dataloader into a batch size of 512
test_data_loader = torch.utils.data.DataLoader(MNIST_dataset_testing,
                                          batch_size=512,
                                          shuffle=True)

#examples = enumerate(test_data_loader.dataset)
#batch_idx, (example_data, example_targets) = next(examples)
#print(batch_idx, example_data[0], example_targets)
#print(example_data.shape)
#print(training_data_loader.dataset,test_data_loader.dataset,test_data_loader.batch_size,test_data_loader.num_workers)
#for data,target in test_data_loader.dataset :
#    print(data,target)
# For veiwing image from dataset
#Image.open(training_data_loader.dataset.imgs[0][0])
#torchvision.transforms.ToPILImage().(MNIST_dataset_training.imgs[0][0])
#transform_function = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.ToTensor(),torchvision.transforms.Grayscale()])
#Image.open(MNIST_dataset_training.imgs[0][0])

# Randomly viewing image from dataset
#print(MNIST_dataset_training.imgs[1000])
#input_image = Image.open(MNIST_dataset_training.imgs[0][0])
input_image = Image.open(training_data_loader.dataset.imgs[0][0])
#input_image = input_image.convert('L',dither=1)
print(type(input_image),input_image.size,input_image.getcolors())
plt.figure()
plt.imshow(input_image,cmap='gray')


# Defining cnn model
class cnn_model(nn.Module) :
    def __init__(self) :
        super(cnn_model , self).__init__()
        self.conv1 = nn.Conv2d(1,20,kernel_size=5)
        self.conv2 = nn.Conv2d(20,40,kernel_size=5)
        self.conv2_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(640,120)
        self.fc2 = nn.Linear(120,10)
        
    def forward(self , input_x) :
        input_x = F.relu(F.max_pool2d(self.conv1(input_x),2))
        input_x = F.relu(F.max_pool2d(self.conv2_dropout(self.conv2(input_x)),2))
        input_x = input_x.view(-1,640)
        input_x = self.fc1(input_x)
        input_x = F.dropout(input_x,training=self.training)
        input_x = self.fc2(input_x)
        return F.log_softmax(input_x)

# Initalizing model parameters over which model to be trained
#learning_rate = 0.01
learning_rate = 0.001
momentum = 0.6
#no_of_epochs = 80
no_of_epochs = 26

Model = cnn_model()
# Loading model to device ( CPU or GPU - if available )
Model.to(device_to_use)
# Initalizing optimizer
optimizer = optim.Adam(Model.parameters(),lr=learning_rate,betas=(0.9,0.999))#,momentum=momentum)

# Variables for storing losses in each epoch
train_losses = []
test_losses = []
train_counter = []
test_counter = [i*len(training_data_loader.dataset) for i in range(no_of_epochs+1)]

#print(test_counter)

# Function for training model for each epoch
def model_training( no_of_epochs ) :
    Model.train()
    # Loss defining
    loss = nn.CrossEntropyLoss()
    # Looping for each batch in training data to calculate loss
    for batch , ( input_Data , input_label ) in enumerate(training_data_loader) :
        #input_Data.to(device_to_use)
        #input_label.to(device_to_use)
        # Loading data to device ( CPU or GPU - if available )
        input_Data , input_label = input_Data.to(device_to_use) , input_label.to(device_to_use)
        # Sets gradients of all model parameters to zero.
        optimizer.zero_grad()
        output = Model(input_Data)
        #train_loss = F.nll_loss(output , input_label)
        # Loss calculation
        train_loss = loss(output , input_label)
        # Computes the derivative of the loss w.r.t. the parameters
        train_loss.backward()
        optimizer.step()
        if batch % 10 == 0 :
            print('Training - Epoch - {} [ {}/{} ] \t Loss - {:.6f}'.
                  format(no_of_epochs,batch*len(input_Data),len(training_data_loader.dataset),train_loss.item()))
            train_losses.append(train_loss.item())
            train_counter.append( (batch*64) + (no_of_epochs-1)*len(training_data_loader.dataset) )
        # Saving model weights for each epoch
        torch.save(Model.state_dict() , '/storage/Suduko_solver_data/model_2.pth')
        torch.save(optimizer.state_dict() , '/storage/Suduko_solver_data/optimizer_2.pth')
        
# Function for test model
def model_testing() :
    # Setting model to eval mode
    Model.eval()
    # Initalize loss and no of correct predictions to 0
    test_loss = 0
    correct_prediction = 0
    # Initalize loss
    loss = nn.CrossEntropyLoss(size_average=False)
    # Model won’t be able to backprop 
    with torch.no_grad() :
        for test_data , test_label in test_data_loader:
            #print(test_data)
            #print(test_data.to(device_to_use))
            # Loading data to device ( CPU or GPU - if available )
            test_data , test_label = test_data.to(device_to_use) , test_label.to(device_to_use)
            #test_label.to(device_to_use)
            output = Model(test_data)
            #test_loss += F.nll_loss(output,test_label,size_average=False).item()
            # Calculating loss and prediction
            test_loss += loss(output,test_label).item()
            prediction = output.data.max(1,keepdim=True)[1]
            correct_prediction += prediction.eq(test_label.data.view_as(prediction)).sum()
        test_loss /= len(test_data_loader.dataset)
        test_losses.append(test_loss)
        print('Test - Average loss - {:.4f} {}/{} '.
                  format(test_loss,correct_prediction,len(test_data_loader.dataset)))

# Ruuning test without any training to get average loss and correct predictions
model_testing()

# Training model for defined no_of_epochs and saving model weights
for x in range(1,no_of_epochs+1) :
    model_training(x)
    model_testing()

#print(test_counter, test_losses)

# Plotting training and test losses for all epochs
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig

#print(train_counter[-1], train_losses[-1])
#model_testing()
#test_counter.append(test_counter[-1]+60000)
#test_counter[-1]
#for x in range(21,26) :
#    model_training(x)
#    test_counter.append(test_counter[-1]+61507)
#    model_testing()
#model_training(37)
#test_counter.append(test_counter[-1]+61507)
#model_testing()
