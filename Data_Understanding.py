import torch 
import torchvision # the package provides the MNIST dataset "handwritten digits"
import torchvision.transforms as transforms # To put the data into tensor form 
import numpy as np
import matplotlib.pyplot as plt

# To run the code on the GPU (Nvidia GeForce GTX 1050 graphic card)

my_device = torch.device('cuda', 0)

#-------Extracting & Transforming data "The dataset is already installed on the desk"------------
train_dataset = torchvision.datasets.MNIST(
    root = './train_dataset/MNIST', train = True, download = False,
    transform = transforms.Compose([
        transforms.ToTensor()]))

test_dataset = torchvision.datasets.MNIST(
    root = './test_dataset/MNIST', train = False, download = False,
    transform = transforms.Compose([
        transforms.ToTensor()]))

#----------------------------------Loading data-------------------------------------------- 
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size = 10, shuffle = True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size = 10, shuffle = True)

#----------------------------dividing the labeling samples----------------------------------

def dis_train_data(d):
    for data in train_loader:
        img, labels = data
        for label in labels:
            d[int(label)] = d[int(label)] + 1
    return d

def dis_test_data(d):
    for data in test_loader:
        img, labels = data
        for label in labels:
            d[int(label)] = d[int(label)] + 1
    return d

# labels_dict_train = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 
#           8:0, 9:0}

# labels_dict_test = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}     

# dis1 = dis_train_data(labels_dict_train)

# dis2 = dis_test_data(labels_dict_test)

# print(dis1)

# print(dis2)


#--------------------------------Dataset Samples----------------------------------

batch = next(iter(train_loader)) # extracting a random batch for visualization purposes

imgs, labels = batch # spliting the list into seperate list

grid = torchvision.utils.make_grid(imgs, nrow = 10)

plt.imshow(grid.permute(1,2,0))













        


    
        
    
    








