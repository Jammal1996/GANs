# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 19:46:22 2023

@author: ranad
"""

# Importing libraries
# --------------------------------------------------------------------
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Initial setup 
# --------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Params
# --------------------------------------------------------------------
num_workers = 0
batch_size = 32

# Loading data
# --------------------------------------------------------------------
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True,
download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
num_workers=num_workers)

# Visualize
# --------------------------------------------------------------------
dataiter = iter(train_loader)
images, labels = dataiter.next()
img = np.squeeze(images[20]) # choose random index number 
fig = plt.figure(figsize = (3,3)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap="gray")




