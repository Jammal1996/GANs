# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 13:34:11 2023

@author: ranad
"""

import torch 
import pickle as pkl
from gen import Generator
from discr import Discriminator
from loss import real_loss, fake_loss
import torch.optim as optim
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

# Initial setup 
# --------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
train = True

# Params
# --------------------------------------------------------------------
num_workers = 0
batch_size = 16
lr = 0.001
num_epochs = 100

# Discriminator hyperparams
# Size of input image to discriminator (28*28)
input_size = 784
# Size of last hidden layer in the discriminator
d_hidden_size = 32
# Size of discriminator output (real or fake)
d_output_size = 1

# Generator hyperparams
# Size of latent vector to give to generator
z_size = 100
# Size of first hidden layer in the generator
g_hidden_size = 32
# Size of discriminator output (generated image)
g_output_size = 784
# --------------------------------------------------------------------

# Loading real data
# --------------------------------------------------------------------
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True,
download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
num_workers=num_workers)
# --------------------------------------------------------------------

# Instantiate the GAN models 
d = Discriminator(input_size, d_hidden_size, d_output_size).to(device)
g = Generator(z_size, g_hidden_size, g_output_size).to(device)

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(d.parameters(), lr)
g_optimizer = optim.Adam(g.parameters(), lr)

# keep track of loss, and generated "fake" samples
losses = []
samples = []

print_every = 400

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size = 16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float().to(device)

# train the network
d.train()
g.train()

dataiter = iter(train_loader)
images, labels = dataiter.next()

if train:
    for epoch in range(num_epochs):
        
        for batch_i, (real_images, _) in enumerate(train_loader):
            
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            ## Important rescaling step ## 
            real_images = real_images*2 - 1  # rescale input images from [0,1) to [-1, 1)
            
            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================
            
            d_optimizer.zero_grad()
            
            # 1. Train with real images
            
            # Compute the discriminator losses on real images 
            # smooth the real labels
            d_real = d(real_images)
            d_real_loss = real_loss(d_real, smooth=True)
            
            # 2. Train with fake images
            
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float().to(device)
            fake_images = g(z)
            
            # Compute the discriminator losses on fake images        
            d_fake = d(fake_images)
            d_fake_loss = fake_loss(d_fake)
            
            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            
            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================
            g_optimizer.zero_grad()
            
            # 1. Train with fake images and flipped labels
            
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float().to(device)
            fake_images = g(z)
            
            # Compute the discriminator losses on fake images 
            # using flipped labels!
            d_fake = d(fake_images)
            g_loss = real_loss(d_fake) # use real loss to flip labels
            
            # perform backprop
            g_loss.backward()
            g_optimizer.step()
            
            # Print some loss stats
            if batch_i % print_every == 0:
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, num_epochs, d_loss.item(), g_loss.item()))
        
        
        ## AFTER EACH EPOCH##
        # append discriminator loss and generator loss
        losses.append((d_loss.item(), g_loss.item()))
        
        # generate and save sample, fake images
        g.eval() # eval mode for generating samples
        samples_z = g(fixed_z)
        samples.append(samples_z)
        g.train() # back to train mode
    
    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    



