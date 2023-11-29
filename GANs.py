import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter




         # ================================================================== #
         #                      Discriminator Network                         #
         # ================================================================== #

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        #Convolution layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)
        #Fletten (dense) layers
        self.fc1 = nn.Linear(12*4*4, 120)
        self.fc2 = nn.Linear(120, 60)
        self.out = nn.Linear(60, 1)
    def forward(self, x):
        # input layer
        x = x
        # hidden conv layer
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        # hidden conv layer
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        # hidden linear layer
        x = x.reshape(-1, 12*4*4)
        x = self.fc1(x)
        x = F.relu(x)
        # hidden linear layer
        x = self.fc2(x)
        x = F.relu(x)
        # output layer
        x = self.out(x)
        # Sigmoid function
        x = F.sigmoid(x)
        return x
    
         # ================================================================== #
         #                        Generator Network                           #
         # ================================================================== #

class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 512)
        self.out = nn.Linear(512, 784)
    def forward(self, x):
        #input layer
        x = x
        #hidden linear layer
        x = self.fc1(x)
        x = F.relu(x)
        #hidden linear layer
        x = self.fc2(x)
        x = F.relu(x)
        #hidden linear layer
        x = self.out(x)
        x = F.tanh(x)
        return x

         # ================================================================== #
         #                      Training Settings                             #
         # ================================================================== #

                        #=================================#
                        #          MNIST dataset          #
                        #=================================#
                        
train_set = torchvision.datasets.MNIST(
    root = './train_dataset/MNIST', train = True,
    download = False, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5])]))


                        #=================================#
                        #         Normlizing Data         #
                        #=================================#
                        
# loader = DataLoader(train_set, batch_size=len(train_set), shuffle = False)

# loader_norm = DataLoader(train_set_norm, batch_size = len(train_set_norm), shuffle = False)

# data = next(iter(loader))

# data_norm = next(iter(loader_norm))

# m = data[0].mean()
# s = data[0].std()

# m_norm = data_norm[0].mean()
# s_norm = data_norm[0].std()


# print(m)
# print(s)

# print(m_norm)
# print(s_norm)

# plt.hist(data[0].flatten())
# plt.axvline(data[0].mean())
# plt.axvline(data[0].std())
# plt.xlabel('pixel value')
# plt.ylabel('number of pixels')


# plt.hist(data_norm[0].flatten())
# plt.axvline(data_norm[0].mean())
# plt.axvline(data_norm[0].std())
# plt.xlabel('pixel value')
# plt.ylabel('number of pixels')

                        #=================================#
                        #           Training Set          #
                        #=================================#
                        
my_device = torch.device('cuda:0')

batch_size = 100

num_epochs = 200

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)

disc_network = Disc().to(my_device)

gen_network = Gen().to(my_device)
                      
disc_optimizer = optim.Adam(disc_network.parameters(), lr = 0.0002)

gen_optimizer = optim.Adam(gen_network.parameters(), lr = 0.0002)

criterion = nn.BCELoss()

        
          # ================================================================== #
          #                      Adversarial Training                          #
          # ================================================================== #

tb_disc_graph = SummaryWriter()

tb_gen_graph = SummaryWriter()

tb_real_imgs = SummaryWriter()

tb_fake_imgs = SummaryWriter()

batch_num = len(train_loader) # Number of batches or Number of iterations

step_graph = 0

step_img = 0


for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(my_device)
                        #=================================#
                        #      Discriminator Training     #
                        #=================================#
                        
        real_labels = torch.ones(batch_size, 1).to(my_device)
        preds = disc_network(imgs)
        disc_loss_real = criterion(preds, real_labels).to(my_device)
    
        fake_labels = torch.zeros(batch_size, 1).to(my_device)
        z_noise_set = torch.randn(batch_size, 64).to(my_device)
        fake_imgs = gen_network(z_noise_set)
        fake_imgs = fake_imgs.reshape(batch_size, 1, 28, 28)
        fake_imgs = fake_imgs.to(my_device)
        preds = disc_network(fake_imgs)
        disc_loss_fake = criterion(preds, fake_labels)

        disc_loss = disc_loss_real + disc_loss_fake
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
        
        
                        #=================================#
                        #         Generator Training      #
                        #=================================#
                        
        z_noise_set = torch.randn(batch_size, 64).to(my_device)
        fake_imgs = gen_network(z_noise_set)
        fake_imgs = fake_imgs.reshape(batch_size, 1, 28, 28)
        fake_imgs = fake_imgs.to(my_device)
        preds = disc_network(fake_imgs)
        gen_loss = criterion(preds, real_labels)
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        tb_disc_graph.add_scalar('Discriminator Loss', disc_loss, global_step = step_graph)
        tb_gen_graph.add_scalar('Generator Loss', gen_loss, global_step = step_graph)
        step_graph = step_graph + 1    
        if (i+1) % 100 == 0:
            print(f"Epoch num[{epoch}/{num_epochs}], Batch num [{i+1}/{batch_num}], Discriminator Loss: {disc_loss:.4f}, Generator Loss: {gen_loss:.4f}")
            img_grid_real = torchvision.utils.make_grid(imgs, normalize=True) 
            img_grid_fake = torchvision.utils.make_grid(fake_imgs, normalize=True) 
            tb_real_imgs.add_image("MNIST Real Images", img_grid_real, global_step = step_img)
            tb_fake_imgs.add_image("MNIST Fake Images", img_grid_fake, global_step = step_img)
            step_img = step_img + 1

   
                    

                        
                    
        
        
        
        
        
        









    
        
        
        