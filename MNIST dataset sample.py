import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0')

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
    def forward(self, x):
        # input layer
        x = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.sigmoid(x)
        return x
    
class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 120)
        self.fc2 = nn.Linear(120, 256)
        self.fc3 = nn.Linear(256, 512)
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
        x = self.fc3(x)
        x = F.relu(x)
        #hidden linear layer
        x = self.out(x)
        x = F.tanh(x)
        return x
    
batch_size = 100

train_set = torchvision.datasets.MNIST(
    root = './train_dataset/FashionMNIST',
    train = True,
    download = False,
    transform = transforms.Compose([
        transforms.ToTensor()
        ]))

D = D()

G = G()

    
data_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)

riterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

num_epochs = 10

total_step = len(data_loader)

for epoch in range(1):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion = nn.BCELoss(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, 64)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, 64)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
        
        
        
        

