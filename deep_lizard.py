import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.tensorboard import SummaryWriter


# To run the code on the GPU (Nvidia GeForce GTX 1050 graphic card)
my_device = torch.device('cuda:0')

train_set = torchvision.datasets.MNIST(
    root = './train_dataset/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
        ]))

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)
        
        self.fc1 = nn.Linear(12*4*4, 120)
        self.fc2 = nn.Linear(120, 60)
        self.out = nn.Linear(60, 10)
        
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
        return x

        
network = Network()

batch_size = 10
lr = 0.01
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)

optimizer = optim.Adam(network.parameters(), lr = lr)

def get_num_correct(preds, labels):
    return preds.argmax(dim = 1).eq(labels).sum().item()


for epoch in range(5):
    total_loss = 0
    total_correct = 0
    for imgs, labels in train_loader:
        preds = network(imgs)
        print(imgs.shape)
        print(labels.shape)
        break
        loss = F.cross_entropy(preds, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss = total_loss + loss.item() * batch_size
        total_correct = total_correct + get_num_correct(preds, labels)
    
    # tb.add_scalar('Loss', total_loss, epoch)
    # tb.add_scalar('Number Correct', total_correct, epoch)
    # tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

    # tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
    # tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
    # tb.add_histogram(
    #     'conv1.weight.grad'
    #     ,network.conv1.weight.grad
    #     ,epoch
    # )
        
    print(f"epoch: {epoch}, total correct: {total_correct}, Loss: {total_loss}")
    
# tb.close()
    
