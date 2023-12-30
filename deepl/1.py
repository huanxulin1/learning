import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from matplotlib import pyplot as plt

batch_size = 512

#utils
def one_hot(x, depth = 10):
    out = torch.zeros(x.size(0), depth)
    idx = torch.LongTensor(x).view(-1, 1)
    out.scatter_(dim = 1, index = idx, value = 1)
    return out

#1 load dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081))
                               ])),
    batch_size = batch_size, shuffle = True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081))
                               ])),
    batch_size = batch_size, shuffle = False
)

x, y = next(iter(train_loader))


#2 create model

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28*28, 512)
        self.fc11 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        #x:{b, 1, 28, 28}
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc11(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x

#3 train
net = Net()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
for epoch in range(3):
    for batch_size , (x, y) in enumerate(train_loader):
        
        x = x.view(x.size(0), 28*28)
        out = net(x)
        y_onehot = one_hot(y)
        loss = F.mse_loss(out, y_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_size % 10 == 0:
            print('epoch = ',epoch,'batch = ', batch_size,'loss = ', loss.item())
        
#4 test
total_correct = 0
for x,y in test_loader:
    x = x.view(x.size(0), 28*28)
    y_onehot = one_hot(y)
    out = net(x)
    pred = out.argmax(dim = 1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_number = len(test_loader.dataset)
acc = total_correct / total_number
print('acc = ',acc)