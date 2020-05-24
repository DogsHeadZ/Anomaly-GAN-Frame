import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
a = torch.cuda.is_available()
print(a)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3))

transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])


train_ds = MNIST(root='./data', train=True, download=True, transform=transform)
train_dl = DataLoader(dataset=train_ds, batch_size=32, shuffle=True, drop_last=True)
for data in train_dl:
    print(data[0])