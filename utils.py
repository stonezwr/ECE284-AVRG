import os
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets 
import numpy as np
import matplotlib.pyplot as plt

def MNIST_dataset():
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Download MNIST dataset and set the valset as the test test
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))])
    test_set = datasets.MNIST('../data/MNIST', download=True, train=False, transform=transform)
    train_set = datasets.MNIST("../data/MNIST", download=True, train=True, transform=transform)
    return train_set, test_set

def CIFAR10_dataset():
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Download MNIST dataset and set the valset as the test test
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_set = datasets.CIFAR10('../data/CIFAR10', download=True, train=False, transform=transform_train)
    train_set = datasets.CIFAR10("../data/CIFAR10", download=True, train=True, transform=transform_test)
    return train_set, test_set
    nn.Conv2d

def MNIST_two_layers():
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.LogSoftmax(dim=1))

    return model

class CIFAR10_ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR10_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

