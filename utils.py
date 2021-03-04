import os
from torch import nn
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets 
import numpy as np


def get_binary_indices(dataset, class_names):
    indices = []
    count = 0 
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_names:
            indices.append(i)
            count += 1
    return indices

def MNIST_logistic_regression():
    model = nn.Linear(784, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    return model

def CIFAR10_logistic_regression():
    model = nn.Linear(32*32*3, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    return model

def MNIST_two_layers():
    input_size = 784
    hidden_sizes = [64, 128]
    output_size = 10

    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0], bias=False),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=False),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size, bias=False),
        nn.LogSoftmax(dim=1))

    return model

class CIFAR10_ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR10_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Logistic_Regression(torch.nn.Module):
    def __init__(self, w_star, rho):
        self.rho = rho
        self.w_star = torch.from_numpy(w_star).cuda()
        self.w_star = self.w_star.type(torch.float32)
        super(Logistic_Regression, self).__init__()

    def loss(self, y, y_hat):
        return torch.mean(torch.log(1+torch.exp(-y*y_hat)))

    def ER(self, train_loader, w):
        with torch.no_grad():
            xy_star = []
            xy = []
            for inputs, labels in train_loader:
                inputs = inputs.cuda()
                labels = labels.cuda()
                labels = (labels-0.5) * 2
                y_star = F.linear(inputs, self.w_star) 
                y = F.linear(inputs, w)
                xy_star.append(y_star*labels)
                xy.append(y*labels)
            xy_star = torch.stack(xy_star).view(-1)
            xy = torch.stack(xy).view(-1)
            v_star = torch.mean(torch.log(1+torch.exp(-xy_star))) + self.rho / 2 * torch.sum(torch.square(self.w_star)) 
            v = torch.mean(torch.log(1+torch.exp(-xy))) + self.rho / 2 * torch.sum(torch.square(w)) 
            ER = v-v_star
            MSD = torch.sum(torch.square(w - self.w_star)) / torch.sum(torch.square(self.w_star))
            return MSD, ER 

