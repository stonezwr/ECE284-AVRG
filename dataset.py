import torch
from torchvision import transforms, datasets 
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import sklearn
import numpy as np


class MNIST_data(Dataset):
    def __init__(self, path, train, transform, batch_size):
        self.batch_size = batch_size
        self.mnist = datasets.MNIST(root=path,
                                        download=True,
                                        train=train,
                                        transform=transform)
        
    def __getitem__(self, index):
        start = index*self.batch_size
        datas = []
        targets = []
        if self.batch_size > 1:
            for i in range(self.batch_size):
                ind = start + i
                data, target = self.mnist[ind]
                datas.append(data)
                targets.append(target)
            datas = torch.stack(datas)
            targets = torch.tensor(targets)
        else:
            datas, targets = self.mnist[index]
            datas.unsqueeze_(0)
            targets = torch.tensor(targets).unsqueeze_(0)
        return datas, targets, index

    def __len__(self):
        return int(len(self.mnist) / self.batch_size)

    def get_dataset(self):
        return self.mnist

class CIFAR10_data(Dataset):
    def __init__(self, path, train, transform, batch_size):
        self.batch_size = batch_size
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=train,
                                        transform=transform)
        
    def __getitem__(self, index):
        start = index*self.batch_size
        datas = []
        targets = []
        if self.batch_size > 1:
            for i in range(self.batch_size):
                ind = start + i
                data, target = self.cifar10[ind]
                datas.append(data)
                targets.append(target)
            datas = torch.stack(datas)
            targets = torch.tensor(targets)
        else:
            datas, targets = self.cifar10[index]
            datas.unsqueeze_(0)
            targets = torch.tensor(targets).unsqueeze_(0)
        return datas, targets, index

    def __len__(self):
        return int(len(self.cifar10) / self.batch_size)

    def get_dataset(self):
        return self.cifar10


class MNIST_binary_data(Dataset):
    def __init__(self, path, train, transform):
        self.mnist = datasets.MNIST(root=path,
                                        download=True,
                                        train=train,
                                        transform=transform)
        self.mnist_binary = []
        self.x = []
        self.y = [] 
        count_0 = 0
        count_1 = 0
        for i in range(len(self.mnist)):
            d, t = self.mnist[i]
            if t == 0 and count_0 < 1000:
                self.x.append(d)
                self.y.append(t)
                count_0 += 1
            if t == 1 and count_1 < 1000:
                self.x.append(d)
                self.y.append(t)
                count_1 += 1
        self.x = torch.stack(self.x, dim=0).numpy()
        self.y = torch.tensor(self.y)
        normalizer = sklearn.preprocessing.Normalizer()
        self.x = torch.tensor(normalizer.fit_transform(self.x))
        self.y = (self.y-.5)*2
        
    def __getitem__(self, index):
        data = self.x[index, :]
        target = self.y[index]
        return data, target, index

    def __len__(self):
        return int(self.y.shape[0])

    def get_samples(self):
        return self.x

    def get_labels(self):
        return self.y

class CIFAR10_binary_data(Dataset):
    def __init__(self, path, train, transform):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=train,
                                        transform=transform)
        self.x = []
        self.y = [] 
        count_0 = 0
        count_1 = 0
        for i in range(len(self.cifar10)):
            d, t = self.cifar10[i]
            if t == 0 and count_0 < 10000:
                self.x.append(d)
                self.y.append(t)
                count_0 += 1
            if t == 1 and count_1 < 10000:
                self.x.append(d)
                self.y.append(t)
                count_1 += 1
        self.x = torch.stack(self.x, dim=0).numpy()
        self.y = torch.tensor(self.y)
        normalizer = sklearn.preprocessing.Normalizer()
        self.x = torch.tensor(normalizer.fit_transform(self.x))
        self.y = (self.y-.5)*2
        
    def __getitem__(self, index):
        data = self.x[index, :]
        target = self.y[index]
        return data, target, index

def MNIST_dataset_dist(rank):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(f"../data/data-{rank}", download=True, train=True, transform=transform)
    test_set = datasets.MNIST(f"../data/data-{rank}", train=False, transform=transform)
    return train_set, test_set

def CIFAR10_dataset_dist(rank):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = datasets.CIFAR10(f"../data/data-{rank}", download=True, train=True, transform=transform_test)
    test_set = datasets.CIFAR10(f'../data/data-{rank}', train=False, transform=transform_train)
    return train_set, test_set

def MNIST_dataset_flat_dist(rank):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))])
    train_set = datasets.MNIST(f"../data/data-{rank}", download=True, train=True, transform=transform)
    test_set = datasets.MNIST(f'../data/data-{rank}', train=False, transform=transform)
    return train_set, test_set

def CIFAR10_dataset_flat_dist(rank):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(-1))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = datasets.CIFAR10(f"../data/data-{rank}", download=True, train=True, transform=transform_test)
    test_set = datasets.CIFAR10(f'../data/data-{rank}', train=False, transform=transform_train)
    return train_set, test_set

def MNIST_dataset():
    def __len__(self):
        return int(self.y.shape[0])

    def get_samples(self):
        return self.x

    def get_labels(self):
        return self.y

        
def MNIST_dataset(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_set = MNIST_data('../data/MNIST', True, transform, batch_size)
    test_set = MNIST_data('../data/MNIST', False, transform, batch_size)
    return train_set, test_set

def CIFAR10_dataset(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = CIFAR10_data("../data/CIFAR10", True, transform_train, batch_size)
    test_set = CIFAR10_data("../data/CIFAR10", False, transform_train, batch_size)
    return train_set, test_set

def MNIST_dataset_flat(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))])
    train_set = MNIST_data("../data/MNIST", True, transform, batch_size)
    test_set = MNIST_data("../data/MNIST", False, transform, batch_size)
    return train_set, test_set

def CIFAR10_dataset_flat(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(-1))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = CIFAR10_data("../data/CIFAR10", True, transform_train, batch_size)
    test_set = CIFAR10_data("../data/CIFAR10", False, transform_train, batch_size)
    return train_set, test_set

def MNIST_dataset_binary_flat():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))])
    train_set = MNIST_binary_data("../data/MNIST", True, transform)
    return train_set

def CIFAR10_dataset_binary_flat():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))])
    train_set = CIFAR10_binary_data("../data/CIFAR10", True, transform_train)
    return train_set
