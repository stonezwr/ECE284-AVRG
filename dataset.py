import torch
from torchvision import transforms, datasets 
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def MNIST_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST("../data/MNIST", download=True, train=True, transform=transform)
    test_set = datasets.MNIST('../data/MNIST', download=True, train=False, transform=transform)
    return train_set, test_set

def CIFAR10_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = datasets.CIFAR10("../data/CIFAR10", download=True, train=True, transform=transform_test)
    test_set = datasets.CIFAR10('../data/CIFAR10', download=True, train=False, transform=transform_train)
    return train_set, test_set

def MNIST_dataset_flat():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))])
    train_set = datasets.MNIST("../data/MNIST", download=True, train=True, transform=transform)
    test_set = datasets.MNIST('../data/MNIST', download=True, train=False, transform=transform)
    return train_set, test_set

def CIFAR10_dataset_flat():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(-1))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = datasets.CIFAR10("../data/CIFAR10", download=True, train=True, transform=transform_test)
    test_set = datasets.CIFAR10('../data/CIFAR10', download=True, train=False, transform=transform_train)
    return train_set, test_set
