import torch
from torch import nn 
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import os 
import time
import logging
import sys

from svrg import SVRG, SVRG_0
from avrg import AVRG
from utils import MNIST_dataset, CIFAR10_dataset, MNIST_two_layers, CIFAR10_ConvNet

parser = argparse.ArgumentParser(description="Train SVRG/AVRG.")
parser.add_argument('--optimizer', type=str, default="SGD",
        help="Choose optimizer (Choices: SVRG, AVRG. default: SGD).")
parser.add_argument('--dataset', type=str, default="MNIST",
        help="datasets (default: MNIST).")
parser.add_argument('--n_epoch', type=int, default=100,
        help="number of training iterations (default: 100).")
parser.add_argument('--lr', type=float, default=0.001,
        help="learning rate (default: 0.001).")
parser.add_argument('--batch_size', type=int, default=256,
        help="batch size (default: 256).")
parser.add_argument('--weight_decay', type=float, default=0.0001,
        help="regularization strength (default: 0.0001).")
parser.add_argument('--gpu', type=int, default=0, 
        help='GPU device to use (default: 0).')
parser.add_argument('--seed', type=int, default=3, 
        help='set seed (default: 3).')

def train_SVRG(model_i, model_0, optimizer_i, optimizer_0, train_loader, loss_fn):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    # outer loop 
    optimizer_0.zero_grad()  
    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model_0(inputs)
        outer_loss = loss_fn(outputs, labels) / len(train_loader) 
        outer_loss.backward()

    # pass the current paramesters of optimizer_0 to optimizer_i
    optimizer_i.set_outer_params(optimizer_0.get_param_groups())

    # inner loop
    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        outputs_i = model_i(inputs)
        loss_i = loss_fn(outputs_i, labels)

        optimizer_i.zero_grad()  
        loss_i.backward()

        outputs_0 = model_0(inputs)
        loss_0 = loss_fn(outputs_0, labels)

        optimizer_0.zero_grad()
        loss_0.backward()

        optimizer_i.step(optimizer_0.get_param_groups())

        # logging 
        _, indices = outputs_i.max(1)
        correct += (indices == labels).sum().data.item()
        total += len(labels)
        total_loss += loss_i
        
        acc = correct * 100 / float(total)
        loss_avg = total_loss / float(total)
        logging.info('| epoch {:3d} | train | {:5d}/{:5d} samples | lr {:.5f} | train_acc {:3.2f}% | train_loss {:.6f}'.format(epoch, total, len(train_loader.dataset), optimizer.param_groups[0]['lr'], acc, loss_avg))
        print('\033[2A')
    
    # update the outer loop 
    optimizer_0.set_param_groups(optimizer_i.get_param_groups())

    return loss_avg, acc

def train(model, optimizer, train_loader, loss_fn):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()

        _, indices = outputs.max(1)
        correct += (indices == labels).sum().data.item()
        total += len(labels)
        total_loss += loss
        
        acc = correct * 100 / float(total)
        loss_avg = total_loss / float(total)
        logging.info('| epoch {:3d} | train | {:5d}/{:5d} samples | lr {:.5f} | train_acc {:3.2f}% | train_loss {:.6f}'.format(epoch, total, len(train_loader.dataset), optimizer.param_groups[0]['lr'], acc, loss_avg))
        print('\033[2A')
    
    return loss_avg, acc


def test(model, test_loader, loss_fn):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        _, indices = outputs.max(1)
        correct += (indices == labels).sum().data.item()
        total += len(labels)
        total_loss += loss
        
        acc = correct * 100 / float(total)
        loss_avg = total_loss / float(total)
        logging.info('| epoch {:3d} | test | {:5d}/{:5d} samples| train_acc {:3.2f}% | train_loss {:.6f}'.format(epoch, total, len(test_loader.dataset), acc, loss_avg))
        print('\033[2A')
    
    return loss_avg, acc

if __name__ == "__main__":
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if not args.optimizer in ['AVRG', 'SVRG', 'SGD']:
        raise ValueError("--optimizer must be 'AVRG' or 'SVRG' or 'SGD'.")

    fname = "log_" + time.strftime("%Y%m%d-%H%M%S") + "_" + args.optimizer + "_" + args.dataset + ".txt"
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(fname)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('Args: {}'.format(args))
    
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load the data
    if args.dataset == "MNIST":
        train_set, val_set = MNIST_dataset()
        NN_model = MNIST_two_layers
    elif args.dataset == "CIFAR10":
        train_set, val_set = CIFAR10_dataset() 
        NN_model = CIFAR10_ConvNet
    else:
        raise ValueError("Unknown dataset")
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # The network
    model = NN_model().cuda()
    if args.optimizer == 'SVRG':
        model_0 = NN_model().cuda()

    lr = args.lr  # learning rate
    n_epoch = args.n_epoch  # the number of epochs

    # The loss function 
    if args.dataset == "CIFAR10":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.NLLLoss()  

    # The optimizer 
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SVRG":
        optimizer = SVRG(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        optimizer_0 = SVRG_0(model_0.parameters())
    elif args.optimizer == "AVRG":
        optimizer = AVRG(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unknown optimizer")

    best_train = 0
    best_test = 0
    for epoch in range(n_epoch):
        t0 = time.time()

        # training 
        if args.optimizer == "SVRG":
            train_loss, train_acc = train_SVRG(model, model_0, optimizer, optimizer_0, train_loader, loss_fn)
        else:
            train_loss, train_acc = train(model, optimizer, train_loader, loss_fn)
        
        best_train = max(train_acc, best_train)

        logging.info('-' * 96)
        logging.info('train | end of epoch {:3d} | time: {:5.2f}s | train acc {:3.2f}% ({:3.2f}%) | train loss {:.6f} | '.format(epoch, (time.time() - t0), train_acc, best_train, train_loss))
        logging.info('-' * 96) 

        t0 = time.time()
        # testing 
        test_loss, test_acc = test(model, test_loader, loss_fn)
        
        best_test = max(test_acc, best_test)

        logging.info('-' * 96)
        logging.info('test | end of epoch {:3d} | time: {:5.2f}s | test acc {:3.2f}% ({:3.2f}%) | test loss {:.6f} | '.format(epoch, (time.time() - t0), test_acc, best_test, test_loss))
        logging.info('-' * 96)
        
