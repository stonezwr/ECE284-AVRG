import torch
from torch import nn 
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import os 
import json
import time

from svrg import SVRG_k, SVRG
from utils import MNIST_dataset, CIFAR10_dataset, MNIST_two_layers, CIFAR10_ConvNet

parser = argparse.ArgumentParser(description="Train SVRG/AVRG.")
parser.add_argument('--optimizer', type=str, default="None",
        help="Choose optimizer (Choices: SVRG, AVRG. default: None(SGD)).")
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
parser.add_argument('-gpu', type=int, default=0, 
        help='GPU device to use (default: 0)')

def accuracy(yhat, labels):
    _, indices = yhat.max(1)
    return (indices == labels).sum().data.item() / float(len(labels))

def train(model_k, model_snapshot, optimizer_k, optimizer_snapshot, train_loader, loss_fn, flatten_img=True):
    model_k.train()
    model_snapshot.train()

    # calculate the mean gradient
    optimizer_snapshot.zero_grad()  # zero_grad outside for loop, accumulate gradient inside
    for images, labels in train_loader:
        images = images.to(device)
        if flatten_img:
            images = images.view(images.shape[0], -1)
        yhat = model_snapshot(images)
        labels = labels.to(device)
        snapshot_loss = loss_fn(yhat, labels) / len(train_loader)
        snapshot_loss.backward()

    # pass the current paramesters of optimizer_0 to optimizer_k 
    u = optimizer_snapshot.get_param_groups()
    optimizer_k.set_u(u)
    
    for images, labels in train_loader:
        images = images.to(device)
        if flatten_img:
            images = images.view(images.shape[0], -1)
        yhat = model_k(images)
        labels = labels.to(device)
        loss_iter = loss_fn(yhat, labels)

        # optimization 
        optimizer_k.zero_grad()
        loss_iter.backward()    

        yhat2 = model_snapshot(images)
        loss2 = loss_fn(yhat2, labels)

        optimizer_snapshot.zero_grad()
        loss2.backward()

        optimizer_k.step(optimizer_snapshot.get_param_groups())

        # logging 
        acc_iter = accuracy(yhat, labels)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    # update the snapshot 
    optimizer_snapshot.set_param_groups(optimizer_k.get_param_groups())
    
    return loss.avg, acc.avg


def test(model, test_loader, loss_fn):
    model.eval()
    loss = AverageCalculator()
    acc = AverageCalculator()

    for images, labels in val_loader:
        images = images.to(device)
        if flatten_img:
            images = images.view(images.shape[0], -1)
        yhat = model(images)
        labels = labels.to(device)

        # logging 
        loss_iter = loss_fn(yhat, labels)
        acc_iter = accuracy(yhat, labels)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    return loss.avg, acc.avg

if __name__ == "__main__":
     try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)
    args_dict = vars(args)

    if not args.optimizer in ['AVRG', 'SVRG']:
        raise ValueError("--optimizer must be 'AVRG' or 'SVRG'.")

    fname = "log_" + time.strftime("%Y%m%d-%H%M%S") + "_" + args.optimizer + "_" + args.nn_model + ".txt"
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(fname)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('Args: {}'.format(args))
    
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.manual_seed_all(43)
    np.random.seed(43)

    # load the data
    if args.dataset == "MNIST":
        train_set, val_set = MNIST_dataset()
        flatten_img = True
        NN_model = MNIST_two_layers
    elif args.dataset == "CIFAR10":
        train_set, val_set = CIFAR10_dataset() 
        flatten_img = False
        NN_model = CIFAR10_ConvNet
    else:
        raise ValueError("Unknown dataset")
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE_LARGE, shuffle=True, num_workers=4)

    model = NN_model().cuda()
    if args.optimizer == 'SVRG':
        model_snapshot = NN_model().cuda()

    lr = args.lr  # learning rate
    n_epoch = args.n_epoch  # the number of epochs

    # The loss function 
    if args.nn_model == "CIFAR10_convnet":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.NLLLoss()  

    # the optimizer 
    if args.optimizer == "None":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SVRG":
        optimizer = SVRG_k(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        optimizer_snapshot = SVRG_Snapshot(model_snapshot.parameters())
    elif args.optimizer == "AVRG":
        optimizer = AVRG(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unknown optimizer")

    # store training stats
    train_loss_all, val_loss_all = [], []
    train_acc_all, val_acc_all = [], []

    for epoch in range(n_epoch):
        t0 = time.time()

        # training 
        if args.optimizer == "SGD":
            train_loss, train_acc = train_epoch_SGD(model, optimizer, train_loader, loss_fn, flatten_img=flatten_img)
        elif args.optimizer == "SVRG":
            train_loss, train_acc = train_epoch_SVRG(model, model_snapshot, optimizer, optimizer_snapshot, train_loader, loss_fn, flatten_img=flatten_img)
        
        logging.info('-' * 96)
        logging.info('train | end of epoch {:3d} | time: {:5.2f}s | train acc {:3.2f}% ({:3.2f}%) | train loss {:.6f} | '.format(epoch, (time.time() - epoch_start_time), train_acc, best_train, train_loss))
        logging.info('-' * 96) 

        # validation 
        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn)

        logging.info('-' * 96)
        logging.info('test_model | end of epoch {:3d} | time: {:5.2f}s | test acc {:3.2f}% ({:3.2f}%) | test loss {:.6f} | '.format(epoch, (time.time() - epoch_start_time), test_acc, best_test, test_loss))
        logging.info('-' * 96)
        
        train_loss_all.append(train_loss)  # averaged loss for the current epoch 
        train_acc_all.append(train_acc)
        val_loss_all.append(val_loss)  
        val_acc_all.append(val_acc)
        
