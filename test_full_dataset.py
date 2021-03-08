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
import copy

from svrg import SVRG, SVRG_0
from avrg import AVRG, AVRG_0
from saga import SAGA
from utils import *
from dataset import *
from vgg import *

import matplotlib.pyplot as plt


def train_SVRG(model_0, model_i, optimizer_0, optimizer_i, train_loader, loss_fn):
    model_0.train()
    model_i.train()
    correct = 0
    total = 0
    total_loss = 0

    # SVRG outer loop 
    optimizer_0.reset_full_grad()  
    for inputs, labels, index in train_loader:
        inputs = inputs.squeeze_(0).cuda()
        labels = labels.squeeze_(0).cuda()
        outputs = model_0(inputs)
        outer_loss = loss_fn(outputs, labels) 
        optimizer_0.zero_grad()  
        outer_loss.backward()
        optimizer_0.update_full_grad()

    # pass the current paramesters of optimizer_0 to optimizer_i
    optimizer_i.set_full_grad(optimizer_0.get_full_grad())

    # inner loop
    for inputs, labels, index in train_loader:
        inputs = inputs.squeeze_(0).cuda()
        labels = labels.squeeze_(0).cuda()
        
        outputs_i = model_i(inputs)
        loss_i = loss_fn(outputs_i, labels)
        outputs_0 = model_0(inputs)
        loss_0 = loss_fn(outputs_0, labels)

        optimizer_i.zero_grad()  
        optimizer_0.zero_grad()
        loss_i.backward()
        loss_0.backward()

        optimizer_i.step(optimizer_0.get_param_groups())

        # logging 
        _, indices = outputs_i.max(1)
        correct += (indices == labels).sum().data.item()
        total += len(labels)
        total_loss += loss_i
        
        acc = correct * 100 / float(total)
        loss_avg = total_loss / float(total)
        logging.info('| epoch {:3d} | train SVRG | {:5d} samples | train_acc {:3.2f}% | train_loss {:.6f}'.format(epoch, total, acc, loss_avg))
        print('\033[2A')
    
    # update the outer loop 
    optimizer_0.set_param_groups(optimizer_i.get_param_groups())

    return loss_avg.item(), acc

def train_AVRG(model_0, model_i, optimizer_0, optimizer_i, train_loader, loss_fn):
    model_0.train()
    model_i.train()
    correct = 0
    total = 0
    total_loss = 0

    # inner loop
    for inputs, labels, index in train_loader:
        inputs = inputs.squeeze_(0).cuda()
        labels = labels.squeeze_(0).cuda()
        
        outputs_i = model_i(inputs)
        loss_i = loss_fn(outputs_i, labels)
        outputs_0 = model_0(inputs)
        loss_0 = loss_fn(outputs_0, labels)

        optimizer_i.zero_grad()  
        optimizer_0.zero_grad()
        loss_i.backward()
        loss_0.backward()

        optimizer_i.step(optimizer_0.get_param_groups())

        # logging 
        _, indices = outputs_i.max(1)
        correct += (indices == labels).sum().data.item()
        total += len(labels)
        total_loss += loss_i
        
        acc = correct * 100 / float(total)
        loss_avg = total_loss / float(total)
        logging.info('| epoch {:3d} | train AVRG | {:5d} samples | train_acc {:3.2f}% | train_loss {:.6f}'.format(epoch, total, acc, loss_avg))
        print('\033[2A')
    
    # update the outer loop 
    optimizer_0.set_param_groups(optimizer_i.get_param_groups())
    optimizer_i.update_g()

    return loss_avg.item(), acc

def train_SGD(model, optimizer, train_loader, loss_fn):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for inputs, labels, index in train_loader:
        inputs = inputs.squeeze_(0).cuda()
        labels = labels.squeeze_(0).cuda()
        
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
        logging.info('| epoch {:3d} | train SGD | {:5d} samples | train_acc {:3.2f}% | train_loss {:.6f}'.format(epoch, total, acc, loss_avg))
        print('\033[2A')
    
    return loss_avg.item(), acc

def train_SAGA(model, optimizer, train_loader, loss_fn):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for inputs, labels, index in train_loader:
        inputs = inputs.squeeze_(0).cuda()
        labels = labels.squeeze_(0).cuda()
        
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()  
        loss.backward()
        optimizer.step(int(index.item()))

        _, indices = outputs.max(1)
        correct += (indices == labels).sum().data.item()
        total += len(labels)
        total_loss += loss
        
        acc = correct * 100 / float(total)
        loss_avg = total_loss / float(total)
        logging.info('| epoch {:3d} | train SAGA | {:5d} samples | train_acc {:3.2f}% | train_loss {:.6f}'.format(epoch, total, acc, loss_avg))
        print('\033[2A')
    
    return loss_avg.item(), acc

def test(model, test_loader, loss_fn):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, labels, index in test_loader:
            inputs = inputs.squeeze_(0).cuda()
            labels = labels.squeeze_(0).cuda()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            _, indices = outputs.max(1)
            correct += (indices == labels).sum().data.item()
            total += len(labels)
            total_loss += loss
            
            acc = correct * 100 / float(total)
            loss_avg = total_loss / float(total)
            logging.info('| epoch {:3d} | test | {:5d} samples| train_acc {:3.2f}% | train_loss {:.6f}'.format(epoch, total, acc, loss_avg))
            print('\033[2A')
    
    return loss_avg.item(), acc


parser = argparse.ArgumentParser(description="Train SVRG/AVRG.")
parser.add_argument('--dataset', type=str, default="MNIST",
        help="datasets (default: MNIST).")
parser.add_argument('--n_epoch', type=int, default=30,
        help="number of training iterations (default: 30).")
parser.add_argument('--lr', type=float, default=0.005,
        help="learning rate (default: 0.005).")
parser.add_argument('--batch_size', type=int, default=100,
        help="batch size (default: 100).")
parser.add_argument('--weight_decay', type=float, default=0.01,
        help="regularization strength (default: 0.01).")
parser.add_argument('--gpu', type=int, default=0, 
        help='GPU device to use (default: 0).')
parser.add_argument('--seed', type=int, default=3, 
        help='set seed (default: 3).')
parser.add_argument('--save_path', type=str, default=None, 
        help='The path to save log and figures')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    exit(0)

if args.save_path is None:
    args.save_path = 'experiment_' + time.strftime("%Y%m%d-%H%M%S")
os.mkdir(args.save_path)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save_path, 'log.txt'))
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
    train_set, test_set = MNIST_dataset_flat(args.batch_size)
    NN_model = MNIST_two_layers
elif args.dataset == "CIFAR10":
    train_set, test_set = CIFAR10_dataset(args.batch_size) 
    NN_model = vgg11
else:
    raise ValueError("Unknown dataset")

train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

# The network
model_SGD = NN_model().cuda()
#Use same initial weights for both networks
model_SVRG_0 = copy.deepcopy(model_SGD) 
model_SVRG_1 = copy.deepcopy(model_SGD) 
model_AVRG_0 = copy.deepcopy(model_SGD) 
model_AVRG_1 = copy.deepcopy(model_SGD) 
model_SAGA = copy.deepcopy(model_SGD) 

lr = args.lr  # learning rate
n_epoch = args.n_epoch  # the number of epochs

loss_fn = nn.CrossEntropyLoss()

N = len(train_loader)

# The optimizer 
optimizer_SGD = torch.optim.SGD(model_SGD.parameters(), lr=lr, weight_decay=args.weight_decay)
optimizer_SVRG_1 = SVRG(model_SVRG_1.parameters(), lr=lr, N=N)
optimizer_SVRG_0 = SVRG_0(model_SVRG_0.parameters())
optimizer_AVRG_1 = AVRG(model_AVRG_1.parameters(), lr=lr, N=N)
optimizer_AVRG_0 = AVRG_0(model_AVRG_0.parameters())
optimizer_SAGA = SAGA(model_SAGA.parameters(), lr=lr, N=N)

best_train_SGD = 0
best_test_SGD = 0

best_train_SVRG = 0
best_test_SVRG = 0

best_train_AVRG = 0
best_test_AVRG = 0

best_train_SAGA = 0
best_test_SAGA = 0

train_acc_list_SGD = []
train_loss_list_SGD = []
test_acc_list_SGD = []
test_loss_list_SGD = []

train_acc_list_SVRG = []
train_loss_list_SVRG = []
test_acc_list_SVRG = []
test_loss_list_SVRG = []

train_acc_list_AVRG = []
train_loss_list_AVRG = []
test_acc_list_AVRG = []
test_loss_list_AVRG = []

train_acc_list_SAGA = []
train_loss_list_SAGA = []
test_acc_list_SAGA = []
test_loss_list_SAGA = []

for epoch in range(n_epoch+1):
    # if epoch == 0:
    #     # first round to get g
    #     train_AVRG(model_AVRG_0, model_AVRG_1, optimizer_AVRG_0, optimizer_AVRG_1, train_loader, loss_fn)
    #     continue

    # training 
    t0 = time.time()

    train_loss_SGD, train_acc_SGD = train_SGD(model_SGD, optimizer_SGD, train_loader, loss_fn)

    best_train_SGD = max(train_acc_SGD, best_train_SGD)
    logging.info('-' * 96)
    logging.info('train SGD | end of epoch {:3d} | time: {:5.2f}s | train acc {:3.2f}% ({:3.2f}%) | train loss {:.6f} | '.format(epoch, (time.time() - t0), train_acc_SGD, best_train_SGD, train_loss_SGD))
    logging.info('-' * 96) 

    train_loss_SVRG, train_acc_SVRG = train_SVRG(model_SVRG_0, model_SVRG_1, optimizer_SVRG_0, optimizer_SVRG_1, train_loader, loss_fn)

    best_train_SVRG = max(train_acc_SVRG, best_train_SVRG)
    logging.info('-' * 96)
    logging.info('train SVRG | end of epoch {:3d} | time: {:5.2f}s | train acc {:3.2f}% ({:3.2f}%) | train loss {:.6f} | '.format(epoch, (time.time() - t0), train_acc_SVRG, best_train_SVRG, train_loss_SVRG))
    logging.info('-' * 96) 

    train_loss_AVRG, train_acc_AVRG= train_AVRG(model_AVRG_0, model_AVRG_1, optimizer_AVRG_0, optimizer_AVRG_1, train_loader, loss_fn)

    best_train_AVRG = max(train_acc_AVRG, best_train_AVRG)
    logging.info('-' * 96)
    logging.info('train AVRG | end of epoch {:3d} | time: {:5.2f}s | train acc {:3.2f}% ({:3.2f}%) | train loss {:.6f} | '.format(epoch, (time.time() - t0), train_acc_AVRG, best_train_AVRG, train_loss_AVRG))
    logging.info('-' * 96) 

    train_loss_SAGA, train_acc_SAGA= train_SAGA(model_SAGA, optimizer_SAGA, train_loader, loss_fn)

    best_train_SAGA = max(train_acc_SAGA, best_train_SAGA)
    logging.info('-' * 96)
    logging.info('train SAGA | end of epoch {:3d} | time: {:5.2f}s | train acc {:3.2f}% ({:3.2f}%) | train loss {:.6f} | '.format(epoch, (time.time() - t0), train_acc_SAGA, best_train_SAGA, train_loss_SAGA))
    logging.info('-' * 96) 

    train_acc_list_SGD.append(train_acc_SGD)
    train_loss_list_SGD.append(train_loss_SGD)
    train_acc_list_SVRG.append(train_acc_SVRG)
    train_loss_list_SVRG.append(train_loss_SVRG)
    train_acc_list_AVRG.append(train_acc_AVRG)
    train_loss_list_AVRG.append(train_loss_AVRG)
    train_acc_list_SAGA.append(train_acc_SAGA)
    train_loss_list_SAGA.append(train_loss_SAGA)
    
    # testing 
    t0 = time.time()

    test_loss_SGD, test_acc_SGD = test(model_SGD, test_loader, loss_fn)
    best_test_SGD = max(test_acc_SGD, best_test_SGD)
    logging.info('-' * 96)
    logging.info('test SGD | end of epoch {:3d} | time: {:5.2f}s | test acc {:3.2f}% ({:3.2f}%) | test loss {:.6f} | '.format(epoch, (time.time() - t0), test_acc_SGD, best_test_SGD, test_loss_SGD))
    logging.info('-' * 96)
    
    test_loss_SVRG, test_acc_SVRG = test(model_SVRG_1, test_loader, loss_fn)
    best_test_SVRG = max(test_acc_SVRG, best_test_SVRG)
    logging.info('-' * 96)
    logging.info('test SVRG | end of epoch {:3d} | time: {:5.2f}s | test acc {:3.2f}% ({:3.2f}%) | test loss {:.6f} | '.format(epoch, (time.time() - t0), test_acc_SVRG, best_test_SVRG, test_loss_SVRG))
    logging.info('-' * 96)
    
    test_loss_AVRG, test_acc_AVRG = test(model_AVRG_1, test_loader, loss_fn)
    best_test_AVRG = max(test_acc_AVRG, best_test_AVRG)
    logging.info('-' * 96)
    logging.info('test AVRG | end of epoch {:3d} | time: {:5.2f}s | test acc {:3.2f}% ({:3.2f}%) | test loss {:.6f} | '.format(epoch, (time.time() - t0), test_acc_AVRG, best_test_AVRG, test_loss_AVRG))
    logging.info('-' * 96)
    
    test_loss_SAGA, test_acc_SAGA = test(model_SAGA, test_loader, loss_fn)
    best_test_SAGA = max(test_acc_SAGA, best_test_SAGA)
    logging.info('-' * 96)
    logging.info('test SAGA | end of epoch {:3d} | time: {:5.2f}s | test acc {:3.2f}% ({:3.2f}%) | test loss {:.6f} | '.format(epoch, (time.time() - t0), test_acc_SAGA, best_test_SAGA, test_loss_SAGA))
    logging.info('-' * 96)
    
    test_acc_list_SGD.append(test_acc_SGD)
    test_loss_list_SGD.append(test_loss_SGD)
    test_acc_list_SVRG.append(test_acc_SVRG)
    test_loss_list_SVRG.append(test_loss_SVRG)
    test_acc_list_AVRG.append(test_acc_AVRG)
    test_loss_list_AVRG.append(test_loss_AVRG)
    test_acc_list_SAGA.append(test_acc_SAGA)
    test_loss_list_SAGA.append(test_loss_SAGA)

epoch_list = np.arange(n_epoch) + 1

fname_train_acc = os.path.join(args.save_path, 'train_acc.png')
fname_train_loss = os.path.join(args.save_path, 'train_loss.png')
fname_test_acc = os.path.join(args.save_path, 'test_acc.png')
fname_test_loss = os.path.join(args.save_path, 'test_loss.png')

plt.figure()
plt.plot(epoch_list, train_acc_list_SGD, label = "SGD")
plt.plot(epoch_list, train_acc_list_SVRG, label = "SVRG")
plt.plot(epoch_list, train_acc_list_AVRG, label = "AVRG")
plt.plot(epoch_list, train_acc_list_SAGA, label = "SAGA")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Curve of Train Accuracy")
plt.legend()
plt.savefig(fname_train_acc)

plt.figure()
plt.plot(epoch_list, train_loss_list_SGD, label = "SGD")
plt.plot(epoch_list, train_loss_list_SVRG, label = "SVRG")
plt.plot(epoch_list, train_loss_list_AVRG, label = "AVRG")
plt.plot(epoch_list, train_loss_list_SAGA, label = "SAGA")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Curve of Train Loss")
plt.legend()
plt.savefig(fname_train_loss)

plt.figure()
plt.plot(epoch_list, test_acc_list_SGD, label = "SGD")
plt.plot(epoch_list, test_acc_list_SVRG, label = "SVRG")
plt.plot(epoch_list, test_acc_list_AVRG, label = "AVRG")
plt.plot(epoch_list, test_acc_list_SAGA, label = "SAGA")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Curve of Test Accuracy")
plt.legend()
plt.savefig(fname_test_acc)

plt.figure()
plt.plot(epoch_list, test_loss_list_SGD, label = "SGD")
plt.plot(epoch_list, test_loss_list_SVRG, label = "SVRG")
plt.plot(epoch_list, test_loss_list_AVRG, label = "AVRG")
plt.plot(epoch_list, test_loss_list_SAGA, label = "SAGA")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Curve of Test Loss")
plt.legend()
plt.savefig(fname_test_loss)
