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
from avrg import AVRG, AVRG_0
from utils import *
from dataset import *

import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression

def LR_scikit_learn(train_loader):
    x = []
    y = []
    for inputs, labels in train_loader:
        labels = (labels-0.5) * 2
        x.append(inputs)
        y.append(labels)
    x = torch.cat(x, dim=0).cpu().detach().numpy()
    y = torch.cat(y).cpu().detach().numpy()
    lr = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=False,
          intercept_scaling=1, max_iter=300, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=1e-08,
          verbose=0, warm_start=False) 
    lr.fit(x, y)
    return lr.coef_

def train_SVRG(model_0, model_i, optimizer_0, optimizer_i, train_loader, loss_fn):
    model_0.train()
    model_i.train()
    total = 0
    total_loss = 0

    # SVRG outer loop 
    optimizer_0.reset_full_grad()  
    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels = (labels-0.5) * 2
        outputs = model_0(inputs)
        outer_loss = loss_fn.loss(outputs, labels)
        optimizer_0.zero_grad()  
        outer_loss.backward()
        optimizer_0.update_full_grad()

    # pass the current paramesters of optimizer_0 to optimizer_i
    optimizer_i.set_full_grad(optimizer_0.get_full_grad())

    # inner loop
    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels = (labels-0.5) * 2
        
        outputs_i = model_i(inputs)
        loss_i = loss_fn.loss(outputs_i, labels)

        optimizer_i.zero_grad()  
        loss_i.backward()

        outputs_0 = model_0(inputs)
        loss_0 = loss_fn.loss(outputs_0, labels)

        optimizer_0.zero_grad()
        loss_0.backward()

        optimizer_i.step(optimizer_0.get_param_groups())

    # update the outer loop 
    optimizer_0.set_param_groups(optimizer_i.get_param_groups())

    # evaluate ER
    MSD, ER = loss_fn.ER(train_loader, optimizer_i.param_groups[0]['params'][0])

    return MSD, ER

def train_AVRG(model_0, model_i, optimizer_0, optimizer_i, train_loader, loss_fn):
    model_0.train()
    model_i.train()
    total = 0
    total_loss = 0

    # inner loop
    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels = (labels-0.5) * 2
        
        outputs_i = model_i(inputs)
        loss_i = loss_fn.loss(outputs_i, labels)

        optimizer_i.zero_grad()  
        loss_i.backward()

        outputs_0 = model_0(inputs)
        loss_0 = loss_fn.loss(outputs_0, labels)

        optimizer_0.zero_grad()
        loss_0.backward()

        optimizer_i.step(optimizer_0.get_param_groups())

    # update the outer loop 
    optimizer_0.set_param_groups(optimizer_i.get_param_groups())
    optimizer_i.update_g()

    # evaluate ER
    MSD, ER = loss_fn.ER(train_loader, optimizer_i.param_groups[0]['params'][0])

    return MSD, ER

def train_SGD(model, optimizer, train_loader, loss_fn):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels = (labels-0.5) * 2
        
        outputs = model(inputs)
        loss = loss_fn.loss(outputs, labels)

        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()

    # evaluate ER
    MSD, ER = loss_fn.ER(train_loader, optimizer.param_groups[0]['params'][0])

    return MSD, ER


parser = argparse.ArgumentParser(description="Train SVRG/AVRG for logistic regression problem.")
parser.add_argument('--dataset', type=str, default="MNIST",
        help="datasets (default: MNIST).")
parser.add_argument('--n_epoch', type=int, default=15,
        help="number of training iterations (default: 15).")
parser.add_argument('--lr', type=float, default=0.001,
        help="learning rate (default: 0.001).")
parser.add_argument('--batch_size', type=int, default=100,
        help="batch size (default: 100).")
parser.add_argument('--weight_decay', type=float, default=0.01,
        help="regularization strength (default: 0.01).")
parser.add_argument('--gpu', type=int, default=0, 
        help='GPU device to use (default: 0).')
parser.add_argument('--seed', type=int, default=43, 
        help='set seed (default: 43).')
parser.add_argument('-save_path', type=str, default=None, 
        help='The path to save log and figures')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    exit(0)

if args.save_path is None:
    args.save_path = 'logistic_' + time.strftime("%Y%m%d-%H%M%S")
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
    train_set, val_set = MNIST_dataset_flat()
    NN_model = MNIST_logistic_regression
elif args.dataset == "CIFAR10":
    train_set, val_set = CIFAR10_dataset_flat() 
    NN_model = CIFAR10_logistic_regression
else:
    raise ValueError("Unknown dataset")

train_indices = get_binary_indices(train_set, [0, 1])
train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices), num_workers=4, drop_last=True)

# The network
model_SGD = NN_model().cuda()
model_SVRG_0 = NN_model().cuda()
model_SVRG_1 = NN_model().cuda()
model_AVRG_0 = NN_model().cuda()
model_AVRG_1 = NN_model().cuda()

lr = args.lr  # learning rate
n_epoch = args.n_epoch  # the number of epochs

coef = LR_scikit_learn(train_loader)

loss_fn = Logistic_Regression(coef, len(train_loader)*args.batch_size)

N = len(train_loader)

# The optimizer 
optimizer_SGD = torch.optim.SGD(model_SGD.parameters(), lr=lr, weight_decay=args.weight_decay)
optimizer_SVRG_1 = SVRG(model_SVRG_1.parameters(), lr=lr, N=N)
optimizer_SVRG_0 = SVRG_0(model_SVRG_0.parameters())
optimizer_AVRG_1 = AVRG(model_AVRG_1.parameters(), lr=lr, N=N)
optimizer_AVRG_0 = AVRG_0(model_AVRG_0.parameters())

list_MSD_SGD = []
list_ER_SGD = []
list_MSD_SVRG = []
list_ER_SVRG = []
list_MSD_AVRG = []
list_ER_AVRG = []

for epoch in range(n_epoch):
    if epoch == 0:
        # first round to get g
        train_AVRG(model_AVRG_0, model_AVRG_1, optimizer_AVRG_0, optimizer_AVRG_1, train_loader, loss_fn)
        continue

    # training 
    t0 = time.time()

    MSD_SGD, ER_SGD = train_SGD(model_SGD, optimizer_SGD, train_loader, loss_fn)

    logging.info('-' * 96)
    logging.info('train SGD | end of epoch {:3d} | time: {:5.2f}s | MSD: {:10f} | ER: ({:10f}) | '.format(epoch, (time.time() - t0), MSD_SGD, ER_SGD))
    logging.info('-' * 96) 

    MSD_SVRG, ER_SVRG = train_SVRG(model_SVRG_0, model_SVRG_1, optimizer_SVRG_0, optimizer_SVRG_1, train_loader, loss_fn)

    logging.info('-' * 96)
    logging.info('train SVRG | end of epoch {:3d} | time: {:5.2f}s | MSD: {:10f} | ER: ({:10f}) | '.format(epoch, (time.time() - t0), MSD_SVRG, ER_SVRG))
    logging.info('-' * 96) 

    MSD_AVRG, ER_AVRG = train_AVRG(model_AVRG_0, model_AVRG_1, optimizer_AVRG_0, optimizer_AVRG_1, train_loader, loss_fn)

    logging.info('-' * 96)
    logging.info('train AVRG | end of epoch {:3d} | time: {:5.2f}s | MSD: {:10f} | ER: ({:10f}) | '.format(epoch, (time.time() - t0), MSD_AVRG, ER_AVRG))
    logging.info('-' * 96) 

    list_MSD_SGD.append(MSD_SGD)
    list_ER_SGD.append(ER_SGD)
    list_MSD_SVRG.append(MSD_SVRG)
    list_ER_SVRG.append(ER_SVRG)
    list_MSD_AVRG.append(MSD_AVRG)
    list_ER_AVRG.append(ER_AVRG)
    
epoch_list = np.arange(n_epoch)

fname = os.path.join(args.save_path, 'losgistic_msd.png')
plt.plot(epoch_list+1, list_MSD_SGD, label = "SGD")
plt.plot(epoch_list*2.5+2.5, list_MSD_SVRG, label = "SVRG")
plt.plot(epoch_list*2+2, list_MSD_AVRG, label = "AVRG")
plt.xlabel("Epoch")
plt.ylabel('$\mathrm{\mathbb{E} } ||w^t_0-w^*||^2/||w^*||^2$')
plt.title("Curve of MSD")
plt.legend()
plt.savefig(fname)

fname = os.path.join(args.save_path, 'losgistic_er.png')
plt.plot(epoch_list+1, list_ER_SGD, label = "SGD")
plt.plot(epoch_list*2.5+2.5, list_ER_SVRG, label = "SVRG")
plt.plot(epoch_list*2+2, list_ER_AVRG, label = "AVRG")
plt.xlabel("Epoch")
plt.ylabel('$\mathrm{\mathbb{E} } J(w^k_i)-J(w^\star)$')
plt.title("Curve of ER")
plt.legend()
plt.savefig(fname)
