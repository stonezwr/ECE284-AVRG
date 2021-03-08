import torch
from torch import nn 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import os 
import time
import logging
import sys
import copy

from vr_algorithm import *
from cost_func import *
from dataset import *

import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser(description="Train SVRG/AVRG for logistic regression problem.")
parser.add_argument('--dataset', type=str, default="MNIST",
        help="datasets (default: MNIST).")
parser.add_argument('--n_epoch', type=int, default=15,
        help="number of training iterations (default: 15).")
parser.add_argument('--gpu', type=int, default=0, 
        help='GPU device to use (default: 0).')
parser.add_argument('--seed', type=int, default=43, 
        help='set seed (default: 43).')
parser.add_argument('--save_path', type=str, default=None, 
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

logging.info("Loading data")
# load the data
if args.dataset == "MNIST":
    train_set = MNIST_dataset_binary_flat()
elif args.dataset == "CIFAR10":
    train_set = CIFAR10_dataset_binary_flat() 
else:
    raise ValueError("Unknown dataset")

train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)

logging.info("Get answer from sklearn")
x = train_set.get_samples().cuda()
y = train_set.get_labels().cuda()
x_n = x.cpu().detach().numpy()
y_n = y.cpu().detach().numpy()
lr = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=False,
          intercept_scaling=1, max_iter=300, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=1e-08,
          verbose=0, warm_start=False) 
lr.fit(x_n, y_n)

N = len(train_loader)

logging.info("Train SVRG")
svrg = VR_algorithm(x, y, lr.coef_.T, logistic_regression, N)
svrg.train(mu=1.0, method='SVRG', train_loader = train_loader, N_epoch= args.n_epoch)

logging.info("Train AVRG")
avrg = VR_algorithm(x, y, lr.coef_.T, logistic_regression, N)
avrg.train(mu=0.5, method='AVRG', train_loader = train_loader, N_epoch= args.n_epoch)

logging.info("Train SAGA")
saga = VR_algorithm(x, y, lr.coef_.T, logistic_regression, N)
saga.train(mu=0.6, method='SAGA', train_loader = train_loader, N_epoch= args.n_epoch)

fname_ER = os.path.join(args.save_path, 'ER.png')
fname_MSD = os.path.join(args.save_path, 'MSD.png')

plt.figure()
plt.semilogy(np.arange(args.n_epoch)*2+2,avrg.ER,'b^-',lw=1.5)
plt.semilogy(np.arange(args.n_epoch)*2.5+2.5,svrg.ER,'ks-',lw=1.5)
plt.semilogy(np.arange(args.n_epoch)+1,saga.ER,'ro-',lw=1.5)

plt.grid('on')
plt.legend(['AVRG','SVRG','SAGA','SAG'],loc=3)
plt.xlabel('Gradients/N')
plt.ylabel('$\mathrm{\mathbb{E} } J(w^k_i)-J(w^\star)$')    

plt.savefig(fname_ER)

plt.figure()
plt.semilogy(np.arange(args.n_epoch)*2+2,avrg.MSD,'b^-',lw=1.5)
plt.semilogy(np.arange(args.n_epoch)*2.5+2.5,svrg.MSD,'ks-',lw=1.5)
plt.semilogy(np.arange(args.n_epoch)+1,saga.MSD,'ro-',lw=1.5)

plt.grid('on')
plt.legend(['AVRG','SVRG','SAGA','SAG'],loc=3)
plt.xlabel('Gradients/N')
plt.ylabel('$\mathrm{\mathbb{E} } ||w^t_0-w^*||^2/||w^*||^2$')

plt.savefig(fname_MSD)
