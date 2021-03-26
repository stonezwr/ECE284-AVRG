import torch
from torch import nn 
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import argparse
import copy
import time

from exact_diff import ExactDiff, train_Exact_Diffusion
from diff_avrg import DiffAVRG, DiffAVRG_0, train_Diffusion_AVRG
import bluefog.torch as bf
from utils import *
from dataset import *
from vgg import *
from lenet import LeNet

def train_ATC_SGD(model, optimizer, train_loader, loss_fn):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()  
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

def test(model, test_loader, loss_fn):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            _, indices = outputs.max(1)
            correct += (indices == labels).sum().data.item()
            total += len(labels)
            total_loss += loss
            
            acc = correct * 100 / float(total)
            loss_avg = total_loss / float(total)
    
    return loss_avg.item(), acc


parser = argparse.ArgumentParser(description="Train Diffusion AVRG")
parser.add_argument('--dataset', type=str, default="MNIST",
        choices=['MNIST','CIFAR10','MNIST_Conv'],
        help="datasets (default: MNIST).")
parser.add_argument('--n_epoch', type=int, default=100,
        help="number of training iterations (default: 100).")
parser.add_argument('--method', default="ExactDiff",
        choices=["ExactDiff","DiffAVRG","DiffAVRG_B","ATC_SGD"],
        help="Diffusion method (default: ExactDiff).")
parser.add_argument('--comm', default="allreduce",
        choices=["allreduce", "neighbor_allreduce"],
        help="Communication Method (default: allreduce).")
parser.add_argument('--lr', type=float, default=0.001,
        help="learning rate (default: 0.001).")
parser.add_argument('--batch_size', type=int, default=100,
        help="batch size (default: 100).")
parser.add_argument('--seed', type=int, default=3, 
        help='set seed (default: 3).')
parser.add_argument('--save_name', type=str, required=True, 
        help='The file_postfix to save log')

args = parser.parse_args()
cudnn.benchmark = True
cudnn.enabled = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

bf.init()

device_id = bf.local_rank() if bf.nccl_built() else bf.local_rank() % torch.cuda.device_count()
torch.cuda.set_device(device_id)
torch.cuda.manual_seed(args.seed)

kwargs = {"num_workers": 4, "pin_memory": True}

# load the data
if args.dataset == "MNIST":
    train_set, test_set = MNIST_dataset_flat_dist(bf.rank())
    NN_model = MNIST_two_layers
elif args.dataset == "MNIST_Conv":
    train_set, test_set = MNIST_dataset_dist(bf.rank())
    NN_model = LeNet
elif args.dataset == "CIFAR10":
    train_set, test_set = CIFAR10_dataset_dist(bf.rank()) 
    NN_model = vgg11
else:
    raise ValueError("Unknown dataset")

train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_set, num_replicas=bf.size(), rank=bf.rank())
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_set, num_replicas=bf.size(), rank=bf.rank())

if args.method == "DiffAVRG":
    train_loader = DataLoader(train_set, batch_size=1, sampler=train_sampler, **kwargs)
else:
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler, **kwargs)

lr = args.lr  # learning rate

if args.method == "ExactDiff":
    L = len(train_sampler)//args.batch_size
elif args.method == "DiffAVRG":
    L = len(train_sampler)
elif args.method == "DiffAVRG_B":
    L = len(train_sampler)//args.batch_size

# The network and optimizer
if args.method == "ExactDiff":
    model = NN_model().cuda()
    bf.broadcast_parameters(model.state_dict(), root_rank=0)
    optimizer = ExactDiff(model.parameters(), lr=lr, L=L, communication_type=args.comm)
elif args.method == "ATC_SGD":
    model = NN_model().cuda()
    bf.broadcast_parameters(model.state_dict(), root_rank=0)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    bf.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = bf.DistributedAdaptThenCombineOptimizer(optimizer, model,
        communication_type=bf.CommunicationType.allreduce if args.comm=="allreduce" else
                           bf.CommunicationType.neighbor_allreduce)
else:
    model_0 = NN_model().cuda()
    model_i = copy.deepcopy(model_0)
    bf.broadcast_parameters(model_0.state_dict(), root_rank=0)
    bf.broadcast_parameters(model_i.state_dict(), root_rank=0)
    optimizer_0 = DiffAVRG_0(model_0.parameters())
    optimizer_i = DiffAVRG(model_i.parameters(), lr=lr, L=L, communication_type=args.comm)

n_epoch = args.n_epoch  # the number of epochs
loss_fn = nn.CrossEntropyLoss()

res_list = []
total_time = 0
for epoch in range(1,n_epoch+1):
    train_sampler.set_epoch(epoch)
    if args.method == "ExactDiff":
        t0 = time.time()
        train_Exact_Diffusion(model, optimizer, train_loader, loss_fn)
        t = time.time()
        train_loss, train_acc = test(model, train_loader, loss_fn)
        test_loss, test_acc = test(model, test_loader, loss_fn)
    elif args.method == "ATC_SGD":
        t0 = time.time()
        train_ATC_SGD(model, optimizer, train_loader, loss_fn)
        t = time.time()
        train_loss, train_acc = test(model, train_loader, loss_fn)
        test_loss, test_acc = test(model, test_loader, loss_fn)
    else:
        t0 = time.time()
        train_Diffusion_AVRG(model_0, model_i, optimizer_0, optimizer_i, train_loader, loss_fn)
        t = time.time()
        train_loss, train_acc = test(model_i, train_loader, loss_fn)
        test_loss, test_acc = test(model_i, test_loader, loss_fn)
    total_time += t-t0
    if bf.rank() == 0:
        print(f"{epoch:3d}/{test_loss:.5f}/{test_acc:.2f}%")
    
    res_list.append([epoch, train_loss, test_loss, train_acc, test_acc])

avg_time = total_time/n_epoch
res_list = bf.allreduce(torch.tensor(res_list))

if bf.rank() == 0:
    print(f"Avg Time Per Epoch: {avg_time:.2f}s")
    with open(f'{args.method}_{args.save_name}.csv', 'w') as f:
        for res in res_list:
            epoch = res[0]
            train_loss = res[1]
            test_loss = res[2]
            train_acc = res[3]
            test_acc = res[4]
            f.write(f"{epoch},{train_loss},{test_loss},{train_acc},{test_acc}\n")