import torch
from torch.optim import Optimizer
import copy


class SVRG(Optimizer):
    def __init__(self, params, lr, N):
        self.N = N
        self.weight_decay = 1/N
        self.lr = lr
        self.j = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if self.weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(self.weight_decay))
        defaults = dict(lr=lr, weight_decay=self.weight_decay)
        super(SVRG, self).__init__(params, defaults)
    
    def get_param_groups(self):
        return self.param_groups

    def set_full_grad(self, new_j):
        """Set the mean gradient for the current epoch. 
        """
        if self.j is None:
            self.j = copy.deepcopy(new_j)
        for j_old, j_new in zip(self.j, new_j):  
            for x, y in zip(j_old, j_new):
                x.data.copy_(y.data)
                x.data.div_(self.N)

    def step(self, params):
        """Performs a single optimization step.
        """
        for q_i_group, q_0_group, j_group in zip(self.param_groups, params, self.j):
            for q_i, q_0, j_0 in zip(q_i_group['params'], q_0_group['params'], j_group):
                if q_i.grad is None:
                    continue
                if q_0.grad is None:
                    continue
                # core SVRG gradient update 
                new_d = q_i.grad.data - q_0.grad.data + j_0.data
                # if self.weight_decay != 0:
                #     new_d.add_(q_i.data, alpha=self.weight_decay)
                q_i.data.add_(new_d, alpha=-self.lr )


class SVRG_0(Optimizer):
    def __init__(self, params):
        defaults = dict()
        self.full_grad = None
        super(SVRG_0, self).__init__(params, defaults)
      
    def reset_full_grad(self):
        if self.full_grad is None:
            self.full_grad = []
            for groups in self.param_groups:
                grads = []
                for p in groups['params']:
                    fg = torch.zeros_like(p)
                    grads.append(fg)
                self.full_grad.append(grads)    
        else:
            for full_grad_group in self.full_grad:
                for fg in full_grad_group:
                    fg.zero_()

    def get_param_groups(self):
        return self.param_groups

    def get_full_grad(self):
        return self.full_grad
    
    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer. 
        """
        for group, new_group in zip(self.param_groups, new_params): 
            for p, q in zip(group['params'], new_group['params']):
                  p.data.copy_(q.data)

    def update_full_grad(self):
        for groups, fg_groups in zip(self.param_groups, self.full_grad):
            for p, fg in zip(groups['params'], fg_groups):
                fg += p.grad



