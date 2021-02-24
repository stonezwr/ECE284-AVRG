import torch
from torch.optim import Optimizer
import copy


class SVRG(Optimizer):
    def __init__(self, params, lr, weight_decay=0):
        print("Using optimizer: SVRG")
        self.j = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SVRG, self).__init__(params, defaults)
    
    def get_param_groups(self):
            return self.param_groups

    def set_outer_params(self, new_j):
        """Set the mean gradient for the current epoch. 
        """
        if self.j is None:
            self.j = copy.deepcopy(new_j)
        for j_old, j_new in zip(self.j, new_j):  
            for x, y in zip(j_old['params'], j_new['params']):
                x.grad = y.grad.clone()

    def step(self, params):
        """Performs a single optimization step.
        """
        for q_i_group, q_0_group, j_group in zip(self.param_groups, params, self.j):
            weight_decay = q_i_group['weight_decay']

            for q_i, q_0, j_0 in zip(q_i_group['params'], q_0_group['params'], j_group['params']):
                if q_i.grad is None:
                    continue
                if q_0.grad is None:
                    continue
                # core SVRG gradient update 
                new_d = q_i.grad.data - q_0.grad.data + j_0.grad.data
                if weight_decay != 0:
                    new_d.add_(weight_decay, q_i.data)
                q_i.data.add_(-q_i_group['lr'], new_d)


class SVRG_0(Optimizer):
    def __init__(self, params):
        defaults = dict()
        super(SVRG_0, self).__init__(params, defaults)
      
    def get_param_groups(self):
            return self.param_groups
    
    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer. 
        """
        for group, new_group in zip(self.param_groups, new_params): 
            for p, q in zip(group['params'], new_group['params']):
                  p.data[:] = q.data[:]

