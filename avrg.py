import torch
from torch.optim import Optimizer
import copy


class AVRG(Optimizer):
    def __init__(self, params, lr, weight_decay=0):
        print("Using optimizer: AVRG")
        self.N = 0
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(AVRG, self).__init__(params, defaults)

        self.old_g = [] 
        self.new_g = []
        for groups in self.param_groups:
            for i in groups['params']:
                self.old_g.append(torch.zeros_like(i))
                self.new_g.append(torch.zeros_like(i))
    
    def get_param_groups(self):
        return self.param_groups

    def update_g(self, n):
        """Set the mean gradient for the current epoch. 
        """
        self.N = n
        for p, q in zip(self.old_g, self.new_g):
            p.data.copy_(q.data)
            q.zero_()

    def step(self, params):
        """Performs a single optimization step.
        """
        for q_i_group, q_0_group in zip(self.param_groups, params):
            weight_decay = q_i_group['weight_decay']

            for q_i, q_0, g_old, g_new in zip(q_i_group['params'], q_0_group['params'], self.old_g, self.new_g):
                if q_i.grad is None:
                    continue
                if q_0.grad is None:
                    continue
                # core AVRG gradient update 
                new_d = q_i.grad.data - q_0.grad.data + g_old
                if weight_decay != 0:
                    new_d.add_(q_i.data, alpha=weight_decay)
                g_new += q_i.grad.data/self.N 
                q_i.data.add_(new_d, alpha=-q_i_group['lr'] )


class AVRG_0(Optimizer):
    def __init__(self, params):
        defaults = dict()
        super(AVRG_0, self).__init__(params, defaults)
      
    def get_param_groups(self):
            return self.param_groups
    
    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer. 
        """
        for group, new_group in zip(self.param_groups, new_params): 
            for p, q in zip(group['params'], new_group['params']):
                p.data.copy_(q.data)

