import torch
from torch.optim import Optimizer
import copy


class AVRG(Optimizer):
    def __init__(self, params, lr, N):
        self.weight_decay = 1/N
        self.N = N
        self.lr = lr
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if self.weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(self.weight_decay))
        defaults = dict(lr=lr, weight_decay=self.weight_decay)
        super(AVRG, self).__init__(params, defaults)

        self.old_g = [] 
        self.new_g = []
        for groups in self.param_groups:
            old_g_group = []
            new_g_group = []
            for p in groups['params']:
                old_g_group.append(torch.zeros_like(p))
                new_g_group.append(torch.zeros_like(p))
            self.old_g.append(old_g_group)
            self.new_g.append(new_g_group)
    
    def get_param_groups(self):
        return self.param_groups

    def update_g(self):
        """Set the mean gradient for the current epoch. 
        """
        for old_g_group, new_g_group in zip(self.old_g, self.new_g):
            for p, q in zip(old_g_group, new_g_group):
                p.data.copy_(q.data)
                q.zero_()

    def step(self, params):
        """Performs a single optimization step.
        """
        for q_i_group, q_0_group, old_g_group, new_g_group in zip(self.param_groups, params, self.old_g, self.new_g):
            for q_i, q_0, g_old, g_new in zip(q_i_group['params'], q_0_group['params'], old_g_group, new_g_group):
                if q_i.grad is None:
                    continue
                if q_0.grad is None:
                    continue
                new_d = q_i.grad.data - q_0.grad.data + g_old
                # if self.weight_decay != 0:
                #     new_d.add_(q_i.data, alpha=self.weight_decay)
                g_new.add_(q_i.grad.data, alpha=1/self.N) 
                q_i.data.add_(new_d, alpha=-self.lr )


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

