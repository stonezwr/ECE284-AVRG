import torch
from torch.optim import Optimizer
import copy


class SAGA(Optimizer):
    def __init__(self, params, lr, N):
        self.weight_decay = 1/N
        self.N = N
        self.lr = lr
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if self.weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(self.weight_decay))
        defaults = dict(lr=lr, weight_decay=self.weight_decay)
        super(SAGA, self).__init__(params, defaults)

        self.last_group = []
        self.avg_group = []
        for groups in self.param_groups:
            group = []
            for p in groups['params']:
                group.append(torch.zeros_like(p))
            self.avg_group.append(group)
    
    def get_param_groups(self):
        return self.param_groups

    def step(self, ind):
        """Performs a single optimization step.
        """
        if ind < len(self.last_group):
            for q_group, a_group, l_group in zip(self.param_groups, self.avg_group, self.last_group[ind]):
                for q, a, l in zip(q_group['params'], a_group, l_group):
                    if q.grad is None:
                        continue

                    a.data = a.data - (l.data-q.grad.data) / self.N
                    l.data.copy_(q.grad.data)
                    new_d = a.data
                    # if self.weight_decay != 0:
                    #     new_d.add_(q.data, alpha=self.weight_decay)
                    q.data.add_(new_d, alpha=-self.lr )
        else:
            l_groups = []
            for q_group, a_group in zip(self.param_groups, self.avg_group):
                l_group = []
                for q, a in zip(q_group['params'], a_group):
                    if q.grad is None:
                        continue

                    a.data = a.data - (-q.grad.data) / self.N
                    l = torch.empty_like(q.grad)
                    l.data.copy_(q.grad.data)
                    l_group.append(l)
                    new_d = a.data
                    # if self.weight_decay != 0:
                    #     new_d.add_(q.data, alpha=self.weight_decay)
                    q.data.add_(new_d, alpha=-self.lr )
                l_groups.append(l_group)
            self.last_group.append(l_groups)

