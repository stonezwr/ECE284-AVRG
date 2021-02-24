import torch
from torch.optim import Optimizer


class AVRG(Optimizer):
    def __init__(self, params, lr, weight_decay=0):
        print("Using optimizer: AVRG")
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(AVRG, self).__init__(params, defaults)
    
    def get_param_groups(self):
            return self.param_groups

    def step(self, params):
        """Performs a single optimization step.
        """
        

