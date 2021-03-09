import torch
from torch.optim import Optimizer
import copy
import bluefog.torch as bf
from tqdm import tqdm

def train_Diffusion_AVRG(model_0, model_i, optimizer_0, optimizer_i, train_loader, loss_fn):
    model_0.train()
    model_i.train()

    # inner loop
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer_i.zero_grad()  
        optimizer_0.zero_grad()

        outputs_i = model_i(inputs)
        loss_i = loss_fn(outputs_i, labels)
        outputs_0 = model_0(inputs)
        loss_0 = loss_fn(outputs_0, labels)

        loss_i.backward()
        loss_0.backward()

        optimizer_i.step(optimizer_0.get_param_groups())

    # update the outer loop 
    optimizer_0.set_param_groups(optimizer_i.get_param_groups())
    optimizer_i.update_g()

class DiffAVRG(Optimizer):
    def __init__(self, params, lr, L, communication_type):
        '''
        lr: Learning rate
        L: Number of batches
        '''
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, L=L)
        super(DiffAVRG, self).__init__(params, defaults)

        self._communication_type = communication_type
        self.lr = lr
        self.L = L
        self._states = {}
        for groups in self.param_groups:
            for p in groups['params']:
                self._states[p] = {'prev_g': torch.zeros_like(p),
                                   'curr_g': torch.zeros_like(p),
                                   'psi': torch.clone(p),
                                   'phi': torch.zeros_like(p),
                                   'handle': None}
    
    def get_param_groups(self):
        return self.param_groups

    def update_g(self):
        """Set the mean gradient for the current epoch. 
        """
        for groups in self.param_groups:
            for p in groups['params']:
                state = self._states[p]
                state['prev_g'].data.copy_(state['curr_g'].data)
                state['curr_g'].zero_()

    def step(self, param_groups):
        """Performs a single optimization step.
        """
        for p_i_group, p_0_group in zip(self.param_groups, param_groups):
            for p_i, p_0 in zip(p_i_group['params'], p_0_group['params']):
                if p_i.grad is None:
                    continue
                if p_0.grad is None:
                    continue
                state = self._states[p_i]
                # core Diffusion-AVRG gradient update 
                with torch.no_grad():
                    amortized_grad = p_i.grad - p_0.grad + state['prev_g']
                    state['curr_g'].add_(p_i.grad, alpha=1.0/self.L)
                    psi = p_i-self.lr*amortized_grad
                    state['phi'].set_(psi + p_i - state['psi'])
                    state['psi'].set_(psi)
                    if self._communication_type == 'allreduce':
                        handle = self._allreduce_data_async(state['phi'])
                    elif self._communication_type == 'neighbor_allreduce':
                        handle = self._neighbor_allreduce_data_async(state['phi'])
                    state['handle'] = handle
        self._synchronize()
    
    def _synchronize(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self._states[p]
                with torch.no_grad():
                    p.set_(bf.synchronize(state['handle']))

    def _neighbor_allreduce_data_async(self, p):
        handle = bf.neighbor_allreduce_nonblocking(p.data)
        return handle

    def _allreduce_data_async(self, p):
        handle = bf.allreduce_nonblocking(p.data, average=True)
        return handle


class DiffAVRG_0(Optimizer):
    def __init__(self, params):
        defaults = dict()
        super(DiffAVRG_0, self).__init__(params, defaults)
      
    def get_param_groups(self):
        return self.param_groups
    
    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer. 
        """
        for group, new_group in zip(self.param_groups, new_params): 
            for p, q in zip(group['params'], new_group['params']):
                p.data.copy_(q.data)

