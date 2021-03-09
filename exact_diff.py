import torch
from torch.optim import Optimizer
import copy
import bluefog.torch as bf
from tqdm import tqdm

def train_Exact_Diffusion(model, optimizer, train_loader, loss_fn):
    model.train()

    # inner loop
    optimizer.zero_grad()  
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
    optimizer.step()

class ExactDiff(Optimizer):
    def __init__(self, params, lr, L, communication_type):
        '''
        lr: Learning rate
        L: Number of batches
        '''
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, L=L)
        super(ExactDiff, self).__init__(params, defaults)

        self._communication_type = communication_type
        self.lr = lr
        self.L = L
        self._q = bf.size()
        self._states = {}
        for groups in self.param_groups:
            for p in groups['params']:
                self._states[p] = {'psi': torch.clone(p),
                                   'phi': torch.zeros_like(p),
                                   'handle': None}
    
    def step(self):
        """Performs a single optimization step.
        """
        scale = self.lr / self._q / self.L
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self._states[p]
                # core Exact Diffusion gradient update 
                with torch.no_grad():
                    psi = p-scale*p.grad
                    state['phi'].set_(psi + p - state['psi'])
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