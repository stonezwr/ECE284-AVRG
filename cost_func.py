import torch


class logistic_regression():
    '''
        solving the min_w mean(\ln[1+exp(-yX*w)]) + rho/2\|w\|^2 LR problem
        X---------N*M (# of instance * # dimension of problem)
        y---------N*1 (# of instance * 1) (should be +/- 1)
        w---------M*1 (# dimension of problem * 1)
        The full gradient is
            rho*w -  mean(exp(-yX*w)/( 1 + exp(-yX*w) )yX)
    '''
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.N = X.shape[0]
        self.M = X.shape[1]

        self.w = torch.zeros(self.M, 1).cuda()
        self.rho = 1/self.N

    def full_gradient(self, w_=None):
        w = self.w if w_ == None else w_

        exp_val = torch.exp(-self.y*self.X.matmul(w).squeeze_(1))
        shape = self.X.shape
        tmp = exp_val/(1+exp_val)*self.y
        tmp = tmp.unsqueeze_(-1).repeat(1, shape[1])
        grad = torch.mean(tmp*self.X, axis=0).unsqueeze(-1)
        grad = self.rho*w - grad
        return grad

    def partial_gradient(self, inputs, labels, w_=None):
        '''
            return a partial gradient by index
            if index is none, the function return the gradient based on one (uniformly) random realization
            if index is not none, the function will return the gradient based on selected index realization
            index can be int or the list of int
            index \in [0, N-1]
        '''
        w = self.w if w_ is None else w_

        exp_val = torch.exp(-labels*inputs.matmul(w))
        tmp = exp_val/(1+exp_val)
        val = tmp * labels * inputs
        d_w = self.rho*w - val.unsqueeze_(-1) 
        return d_w

    def func_value(self, w_=None):
        w = self.w if w_ is None else w_

        return torch.mean(torch.log(1+torch.exp(-self.X.matmul(w)*self.y))) + self.rho/2 * torch.sum(torch.square(w))

    def _update_w(self, gradient, mu=0.01):
        self.w -= mu*gradient

    def _reset_w(self, rand=False):
        if rand:
            self.w = torch.randn(self.M, 1).cuda()*0.01
        else:
            self.w.zero_()

