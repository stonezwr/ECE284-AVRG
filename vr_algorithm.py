import copy
import numpy as np
import torch


class VR_algorithm():
    def __init__(self, X, y, w_star, cost_model, N):
        self.cost_model = cost_model(X, y) 
        self.w_star = torch.from_numpy(w_star).cuda().type(torch.float32)
        self.norm_w_star =  torch.sum(torch.square(self.w_star))
        self.N = N

        self.option = {'SVRG': self.SVRG_step,
                       'AVRG': self.AVRG_step,
                       'SAGA': self.SAGA_step,
                       'SAG': self.SAG_step}


    def SVRG_step(self, ite, epoch, inputs, labels, idx):
        epoch_per_FG =  2
        using_sgd = 1

        if ite == 0 and epoch % epoch_per_FG == 0:
            self.w_at_start = copy.deepcopy(self.cost_model.w)
            self.grad_full_at_start = self.cost_model.full_gradient()

        grad = self.cost_model.partial_gradient(inputs, labels)
        grad_at_start = self.cost_model.partial_gradient(inputs, labels, w_ = self.w_at_start)

        grad_modified = grad - grad_at_start + self.grad_full_at_start if epoch>=using_sgd else grad

        return grad_modified

    def AVRG_step(self, ite, epoch, inputs, labels, idx):
        using_sgd = 1

        if ite == 0:
            self.w_at_start = copy.deepcopy(self.cost_model.w)
            self.grad_full_at_start = copy.deepcopy(self.grad_full_at_start_next) if epoch != 0 else torch.zeros_like(self.cost_model.w).cuda()
            self.grad_full_at_start_next = torch.zeros(self.grad_full_at_start.shape).cuda()

        grad = self.cost_model.partial_gradient(inputs, labels)
        grad_at_start = self.cost_model.partial_gradient(inputs, labels, w_ = self.w_at_start)

        grad_modified = grad - grad_at_start + self.grad_full_at_start if epoch>=using_sgd else grad

        self.grad_full_at_start_next += (grad / self.N)

        return grad_modified

    def SAGA_step(self, ite, epoch, inputs, labels, idx):
        using_sgd = 1

        # SAGA particular: initialize the memory:
        if ite == 0 and epoch == 0:
            self.grad_at_last =  torch.zeros(self.cost_model.M, self.cost_model.N).cuda()
            self.grad_avg = torch.zeros(self.cost_model.M, 1).cuda()

        grad = self.cost_model.partial_gradient(inputs, labels)

        grad_modified = grad - self.grad_at_last[:,idx] + self.grad_avg if epoch>=using_sgd else grad

        self.grad_avg = self.grad_avg - (self.grad_at_last[:,idx] - grad)/self.N
        self.grad_at_last[:,idx] = grad

        return grad_modified

    def SAG_step(self, ite, epoch, inputs, labels, idx):
        using_sgd = 1

        # SAG particular: initialize the memory:
        if ite == 0 and epoch == 0:
            self.grad_at_last =  torch.zeros(self.cost_model.M, self.cost_model.N).cuda()
            self.grad_avg = torch.zeros(self.cost_model.M, 1).cuda()

        grad = self.cost_model.partial_gradient(inputs, labels)
        self.grad_avg = self.grad_avg - (self.grad_at_last[:,idx] - grad)/self.N
        self.grad_at_last[:,idx] = grad

        return self.grad_avg if epoch>=using_sgd else grad

    def train(self, mu, method, train_loader, N_epoch=10):
        self.MSD = []
        self.ER = []
        for epoch in range(N_epoch):
            print("epoch: ", str(epoch))
            ite = 0 
            for inputs, labels, index in train_loader:
                inputs = inputs.squeeze_(0).cuda()
                labels = labels.squeeze_(0).cuda()
                grad_modifed = self.option[method](ite, epoch, inputs, labels, index)
                self.cost_model._update_w(grad_modifed, mu)
                ite += 1

            msd_ = torch.sum( (self.cost_model.w - self.w_star)*(self.cost_model.w - self.w_star) ) / self.norm_w_star
            err_ = self.cost_model.func_value() - self.cost_model.func_value(w_ = self.w_star)
            self.MSD.append(msd_.item())
            self.ER.append(err_.item())
        print(self.MSD)
        print(self.ER)

