import torch
import abc

import numpy as np

from abc import ABC
from torch.autograd import Variable


class OurModel(torch.nn.Module):
    def __init__(self, n):
        super(OurModel, self).__init__()
        self.linear = torch.nn.Linear(n, 1, bias=False)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


class Optimizer(ABC):
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    @abc.abstractmethod
    def optimize(self, x_data, y_data, oldweight, regularizerterm):
        torch_oldweight = torch.from_numpy(np.array(oldweight, dtype=np.float32))
        self.model.linear.weight.data = torch_oldweight
        for iterinner in range(40):
            self.optimizer.zero_grad()
            y_pred = self.model(x_data)
            loss1 = self.criterion(y_pred, y_data)
            loss2 = 1 / (2 * regularizerterm) * torch.mean((self.model.linear.weight - torch_oldweight) ** 2)  # + 10000*torch.mean((model.linear.bias+0.5)**2)#model.linear.weight.norm(2)
            loss = loss1 + loss2
            loss.backward()
            self.optimizer.step()


class LinearOptimizer(Optimizer):
    def __init__(self, model):
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.RMSprop(model.parameters())
        super(LinearOptimizer, self).__init__(model, optimizer, criterion)

    def optimize(self, x_data, y_data, oldweight, regularizerterm):
        super(LinearOptimizer, self).optimize(x_data, y_data, oldweight, regularizerterm)


class LogisticOptimizer(Optimizer):
    def __init__(self, model):
        criterion = torch.nn.BCELoss(reduction='mean')
        optimizer = torch.optim.RMSprop(model.parameters())
        super(LogisticOptimizer, self).__init__(model, optimizer, criterion)

    def optimize(self, x_data, y_data, oldweight, regularizerterm):
        super(LogisticOptimizer, self).optimize(x_data, y_data, oldweight, regularizerterm)


def get_gamma_vec(B):
    Gamma_vec = np.array((1.0 / (np.sum(abs(B), 0)))).ravel()
    return Gamma_vec


def prepare_data_for_algorithm1(B, X, Y, loss_func='linear_reg'):
    Gamma_vec = get_gamma_vec(B)

    data = []
    for i in range(len(X)):
        _, n = X[i].shape
        model = OurModel(n)
        if loss_func == 'linear_reg':
            optimizer = LinearOptimizer(model)
        elif loss_func == 'logistic_reg':
            optimizer = LogisticOptimizer(model)
        else:
            print('invalid loss_func')
        data.append({'features': Variable(torch.from_numpy(X[i])).to(torch.float32),
                     'label': Variable(torch.from_numpy(np.array(Y[i]))).to(torch.float32),
                     'degree': Gamma_vec[i],
                     'optimizer': optimizer})
    return data


def get_matrices(weight_vec, B):
    Sigma = np.diag(np.full(weight_vec.shape, 0.9 / 2))

    D = B

    Gamma_vec = get_gamma_vec(B)
    Gamma = np.diag(Gamma_vec)

    if np.linalg.norm(np.dot(Sigma ** 0.5, D).dot(Gamma ** 0.5), 2) > 1:
        print ('product norm', np.linalg.norm(np.dot(Sigma ** 0.5, D).dot(Gamma ** 0.5), 2))
        # raise Exception('higher than 1')
    return Sigma, Gamma, Gamma_vec, D


def get_preprocessed_matrices(samplingset, Gamma_vec, X, Y):
    MTX1_INV = {}
    MTX2 = {}
    for i in samplingset:
        mtx1 = 2 * Gamma_vec[i] * np.dot(X[i].T, X[i]).astype('float64')
        if mtx1.shape:
            mtx1 += 1 * np.eye(mtx1.shape[0])
            mtx_inv = np.linalg.inv(mtx1)
        else:
            mtx1 += 1
            mtx_inv = 1.0 / mtx1
        MTX1_INV[i] = mtx_inv

        MTX2[i] = 2 * Gamma_vec[i] * np.dot(X[i].T, Y[i]).T
    return MTX1_INV, MTX2