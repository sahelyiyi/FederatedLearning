import torch

import numpy as np

from abc import ABC
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable

from algorithm_utils import get_matrices


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


class LinearOptimizer(Optimizer):
    def __init__(self, model):
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.RMSprop(model.parameters())
        super(LinearOptimizer, self).__init__(model, optimizer, criterion)


class LogisticOptimizer(Optimizer):
    def __init__(self, model):
        criterion = torch.nn.BCELoss(reduction='mean')
        optimizer = torch.optim.RMSprop(model.parameters())
        super(LogisticOptimizer, self).__init__(model, optimizer, criterion)


def algorithm_1(K, B, weight_vec, X, Y, samplingset, lambda_lasso, score_func=mean_squared_error, loss_func='linear_reg'):
    Sigma, Gamma, Gamma_vec, D = get_matrices(weight_vec, B)

    E, N = B.shape
    m, n = X[0].shape

    data = []
    for i in range(len(X)):
        model = OurModel(n)
        if loss_func == 'linear_reg':
            optimizer = LinearOptimizer(model)
        elif loss_func == 'logistic_reg':
            optimizer = LogisticOptimizer(model)
        else:
            print('invalid loss_func')
        data.append({"features": Variable(torch.from_numpy(X[i])).to(torch.float32),
                     "label": Variable(torch.from_numpy(np.array(Y[i]))).to(torch.float32),
                     'degree': Gamma_vec[i],
                     'optimizer': optimizer})

    new_w = np.array([np.zeros(n) for i in range(N)])
    new_u = np.array([np.zeros(n) for i in range(E)])

    iteration_scores = []
    limit = np.array([np.zeros(n) for i in range(E)])
    for i in range(n):
        limit[:, i] = lambda_lasso * weight_vec

    for iterk in range(K):
        if iterk % 100 == 0:
            print ('iter:', iterk)
        prev_w = np.copy(new_w)

        hat_w = new_w - np.dot(Gamma, np.dot(D.T, new_u))

        for i in range(N):
            optimizer = data[i]['optimizer']
            if i in samplingset:
                optimizer.optimize(data[i]['features'], data[i]['label'], hat_w[i], data[i]['degree'])
                new_w[i] = optimizer.model.linear.weight.data.numpy()
            else:
                new_w[i] = hat_w[i]

        tilde_w = 2 * new_w - prev_w
        new_u = new_u + np.dot(Sigma, np.dot(D, tilde_w))

        normalized_u = np.where(abs(new_u) >= limit)
        new_u[normalized_u] = limit[normalized_u] * new_u[normalized_u] / abs(new_u[normalized_u])

        Y_pred = []
        for i in range(N):
            Y_pred.append(np.dot(X[i], new_w[i]))

        # iteration_scores.append(score_func(Y.reshape(N, m), Y_pred))

    # print (np.max(abs(new_w - prev_w)))

    return iteration_scores, new_w
