import torch

import numpy as np

from torch.autograd import Variable

from algorithm.optimizer import TorchLinearModel, LinearModel, LinearOptimizer, TorchLinearOptimizer, \
    TorchLogisticOptimizer


def get_gamma_vec(B):
    Gamma_vec = np.array((1.0 / (np.sum(abs(B), 0)))).ravel()
    return Gamma_vec


def prepare_linear_reg_data_for_algorithm1(X, Y, Gamma_vec, samplingset):
    MTX1_INV, MTX2 = get_preprocessed_matrices(samplingset, Gamma_vec, X, Y)

    data = {}
    for i in range(len(X)):
        data[i] = {
            'features': X[i],
            'degree': Gamma_vec[i]
        }

        if i in samplingset:
            model = LinearModel(MTX1_INV[i], MTX2[i])
            optimizer = LinearOptimizer(model)

            data[i].update({
                'label': Y[i],
                'optimizer': optimizer}
            )

    return data


def prepare_data_for_algorithm1(B, X, Y, samplingset, loss_func='linear_reg'):
    Gamma_vec = get_gamma_vec(B)

    if loss_func == 'linear_reg':
        return prepare_linear_reg_data_for_algorithm1(X, Y, Gamma_vec, samplingset)

    data = {}
    for i in range(len(X)):

        data[i] = {
            'features': Variable(torch.from_numpy(X[i])).to(torch.float32),
            'degree': Gamma_vec[i]
        }

        if i in samplingset:
            _, n = X[i].shape
            model = TorchLinearModel(n)
            if loss_func == 'torch_linear_reg':
                optimizer = TorchLinearOptimizer(model)
            elif loss_func == 'logistic_reg':
                optimizer = TorchLogisticOptimizer(model)
            else:
                raise Exception('invalid loss_func')

            data[i].update({
                'label': Variable(torch.from_numpy(np.array(Y[i]))).to(torch.float32),
                'optimizer': optimizer
            })

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
