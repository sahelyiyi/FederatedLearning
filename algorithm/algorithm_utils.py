import torch

import numpy as np

from torch.autograd import Variable

from algorithm.optimizer import TorchLinearModel, LinearModel, LinearOptimizer, TorchLinearOptimizer, \
    TorchLogisticOptimizer


def prepare_data_for_algorithm1(B, X, Y, samplingset, loss_func='linear_reg'):
    '''
    :param B: the adjacency matrix of the graph
    :param X: a list containing the feature vectors of the nodes of the graph
    :param Y: a list containing the labels of the nodes of the graph
    :param samplingset: the sampling set for the algorithm 1
    :param loss_func: the selected loss function for the optimizer

    :return: datapoints: a dictionary containing the data of each node needed for the algorithm 1
    '''

    node_degrees = np.array((1.0 / (np.sum(abs(B), 0)))).ravel()
    '''
    node_degrees: a list containing the nodes degree for the alg1 (1/N_i)
    '''

    datapoints = {}
    for i in range(len(X)):

        if 'torch' in loss_func:
            features = Variable(torch.from_numpy(X[i])).to(torch.float32)
        else:
            features = X[i]

        datapoints[i] = {
            'features': features,
            'degree': node_degrees[i]
        }

        if i in samplingset:
            _, n = X[i].shape

            if 'torch' in loss_func:
                label = Variable(torch.from_numpy(np.array(Y[i]))).to(torch.float32)
            else:
                label = Y[i]

            if loss_func == 'linear_reg':
                model = LinearModel(node_degrees[i], features, label)
                optimizer = LinearOptimizer(model)

            elif loss_func == 'torch_linear_reg':
                model = TorchLinearModel(n)
                optimizer = TorchLinearOptimizer(model)

            elif loss_func == 'logistic_reg':
                model = TorchLinearModel(n)
                optimizer = TorchLogisticOptimizer(model)

            else:
                raise Exception('invalid loss_func')
            '''
            model : the model for the node i 
            optimizer : the optimizer for the node i 
            '''

            datapoints[i].update({
                'label': label,
                'optimizer': optimizer
            })

    return datapoints

