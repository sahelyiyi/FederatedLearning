import numpy as np
import random

from regression_lasso.main import reg_run


def get_graph_data(N):
    E = int(N * (N - 1) / 2)
    Y = np.array([[2] for i in range(N)])
    X = np.array([[[1, 1, 1]] for i in range(N)])
    m, n = X[0].shape

    B = np.zeros((E, N))  # incidence matrix
    cnt = 0
    for i in range(N):
        for j in range(i + 1, N):
            B[cnt, i] = 1
            B[cnt, j] = -1
            cnt += 1

    weight_vec = 2 * np.ones(E)

    return B, weight_vec, Y, X


def run_reg_complete(K=300, N=100, lambda_lasso=1/3, M=0.2):
    B, weight_vec, Y, X = get_graph_data(N)

    samplingset = random.sample([i for i in range(N)], k=int(M * N))

    return reg_run(K, B, weight_vec, Y, X, samplingset, lambda_lasso, method='norm')
