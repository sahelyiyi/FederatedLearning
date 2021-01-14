import random
from stochastic_block_model import get_B_and_weight_vec
from regression_lasso.main import *


def run_reg_sbm_2blocks(K, lambda_lasso, m, n, N1=150, N2=150, M=0.2):
    B, weight_vec = get_B_and_weight_vec([N1, N2], pout=0.001, mu_in=40, mu_out=10)
    E, N = B.shape

    X = []
    for i in range(N):
        sigma, mu = 1.0, 0.0
        g = np.random.normal(mu, sigma, (m, n))
        X.append(g)
    X = np.array(X)

    W1 = np.random.random(n)
    W1 = np.array([2, 2])
    W2 = np.random.normal(5, 2, n)
    W2 = np.array([-2, 2])
    # print (W1, W2)

    Y = []
    W = []
    for i in range(N):
        if i < N1:
            Y.append(np.dot(X[i], W1))
            W.append(W1)
        else:
            Y.append(np.dot(X[i], W2))
            W.append(W2)
    Y = np.array(Y)
    W = np.array(W)

    samplingset = random.sample([i for i in range(N)], k=int(M * N))
    return reg_run(K, B, weight_vec, Y, X, samplingset, lambda_lasso, method='norm')


def run_reg_sbm_4blocks(K, lambda_lasso, m=5, n=2, M=0.2):
    block_sizes = [70, 10, 50, 100, 150]
    blocks_num = len(block_sizes)
    B, weight_vec = get_B_and_weight_vec(block_sizes, pout=0.001, mu_in=40, mu_out=10)
    E, N = B.shape

    X = []
    for i in range(N):
        sigma, mu = 1.0, 0.0
        g = np.random.normal(mu, sigma, (m, n))
        X.append(g)
    X = np.array(X)

    block_Ws = []
    for i in range(blocks_num):
        block_Ws.append(np.random.random(n))

    Y = []
    W = []
    cnt = 0
    for j in range(blocks_num):
        for i in range(block_sizes[j]):
            block_W = block_Ws[j]
            Y.append(np.dot(X[cnt], block_W))
            W.append(block_W)
            cnt += 1

    Y = np.array(Y)
    W = np.array(W)

    samplingset = random.sample([i for i in range(N)], k=int(M * N))
    return reg_run(K, B, weight_vec, Y, X, samplingset, lambda_lasso, method='norm')

# run_reg_sbm_2blocks(K=2000, lambda_lasso=0.5, m=5, n=15)
# run_reg_sbm_2blocks(K=2000, lambda_lasso=1, m=5, n=2) -> 0.0001

