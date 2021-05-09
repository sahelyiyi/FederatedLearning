import numpy as np

from algorithm.algorithm_utils import get_matrices
from algorithm.penalty import get_penalty


def algorithm_1(K, B, weight_vec, data, true_labels, samplingset, lambda_lasso, penalty_func_name='norm1', score_func=None):
    Sigma, Gamma, Gamma_vec, D = get_matrices(weight_vec, B)

    E, N = B.shape
    m, n = data[0]['features'].shape

    penalty_func = get_penalty(penalty_func_name, lambda_lasso, weight_vec, Sigma, n)

    new_w = np.array([np.zeros(n) for i in range(N)])
    new_u = np.array([np.zeros(n) for i in range(E)])

    iteration_scores = []
    for iterk in range(K):
        # if iterk % 100 == 0:
        #     print ('iter:', iterk)
        prev_w = np.copy(new_w)

        hat_w = new_w - np.dot(Gamma, np.dot(D.T, new_u))

        for i in range(N):
            if i in samplingset:
                optimizer = data[i]['optimizer']
                new_w[i] = optimizer.optimize(data[i]['features'], data[i]['label'], hat_w[i], data[i]['degree'])
            else:
                new_w[i] = hat_w[i]

        tilde_w = 2 * new_w - prev_w
        new_u = new_u + np.dot(Sigma, np.dot(D, tilde_w))

        new_u = penalty_func.update(new_u)

        if score_func:
            Y_pred = []
            for i in range(N):
                Y_pred.append(np.dot(data[i]['features'], new_w[i]))

            iteration_scores.append(score_func(true_labels.reshape(N, m), Y_pred))

    # print (np.max(abs(new_w - prev_w)))

    return iteration_scores, new_w
