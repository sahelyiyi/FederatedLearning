import numpy as np

from sklearn.metrics import mean_squared_error


def get_matrices(weight_vec, B):
    Sigma = np.diag(np.full(weight_vec.shape, 0.9 / 2))

    D = B

    Gamma_vec = np.array((1.0 / (np.sum(abs(B), 0)))).ravel()
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


def algorithm_1(K, B, weight_vec, X, Y, samplingset, lambda_lasso, score_func=mean_squared_error):
    Sigma, Gamma, Gamma_vec, D = get_matrices(weight_vec, B)

    E, N = B.shape
    m, n = X[0].shape

    MTX1_INV, MTX2 = get_preprocessed_matrices(samplingset, Gamma_vec, X, Y)

    hat_w = np.array([np.zeros(n) for i in range(N)])
    new_w = np.array([np.zeros(n) for i in range(N)])
    prev_w = np.array([np.zeros(n) for i in range(N)])
    new_u = np.array([np.zeros(n) for i in range(E)])

    # K = 1000
    iteration_scores = []
    limit = np.array([np.zeros(n) for i in range(E)])
    for i in range(n):
        limit[:, i] = lambda_lasso * weight_vec

    for iterk in range(K):
        # if iterk % 100 == 0:
        #     print ('iter:', iterk)
        prev_w = np.copy(new_w)

        hat_w = new_w - np.dot(Gamma, np.dot(D.T, new_u))  # could  be negative

        for i in range(N):
            if i in samplingset:
                mtx2 = hat_w[i] + MTX2[i]
                mtx_inv = MTX1_INV[i]

                new_w[i] = np.dot(mtx_inv, mtx2)
            else:
                new_w[i] = hat_w[i]

        tilde_w = 2 * new_w - prev_w
        new_u = new_u + np.dot(Sigma, np.dot(D, tilde_w))  # chould be negative

        normalized_u = np.where(abs(new_u) >= limit)
        new_u[normalized_u] = limit[normalized_u] * new_u[normalized_u] / abs(new_u[normalized_u])

        Y_pred = []
        for i in range(N):
            Y_pred.append(np.dot(X[i], new_w[i]))

        iteration_scores.append(score_func(Y.reshape(N, m), Y_pred))

    # print (np.max(abs(new_w - prev_w)))

    return iteration_scores, new_w
