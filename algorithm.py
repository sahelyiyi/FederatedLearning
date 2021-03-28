import contextlib
import functools
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod


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


class Optimizer(ABC):

    @abstractmethod
    def optimize(self, idx, hat_w):
        pass


class LinearOptimizer(Optimizer):

    def __init__(self, samplingset, Gamma_vec, X, Y):
        super(Optimizer).__init__()
        self.MTX1_INV, self.MTX2 = get_preprocessed_matrices(samplingset, Gamma_vec, X, Y)

    def optimize(self, idx, hat_w):
        mtx2 = hat_w[idx] + self.MTX2[idx]
        mtx_inv = self.MTX1_INV[idx]

        return np.dot(mtx_inv, mtx2)


class LogisticOptimizer(Optimizer):

    def __init__(self, tau, X, Y):
        super(Optimizer).__init__()
        self.tau = tau
        self.X = tf.constant(X, dtype=tf.float64)
        self.Y = tf.constant(Y, dtype=tf.float64)

    def optimize(self, idx, hat_w):
        def make_val_and_grad_fn(value_fn):
            @functools.wraps(value_fn)
            def val_and_grad(x):
                return tfp.math.value_and_gradient(value_fn, x)

            return val_and_grad

        @contextlib.contextmanager
        def timed_execution():
            t0 = time.time()
            yield
            dt = time.time() - t0
            print('Evaluation took: %f seconds' % dt)

        def np_value(tensor):
            if isinstance(tensor, tuple):
                return type(tensor)(*(np_value(t) for t in tensor))
            else:
                return tensor.numpy()

        def run(optimizer):
            optimizer()
            # with timed_execution():
            result = optimizer()
            return np_value(result)

        def regression_loss(params):
            labels = Y[idx]
            feature = X[idx]
            new_params = tf.expand_dims(params, 1)
            logits = tf.matmul(feature, new_params)
            labels = tf.expand_dims(labels, 1)

            w = tf.expand_dims(tf.constant(hat_w[idx], dtype=tf.float64), 1)

            penalty_var = tf.math.subtract(w, params)
            loss_penalty = regularization_factor * tf.nn.l2_loss(penalty_var)

            mse_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

            total_loss = mse_loss + loss_penalty

            return total_loss

        @tf.function
        def l1_regression_with_lbfgs():
            return tfp.optimizer.lbfgs_minimize(
                make_val_and_grad_fn(regression_loss),
                initial_position=tf.constant(start),
                tolerance=1e-8)

        dim = len(hat_w[idx])
        start = np.random.randn(dim)
        X = self.X
        Y = self.Y

        regularization_factor = 1/(2*self.tau[idx])

        results = run(l1_regression_with_lbfgs)
        minimum = results.position
        return minimum


def algorithm_1(K, B, weight_vec, X, Y, samplingset, lambda_lasso, score_func=mean_squared_error, loss_func='linear_reg'):
    Sigma, Gamma, Gamma_vec, D = get_matrices(weight_vec, B)

    E, N = B.shape
    m, n = X[0].shape

    if loss_func == 'linear_reg':
        optimizer = LinearOptimizer(samplingset, Gamma_vec, X, Y)
    elif loss_func == 'logistic_reg':
        optimizer = LogisticOptimizer(Gamma_vec, X, Y)
    else:
        print('invalid loss_func')
        return

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
            if i in samplingset:
                new_w[i] = optimizer.optimize(i, hat_w)
            else:
                new_w[i] = hat_w[i]

        tilde_w = 2 * new_w - prev_w
        new_u = new_u + np.dot(Sigma, np.dot(D, tilde_w))

        normalized_u = np.where(abs(new_u) >= limit)
        new_u[normalized_u] = limit[normalized_u] * new_u[normalized_u] / abs(new_u[normalized_u])

        Y_pred = []
        for i in range(N):
            Y_pred.append(np.dot(X[i], new_w[i]))

        iteration_scores.append(score_func(Y.reshape(N, m), Y_pred))

    # print (np.max(abs(new_w - prev_w)))

    return iteration_scores, new_w
