import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict

from utils import nmse_func


def get_algorithm_1_scores(X, Y, new_w, samplingset, not_samplingset):
    Y_pred = []
    for i in range(len(X)):
        Y_pred.append(np.dot(X[i], new_w[i]))
    Y_pred = np.array(Y_pred)

    alg_1_score = {'total': mean_squared_error(Y, Y_pred),
                   'train': mean_squared_error(Y[samplingset], Y_pred[samplingset]),
                   'test': mean_squared_error(Y[not_samplingset], Y_pred[not_samplingset])}

    return alg_1_score


def get_linear_regression_score(x, y, decision_tree_samplingset, decision_tree_not_samplingset):
    model = LinearRegression().fit(x[decision_tree_samplingset], y[decision_tree_samplingset])
    pred_y = model.predict(x)
    linear_regression_score = {'total': mean_squared_error(y, pred_y),
                               'train': mean_squared_error(y[decision_tree_samplingset],
                                                           pred_y[decision_tree_samplingset]),
                               'test': mean_squared_error(y[decision_tree_not_samplingset],
                                                          pred_y[decision_tree_not_samplingset])}

    return linear_regression_score


def get_decision_tree_score(x, y, decision_tree_samplingset, decision_tree_not_samplingset):
    max_depth = 2
    regressor = DecisionTreeRegressor(max_depth=max_depth)
    regressor.fit(x[decision_tree_samplingset], y[decision_tree_samplingset])
    pred_y = regressor.predict(x)
    decision_tree_score = {'total': mean_squared_error(y, pred_y),
                           'train': mean_squared_error(y[decision_tree_samplingset],
                                                       pred_y[decision_tree_samplingset]),
                           'test': mean_squared_error(y[decision_tree_not_samplingset],
                                                      pred_y[decision_tree_not_samplingset])}
    return decision_tree_score


def get_scores(X, Y, new_w, samplingset):

    N = len(X)
    m, n = X[0].shape

    not_samplingset = [i for i in range(N) if i not in samplingset]

    alg_1_score = get_algorithm_1_scores(X, Y, new_w, samplingset, not_samplingset)

    y = Y.reshape(-1, 1)
    x = X.reshape(-1, n)
    decision_tree_samplingset = []
    for item in samplingset:
        for i in range(m):
            decision_tree_samplingset.append(m * item + i)
    decision_tree_not_samplingset = [i for i in range(m * N) if i not in decision_tree_samplingset]

    linear_regression_score = get_linear_regression_score(x, y, decision_tree_samplingset, decision_tree_not_samplingset)

    decision_tree_score = get_decision_tree_score(x, y, decision_tree_samplingset, decision_tree_not_samplingset)

    return alg_1_score, linear_regression_score, decision_tree_score
