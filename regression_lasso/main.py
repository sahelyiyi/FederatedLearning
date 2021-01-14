import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error
from collections import defaultdict

from utils import algorithm, nmse_func


def reg_run(K, B, weight_vec, Y, X, samplingset, lambda_lasso=0.1, method=None):
    functions = {
        'mean_squared_error': mean_squared_error,
        'normalized_mean_squared_error': nmse_func,
        'mean_absolute_error': mean_absolute_error
    }

    if method == 'log':
        default_score_func = mean_squared_log_error
    elif method == 'norm':
        default_score_func = nmse_func
    else:
        default_score_func = mean_squared_error

    E, N = B.shape
    m, n = X[0].shape
    not_samplingset = [i for i in range(N) if i not in samplingset]
    iteration_scores, new_w = algorithm(K, B, weight_vec, X, Y, samplingset, lambda_lasso)

    Y_pred = []
    for i in range(N):
        Y_pred.append(np.dot(X[i], new_w[i]))
    Y_pred = np.array(Y_pred)

    our_score = defaultdict(dict)
    for score_func_name, score_func in functions.items():
        our_score['total'][score_func_name] = score_func(Y, Y_pred)
        our_score['train'][score_func_name] = score_func(Y[samplingset], Y_pred[samplingset])
        our_score['test'][score_func_name] = score_func(Y[not_samplingset], Y_pred[not_samplingset])

    y = Y.reshape(-1, 1)
    x = X.reshape(-1, n)
    decision_tree_samplingset = []
    for item in samplingset:
        for i in range(m):
            decision_tree_samplingset.append(m * item + i)
    decision_tree_not_samplingset = [i for i in range(m*N) if i not in decision_tree_samplingset]

    model = LinearRegression().fit(x[decision_tree_samplingset], y[decision_tree_samplingset])
    pred_y = model.predict(x)
    linear_regression_score = defaultdict(dict)
    for score_func_name, score_func in functions.items():
        linear_regression_score['total'][score_func_name] = score_func(y, pred_y)
        linear_regression_score['train'][score_func_name] = score_func(y[decision_tree_samplingset], pred_y[decision_tree_samplingset])
        linear_regression_score['test'][score_func_name] = score_func(y[decision_tree_not_samplingset], pred_y[decision_tree_not_samplingset])

    max_depth = 2
    regressor = DecisionTreeRegressor(max_depth=max_depth)
    regressor.fit(x[decision_tree_samplingset], y[decision_tree_samplingset])
    pred_y = regressor.predict(x)
    decision_tree_score = defaultdict(dict)
    for score_func_name, score_func in functions.items():
        decision_tree_score['total'][score_func_name] = score_func(y, pred_y)
        decision_tree_score['train'][score_func_name] = score_func(y[decision_tree_samplingset], pred_y[decision_tree_samplingset])
        decision_tree_score['test'][score_func_name] = score_func(y[decision_tree_not_samplingset], pred_y[decision_tree_not_samplingset])

    return iteration_scores, our_score, linear_regression_score, decision_tree_score
