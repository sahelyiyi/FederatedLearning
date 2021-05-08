from algorithm import algorithm_1
from algorithm_utils import prepare_data_for_algorithm1

from regression_lasso.scores import get_scores


def reg_run(K, B, weight_vec, Y, X, samplingset, lambda_lasso=0.1):

    data = prepare_data_for_algorithm1(B, X, Y, loss_func='linear_reg')

    iteration_scores, new_w = algorithm_1(K, B, weight_vec, data, Y, samplingset, lambda_lasso)

    our_score, linear_regression_score, decision_tree_score = get_scores(X, Y, new_w, samplingset)

    return iteration_scores, our_score, linear_regression_score, decision_tree_score
