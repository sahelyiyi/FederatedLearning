from algorithm import algorithm_1

from scores import get_scores


def reg_run(K, B, weight_vec, Y, X, samplingset, lambda_lasso=0.1):

    iteration_scores, new_w = algorithm_1(K, B, weight_vec, X, Y, samplingset, lambda_lasso)

    our_score, linear_regression_score, decision_tree_score = get_scores(X, Y, new_w, samplingset)

    return iteration_scores, our_score, linear_regression_score, decision_tree_score
