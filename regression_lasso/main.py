from algorithm.main import algorithm_1
from algorithm.algorithm_utils import prepare_data_for_algorithm1

from regression_lasso.scores import get_scores


def reg_run(K, B, weight_vec, Y, X, samplingset, lambda_lasso=0.1, loss_func='linear_reg', penalty_func='norm1'):

    data = prepare_data_for_algorithm1(B, X, Y, samplingset, loss_func)

    iteration_scores, new_w = algorithm_1(K, B, weight_vec, data, Y, samplingset, lambda_lasso, penalty_func)

    alg1_score, linear_regression_score, decision_tree_score = get_scores(X, Y, new_w, samplingset)

    return {
        'alg1_score': alg1_score,
        'linear_regression_score': linear_regression_score,
        'decision_tree_score': decision_tree_score
    }
