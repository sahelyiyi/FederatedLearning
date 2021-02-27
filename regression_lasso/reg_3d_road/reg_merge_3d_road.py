import random

from regression_lasso.main import reg_run
from regression_lasso.reg_3d_road.reg_3d_road_utils import load_data, get_graph_data


def run_reg_merge_3d_road(K=1000, lambda_lasso=0.1):
    data = load_data()

    B, weight_vec, Y, X = get_graph_data(data)

    N, E = B.shape
    samplingset = random.sample([i for i in range(N)], k=int(0.7 * N))

    # lambda_lasso = 0.1  # nLasso parameter
    # lambda_lasso = 0.08  # nLasso parameter

    return reg_run(K, B, weight_vec, Y, X, samplingset, lambda_lasso)


# lambda=0.1, M=0.7
# NMSE 0.20356074162702414
# (new) NMSE 0.22
# linear_regression_score 0.37282467073282244
# decision_tree_score 0.2575542348747358

# lambda=0.08, M=0.7
# NMSE 0.21768699584301643

# lambda=0.05, M=0.7
# NMSE 0.22918163759820392

# lambda=0.5, M=0.6
# NMSE 0.2747315880798612
# linear_regression_score 0.372930118135755
# decision_tree_score 0.5804098158147778

# lambda=0.2, M=0.6
# NMSE 0.2826136578917171
