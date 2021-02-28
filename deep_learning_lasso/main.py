import numpy as np

from algorithm import algorithm_1
from deep_learning_lasso.models import get_base_model_output
from deep_learning_lasso.tensorflow_dataset_utils import get_B_and_weight_vec, load_trained_data
from deep_learning_lasso.scores import save_figures


def deep_learning_run(lambda_lasso, K=1000, train_data_dir='deep_learning_lasso/deep_learning_data'):
    # calculate base model output and true labels for all images
    base_model_output, true_labels = get_base_model_output()

    # load trained data from saved models
    all_models_train_images, all_weights, X = load_trained_data(train_data_dir, base_model_output)

    # create B and weight_vec
    B, weight_vec = get_B_and_weight_vec(all_models_train_images)
    E, N = B.shape

    # create Y and W
    Y = []
    W = []
    for i in range(N):
        weights = all_weights[i]

        w1 = np.array(weights[-2]).flatten()
        w2 = weights[-1]
        w = np.concatenate((w1, w2))
        W.append(w)

        Y.append(X[i].dot(w))

    Y = np.array(Y)
    W = np.array(W)

    # choose sampling set for alg1
    M=0.2
    # samplingset = random.sample([i for i in range(N)], k=int(M * N))
    samplingset = [53, 92, 99, 19, 16, 32, 6, 9, 39, 43, 34, 54, 23, 8, 13, 88, 1, 62, 22, 60]

    # alg1
    iteration_scores, alg1_estimated_weights = algorithm_1(K, B, weight_vec, X, Y, samplingset, lambda_lasso)

    # save the orange and blue fig
    save_figures(alg1_estimated_weights, W, lambda_lasso, base_model_output, true_labels)

    return iteration_scores, alg1_estimated_weights
