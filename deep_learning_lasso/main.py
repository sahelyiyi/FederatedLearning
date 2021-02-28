import numpy as np

from algorithm import algorithm_1
from deep_learning_lasso.models import get_base_model_output
from deep_learning_lasso.tensorflow_dataset_utils import get_B_and_weight_vec, load_trained_data
from deep_learning_lasso.scores import save_figures


def get_Y_and_W(X, all_weights):
    Y = []
    W = []
    for i in range(len(X)):
        weights = all_weights[i]

        w1 = np.array(weights[-2]).flatten()
        w2 = weights[-1]
        w = np.concatenate((w1, w2))
        W.append(w)

        Y.append(X[i].dot(w))

    Y = np.array(Y)
    W = np.array(W)

    return Y, W


def deep_learning_run(lambda_lasso, K=1000, train_data_dir='deep_learning_lasso/deep_learning_data'):

    # calculate base model output and true labels for all images
    base_model_output, true_labels = get_base_model_output()
    '''
    base_model_output: the output of the base(pre-trained) model for all the images
    true_labels: the true label of all the images (which is 0 or 1 for each image)
    '''

    # load trained data from saved models in train_data_dir
    trained_models_train_images, trained_models_weights, X = load_trained_data(train_data_dir, base_model_output)
    '''
    trained_models_train_images: A list that contains the images used for training each model
    trained_models_weights : A list of that contains the weight of the new model based on training each model
    X : A list that contains the output of the base model for trainset of each model, which is the features used for algorithm 1
    '''

    # create B and weight_vec for the empirical graph G
    B, weight_vec = get_B_and_weight_vec(trained_models_train_images)
    E, N = B.shape
    '''
    B : Incidence matrix if the empirical graph G
    weight_vec : Wight of each edge of the empirical graph G
    '''

    # calculate the labels(Y) and weights(W) of the empirical graph G
    Y, W = get_Y_and_W(X, trained_models_weights)
    '''
    Y : The lables of the nodes for the algorihtm 1
    W : The weights of the nodes for the algorihtm 1
    '''

    # choose sampling set for alg1
    M=0.2
    # samplingset = random.sample([i for i in range(N)], k=int(M * N))
    samplingset = [53, 92, 99, 19, 16, 32, 6, 9, 39, 43, 34, 54, 23, 8, 13, 88, 1, 62, 22, 60]
    '''
    samplingset : The samplingset used for algorithm 1
    '''

    # alg1
    iteration_scores, alg1_estimated_weights = algorithm_1(K, B, weight_vec, X, Y, samplingset, lambda_lasso)
    '''
    alg1_estimated_weights : The estimated weights by algorithm 1
    '''

    # save the orange and blue fig
    save_figures(alg1_estimated_weights, W, lambda_lasso, base_model_output, true_labels)

    return iteration_scores, alg1_estimated_weights
