import random
import numpy as np

from algorithm import algorithm_1
from algorithm_utils import prepare_data_for_algorithm1
from deep_learning_lasso.models import get_base_model_output
from deep_learning_lasso.tensorflow_dataset_utils import get_B_and_weight_vec, load_trained_data
from deep_learning_lasso.scores import save_figures


def get_Y_and_W(X, trained_models_weights):
    '''
    :param X: A list of the features of algorithm 1
    :param trained_models_weights: A list containing the weight of the new model based on training each model
    '''

    Y = []
    W = []
    for i in range(len(X)):

        # The weights of the trainable layers (the new model) of the i_th trained model
        weights = trained_models_weights[i]

        # the weights of the dense layer of the model
        w1 = np.array(weights[-2]).flatten()

        # the bias of the dense layer
        w2 = weights[-1]

        # combining the weights and the bias of the dense layer of the model, which is the weight of the node for alg1
        w = np.concatenate((w1, w2))
        W.append(w)

        # the label of the i_th node for alg1
        Y.append(X[i].dot(w))

    Y = np.array(Y)
    W = np.array(W)

    return Y, W


def deep_learning_run(lambda_lasso, K=1000, train_data_dir='deep_learning_lasso/new_deeplarning_data'):

    # calculate base model output and true labels for all images
    base_model_output, true_labels = get_base_model_output()
    '''
    base_model_output: A list containing the output of the base(pre-trained) model for all the images
    true_labels: A list containing the true label of all the images (which is 0 or 1 for each image)
    '''

    # load trained data from saved models in train_data_dir
    trained_models_train_images, trained_models_weights, X, Y = load_trained_data(train_data_dir, base_model_output, true_labels)
    '''
    trained_models_train_images: A list containing the images used for training each model
    trained_models_weights : A list containing the weight of the new model based on training each model
    X : A list containing the output of the base model for trainset of each model, which is the features of algorithm 1
    Y : A list containing the true labels for trainset of each model, which is the labels of algorithm 1
    '''

    # create B and weight_vec for the empirical graph G
    B, weight_vec = get_B_and_weight_vec(trained_models_train_images)
    E, N = B.shape
    '''
    B : Incidence matrix of the empirical graph G
    weight_vec : Wight of each edge of the empirical graph G
    '''

    # calculate the weights(W) of the empirical graph G
    _, W = get_Y_and_W(X, trained_models_weights)
    print ("hereee", Y.shape, W.shape, X.shape, true_labels.shape)
    '''
    W : The weights of the nodes for the algorihtm 1
    '''

    # choose sampling set for alg1
    M=0.2
    samplingset = random.sample([i for i in range(N)], k=int(M * N))
    # samplingset = [53, 92, 99, 19, 16, 32, 6, 9, 39, 43, 34, 54, 23, 8, 13, 88, 1, 62, 22, 60]
    '''
    samplingset : The samplingset selected for algorithm 1
    '''

    # alg1
    print ('start alg')
    data = prepare_data_for_algorithm1(B, X, Y, loss_func='linear_reg')
    _, alg1_estimated_weights = algorithm_1(K, B, weight_vec, data, Y, samplingset, lambda_lasso)
    '''
    alg1_estimated_weights : The estimated weights by algorithm 1
    '''

    # save the orange and blue fig
    save_figures(alg1_estimated_weights, W, lambda_lasso, base_model_output, true_labels)

    return alg1_estimated_weights
