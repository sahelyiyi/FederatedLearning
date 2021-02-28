import numpy as np
import matplotlib.pyplot as plt

from deep_learning_lasso.models import get_new_model


def get_new_model_output(new_model_weights, base_model_output):
    new_model_weights = [np.array(new_model_weights[:-1]).reshape(-1, 1), np.array(new_model_weights[-1:])]

    new_model = get_new_model()
    new_model.set_weights(new_model_weights)

    model_predicts = new_model.predict(base_model_output).flatten()
    model_predicts[model_predicts > 0] = 1
    model_predicts[model_predicts <= 0] = 0

    return model_predicts


def save_figures(alg1_estimated_weights, original_weights, lambda_lasso, base_model_output, true_labels):
    '''

    :param alg1_estimated_weights: the weights of the models estimated by algorithm 1
    :param original_weights: the weights of the models based on network's training
    :param lambda_lasso: lambda_lasso parameter used for algorithm 1
    :param base_model_output: the output of the base model(pre-trained model) for all the images
    :param true_labels: the true label of all the images

    '''

    N = len(alg1_estimated_weights)

    alq1_scores = []  # blue curve
    trained_model_scores = []  # orange curve

    for i in range(N):

        # the trained model output for all images
        trained_model_output = get_new_model_output(original_weights[i], base_model_output)
        # orange curve
        trained_model_score = np.where(true_labels == trained_model_output)[0].shape[0] / len(true_labels)
        trained_model_scores.append(trained_model_score)

        # alg1 output for all images
        alg1_output = get_new_model_output(alg1_estimated_weights[i], base_model_output)
        # blue curve
        alg1_score = np.where(true_labels == alg1_output)[0].shape[0] / len(alg1_output)
        alq1_scores.append(alg1_score)

    x_axis = [i for i in range(N)]
    plt.close()
    plt.plot(x_axis, alq1_scores, label='our')
    plt.plot(x_axis, trained_model_scores, label='deep learning')
    plt.title('alg1 vs trained accuracy')
    plt.xlabel('model')
    plt.ylabel('accuracy')
    plt.legend(loc="lower left")
    plt.savefig('deep_learning_lasso/train_accuracy_%s.png' % lambda_lasso)
