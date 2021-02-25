import numpy as np
import matplotlib.pyplot as plt

from deep_learning_lasso.deep_learning_utils import get_new_model
from deep_learning_lasso.tensorflow_dataset_utils import get_base_model_outputs


def get_alg1_output(model_new_w, pre_trained_results):
    alg_1_weights = [np.array(model_new_w[:-1]).reshape(-1, 1), np.array(model_new_w[-1:])]
    new_model = get_new_model()
    new_model.set_weights(alg_1_weights)
    our_predicts = new_model.predict(pre_trained_results).flatten()
    our_predicts[our_predicts > 0] = 1
    our_predicts[our_predicts <= 0] = 0
    return our_predicts


def save_figures(new_w, W, lambda_lasso, base_model_output, true_labels):

    N = len(new_w)

    alq1_scores = []  # blue curve
    trained_model_scores = []  # orange curve

    for i in range(N):

        # the trained model output for all images
        trained_model_output = get_alg1_output(W[i], base_model_output)
        # orange curve
        trained_model_score = np.where(true_labels == trained_model_output)[0].shape[0] / len(true_labels)
        trained_model_scores.append(trained_model_score)

        # alg1 output for all images
        alg1_output = get_alg1_output(new_w[i], base_model_output)
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
