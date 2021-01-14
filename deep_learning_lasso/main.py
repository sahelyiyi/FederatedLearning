import os
import json
from collections import defaultdict
import tensorflow_datasets as tfds
import tensorflow as tf

from deep_learning_lasso.deep_learning_utils import *
from utils import algorithm


def get_pre_trained_model_results():
    base_model = get_base_model()

    (train_ds,), metadata = tfds.load(
        "cats_vs_dogs",
        split=["train[:100%]"],
        shuffle_files=True,
        with_info=True,
    )

    size = (Image_Width, Image_Height)

    train_ds = train_ds.map(lambda item: (tf.image.resize(item['image'], size), item['label']))

    batch_size = 32

    train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)

    all_predicts = base_model.predict(train_ds)

    true_labels = []
    for obj in train_ds:
        true_labels += list(np.array(obj[1]))
    true_labels = np.array(true_labels)

    return all_predicts, true_labels


def get_origin_predicts(train_data, pre_trained_results, true_labels):
    extra_model = get_extra_layers()
    extra_model.fit(pre_trained_results[np.where(train_data == 1)], true_labels[np.where(train_data == 1)])
    origin_predicts = extra_model.predict(pre_trained_results).flatten()
    origin_predicts[origin_predicts > 0] = 1
    origin_predicts[origin_predicts <= 0] = 0
    return origin_predicts, extra_model.get_weights()


def parse_trained_data(data, new_data, pre_trained_results, all_images_indices, true_labels):
    all_images_size = len(all_images_indices.keys())

    extra_model = get_extra_layers()
    all_train_data = []
    all_weights = []
    all_scores = []
    X = []
    all_origin_predicts = []
    for item in data:
        # print ('.')
        train_data = item['train_df']
        train_images_vector = np.zeros(all_images_size)
        item_predicts = []
        for train_image_name in train_data:
            index = all_images_indices[train_image_name]
            train_images_vector[index] = 1
            item_predict = np.concatenate((pre_trained_results[index], [1]))
            item_predicts.append(item_predict)

        X.append(np.array(item_predicts))

        all_train_data.append(train_images_vector)

        model_weights = []
        for weight in item['weights'][-2:]:
            model_weights.append(np.array(weight))
        model_weights = np.array(model_weights)
        all_weights.append(model_weights)

        extra_model.set_weights(model_weights)
        origin_predicts = extra_model.predict(pre_trained_results).flatten()
        origin_predicts[origin_predicts > 0] = 1
        origin_predicts[origin_predicts <= 0] = 0
        all_origin_predicts.append(origin_predicts)

        score = np.where(true_labels == origin_predicts)[0].shape[0] / len(true_labels)
        all_scores.append(score)

    for train_data in new_data:
        train_images_vector = np.zeros(all_images_size)
        item_predicts = []
        for train_image_name in train_data:
            index = all_images_indices[train_image_name]
            train_images_vector[index] = 1
            item_predict = np.concatenate((pre_trained_results[index], [1]))
            item_predicts.append(item_predict)
        X.append(np.array(item_predicts))
        all_train_data.append(train_images_vector)

        origin_predicts, model_weights = get_origin_predicts(train_images_vector, pre_trained_results)
        all_origin_predicts.append(origin_predicts)
        all_weights.append(model_weights)

        score = np.where(true_labels == origin_predicts)[0].shape[0] / len(true_labels)
        all_scores.append(score)

    X = np.array(X)

    return all_train_data, all_scores, all_weights, all_origin_predicts, X


def get_dist(first_node, second_node):
    all_equals = np.where(first_node == second_node)[0]
    equal_train_images = np.where(first_node[all_equals] == 1)[0]
    dist = len(equal_train_images) / len(np.where(first_node == 1)[0])
    return dist


def check_accuracy_and_trainset_correlation(data, all_scores, all_train_data):
    train_nns_size = len(data)
    train_dist_mtx = np.zeros((train_nns_size, train_nns_size))
    scores_mtx = np.zeros((train_nns_size, train_nns_size))
    for i in range(train_nns_size):
        for j in range(train_nns_size):
            train_dist_mtx[i, j] = get_dist(all_train_data[i], all_train_data[j])
            scores_mtx[i, j] = np.sqrt((all_scores[i] - all_scores[j]) ** 2)
    pearson = np.corrcoef(train_dist_mtx.flatten(), scores_mtx.flatten())
    # print (pearson)


def get_B_and_weight_vec(all_train_data):
    N = len(all_train_data)
    E = int(N * (N - 1) / 2)

    dist_mtx = np.zeros((N, N))
    weight_vec = np.zeros(E)
    B = np.zeros((E, N))
    cnt = 0
    neigh_cnt = 3
    for i in range(N):
        node_dists = []
        for j in range(N):
            if j == i:
                continue
            node_dists.append(get_dist(all_train_data[i], all_train_data[j]))
        node_dists.sort(reverse=True)
        node_cnt = 0
        for j in range(N):
            if node_cnt >= neigh_cnt:
                continue
            if j == i:
                continue
            dist = get_dist(all_train_data[i], all_train_data[j])
            if dist == 0 or dist < node_dists[neigh_cnt]:
                continue
            node_cnt += 1
            dist_mtx[i, j] = dist
            B[cnt][i] = 1
            B[cnt][j] = -1
            weight_vec[cnt] = dist
            cnt += 1

    B = B[:cnt, :]
    weight_vec = weight_vec[:cnt]
    return B, weight_vec


def deep_learning_run(lambda_lasso, K=1000, train_data_dir='deep_learning_lasso/deep_learning_data'):
    data = []
    for filename in sorted(os.listdir(train_data_dir)):
        if '.json' not in filename:
            continue
        num = filename.split('_')[-1].replace('.json', '')
        if int(num) >= 100:
            continue
        with open(os.path.join(train_data_dir, filename), 'r') as f:
            data.append(json.load(f))

    all_images_indices = {}
    cnt = 0
    for train_data in data:
        for i, file_name in enumerate(train_data['train_df']):
            if file_name not in all_images_indices:
                all_images_indices[file_name] = cnt
                cnt += 1
    pre_trained_results, true_labels = get_pre_trained_model_results()

    new_data = []
    all_train_data, all_scores, all_weights, all_origin_predicts, X = parse_trained_data(data, new_data, pre_trained_results, all_images_indices, true_labels)

    B, weight_vec = get_B_and_weight_vec(all_train_data)

    E, N = B.shape
    m, n = X[0].shape

    edges = np.where(B!=0)[1]
    neighs = defaultdict(list)
    for i in range(len(edges)):
        if i % 2 == 1:
            continue
        v = edges[i]
        u = edges[i+1]
        neighs[v].append(u)
        neighs[u].append(v)

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

    M=0.2
    # samplingset = random.sample([i for i in range(N)], k=int(M * N))
    samplingset = [53, 92, 99, 19, 16, 32, 6, 9, 39, 43, 34, 54, 23, 8, 13, 88, 1, 62, 22, 60]

    iteration_scores, new_w = algorithm(K, B, weight_vec, X, Y, samplingset, lambda_lasso)
    return iteration_scores, new_w
    # scores = []
    # for i in range(N):
    #     if i % 10 == 0:
    #         print (i)
    #     origin_predicts = all_origin_predicts[i]
    #
    #     weights = [np.array(new_w[i][:-1]).reshape(-1, 1), np.array(new_w[i][-1:])]
    #     extra_model = get_extra_layers()
    #     extra_model.set_weights(weights)
    #     our_predicts = extra_model.predict(pre_trained_results).flatten()
    #     our_predicts[our_predicts > 0] = 1
    #     our_predicts[our_predicts <= 0] = 0
    #
    #     estimate_score = np.where(origin_predicts == our_predicts)[0].shape[0] / len(our_predicts)
    #     our_score = np.where(true_labels == our_predicts)[0].shape[0] / len(our_predicts)
    #     origin_score = np.where(true_labels == origin_predicts)[0].shape[0] / len(true_labels)
    #     scores.append({'estimate_score': estimate_score, 'our_score': our_score, 'origin_score': origin_score})
    #
    # lambda_score = {
    #     'estimate_scores': list(map(lambda d: d['estimate_score'], scores)),
    #     'our_scores': list(map(lambda d: d['our_score'], scores)),
    #     'origin_scores': list(map(lambda d: d['origin_score'], scores)),
    # }

    # for lambda_lasso in lambda_scores:
    #     rate = lambda_scores[lambda_lasso]['origin_scores']
    #     print(sorted(rate)[-5:])
    #     rate = lambda_scores[lambda_lasso]['our_scores']
    #     print(sorted(rate)[-5:])
    #     print()
    #
    # x_axis = [i for i in range(N)]
    # for lambda_lasso in lambda_scores:
    #     plt.close()
    #     plt.plot(x_axis, lambda_scores[lambda_lasso]['our_scores'], label='our')
    #     plt.plot(x_axis, lambda_scores[lambda_lasso]['origin_scores'], label='deep learning')
    #     plt.title('our vs origin accuracy')
    #     plt.xlabel('train model')
    #     plt.ylabel('accuracy')
    #     plt.legend(loc="lower left")
    #     plt.savefig('deep_learning_lasso/train_accuracy_%s.png' % lambda_lasso)
    #
    # plt.close()
    # plt.plot(x_axis, lambda_scores[lambda_lasso]['origin_scores'], label='deep learning')
    # x_axis = [i for i in range(N)]
    # for lambda_lasso in lambda_scores:
    #     plt.plot(x_axis, lambda_scores[lambda_lasso]['our_scores'], label='lambda=%s' % lambda_lasso)
    #
    # plt.title('our vs origin accuracy')
    # plt.xlabel('train model')
    # plt.ylabel('accuracy')
    # plt.legend(loc="lower left")
    # plt.savefig('deep_learning_lasso/train_accuracy_all.png')
