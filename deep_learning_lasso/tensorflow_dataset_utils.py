import json
import os

import numpy as np


def get_dist(first_node, second_node):
    all_equals = np.where(first_node == second_node)[0]
    equal_train_images = np.where(first_node[all_equals] == 1)[0]
    dist = len(equal_train_images) / len(np.where(first_node == 1)[0])
    return dist


def get_B_and_weight_vec(all_models_train_images, neigh_cnt=3):
    N = len(all_models_train_images)
    E = int(N * (N - 1) / 2)

    dist_mtx = np.zeros((N, N))
    weight_vec = np.zeros(E)
    B = np.zeros((E, N))
    cnt = 0
    for i in range(N):
        node_dists = []
        for j in range(N):
            if j == i:
                continue
            node_dists.append(get_dist(all_models_train_images[i], all_models_train_images[j]))
        node_dists.sort(reverse=True)

        node_cnt = 0
        for j in range(N):
            if node_cnt >= neigh_cnt:
                continue
            if j == i:
                continue
            dist = get_dist(all_models_train_images[i], all_models_train_images[j])
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


def get_model_train_images_data(train_data_image_names, all_images_indices, all_images_size, base_model_outputs):
    train_images_vector = np.zeros(all_images_size)
    base_model_output = []
    for train_image_name in train_data_image_names:
        index = all_images_indices[train_image_name]
        train_images_vector[index] = 1
        item_predict = np.concatenate((base_model_outputs[index], [1]))  # [1] is for the bias (b)
        base_model_output.append(item_predict)
    return base_model_output, train_images_vector


def get_all_images_dict(data):
    all_images_indices = {}
    cnt = 0
    for train_data in data:
        for i, file_name in enumerate(train_data['train_df']):
            if file_name not in all_images_indices:
                all_images_indices[file_name] = cnt
                cnt += 1
    return all_images_indices


def get_trained_model_weights(raw_model_weights):
    model_weights = []
    for weight in raw_model_weights[-2:]:
        model_weights.append(np.array(weight))
    model_weights = np.array(model_weights)
    return model_weights


# get the trained dataset and weights of each trained model and also the features of algorithm 1
def parse_saved_data(data, base_model_output):

    all_images_indices = get_all_images_dict(data)
    '''
    all_images_indices: a dictionary from image_name to index
    '''
    all_images_size = len(all_images_indices.keys())
    '''
    all_images_size : total number of images of the (tensorflow) dataset
    '''

    all_models_train_images = []
    all_models_weights = []
    X = []
    for model_data in data:

        base_model_train_images_output, model_train_images = get_model_train_images_data(model_data['train_df'], all_images_indices, all_images_size, base_model_output)
        '''
        base_model_train_images_output: the output of the base model for the training dataset of this model
        model_train_images: a vector from 0/1 with the size of "all_images_size", model_train_images[i] = 1 if 
                the image with the index i is in the train dataset of this model otherwise model_train_images[i] = 0
        '''

        X.append(np.array(base_model_train_images_output))

        all_models_train_images.append(model_train_images)

        model_weights = get_trained_model_weights(model_data['weights'])
        '''
        model_weights: the weights of this model for the new model (trainable layers)
        '''
        all_models_weights.append(model_weights)

    X = np.array(X)

    return all_models_train_images, all_models_weights, X


# read the trained models data from saved files
def read_trained_data_from_saved_files(train_data_dir):
    data = []
    for filename in sorted(os.listdir(train_data_dir)):
        if '.json' not in filename:
            continue
        num = filename.split('_')[-1].replace('.json', '')
        if int(num) >= 100:
            continue
        with open(os.path.join(train_data_dir, filename), 'r') as f:
            data.append(json.load(f))
    return data


def load_trained_data(train_data_dir, base_model_output):

    # read the trained models data from saved files
    data = read_trained_data_from_saved_files(train_data_dir)
    '''
    data: saved data from the trained models
    '''

    # get the trained dataset and weights of each trained model and also the features of algorithm 1
    all_models_train_images, all_models_weights, X = parse_saved_data(data, base_model_output)
    '''
    all_models_train_images: A list containing the images used for training each model
    all_models_weights : A list containing the weight of the new model based on training each model
    X : A list containing the output of the base model for trainset of each model, which is the features of algorithm 1
    '''

    return all_models_train_images, all_models_weights, X
