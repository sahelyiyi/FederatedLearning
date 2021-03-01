import json
import datetime
import numpy as np
import tensorflow_datasets as tfds

from deep_learning_lasso.deep_learning_utils import EPOCHS
from deep_learning_lasso.models import get_NN_model
from deep_learning_lasso.train_models.train_utils import NpEncoder, split_dataset


n_models = 200

# train different models with different train/validation/test datasets and save its data
for i in range(n_models):

    # load the data from tensorflow dataset
    dataset, metadata = tfds.load(
        "cats_vs_dogs",
        shuffle_files=True,
        with_info=True,
    )
    dataset = dataset['train']

    # prepare train/validation/test datasets for training the model
    (train_ds, train_df), (validation_ds, validate_df), (test_ds, test_df) = split_dataset(dataset, train_ratio=0.75, val_ratio=0.15, test_ratio=0.1)
    '''
    train_ds: dataset of the training data for the model (we need dataset to train the model)
    train_df: dataframe of the training data for the model (we need dataframe to save the model's data) 

    validation_ds: dataset of the validation data for the model
    validate_df: dataframe of the validation data for the model

    test_ds: dataset of the test data for the model
    test_df: dataframe of the test data for the model
    '''
    # create the model
    model = get_NN_model()

    # train the model for the selected train and validation datasets
    start = datetime.datetime.now()
    model.fit(train_ds, epochs=EPOCHS, validation_data=validation_ds)
    print(datetime.datetime.now() - start)

    # calculate the predicted labels of the trained model for the test dataset
    pred_labels = model.predict(test_ds).flatten()
    pred_labels[pred_labels <= 0] = 0
    pred_labels[pred_labels > 0] = 1

    # obtain the true labels for the test dataset
    true_labels = []
    for obj in test_ds:
        true_labels += list(np.array(obj[1]))
    true_labels = np.array(true_labels)

    # calculate the accuracy of the model for the test dataset
    accuracy = np.where(true_labels == pred_labels)[0].shape[0] / test_df.shape[0]
    print ('\n\n\nthe accuracy is: ', accuracy)

    # the weights of the trainable layers of the model
    weights = model.get_weights()[-2:]

    # useful model's data to save
    model_data = {
        'score': accuracy,
        'train_df': train_df['filename'].values,
        'validate_df': validate_df['filename'].values,
        'test_df': test_df['filename'].values,
        'weights': weights,
    }

    # save trained data
    with open('deep_learning_lasso/deep_learning_data/new_deeplearning_%d.json' % i, 'w') as f:
        f.write(json.dumps(model_data, cls=NpEncoder))
