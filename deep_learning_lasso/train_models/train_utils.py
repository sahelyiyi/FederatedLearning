import json

import tensorflow as tf
import numpy as np
import pandas as pd

from deep_learning_lasso.deep_learning_utils import Image_Size, BATCH_SIZE


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# convert pandas.dataset to pandas.dataframe
def convert_dataset_to_dataframe(dataset):
    file_names = []
    labels = []
    for obj in dataset:
        file_names.append(obj['image/filename'].numpy().decode('utf-8'))
        labels.append(obj['label'].numpy())
    df = pd.DataFrame(data={'filename': file_names, 'label': labels})
    return df


# prepare train/validation/test datasets for training the model
def split_dataset(dataset, train_ratio, val_ratio, test_ratio):
    # ds_size = len(list(ds))
    ds_size = 23262

    train_size = int(ds_size * train_ratio)
    validate_size = int(ds_size * val_ratio)
    test_size = int(ds_size * test_ratio)

    # split the dataset to train/validation/test datasets based on their ratios
    train_ds = dataset.take(train_size)
    validation_ds = dataset.skip(train_size).take(validate_size)
    test_ds = dataset.skip(train_size + validate_size).take(test_size)

    # convert train/validation/test pandas.dataset to pandas.dataframe in order to save them
    train_df = convert_dataset_to_dataframe(train_ds)
    validate_df = convert_dataset_to_dataframe(validation_ds)
    test_df = convert_dataset_to_dataframe(test_ds)

    # resize the images of train/validation/test dataset to the standard size
    train_ds = train_ds.map(lambda item: (tf.image.resize(item['image'], Image_Size), item['label']))
    validation_ds = validation_ds.map(lambda item: (tf.image.resize(item['image'], Image_Size), item['label']))
    test_ds = test_ds.map(lambda item: (tf.image.resize(item['image'], Image_Size), item['label']))

    # prepare train/validation/test dataset for training the model
    train_ds = train_ds.cache().batch(BATCH_SIZE).prefetch(buffer_size=10)
    validation_ds = validation_ds.cache().batch(BATCH_SIZE).prefetch(buffer_size=10)
    test_ds = test_ds.cache().batch(BATCH_SIZE).prefetch(buffer_size=10)

    return (train_ds, train_df), (validation_ds, validate_df), (test_ds, test_df)
