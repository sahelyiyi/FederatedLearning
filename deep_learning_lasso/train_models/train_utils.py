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


def convert_dataset_to_dataframe(dataset):
    file_names = []
    labels = []
    for obj in dataset:
        file_names.append(obj['image/filename'].numpy().decode('utf-8'))
        labels.append(obj['label'].numpy())
    df = pd.DataFrame(data={'filename': file_names, 'label': labels})
    return df


def get_dataframe_and_datasets(ds):
    ds = ds['train']
    # ds_size = len(list(ds))
    ds_size = 23262
    train_size = int(ds_size * 0.75)
    validate_size = int(ds_size * 0.15)
    test_size = int(ds_size * 0.1)

    train_ds = ds.take(train_size)
    validation_ds = ds.skip(train_size).take(validate_size)
    test_ds = ds.skip(train_size + validate_size).take(test_size)

    train_df = convert_dataset_to_dataframe(train_ds)
    validate_df = convert_dataset_to_dataframe(validation_ds)
    test_df = convert_dataset_to_dataframe(test_ds)

    train_ds = train_ds.map(lambda item: (tf.image.resize(item['image'], Image_Size), item['label']))
    validation_ds = validation_ds.map(lambda item: (tf.image.resize(item['image'], Image_Size), item['label']))
    test_ds = test_ds.map(lambda item: (tf.image.resize(item['image'], Image_Size), item['label']))

    train_ds = train_ds.cache().batch(BATCH_SIZE).prefetch(buffer_size=10)
    validation_ds = validation_ds.cache().batch(BATCH_SIZE).prefetch(buffer_size=10)
    test_ds = test_ds.cache().batch(BATCH_SIZE).prefetch(buffer_size=10)

    return train_ds, train_df, validation_ds, validate_df, test_ds, test_df
