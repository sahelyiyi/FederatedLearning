import json
import datetime
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf

from deep_learning_lasso.deep_learning_utils import *
from deep_learning_lasso.deep_learning_utils import BATCH_SIZE, EPOCHS
from deep_learning_lasso.models import get_NN_model


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


data = []
for i in range(200):

    # get the model
    model = get_NN_model()

    # prepare train, validation, and test datasets
    ds, metadata = tfds.load(
        "cats_vs_dogs",
        shuffle_files=True,
        with_info=True,
    )

    train_ds, train_df, validation_ds, validate_df, test_ds, test_df = get_dataframe_and_datasets(ds)

    # train the mode for the selected train and validation datasets
    start = datetime.datetime.now()
    model.fit(train_ds, epochs=EPOCHS, validation_data=validation_ds)
    print(datetime.datetime.now() - start)

    # calculate the accuracy of the model for the test dataset
    pred_labels = model.predict(test_ds).flatten()
    pred_labels[pred_labels <= 0] = 0
    pred_labels[pred_labels > 0] = 1

    true_labels = []
    for obj in test_ds:
        true_labels += list(np.array(obj[1]))
    true_labels = np.array(true_labels)

    score = np.where(true_labels == pred_labels)[0].shape[0] / test_df.shape[0]
    print ('\n\n\nscore is: ', score)

    data.append({
        'score': score,
        'train_df': train_df['filename'].values,
        'validate_df': validate_df['filename'].values,
        'test_df': test_df['filename'].values,
        'weights': model.get_weights()[-2:],
    })

    # save trained data
    with open('deep_learning_data/new_deeplearning_%d.json' % i, 'w') as f:
        f.write(json.dumps(data, cls=NpEncoder))
