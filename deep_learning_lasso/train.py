import json
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf


from deep_learning_lasso.deep_learning_utils import *


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


def train_model(model, train_generator, total_train, validation_generator, total_validate, batch_size, callbacks):
    # epochs = 10
    epochs = 2

    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate // batch_size,
        steps_per_epoch=total_train // batch_size,
        callbacks=callbacks
    )


def get_df_from_ds(dataset):
    file_names = []
    labels = []
    for obj in dataset:
        file_names.append(obj['image/filename'].numpy().decode('utf-8'))
        labels.append(obj['label'].numpy())
    df = pd.DataFrame(data={'filename': file_names, 'label': labels})
    return df


scores_data = []
for i in range(10):
    model = get_NN_model()
    ds, metadata = tfds.load(
        "cats_vs_dogs",
        shuffle_files = True,
        with_info = True,
    )

    ds = ds['train']
    # ds_size = len(list(ds))
    ds_size = 23262
    train_size = int(ds_size*0.75)
    validate_size = int(ds_size*0.15)
    test_size = int(ds_size*0.1)

    train_ds = ds.take(train_size)
    validation_ds = ds.skip(train_size).take(validate_size)
    test_ds = ds.skip(train_size+validate_size).take(test_size)

    train_df = get_df_from_ds(train_ds)
    validate_df = get_df_from_ds(validation_ds)
    test_df = get_df_from_ds(test_ds)

    size = (Image_Width, Image_Height)

    train_ds = train_ds.map(lambda item: (tf.image.resize(item['image'], size), item['label']))
    validation_ds = validation_ds.map(lambda item: (tf.image.resize(item['image'], size), item['label']))
    test_ds = test_ds.map(lambda item: (tf.image.resize(item['image'], size), item['label']))

    batch_size = 32

    train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

    start = datetime.datetime.now()
    epochs = 3
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
    print(datetime.datetime.now() - start)
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    # model.fit(train_ds, validation_data=validation_ds, callbacks=[callback])

    pred_labels = model.predict(test_ds).flatten()
    pred_labels[pred_labels<=0] = 0
    pred_labels[pred_labels>0] = 1

    true_labels = []
    for obj in test_ds:
        true_labels += list(np.array(obj[1]))
    true_labels = np.array(true_labels)

    score = np.where(true_labels == pred_labels)[0].shape[0] / test_df.shape[0]
    print ('\n\n\nscore is: ', score)

    scores_data.append({
        'score': score,
        'train_df': train_df['filename'].values,
        'validate_df': validate_df['filename'].values,
        'test_df': test_df['filename'].values,
        'weights': model.get_weights()[-2:],
    })

with open('new_deeplearning.json', 'w') as f:
    f.write(json.dumps(scores_data, cls=NpEncoder))
