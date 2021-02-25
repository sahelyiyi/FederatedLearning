import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

from deep_learning_lasso.models import get_base_model

Image_Width = 150
Image_Height = 150
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3

BATCH_SIZE = 32
EPOCHS = 3


def get_base_model_output():
    base_model = get_base_model()

    (train_ds,), metadata = tfds.load(
        "cats_vs_dogs",
        split=["train[:100%]"],
        shuffle_files=True,
        with_info=True,
    )

    train_ds = train_ds.map(lambda item: (tf.image.resize(item['image'], Image_Size), item['label']))

    train_ds = train_ds.cache().batch(BATCH_SIZE).prefetch(buffer_size=10)

    base_model_outputs = base_model.predict(train_ds)

    true_labels = []
    for obj in train_ds:
        true_labels += list(np.array(obj[1]))
    true_labels = np.array(true_labels)

    return base_model_outputs, true_labels
