import keras
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

from keras import layers

from deep_learning_lasso.deep_learning_utils import Image_Width, Image_Height, Image_Channels, Image_Size, BATCH_SIZE


def get_base_model_data():
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(Image_Width, Image_Height, Image_Channels),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(Image_Width, Image_Height, Image_Channels))
    x = data_augmentation(inputs)  # Apply random data augmentation

    # Pre-trained Xception weights requires that input be normalized
    # from (0, 255) to a range (-1., +1.), the normalization layer
    # does the following, outputs = (inputs - mean) / sqrt(var)
    norm_layer = keras.layers.experimental.preprocessing.Normalization()
    mean = np.array([127.5] * 3)
    var = mean ** 2
    # Scale inputs to [-1, +1]
    x = norm_layer(x)
    norm_layer.set_weights([mean, var])

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    return x, inputs


# get the base(pre-trained) model
def get_base_model():
    x, inputs = get_base_model_data()
    outputs = keras.layers.GlobalAveragePooling2D()(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    return model


# get the new(trainable) model
def get_new_model():
    inputs = keras.Input(shape=(2048,))

    x = keras.layers.Dropout(0.2)(inputs)
    outputs = keras.layers.Dense(1)(x)

    extra_model = keras.Model(inputs, outputs)

    extra_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    return extra_model


# get the model
def get_NN_model():
    # base model
    x, inputs = get_base_model_data()

    # new model
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    for layer in model.layers[:-1]:
        layer.trainable = False

    return model


# calculate base model output and true labels for all images
def get_base_model_output():

    # get the base model
    base_model = get_base_model()

    # load the data from tensorflow dataset (all the images)
    (dataset,), metadata = tfds.load(
        "cats_vs_dogs",
        split=["train[:100%]"],
        shuffle_files=True,
        with_info=True,
    )

    # resize the images of the dataset to the standard size
    dataset = dataset.map(lambda item: (tf.image.resize(item['image'], Image_Size), item['label']))
    dataset = dataset.cache().batch(BATCH_SIZE).prefetch(buffer_size=10)

    # get the output of the base model for the dataset
    base_model_outputs = base_model.predict(dataset)
    '''
    base_model_output: A list containing the output of the base(pre-trained) model for all the images
    '''

    # obtain the true labels for the dataset
    true_labels = []
    for obj in dataset:
        true_labels += list(np.array(obj[1]))
    true_labels = np.array(true_labels)
    '''
    true_labels: A list containing the true label of all the images (which is 0 or 1 for each image)
    '''

    return base_model_outputs, true_labels
