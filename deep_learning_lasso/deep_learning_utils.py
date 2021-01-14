import keras
import numpy as np

from keras import layers
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

Image_Width = 150
Image_Height = 150
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3


def get_new_model(model):
    model.layers.pop()
    model.outputs = [model.layers[-2].output]

    x = Dense(2, activation='softmax')(model.layers[-2].output)

    model = Model(model.input, x)

    for layer in model.layers[:-1]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    return model


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


def get_extra_layers():
    inputs = keras.Input(shape=(2048,))
    x = keras.layers.Dropout(0.2)(inputs)  # Regularize with dropout
    outputs = keras.layers.Dense(1)(x)
    extra_model = keras.Model(inputs, outputs)
    extra_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    return extra_model


def get_NN_model():
    x, inputs = get_base_model_data()
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
