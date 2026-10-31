"""Keras model definitions for the ANN baseline and CNN."""

import tensorflow as tf
from tensorflow.keras import layers

from .data import IMAGE_SIZE


def build_cnn(num_classes: int = 3) -> tf.keras.Model:
    """Build a compact CNN suitable for the course dataset and CPU training."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    x = layers.RandomFlip("horizontal", name="random_flip")(x)
    x = layers.RandomRotation(0.06, name="random_rotation")(x)

    for filters in (16, 32, 64):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_scores")(x)
    model = tf.keras.Model(inputs, outputs, name="rps_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_ann(num_classes: int = 3) -> tf.keras.Model:
    """Build a small dense baseline for comparison with the CNN."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_scores")(x)
    model = tf.keras.Model(inputs, outputs, name="rps_ann_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
