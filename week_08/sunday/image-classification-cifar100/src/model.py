"""Keras model definitions for CIFAR-100."""

import tensorflow as tf
from tensorflow.keras import layers

from .data import IMAGE_SIZE


def build_cnn(num_classes: int = 100) -> tf.keras.Model:
    """Build a compact CNN for 32 × 32 CIFAR-100 images."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    x = layers.RandomFlip("horizontal", name="random_flip")(x)
    x = layers.RandomRotation(0.06, name="random_rotation")(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_scores")(x)
    model = tf.keras.Model(inputs, outputs, name="cifar100_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenetv2(num_classes: int = 100, weights="imagenet") -> tf.keras.Model:
    """Build a CIFAR-100 classifier using ImageNet-pretrained MobileNetV2."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(*IMAGE_SIZE, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_scores")(x)
    model = tf.keras.Model(inputs, outputs, name="cifar100_mobilenetv2_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
