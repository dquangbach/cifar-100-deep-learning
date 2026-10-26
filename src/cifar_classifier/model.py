"""The exact WRN-16-4 architecture from build_wrn_cifar_100.ipynb."""

import tensorflow as tf

from .config import IMAGE_SIZE, NUM_CLASSES


def build_wrn_16_4() -> tf.keras.Model:
    layers = tf.keras.layers

    def block(inputs: tf.Tensor, channels: int, stride: int) -> tf.Tensor:
        x = layers.BatchNormalization()(inputs)
        x = layers.Activation("relu")(x)
        shortcut = inputs
        if stride != 1 or inputs.shape[-1] != channels:
            shortcut = layers.Conv2D(
                channels, 1, strides=stride, padding="same", use_bias=False
            )(inputs)
        x = layers.Conv2D(
            channels, 3, strides=stride, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(channels, 3, padding="same", use_bias=False)(x)
        return layers.Add()([shortcut, x])

    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    x = layers.Conv2D(16, 3, padding="same", use_bias=False)(inputs)
    for channels, first_stride in ((64, 1), (128, 2), (256, 2)):
        x = block(x, channels, first_stride)
        x = block(x, channels, 1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="wrn_16_4_cifar100")
