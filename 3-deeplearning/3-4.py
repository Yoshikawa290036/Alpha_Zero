import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def conv(filters, kernel_size, lenstrides=1):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides=lenstrides, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001))


def first_residual_unit(filters, strides):
    def f(x):
        x = tf.keras.layers.BatchNormalization()(x)
        b = tf.keras.layers.Activation('relu')(x)

        x = conv(filters // 4, 1, strides)(b)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = conv(filters // 4, 3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.leyers.Activation('relu')(x)

        x = conv(filters, 1)(x)

        sc = conv(filters, 1, strides)(b)

        return tf.keras.layers.Add()([x, sc])
    return f


def main():
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.cifar10.load_data()

    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)
    # print(train_labels.shape)


if __name__ == '__main__':
    main()
