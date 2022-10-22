import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def conv(filters, kernel_size, lenstrides=1):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides=lenstrides, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001))


def main():
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.cifar10.load_data()

    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)
    # print(train_labels.shape)


if __name__ == '__main__':
    main()
