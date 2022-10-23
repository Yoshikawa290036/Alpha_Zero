from gc import callbacks
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
        x = tf.keras.layers.Activation('relu')(x)

        x = conv(filters, 1)(x)

        sc = conv(filters, 1, strides)(b)

        return tf.keras.layers.Add()([x, sc])
    return f


def residual_unit(filters):
    def f(x):
        sc = x
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = conv(filters//4, 1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = conv(filters//4, 3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = conv(filters, 1)(x)

        return tf.keras.layers.Add()([x, sc])
    return f


def residual_block(filters, strides, unit_size):
    def f(x):
        x = first_residual_unit(filters, strides)(x)
        for i in range(unit_size-1):
            x = residual_unit(filters)(x)
        return x
    return f


def step_decay(epoch):
    x = 0.1
    if epoch >= 80:
        x = 0.01
    if epoch >= 120:
        x = 0.001
    return x


def main():
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.cifar10.load_data()

    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)
    # print(train_labels.shape)

    input = tf.keras.layers.Input(shape=(32, 32, 3))
    x = conv(16, 3)(input)
    x = residual_block(64, 1, 18)(x)
    x = residual_block(128, 2, 18)(x)
    x = residual_block(256, 2, 18)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    output = tf.keras.layers.Dense(
        10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)

    model = tf.keras.models.Model(inputs=input, outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(momentum=0.9), metrics=['acc'])

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True
    )
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )

    for data in (train_gen, test_gen):
        data.fit(train_images)

    lr_decay = tf.keras.callbacks.LearningRateScheduler(step_decay)

    batch_size = 128
    history = model.fit_generator(
        train_gen.flow(train_images, train_labels, batch_size=batch_size),
        epochs=200,
        steps_per_epoch=train_images.shape[0]//batch_size,
        validation_data=test_gen.flow(
            test_images, test_labels, batch_size=batch_size),
        validation_steps=test_images.shape[0]//batch_size,
        callbacks=[lr_decay]
    )

    model.save('3-4.h5')

    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.ylim([0.5, 1])
    plt.show()

    batch_size = 128
    test_loss, test_acc = model.evaluate_generator(
        test_gen.flow(test_images, test_labels, batch_size=batch_size),
        steps=10
    )
    print('loss : {:.4f}      \nacc  : {:.4f}'.format(test_loss, test_acc))


if __name__ == '__main__':
    main()
