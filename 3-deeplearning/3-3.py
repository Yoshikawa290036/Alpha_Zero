from gc import callbacks
from multiprocessing import pool
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(train_images[i])
# plt.show()

train_images = train_images.astype('float32')/255.0
test_images = test_images.astype('float32')/255.0

# print(train_labels[:10])

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)
print(train_labels.shape)

model = tf.keras.models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',
          padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['acc'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)


history = model.fit(train_images, train_labels,
                    batch_size=352, epochs=500, validation_split=0.1, callbacks=[early_stop])

model.save('convolution.h5')

plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.ylim([0.5, 1])
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('loss : {:.4f}      \nacc  : {:.4f}'.format(test_loss, test_acc))
