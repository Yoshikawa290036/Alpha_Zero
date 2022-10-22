from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape, train_labels.shape,
      test_images.shape, test_labels.shape)


# for i in range(10):
#     plt.subplot(1,10,i+1)
#     plt.imshow(train_images[i], 'gray')
# plt.show()
# print(train_images[0])

# train_images, test_images = train_images/255.0, test_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # tf.keras.layers.Dense(input_shape=(28,28), activation='sigmoid'),
    tf.keras.layers.Dense(784, activation='sigmoid'),
    tf.keras.layers.Dense(256, activation='sigmoid'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10)
])

predictions = model(train_images[:1]).numpy()
print(predictions)

tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['acc'])

history = model.fit(train_images, train_labels,
                    batch_size=2700, epochs=50, validation_split=0.1)

plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss, test_acc)
