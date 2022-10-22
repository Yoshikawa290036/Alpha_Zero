from gc import callbacks
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data,
                             test_labels) = tf.keras.datasets.boston_housing.load_data()

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                'AGE', 'DIS', 'RAD', 'TAX', 'PARATIO', 'B', 'LSTAT']
df = pd.DataFrame(train_data, columns=column_names)
print(df.tail())

order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean)/std
test_data = (test_data - mean)/std

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(13,)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['mae'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=500,
                    validation_split=0.2, callbacks=[early_stop])

# print(history)

plt.plot(history.history['mae'], label="train mae")
plt.plot(history.history['val_mae'], label='val mae')
plt.xlabel('epochs')
plt.ylabel('mae @1000$')
plt.legend(loc='best')
plt.ylim([0, 5])
plt.show()


test_loss, test_mae = model.evaluate(test_data, test_labels)
print('loss : {:.4f}      \nmae  : {:.4f}'.format(test_loss, test_mae))
