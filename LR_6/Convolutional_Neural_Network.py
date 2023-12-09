import tensorflow as tf
from keras.src.layers import Dense
from tensorflow.keras import datasets, layers, models
import time
import numpy as np

start_time = time.time()

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    epochs=5,
                    validation_data=(test_images, test_labels))

model.save("./models/cnn_model_2.keras")

print('==============================================================')

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Потери на тестовых данных:', test_loss)
print('Точность модели на тестовых данных:', test_acc)

accuracy = test_acc
percent_correct = accuracy * 100
print('Процент корректной работы:', percent_correct)

end_time = time.time()
print('Затраченное время:', end_time - start_time)
print('==============================================================')
