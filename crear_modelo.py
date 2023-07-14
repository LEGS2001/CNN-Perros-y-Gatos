from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import csv
from config import SIZE, NUM_TRAINING_IMAGES, NUM_TESTING_IMAGES, EPOCHS

train_images = []
train_labels = []
test_images = []
test_labels = []

with open('training.csv', "r") as file:
  csv_reader = csv.reader(file)
  for row in csv_reader:
    label = int(row[0])
    image = np.array([int(val) for val in row[1:]], dtype=np.uint8)  # Convert image values to int
    train_images.append(image)
    train_labels.append(label)

with open('testing.csv', "r") as file:
  csv_reader = csv.reader(file)
  for row in csv_reader:
    label = int(row[0])
    image = np.array([int(val) for val in row[1:]], dtype=np.uint8)  # Convert image values to int
    test_images.append(image)
    test_labels.append(label)
    
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

inputs = keras.Input(shape=(SIZE, SIZE, 1))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

train_images = train_images.reshape((-1, SIZE, SIZE, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((-1, SIZE, SIZE, 1))
test_images = test_images.astype("float32") / 255
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.3f}")

model.save('modelo.keras')