import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import random
import csv
from config import SIZE

class_names = ['Cat', 'Dog']
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(2))
  plt.yticks([])
  thisplot = plt.bar(range(2), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

model = keras.models.load_model('modelo.keras')

test_images = []
test_labels = []
with open('testing.csv', "r") as file:
  csv_reader = csv.reader(file)
  for row in csv_reader:
    label = int(row[0])
    image = np.array([int(val) for val in row[1:]], dtype=np.uint8)  # Convert image values to int
    test_images.append(image)
    test_labels.append(label)

test_images = np.array(test_images)
test_images = test_images.reshape((-1, SIZE, SIZE, 1))
test_labels = np.array(test_labels)

predictions = model.predict(test_images)


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
indice_inicial = random.randint(0, len(predictions)-20 )
indice_final = indice_inicial + num_images

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(indice_inicial, indice_final):
    indice = i - indice_inicial
    plt.subplot(num_rows, 2*num_cols, 2*indice+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*indice+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()