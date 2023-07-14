import csv
import os
from PIL import Image
from config import SIZE, NUM_TRAINING_IMAGES, NUM_TESTING_IMAGES

cats = os.listdir(f'PetImages/Cat')
dogs = os.listdir(f'PetImages/Dog')

with open(f'training.csv', "w", newline="") as csvfile:
    writer = csv.writer(csvfile)             
    for i in range(NUM_TRAINING_IMAGES // 2):    

        # Agrega un gato al training.csv
        image = Image.open(f'PetImages/Cat/{cats[i]}')
        image = image.resize((SIZE, SIZE))
        image = image.convert("L")
        # Get the width and height of the image
        width, height = image.size

        row = []
        for x in range(width):
            for y in range(height):
                row.append(image.getpixel((x, y)))
        row.insert(0, '0')
        writer.writerow(row)

        # Agrega un perro al training.csv
        image = Image.open(f'PetImages/Dog/{dogs[i]}')
        image = image.resize((SIZE, SIZE))
        image = image.convert("L")
        # Get the width and height of the image
        width, height = image.size

        row = []
        for x in range(width):
            for y in range(height):
                row.append(image.getpixel((x, y)))
        row.insert(0, '1')
        writer.writerow(row)

        print(f'Generando training set... [{((i/(NUM_TRAINING_IMAGES//2)) * 100):.2f}]%')

with open(f'testing.csv', "w", newline="") as csvfile:
    writer = csv.writer(csvfile)             
    for i in range(NUM_TRAINING_IMAGES // 2, NUM_TRAINING_IMAGES // 2 + NUM_TESTING_IMAGES // 2):    

        # Agrega un gato al training.csv
        image = Image.open(f'PetImages/Cat/{cats[i]}')
        image = image.resize((SIZE, SIZE))
        image = image.convert("L")
        # Get the width and height of the image
        width, height = image.size

        row = []
        for x in range(width):
            for y in range(height):
                row.append(image.getpixel((x, y)))
        row.insert(0, '0')
        writer.writerow(row)

        # Agrega un perro al training.csv
        image = Image.open(f'PetImages/Dog/{dogs[i]}')
        image = image.resize((SIZE, SIZE))
        image = image.convert("L")
        # Get the width and height of the image
        width, height = image.size

        row = []
        for x in range(width):
            for y in range(height):
                row.append(image.getpixel((x, y)))
        row.insert(0, '1')
        writer.writerow(row)

        print(f'Generando testing set... [{(((i - NUM_TRAINING_IMAGES // 2)/(NUM_TESTING_IMAGES // 2)) * 100):.2f}]%')