import os

import os
import numpy as np
import pandas as pd
from PIL import Image

# Define the size of the images
image_size = (92, 112)


# Load the dataset from the given path
def load_dataset(dataset_path):
    # Initialize the data matrix and label vector
    D = np.zeros((400, image_size[0] * image_size[1]))
    y = np.zeros(400)

    # Loop over all the subjects
    for i in range(1, 41):
        # Loop over all the images for the current subject
        for j in range(1, 11):
            # Load the image and convert it to a numpy array
            image_path = os.path.join(dataset_path, 's' + str(i), str(j) + '.pgm')
            with Image.open(image_path) as img:
                img = img.resize((92, 112))  # Resize the image
                img = np.asarray(img, int).flatten()  # Convert the image to a numpy array

                D[(i - 1) * 10 + j - 1] = img
            # Set the label for the current image
            y[(i - 1) * 10 + j - 1] = int(i)

    # Print the shapes of the data matrix and label vector
    assert D.shape == (400, 10304)
    print(pd.DataFrame(D))
    print('D shape:', D.shape)  # should print (400, 10304) if it is true
    assert y.shape == (400,)
    print('y shape:', y.shape)  # should print (400,) if it is true

    return D, y


# Split the dataset into training and testing sets
def split_dataset_even_odd(D, y):

    # Split the data into training and testing sets
    training, training_labels = D[::2], y[::2]
    testing, testing_labels = D[1::2], y[1::2]

    return training, training_labels, testing, testing_labels

def split_dataset_seven_three(D, y):
    # Split the array based on index % 10
    training, training_labels = D[np.mod(np.arange(len(D)), 10) < 7], y[np.mod(np.arange(len(y)), 10) < 7] # indices with (index % 10) < 7
    testing, testing_labels = D[np.mod(np.arange(len(D)), 10) >= 7],  y[np.mod(np.arange(len(y)), 10) >= 7] # indices with (index % 10) >= 7
    return training, training_labels, testing, testing_labels
