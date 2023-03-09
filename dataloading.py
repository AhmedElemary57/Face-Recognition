import os

import numpy as np
from PIL import Image

# Define the path to the dataset folder
# this is the path to the dataset in colab files.
dataset_path = 'C:\\archive'

# Define the size of the images
image_size = (112, 92)


def load_data():
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
                img = img.resize(image_size)
                img = np.asarray(img, float).flatten()
                D[(i - 1) * 10 + j - 1] = img
            # Set the label for the current image
            y[(i - 1) * 10 + j - 1] = int(i)

    # Print the shapes of the data matrix and label vector
    assert D.shape == (400, 10304)
    print('D shape:', D.shape)  # should print (400, 10304) if it is true
    assert y.shape == (400,)
    print('y shape:', y.shape)  # should print (400,) if it is true

    # Split training data from the testing data
    return D[::2], y[::2], D[1::2], y[1::2]
