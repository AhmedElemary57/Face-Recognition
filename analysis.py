import os

import numpy as np
from PIL import Image

import lda

# Define the size of the images
image_size = (92, 112)


def recognize(img_path):
    # Initialize the data matrix and label vector
    D = np.zeros((400, image_size[0] * image_size[1]))
    y = np.zeros(400)

    # Loop over all the subjects
    for i in range(1, 41):
        # Loop over all the images for the current subject
        for j in range(1, 11):
            # Load the image and convert it to a numpy array
            image_path = os.path.join(img_path, 'archive', 's' + str(i),
                                      str(j) + '.pgm')  # Set the path to the image file
            with Image.open(image_path) as img:  # Open the image file using PIL library
                img = img.resize(image_size)  # Resize the image to the specified image size
                img = np.asarray(img, float).flatten()  # Convert the image to a numpy array and flatten it
                D[(i - 1) * 10 + j - 1] = img  # Add the flattened image array to the data matrix
            # Set the label for the current image
            y[(i - 1) * 10 + j - 1] = 0  # Set the label of the current image to 0 since it is a face image

    # Load the non-face images
    path = os.path.join(img_path, 'non-faces')  # Set the path to the non-face images folder
    i = 200  # Initialize the index variable to 200
    img_list = os.listdir(path)  # Get a list of all the image files in the non-face images folder
    for image_name in img_list:
        image_path = os.path.join(path, image_name)  # Set the path to the current non-face image
        with Image.open(image_path) as img:  # Open the image file using PIL library
            img = img.convert('L')  # Convert the image to grayscale
            img = img.resize(image_size)  # Resize the image to the specified image size
            img = np.asarray(img, float).flatten()  # Convert the image to a numpy array and flatten it
            D[i] = img  # Add the flattened image array to the data matrix
        # Set the label for the current image
        y[i] = 1  # Set the label of the current image to 1 since it is a non-face image
        i += 1  # Increment the index variable

    # Return the training and testing data and label vectors
    return D[::2], y[::2], D[1::2], y[1::2]


training, training_labels, testing, testing_labels = recognize('./data')

projection_matrix = lda.get_projection_matrix(training, training_labels, 39, 100, 1)

training_data_projected = lda.project_data(training, projection_matrix)
testing_data_projected = lda.project_data(testing, projection_matrix)

print(lda.calculate_accuracy(1, training_data_projected, training_labels, testing_data_projected, testing_labels))
lda.plot_lda_accuracy(training_data_projected, training_labels, testing_data_projected, testing_labels)
