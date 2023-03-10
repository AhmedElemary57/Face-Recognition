import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os
from PIL import Image
import cv2
import LDA as lda
dataset_path = './data'

# Define the size of the images
image_size = (92, 112)


def load_faces_non_faces():
    # Initialize the data matrix and label vector
    D = np.zeros((400, image_size[0] * image_size[1]))
    y = np.zeros(400)

    # Loop over all the subjects
    for i in range(1, 41):
        # Loop over all the images for the current subject
        for j in range(1, 11):
            # Load the image and convert it to a numpy array
            image_path = os.path.join(dataset_path, 'archive', 's' + str(i), str(j) + '.pgm')
            with Image.open(image_path) as img:
                img = img.resize(image_size)
                img = np.asarray(img, float).flatten()
                D[(i - 1) * 10 + j - 1] = img
            # Set the label for the current image
            y[(i - 1) * 10 + j - 1] = 0

    path = os.path.join(dataset_path, 'non-faces')
    i = 200
    img_list = os.listdir(path)
    for image_name in img_list:
        image_path = os.path.join(path, image_name)
        with Image.open(image_path) as img:
            img = img.convert('L')
            img = img.resize(image_size)
            img = np.asarray(img, float).flatten()
            D[i] = img
        # Set the label for the current image
        y[i] = 1
        i += 1


    return D[::2], y[::2], D[1::2], y[1::2]


training, training_labels, testing, testing_labels = load_faces_non_faces()


projection_matrix = lda.get_projection_matrix(training,training_labels, 39, 100, 0)

training_data_projected = lda.project_data(training, projection_matrix)
testing_data_projected = lda.project_data(testing, projection_matrix)

print(lda.calculate_accuracy(1, training_data_projected,  training_labels, testing_data_projected, testing_labels))
lda.plot_lda_accuracy(training_data_projected,  training_labels, testing_data_projected, testing_labels)







