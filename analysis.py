import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

import pca
import LDA as lda


# Define the size of the images
image_size = (92, 112)


def _load(img_path, number_of_non_faces, number_of_images_loaded):
    """

    Args:
        img_path: path of images to run face recognition on

    Returns:
        tuple of (
        training: data to be trained,
        training_labels: labels (classes) of the training data,
        testing: data to be tested on the trained data,
        testing_labels: labels (classes) of the testing data
        )
        :param number_of_non_faces:

    """
    # Initialize the data matrix and label vector
    D = np.zeros((number_of_images_loaded, image_size[0] * image_size[1]))
    y = np.zeros(number_of_images_loaded)

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
    img_list = img_list[:number_of_non_faces]
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


def recognize(img_path, number_of_non_faces, number_of_group_samples, method=0):
    """

    Args:
        img_path: the path of image to run face recognition on
        method: boolean (0|1), 0 means PCA analysis, 1 means LDA analysis.
        number_of_non_faces: the number of non faces images loaded to train
    Returns:
        None

    """
    training, training_labels, testing, testing_labels = _load(img_path, number_of_non_faces, 400 + number_of_non_faces)
    print('Training Data\n', pd.DataFrame(training))
    print('Testing Data\n', pd.DataFrame(testing))

    if method:  # PCA
        pca.apply_pca(training, training=1)
        pca.apply_pca(testing, training=0)
        print('Accuracy = ' + str(pca.accuracy(training, training_labels, testing, testing_labels, 1) * 100) + '%')
        pca.plot_accuracy(training, testing, training_labels, testing_labels)

    if not method:  # LDA
        projection_matrix = lda.get_projection_matrix(training, training_labels, 1, number_of_group_samples, 1)

        training_data_projected = lda.project_data(training, projection_matrix)
        testing_data_projected = lda.project_data(testing, projection_matrix)

        print('Accuracy = ' + str(
            lda.calculate_accuracy(1, training_data_projected, training_labels, testing_data_projected,
                                   testing_labels, testing) * 100) + '%')
        lda.plot_lda_accuracy(training_data_projected, training_labels, testing_data_projected, testing_labels, testing)


def get_false_recognitions(tested_labels, predicted_labels):
    """
    Args:
        tested_labels: np array of tested data labels
        predicted_labels: np array of knn predicted data labels

    Returns:
        np array of tuples indicating the false recognitions spotted in tested_labels.
    """
    false_recognitions = []
    for i in range(len(tested_labels)):
        # Check if the identification is not the same as the prediction
        if tested_labels[i] != predicted_labels[i]:
            # Check if the prediction is in the range of faces.
            if np.isin(predicted_labels[i], tested_labels):
                false_recognitions.append(
                    'Photo No. ' + str(i + 1) + ' --> Recognition: a face of someone, prediction: a '
                                                'face of another one.')
            else:
                false_recognitions.append(
                    'Photo No. ' + str(i + 1) + ' --> Recognition: a face of someone, prediction: '
                                                'not a face')

    return false_recognitions

def samples_of_failed_classification(test_data, test_labels, predicted_labels, failure_cases):
    # Select a subset of the failure cases to display
    num_failure_cases_to_display =10
    failure_case_indices_to_display = random.sample(failure_cases, num_failure_cases_to_display)

    # Display the failure cases
    fig, axs = plt.subplots(5, 5, figsize=(12, 12))
    axs = axs.flatten()
    for i, index in enumerate(failure_case_indices_to_display):
        img = np.array(test_data[index], dtype='int')
        true_label = test_labels[index]
        predicted_label = predicted_labels[index]
        axs[i].imshow(img.reshape(112, 92), cmap='gray')
        axs[i].set_title(f'True label: {true_label}, Predicted label: {predicted_label}',color='red')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def accuracy_vs_number_of_non_faces():
    folder_path = "./data"

    num_of_non_faces= [50,100,150,200]
    accuracies =[]
    for i in num_of_non_faces:
        training, training_labels, testing, testing_labels = _load(folder_path, i,
                                                                   400 + i)
        projection_matrix= lda.get_projection_matrix(training,training_labels,1,[200,i],0)
        testing_projected= lda.project_data(testing,projection_matrix)
        training_projected= lda.project_data(training,projection_matrix)
        accuracy = lda.calculate_accuracy(1,training_projected,training_labels,testing_projected,testing_labels,testing)
        accuracies.append(accuracy)


    plt.plot(num_of_non_faces, accuracies)
    plt.xlabel('num_of_non_faces')
    plt.ylabel('accuracy')
    plt.show()

"""folder_path = "./data"
recognize(folder_path,200,[200,200],0)"""