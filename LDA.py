import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import analysis


def means_vectors(data, labels):
    """
    Calculate the mean vector for each class.

    Parameters:
        data: numpy ndarray, array of training samples.
        labels: numpy ndarray, array of labels of the training samples.

    Returns:
    means_df: pandas dataframe, dataframe containing the mean vector for each class.
    """
    grouped_df = pd.DataFrame(data).groupby(labels)
    means_df = grouped_df.mean()
    return means_df


def between_class_scatter_matrix(means, overall_mean, n):
    """
    Computes the between-class scatter matrix.

    Parameters:
        means: numpy ndarray, 2-dimensional array containing the mean of each class.
        overall_mean: numpy ndarray, The overall mean of all the samples.
        n: numpy ndarray, 1-dimensional array containing the number of samples in each class.

    Returns:
    sb: numpy ndarray, The between-class scatter matrix.
    """

    sb = np.zeros((overall_mean.shape[0], overall_mean.shape[0]))
    for i in range(means.shape[0]):
        mean_diff = means[i] - overall_mean.T
        sb += n[i] * np.outer(mean_diff, mean_diff.T)
    return sb


def centralize(data, means, labels):
    """
    Subtract the class mean from each training sample.

    Parameters:
        data: numpy ndarray, array of training samples.
        means: numpy ndarray, 2-dimensional array containing the mean of each class.
        labels: numpy ndarray, array of labels of the training samples.

    Returns:
    z: numpy ndarray, array of centralized training samples.
    """
    z = np.zeros(data.shape)
    for i in range(data.shape[0]):
        z[i, :] = data[i, :] - means[int(labels[i]) - 1, :]
    return z


def within_class_scatter_matrix(z):
    """
    Computes the within-class scatter matrix.

    Parameters:
        z: numpy ndarray, array of centralized training samples.

    Returns:
        s: numpy ndarray, the within-class scatter matrix.
    """
    s = np.dot(z.T, z)
    return s


def get_eigens(data):
    """
    Computes the eigenvalues and eigenvectors of the input matrix.

    Parameters:
        data: numpy ndarray, the input matrix.

    Returns:
        sorted_eigenvalues: numpy ndarray, sorted eigenvalues in descending order.
        sorted_eigenvectors: numpy ndarray, eigenvectors sorted using the same indices as the eigenvalues.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(data)
    # Sort the eigenvalues in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sort_indices]
    # Sort the eigenvectors using the same indices
    sorted_eigenvectors = eigenvectors[:, sort_indices]
    return sorted_eigenvalues, sorted_eigenvectors


def get_projection_matrix(training_data, training_data_labels, dimensions_needed, number_of_group_samples, load):
    """
    Computes the projection matrix needed for linear discriminant analysis using the given training data.

    Parameters:
        training_data: array-like object containing the training data.
        training_data_labels: array-like object containing the labels of the training data.
        number_of_group_samples: number of samples in each class
        dimensions_needed: the number of dimensions needed for the projection matrix.

    Returns:
        eigenvectors: array-like object containing the projection matrix.

    """
    if load == 1:
        return pd.read_csv('projection_matrix.csv').values
    # Compute the mean vector for each class in the training data
    means = means_vectors(pd.DataFrame(training_data), training_data_labels).values
    print("Means:", pd.DataFrame(means))

    # Compute the overall sample mean of the training data
    overall_mean = pd.DataFrame(training_data).mean().values
    print("Overall Mean:", pd.DataFrame(overall_mean))

    # Compute the number of samples in each class (which is assumed to be equal) and equal 5 sample for each class.
    if isinstance(number_of_group_samples, (list, tuple)):
        # If x is a list or tuple
        n = number_of_group_samples
    else:
        # If x is a number
        n = np.zeros(means.shape[0]) + number_of_group_samples

    print("Number of items in each group:", pd.DataFrame(n))

    # Compute between-class scatter matrix
    sb = between_class_scatter_matrix(means, overall_mean, n)
    print("Between-class scatter matrix:", pd.DataFrame(sb))

    # Centralize the data
    z = centralize(training_data, means, training_data_labels)
    print("z:", pd.DataFrame(z))

    # Compute the within-class scatter matrix
    s = within_class_scatter_matrix(z)
    s_inverse = np.linalg.inv(s)
    print("s matrix:", pd.DataFrame(s))
    print("s_inverse:", pd.DataFrame(s_inverse))

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = get_eigens(np.matmul(s_inverse, sb))
    print("Eigenvalues:", pd.DataFrame(eigenvalues))
    print("Eigenvectors:", eigenvectors)

    # Select the top k eigenvectors and return the projection matrix
    projected_data = eigenvectors[:, :dimensions_needed]
    pd.DataFrame(projected_data).to_csv('projection_matrix.csv', index=False)
    return projected_data


def project_data(data, projection_matrix):
    """
    Projects the input data onto a lower-dimensional space using a projection matrix

    Parameters:
    data (numpy.ndarray): Input data to be projected.
    projection_matrix (numpy.ndarray): The projection matrix calculated from training.

    Returns:
    numpy.ndarray: Projected data.

    """
    new_data = np.dot(data, projection_matrix)
    return new_data


def calculate_accuracy(number_of_neighbors, training_data_projected, training_data_labels, testing_data_projected,
                       testing_data_labels, test_data):
    """
    Calculates the accuracy of KNN classification on the projected data.

    Parameters:
        number_of_neighbors (int): Number of nearest neighbors to consider for classification.
        training_data_projected (numpy.ndarray): Training data projected to be used for KNN classification.
        training_data_labels (numpy.ndarray): Labels for the training data.
        testing_data_projected (numpy.ndarray): Testing data projected to be used for KNN classification.
        testing_data_labels (numpy.ndarray): Labels for the testing data.
    Returns:
        float: Accuracy of KNN classification.

    """
    knn = KNeighborsClassifier(number_of_neighbors)
    # Train the model using the training set
    knn.fit(training_data_projected, training_data_labels)
    # Predict the response for test dataset
    y_predict = knn.predict(testing_data_projected)
    accuracy = metrics.accuracy_score(testing_data_labels, y_predict)

    false_recognitions = analysis.get_false_recognitions(testing_data_labels, y_predict)
    incorrect_indices = [i for i in range(len(testing_data_labels)) if testing_data_labels[i] != y_predict[i]]

    #analysis.samples_of_failed_classification(test_data, testing_data_labels, y_predict, incorrect_indices)
    for false_recognition in false_recognitions:
        print(false_recognition)

    return accuracy


def plot_lda_accuracy(training_data_projected, training_data_labels, testing_data_projected, testing_data_labels, test_data):
    """
    plot the accuracy of lda testing data.

    Parameters:
        training_data_projected (numpy.ndarray): Training data projected to be used for KNN classification.
        training_data_labels (numpy.ndarray): Labels for the training data.
        testing_data_projected (numpy.ndarray): Testing data projected to be used for KNN classification.
        testing_data_labels (numpy.ndarray): Labels for the testing data.
    Returns:
        float: Accuracy of KNN classification.
        """
    k_values = [1, 3, 5, 7]
    accuracies = []
    for k in k_values:
        accuracies.append(calculate_accuracy(k, training_data_projected, training_data_labels, testing_data_projected,
                                             testing_data_labels, test_data))
    plt.plot(k_values, accuracies)
    plt.xlabel('k_values')
    plt.ylabel('accuracy')
    plt.title('LDA accuracy vs KNN')
    plt.show()
