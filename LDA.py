import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

"""
Calculate the mean vector for each class.

Args:
data: numpy ndarray, array of training samples.
labels: numpy ndarray, array of labels of the training samples.

Returns:
means_df: pandas dataframe, dataframe containing the mean vector for each class.
"""


def means_vectors(data, labels):
    grouped_df = pd.DataFrame(data).groupby(labels)
    means_df = grouped_df.mean()
    return means_df


"""
Computes the between-class scatter matrix.

Args:
means: numpy ndarray, 2-dimensional array containing the mean of each class.
overall_mean: numpy ndarray, The overall mean of all the samples.
n: numpy ndarray, 1-dimensional array containing the number of samples in each class.

Returns:
SB: numpy ndarray, The between-class scatter matrix.
"""


def between_class_scatter_matrix(means, overall_mean, n):
    SB = np.zeros((overall_mean.shape[0], overall_mean.shape[0]))
    for i in range(means.shape[0]):
        mean_diff = means[i] - overall_mean.T
        SB += n[i] * np.outer(mean_diff, mean_diff.T)
    return SB


"""
Subtract the class mean from each training sample.

Args:
data: numpy ndarray, array of training samples.
means: numpy ndarray, 2-dimensional array containing the mean of each class.
labels: numpy ndarray, array of labels of the training samples.

Returns:
Z: numpy ndarray, array of centralized training samples.
"""


def centralize(data, means, labels):
    Z = np.zeros(data.shape)
    for i in range(data.shape[0]):
        Z[i, :] = data[i, :] - means[int(labels[i]) - 1, :]
    return Z


"""
Computes the within-class scatter matrix.

Args:
Z: numpy ndarray, array of centralized training samples.

Returns:
S: numpy ndarray, the within-class scatter matrix.
"""


def within_class_scatter_matrix(Z):
    S = np.dot(Z.T, Z)
    return S


"""
Computes the eigenvalues and eigenvectors of the input matrix.

Args:
data: numpy ndarray, the input matrix.

Returns:
sorted_eigenvalues: numpy ndarray, sorted eigenvalues in descending order.
sorted_eigenvectors: numpy ndarray, eigenvectors sorted using the same indices as the eigenvalues.
"""


def get_eigens(data):
    eigenvalues, eigenvectors = np.linalg.eigh(data)
    # Sort the eigenvalues in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sort_indices]
    # Sort the eigenvectors using the same indices
    sorted_eigenvectors = eigenvectors[:, sort_indices]
    return sorted_eigenvalues, sorted_eigenvectors


"""
Computes the projection matrix needed for linear discriminant analysis using the given training data.

Parameters:
    - training_data: array-like object containing the training data.
    - training_data_labels: array-like object containing the labels of the training data.
    - dimensions_needed: the number of dimensions needed for the projection matrix.

Returns:
    - eigenvectors: array-like object containing the projection matrix.
"""


def get_projection_matrix(training_data, training_data_labels, dimensions_needed):
    # Compute the mean vector for each class in the training data
    means = means_vectors(pd.DataFrame(training_data), training_data_labels).values
    print("Means:", pd.DataFrame(means))

    # Compute the overall sample mean of the training data
    overall_mean = pd.DataFrame(training_data).mean().values
    print("Overall Mean:", pd.DataFrame(overall_mean))

    # Compute the number of samples in each class (which is assumed to be equal) and equal 5 sample for each class.
    n = np.zeros(means.shape[0]) + 5
    print("Number of items in each group:", pd.DataFrame(n))

    # Compute between-class scatter matrix
    sb = between_class_scatter_matrix(means, overall_mean, n)
    print("Between-class scatter matrix:", pd.DataFrame(sb))

    # Centralize the data
    Z = centralize(training_data, means, training_data_labels)
    print("Z:", pd.DataFrame(Z))

    # Compute the within-class scatter matrix
    S = within_class_scatter_matrix(Z)
    S_inverse = np.linalg.inv(S)
    print("S matrix:", pd.DataFrame(S))
    print("S_inverse:", pd.DataFrame(S_inverse))

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = get_eigens(np.matmul(S_inverse, sb))
    print("Eigenvalues:", pd.DataFrame(eigenvalues))
    print("Eigenvectors:", eigenvectors)

    # Select the top k eigenvectors and return the projection matrix
    eigenvectors = eigenvectors[:, :dimensions_needed]
    pd.DataFrame(eigenvectors).to_csv('projection.csv', index=False)
    return eigenvectors


"""
Projects the input data onto a lower-dimensional space using a projection matrix

Parameters:
data (numpy.ndarray): Input data to be projected.
data_labels (numpy.ndarray): Labels for the input data.
dimensions_needed (int): Number of dimensions in the lower-dimensional space.
load_or_compute (int): Specifies whether to compute a new projection matrix (1) or load a precomputed one (0).

Returns:
numpy.ndarray: Projected data.

"""


def project_data(data, data_labels, dimensions_needed, load_or_compute):
    if (load_or_compute == 1):
        projection_matrix = get_projection_matrix(data, data_labels, dimensions_needed)
    else:
        projection_matrix = pd.read_csv('projection.csv')
    print("projection_matrix", pd.DataFrame(projection_matrix))
    print("training_data", pd.DataFrame(data))

    new_data = np.dot(data, projection_matrix)

    return new_data


"""
Calculates the accuracy of KNN classification on the projected data.

Parameters:
    number_of_neighbors (int): Number of nearest neighbors to consider for classification.
    training_data (numpy.ndarray): Training data to be used for KNN classification.
    training_data_labels (numpy.ndarray): Labels for the training data.
    testing_data (numpy.ndarray): Testing data to be used for KNN classification.
    testing_data_labels (numpy.ndarray): Labels for the testing data.
    dimensions_needed (int): Number of dimensions in the lower-dimensional space.
    load_or_compute (int): Specifies whether to compute a new projection matrix (1) or load a precomputed one (0).
Returns:
    float: Accuracy of KNN classification.

"""


def calculate_accuracy(number_of_neighbors, training_data, training_data_labels, testing_data, testing_data_labels,
                       dimensions_needed, load_or_compute):
    training_data_projected = project_data(training_data, training_data_labels, dimensions_needed, load_or_compute)
    testing_data_projected = project_data(testing_data, testing_data_labels, dimensions_needed, load_or_compute)
    knn = KNeighborsClassifier(number_of_neighbors)
    # Train the model using the training set
    knn.fit(training_data_projected, training_data_labels)
    # Predict the response for test dataset
    y_predict = knn.predict(testing_data_projected)
    accuracy = metrics.accuracy_score(testing_data_labels, y_predict)
    return accuracy


def plot_lda_accuracy(training_data, training_data_labels, testing_data, testing_data_labels, dimensions_needed,
                      load_or_compute):
    k_values = [1, 3, 5, 7]
    accuracies = []
    for k in k_values:
        accuracies.append(calculate_accuracy(k, training_data, training_data_labels, testing_data, testing_data_labels,
                                             dimensions_needed, load_or_compute))
    plt.plot(k_values, accuracies)
    plt.xlabel('k_values')
    plt.ylabel('accuracy')
    plt.title('LDA accuracy vs KNN')
    plt.show()
