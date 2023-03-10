import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from LDA import get_eigens


def reduce(dimensions, eigenvalues, alpha=0.8):
    """
    Auxiliary function to reduce dimensionality of the data matrix.

    @type dimensions: int
    @param dimensions: An integer indicating dimensionality of the data.

    @type eigenvalues: array[int]
    @param eigenvalues: A NumPy array of integers holding the eigenvalues of cov matrix.

    @type alpha: float
    @param alpha: A float type number to determine the accuracy.
    @default alpha: 0.8
    """
    # Choose the best dimensionality reduction, we reduce dimensionality on the training set only
    lambdasum = np.sum(eigenvalues)
    r = dimensions
    for i in range(dimensions, 0, -1):
        ratio = np.sum(eigenvalues[:i]) / lambdasum
        if ratio >= alpha:
            r = i
    print("=== === Data can be reduced to " + str(r) + " dimensions. === ===")
    return r


def apply_pca(data, training):
    """
    Applies the pca on a data matrix.

    @:param
    @:type data: Pandas DataFrame

    @:return
    @:rtype tuple(
        A @:type Pandas.DataFrame, The projection matrix after applying PCA
        mean @:type NumPy.array, The mean vector.
        )
    """
    if training:
        # Compute the mean vector
        mean = data.mean()

        # Compute the centralized matrix
        Z = data - mean

        # Compute the covariance matrix of Z
        covariance = Z.cov()

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = get_eigens(covariance)

        # Apply the dimensionality reduction
        r = reduce(covariance.shape[0], eigenvalues)

        # Compute the projection matrix
        U = eigenvectors[:, :r]

        print(Z.shape)
        print(U.shape)

        # The projection operation
        A = np.dot(Z, U)

        # Keep the data in files
        pd.DataFrame(A).to_csv('training.csv')
        pd.DataFrame(U).to_csv('projection.csv')
        pd.DataFrame(mean).to_csv('training_mean.csv')

    else:
        mean = pd.read_csv('training_mean.csv')
        mean = mean.drop(mean.columns[0], axis=1)
        mean = np.squeeze(np.array(mean), axis=1)

        U = pd.read_csv('projection.csv')
        U = U.drop(U.columns[0], axis=1)

        Z = data.subtract(mean, axis=1)
        A = np.dot(Z, U)
        pd.DataFrame(A).to_csv('testing.csv')

    return pd.DataFrame(A), mean


def accuracy(training, training_labels, testing, testing_labels, neighbours):
    """
    Trains the training and testing data to check the accuracy of the training.

    @:param
    @:type training: np array

    @:param
    @:type testing: np array

    @:param
    @:type neighbours: number

    @:param
    @:type compute: boolean (0 or 1)

    @:returns
    @:rtype accuracy: number
    """

    # Always try to read from the saved reduced data files.
    try:
        training_reduced = pd.read_csv('training.csv')
        training_reduced = training_reduced.drop(training_reduced.columns[0], axis=1)
        print(training_reduced)
    # Compute the reduced data whenever a file is not found.
    except FileNotFoundError:
        training_reduced = apply_pca(training, training=1)
    try:
        testing_reduced = pd.read_csv('testing.csv')
        testing_reduced = testing_reduced.drop(testing_reduced.columns[0], axis=1)
        print(training_reduced)
    except FileNotFoundError:
        testing_reduced = apply_pca(testing, training=0)

    knn = KNeighborsClassifier(neighbours)
    knn.fit(training_reduced, training_labels)
    prediction = knn.predict(testing_reduced)
    return metrics.accuracy_score(testing_labels, prediction)


def plot_accuracy(training, testing, training_labels, testing_labels):
    """
    Plots the accuracy of the pca against the knn

    @:param
    @:type training: Pandas DataFrame

    @:param
    @:type training_labels: Pandas DataFrame

    @:param
    @:type testing: Pandas DataFrame

    @:param
    @:type testing_labels: Pandas DataFrame

    @:param
    @:type compute: boolean (0 or 1)

    @:returns
    @:rtype void
    """
    ks = [1, 3, 5, 7]
    accuracies = []
    for k in ks:
        accuracies.append(accuracy(training, testing, training_labels, testing_labels, k))
    plt.plot(ks, accuracies)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("PCA Accuracy vs KNN")
    plt.show()
