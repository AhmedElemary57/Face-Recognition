import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import analysis

from LDA import get_eigens


def _reduce(dimensions, eigenvalues, alpha=0.8):
    """
    Auxiliary function to reduce dimensionality of the data matrix.

    @:type dimensions: int
    @:param dimensions: An integer indicating dimensionality of the data.

    @:type eigenvalues: array[int]
    @:param eigenvalues: A NumPy array of integers holding the eigenvalues of cov matrix.

    @:type alpha: float
    @:param alpha: A float type number to determine the accuracy.
    @:default alpha: 0.8
    """
    # Choose the best dimensionality reduction, we reduce dimensionality on the training set only
    lambdasum = np.sum(eigenvalues)
    r = dimensions
    for i in range(dimensions, 0, -1):
        ratio = np.sum(eigenvalues[:i]) / lambdasum
        if ratio >= alpha:
            r = i
    print("====== Data can be reduced to " + str(r) + " dimensions. ======")
    return r


def apply_pca(data, training, alpha=0.8):
    """
    Applies the pca on a data matrix.

    @:type data: NumPy Array
    @:param data: the data to apply pca on

    @:type training: boolean
    @:param training: boolean (0, 1) to tell whether the application is on training or testing data

    @:type alpha: float
    @:param alpha: a number between 0 and 1 indicating tolerance percentage

    @:return
    @:rtype tuple(
        A @:type Pandas.DataFrame, The projected matrix after applying PCA
        U @:type Pandas.DataFrame, The projection matrix to apply PCA
        mean @:type NumPy.array, The mean vector.
        )
    """
    # Convert the data to a pandas dataframe
    data = pd.DataFrame(data)

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
        r = _reduce(covariance.shape[0], eigenvalues, alpha)

        # Compute the projection matrix
        U = eigenvectors[:, :r]

        # The projection operation
        A = np.dot(Z, U)

        # Keep the data in files
        pd.DataFrame(A).to_csv('pca_trained.csv')
        pd.DataFrame(U).to_csv('pca_projection.csv')
        pd.DataFrame(mean).to_csv('pca_mean.csv')

    else:
        try:
            mean = pd.read_csv('pca_mean.csv')
            mean = mean.drop(mean.columns[0], axis=1)
            mean = np.squeeze(np.array(mean), axis=1)

            U = pd.read_csv('pca_projection.csv')
            U = U.drop(U.columns[0], axis=1)
        except FileNotFoundError:
            print('Error: No trained data found. Cannot test data on null.')
            return

        Z = data.subtract(mean, axis=1)
        A = np.dot(Z, U)
        pd.DataFrame(A).to_csv('pca_tested.csv')

    return pd.DataFrame(A), pd.DataFrame(U), mean


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
        trained = pd.read_csv('pca_trained.csv')
        trained = trained.drop(trained.columns[0], axis=1)
    # Compute the reduced data whenever a file is not found.
    except FileNotFoundError:
        trained = apply_pca(training, training=1)
    try:
        tested = pd.read_csv('pca_tested.csv')
        tested = tested.drop(tested.columns[0], axis=1)
    except FileNotFoundError:
        tested = apply_pca(testing, training=0)

    knn = KNeighborsClassifier(neighbours)
    knn.fit(trained, training_labels)
    prediction = knn.predict(tested)
    acc = metrics.accuracy_score(testing_labels, prediction)

    false_recognitions = analysis.get_false_recognitions(np.array(testing_labels), np.array(prediction))
    for false_recognition in false_recognitions:
        print(false_recognition)

    return acc


def plot_accuracy(training, training_labels, testing, testing_labels):
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
        accuracies.append(accuracy(training, training_labels, testing, testing_labels, k))

    plt.plot(ks, accuracies)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("PCA Accuracy vs KNN")
    plt.show()
