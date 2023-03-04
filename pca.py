import numpy as np
import pandas as pd

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
def reduce(dimensions, eigenvalues, alpha = 0.8):
    # Choose the best dimensionality reduction, we reduce dimensionality on the training set only
    lambdasum = np.sum(eigenvalues)
    r = dimensions
    for i in range(dimensions, 0, -1):
        ratio = np.sum(eigenvalues[:i]) / lambdasum
        if ratio >= alpha:
            r = i
    print("=== === Data can be reduced to " + r + " dimensions. === ===")
    return r


"""
Applies the pca on a data matrix.

@type data: np array
@param data: A NumPy array

@rtype: tuple
@return: a tuple of (
    A: @type Pandas.DataFrame, The projection matrix after applying PCA
    mean: @type NumPy.array, The mean vector.
    )
"""
def apply_pca(data):
    # Compute the mean vector
    mean = data.mean()

    # Compute the centralized matrix
    Z = data - mean

    # Compute the covariance matrix of Z
    covariance = Z.cov()

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # Apply the dimensionality reduction
    r = reduce(len(covariance), eigenvalues)

    # Compute the projection matrix
    U = eigenvectors[:r]

    print(U.shape)
    print(data.shape)

    # The projection operation
    A = np.dot(Z, np.transpose(U))

    return pd.DataFrame(A), mean