import numpy as np
import pandas as pd

# Choose the best dimensionality reduction, we reduce dimensionality on the training set only
def reduce(dimensions, eigenvalues):
    alpha = .8
    lambdasum = np.sum(eigenvalues)
    r = dimensions
    for i in range(dimensions, 0, -1):
        ratio = np.sum(eigenvalues[:i]) / lambdasum
        if ratio >= alpha:
            r = i
    print(r)
    return r


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