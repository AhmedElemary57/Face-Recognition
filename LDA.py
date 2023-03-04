import pandas as pd
import numpy as np

def means_vectors(data, lables):
    # Reshape the DataFrame into groups for each label
    grouped_df = data.groupby(lables)
    # Calculate the mean for each group
    means_df = grouped_df.mean()
    return means_df

def between_class_scatter_matrix(means, overall_mean, n):
    """
       Computes the between-class scatter matrix.

       Parameters:
       means (ndarray): A 2-dimensional array containing the mean of each class.
       overall_mean (ndarray): The overall mean of all the samples.
       n (ndarray): A 1-dimensional array containing the number of samples in each class.

       Returns:
       SB (ndarray): The between-class scatter matrix.
       """
    SB = np.zeros((overall_mean.shape[0], overall_mean.shape[0]))
    for i in range(means.shape[0]):
        mean_diff = means[i, :] - overall_mean
        SB += n[i] * mean_diff.dot(mean_diff.T)
    return SB

def centralize( data, means, labels ):
    Z = np.zeros(data.shape)
    for i in range(data.shape[0]): # for each row 'sample' do Di- mean vector
        Z[i,:] = data[i,:] - means[(int) (labels[i]) -1,:]
    return Z

def within_class_scatter_matrix(Z):
    S = np.dot(Z.T,Z)
    return S

def get_eigns(data):
    eigenvalues, eigenvectors = np.linalg.eig(data)
    eigenvectors= eigenvectors.T
    idxs = np.argsort(abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]
    return eigenvalues, eigenvectors
def LDA(trainingData, trainingDataLabels, testingData, testingDataLabels, dimensions_needed):
    # Compute the mean vector for each class in the training data
    means = means_vectors(pd.DataFrame(trainingData), trainingDataLabels).values
    # Compute the overall sample mean of the training data
    overall_mean = pd.DataFrame(trainingData).mean().values
    print(pd.DataFrame(means))
    print(pd.DataFrame(overall_mean))
    # Compute the number of samples in each class (which is assumed to be equal) and equal 5 sample for each class.
    n = np.zeros(means.shape[0])+5
    print(n)

    sb = between_class_scatter_matrix(means, overall_mean, n)
    print(pd.DataFrame(sb))
    Z = centralize(trainingData, means, trainingDataLabels)
    print("Z",pd.DataFrame(Z))
    S = within_class_scatter_matrix(Z)
    S_inverse = np.linalg.inv(S)
    print("S matrix",pd.DataFrame(S))
    print("S_inverse ",pd.DataFrame(S_inverse))
    eigenvalues, eigenvectors = get_eigns(S_inverse.dot(sb))
    print("eigenvalues", pd.DataFrame(eigenvalues))
    print("eigenvectors", pd.DataFrame(eigenvectors))
    eigenvalues = eigenvalues[0: dimensions_needed]
    eigenvectors = eigenvectors[0: dimensions_needed] # this should be the needed projection matrix
    return eigenvectors

import dataloading as dl
import pca
import numpy as np
import pandas as pd


trainingData, trainingDataLabels, testingData, testingDataLabels = dl.load_data()

print(pd.DataFrame(LDA(trainingData, trainingDataLabels, testingData, testingDataLabels, 39)))

