import numpy as np


def means_vectors(data, lables):
    # Reshape the DataFrame into groups for each lable
    grouped_df = data.groupby(lables)
    # Calculate the mean for each group
    means_df = grouped_df.mean()
    return means_df


def compute_between_class_scatter_matrix(means, overall_mean, n):
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


def centralize(data, means, labels):
    Z = np.zeros(data.shape)
    for i in range(data.shape[0]):  # for each row 'sample' do Di- mean vector
        Z[i, :] = data[i, :] - means[(int)(labels[i]) - 1, :]
    return Z


def within_class_scatter_matrix(Z):
    S = np.dot(Z.T, Z)
    return S;
