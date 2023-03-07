import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def means_vectors(data, lables):
    print(data)
    # Reshape the DataFrame into groups for each label
    grouped_df = data.groupby(lables)
    # Calculate the mean for each group
    means_df = grouped_df.mean()
    print(means_df)
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
        mean_diff = means[i] - overall_mean.T
        SB += n[i] * np.outer(mean_diff, mean_diff.T)
    return SB

def centralize( data, means, labels ):
    Z = np.zeros(data.shape)
    for i in range(data.shape[0]): # for each row 'sample' do Di- mean vector
        Z[i,:] = data[i,:] - means[(int) (labels[i]) -1,:]
    return Z

def within_class_scatter_matrix(Z):
    S = np.dot(Z.T, Z)
    return S

def get_eigns(data):
    eigenvalues, eigenvectors = np.linalg.eigh(data)
    return eigenvalues, eigenvectors
def get_projection_matrix(trainingData, trainingDataLabels, testingData, testingDataLabels, dimensions_needed):
    # Compute the mean vector for each class in the training data
    means = means_vectors(pd.DataFrame(trainingData), trainingDataLabels).values
    print("means", pd.DataFrame(means))
    # Compute the overall sample mean of the training data
    overall_mean = pd.DataFrame(trainingData).mean().values
    print("overall_mean", pd.DataFrame(overall_mean))
    # Compute the number of samples in each class (which is assumed to be equal) and equal 5 sample for each class.
    n = np.zeros(means.shape[0])+5
    print(n)

    sb = between_class_scatter_matrix(means, overall_mean, n)
    print("between_class_scatter_matrix", pd.DataFrame(sb))
    Z = centralize(trainingData, means, trainingDataLabels)
    print("Z",pd.DataFrame(Z))
    S = within_class_scatter_matrix(Z)
    S_inverse = np.linalg.inv(S)
    print("S matrix", pd.DataFrame(S))
    print("S_inverse ", pd.DataFrame(S_inverse))
    eigenvalues, eigenvectors = np.linalg.eigh(np.matmul(S_inverse, sb))
    print("eigenvalues", pd.DataFrame(eigenvalues))
    print("eigenvectors", eigenvectors)
    eigenvalues = eigenvalues[0: dimensions_needed]
    eigenvectors = eigenvectors[:, :dimensions_needed] # this should be the needed projection matrix
    pd.DataFrame(eigenvectors).to_csv('projection.csv', index=False)
    return eigenvectors


def project_training_testing(training_data, training_dataLabels, testing_data, testing_dataLabels, dimensions_needed, load_or_compute):
    if(load_or_compute == 1):
        projection_matrix = get_projection_matrix(trainingData, trainingDataLabels, testingData, testingDataLabels, dimensions_needed)
    else:
        projection_matrix = pd.read_csv('projection.csv').applymap(lambda x: complex(x)).values

    print("projection_matrix", pd.DataFrame(projection_matrix))
    print("training_data", pd.DataFrame(training_data))

    new_training_data = np.dot(training_data, projection_matrix)
    new_testing_data = np.dot(testing_data, projection_matrix)

    return new_training_data, new_testing_data

def calculate_accuracy(training_data, training_dataLabels, testing_data, testing_dataLabels, dimensions_needed, load_or_compute):
    no_of_neighbours = [1, 3, 5, 7]  # For KNN
    NEW_TRAIN , NEW_TEST = project_training_testing(training_data, training_dataLabels, testing_data, testing_dataLabels, dimensions_needed, load_or_compute)
    for n in no_of_neighbours:
        # Create KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=n)
        # Train the model using the training set
        training_dataLabels = np.ravel(testing_dataLabels)
        knn.fit(NEW_TRAIN, training_dataLabels)
        # Predict the response for test dataset
        y_pred = knn.predict(NEW_TEST)
        accuary = metrics.accuracy_score(testing_dataLabels, y_pred)
        print(f"{n}                       {accuary}\n")

import dataloading as dl
import pca
import numpy as np
import pandas as pd


trainingData, trainingDataLabels, testingData, testingDataLabels = dl.load_data()
project_training_testing(trainingData, trainingDataLabels, testingData, testingDataLabels, 39, 1)

# Perform LDA
lda = LinearDiscriminantAnalysis(n_components=39)
lda.fit(trainingData, trainingDataLabels)

print(lda.coef_.T)

