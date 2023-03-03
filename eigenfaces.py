import os
import numpy as np
import pandas as pd
from PIL import Image

# Define the path to the dataset folder
# this is the path to the dataset in colab files.
dataset_path = 'C:\\Users\\Zeyad\'s G3\\Desktop\\archive'

# Define the size of the images
image_size = (112, 92)

# Initialize the data matrix and label vector
D = np.zeros((400, image_size[0] * image_size[1]))
y = np.zeros(400)
z = np.zeros(400)

# Loop over all the subjects
for i in range(1, 41):
    # Loop over all the images for the current subject
    for j in range(1, 11):
        # Load the image and convert it to a numpy array
        image_path = os.path.join(dataset_path, 's'+str(i), str(j) + '.pgm')
        with Image.open(image_path) as img:
            img = img.resize(image_size)
            img = np.asarray(img, float).flatten()
            D[(i - 1) * 10 + j - 1] = img
        # Set the label for the current image
        y[(i - 1) * 10 + j - 1] = i

# Print the shapes of the data matrix and label vector
assert D.shape == (400, 10304)
print('D shape:', D.shape)  # should print (400, 10304) if it is true
assert y.shape == (400,)
print('y shape:', y.shape)  # should print (400,) if it is true

# Split training data from the testing data
trainingData = D[::2]
trainingDataLabels = y[::2]

testingData = D[1::2]
testingDataLabels = y[1::2]

# Print the results
print('Training Data\n', pd.DataFrame(trainingData))
print('Testing Data\n', pd.DataFrame(testingData))

# Compute the mean vector to each data
xBarTraining = trainingData.mean()
xBarTesting = testingData.mean()

# Centralize the data
ZTraining = np.transpose(np.transpose(trainingData) - xBarTraining)
ZTesting = np.transpose(np.transpose(testingData) - xBarTesting)

# Compute the covariance matrices
covTraining = np.cov(np.transpose(ZTraining))
covTesting = np.cov(np.transpose(ZTesting))

pd.DataFrame(covTraining).to_csv('trained_cov.csv')
pd.DataFrame(covTesting).to_csv('testing_cov.csv')

# Compute the eigen values and vectors of each cov matrix
lambdaTraining, UTraining = np.linalg.eig(covTraining)
lambdaTesting, UTesting = np.linalg.eig(covTesting)

# Choose the best dimensionality reduction, we reduce dimensionality on the training set only
alpha = .95
lambdasum = np.sum(lambdaTraining)
r = len(ZTraining[0])
for i in range(len(ZTraining[0]), 0, -1):
    ratio = np.sum(lambdaTraining[:i]) / lambdasum
    if ratio >= alpha:
        r = i
print(r)

# After choosing r we construct the projection matrix
UTraining = UTraining[:r]
UTesting = UTesting[:r]

# The projection operation
print(pd.DataFrame(UTraining))
print(pd.DataFrame(trainingData))
