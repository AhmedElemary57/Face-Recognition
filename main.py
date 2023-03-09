import pandas as pd

import LDA as lda
import dataloading as dl
import pca

trainingData, trainingDataLabels, testingData, testingDataLabels = dl.load_data()

# Print the results
print('Training Data\n', pd.DataFrame(trainingData))
print('Testing Data\n', pd.DataFrame(testingData))

"""data = pd.DataFrame(trainingData)
A, mean = pca.apply_pca(data)

# Keep the data in files
A.to_csv('reduced_training_set.csv')
file = open('mean.txt', 'w')
file.write(str(mean))
file.close()
# test lda"""
def save_df(np_array, path):
    pd.DataFrame(pd.DataFrame(np_array)).to_csv(path, index=False)
def read_csv(path):
    return pd.read_csv(path)

projection_matrix = lda.get_projection_matrix(trainingData, trainingDataLabels, 39, 0)
training_data_projected = lda.project_data(trainingData, projection_matrix)
save_df(training_data_projected, 'training_data_projected.csv')
testing_data_projected = lda.project_data(testingData, projection_matrix)
save_df(testing_data_projected, 'testing_data_projected.csv')

print(lda.calculate_accuracy(1, training_data_projected,  testing_data_projected, trainingDataLabels, testingDataLabels))
lda.plot_lda_accuracy(trainingData, trainingDataLabels, testingData, testingDataLabels)
