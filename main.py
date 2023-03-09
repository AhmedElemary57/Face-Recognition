import pandas as pd

import LDA as lda
import dataloading as dl
import pca

trainingData, trainingDataLabels, testingData, testingDataLabels = dl.load_data()

# Print the results
print('Training Data\n', pd.DataFrame(trainingData))
print('Testing Data\n', pd.DataFrame(testingData))

data = pd.DataFrame(trainingData)
A, mean = pca.apply_pca(data)

# Keep the data in files
A.to_csv('reduced_training_set.csv')
file = open('mean.txt', 'w')
file.write(str(mean))
file.close()
# test lda
print(lda.calculate_accuracy(1, trainingData, trainingDataLabels, testingData, testingDataLabels, 39, 0))
lda.plot_lda_accuracy(trainingData, trainingDataLabels, testingData, testingDataLabels, 39, 0)
