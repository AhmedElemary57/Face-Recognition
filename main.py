import pandas as pd

import LDA as lda
import dataloading as dl
import pca

training, training_labels, testing, testing_labels = dl.load_data('C:\\archive')

# Print the results
print('Training Data\n', pd.DataFrame(training))
print('Testing Data\n', pd.DataFrame(testing))

# training = pd.DataFrame(training)
# testing = pd.DataFrame(testing)
# pca.apply_pca(training, training=1)
# pca.apply_pca(testing, training=0)

print(pca.accuracy(training, training_labels, testing, testing_labels, 1))
pca.plot_accuracy(training, testing, training_labels, testing_labels)

# # test lda
# print(lda.calculate_accuracy(1, trainingData, trainingDataLabels, testingData, testingDataLabels, 39, 0))
# lda.plot_lda_accuracy(trainingData, trainingDataLabels, testingData, testingDataLabels, 39, 0)
