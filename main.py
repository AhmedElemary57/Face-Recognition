import pandas as pd

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
