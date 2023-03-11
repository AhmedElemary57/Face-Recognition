import pandas as pd

import lda
import dataloading as dl
import pca

training, training_labels, testing, testing_labels = dl.load_data()

# Print the results
print('Training Data\n', pd.DataFrame(training))
print('Testing Data\n', pd.DataFrame(testing))

"""training = pd.DataFrame(training)
testing = pd.DataFrame(testing)
pca.apply_pca(training, training=1)
pca.apply_pca(testing, training=0)

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

projection_matrix = lda.get_projection_matrix(training, training_labels, 39, 5, 1)
training_data_projected = lda.project_data(training, projection_matrix)
testing_data_projected = lda.project_data(testing, projection_matrix)

print(lda.calculate_accuracy(1, training_data_projected,  training_labels, testing_data_projected, testing_labels))
lda.plot_lda_accuracy(training_data_projected,  training_labels, testing_data_projected, testing_labels)
