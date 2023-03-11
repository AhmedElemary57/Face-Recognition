import pandas as pd

import lda
import dataloading as dl
import pca

training, training_labels, testing, testing_labels = dl.load_data('C:\\archive')

# Print the results
print('Training Data\n', pd.DataFrame(training))
print('Testing Data\n', pd.DataFrame(testing))

training = pd.DataFrame(training)
testing = pd.DataFrame(testing)
# pca.apply_pca(training, training=1)
pca.apply_pca(testing, training=0)

print(pca.accuracy(training, training_labels, testing, testing_labels, 1))
pca.plot_accuracy(training, testing, training_labels, testing_labels)

projection_matrix = lda.get_projection_matrix(training, training_labels, 39, 5, 1)
training_data_projected = lda.project_data(training, projection_matrix)
testing_data_projected = lda.project_data(testing, projection_matrix)

print(lda.calculate_accuracy(1, training_data_projected,  training_labels, testing_data_projected, testing_labels))
lda.plot_lda_accuracy(training_data_projected,  training_labels, testing_data_projected, testing_labels)
