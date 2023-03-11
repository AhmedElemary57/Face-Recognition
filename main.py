import pandas as pd

import dataloading as dl
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import pca
import LDA as lda


D, y = dl.load_dataset('./data/archive')

training, training_labels, testing, testing_labels = dl.split_dataset_even_odd(D, y)
# Print the results
print('Training Data\n', pd.DataFrame(training))

print('Testing Data\n', pd.DataFrame(testing))

"""training = pd.DataFrame(training)
testing = pd.DataFrame(testing)
# pca.apply_pca(training, training=1)
pca.apply_pca(testing, training=0)

print(pca.accuracy(training, training_labels, testing, testing_labels, 1))
pca.plot_accuracy(training, testing, training_labels, testing_labels)
"""
"""projection_matrix = lda.get_projection_matrix(training, training_labels, 39, 5, 1)
training_data_projected = lda.project_data(training, projection_matrix)
testing_data_projected = lda.project_data(testing, projection_matrix)

print(lda.calculate_accuracy(1, training_data_projected,  training_labels, testing_data_projected, testing_labels, testing))
lda.plot_lda_accuracy(training_data_projected,  training_labels, testing_data_projected, testing_labels, testing)
"""
fig, axs = plt.subplots(2, 2, figsize=(23,33))
axs = axs.flatten()
img_arr = np.array(testing[0], dtype='int').reshape(92, 112)

true_label = testing_labels[0]
axs[0].imshow(img_arr, cmap='gray')
axs[0].set_title(f'True label: {true_label}, Predicted label: {7}')
axs[0].axis('off')
plt.tight_layout()
plt.show()