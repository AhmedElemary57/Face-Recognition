# Import necessary libraries
from sklearn import metrics
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
import time
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import dataloading as dl



def kernel_pca_transform(training, testing, n_components=39, kernel='rbf'):
    # create an instance of the KernelPCA class
    kpca = KernelPCA(n_components=n_components, kernel=kernel, alpha=0.8)
    # scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(training)
    X_test = scaler.transform(testing)
    # fit the KernelPCA model on the training data
    kpca.fit(X_train)

    # transform the training and testing data using the fitted model
    X_train_kpca = kpca.transform(X_train)
    X_test_kpca = kpca.transform(X_test)
    kpca.fit(X_train_kpca)

    return X_train_kpca, X_test_kpca


def qda_accuracy(training, training_labels, testing, testing_labels, k):
    # create an instance of the SelectKBest class
    selector = SelectKBest(chi2, k=k)

    # fit the selector on the training data and transform both the training and testing data
    training = selector.fit_transform(training, training_labels)
    testing = selector.transform(testing)

    # create an instance of the QuadraticDiscriminantAnalysis class
    qda = QuadraticDiscriminantAnalysis()

    # fit the QDA model on the training data
    qda.fit(training, training_labels)

    # predict the labels for the testing data
    predictions = qda.predict(testing)

    # calculate the accuracy of the predictions
    accuracy = accuracy_score(testing_labels, predictions)

    return accuracy


def accuracy_knn(training_data_projected, training_data_labels, testing_data_projected,
                       testing_data_labels, k):
    # Initialize the k-NN classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier on the training data
    knn.fit(training_data_projected, training_data_labels)

    # Test the classifier on the testing data
    y_predict = knn.predict(testing_data_projected)

    # Compute the accuracy of the classifier
    accuracy = metrics.accuracy_score(testing_data_labels, y_predict)

    return accuracy


D, y = dl.load_dataset('./data/archive')
training, training_labels, testing, testing_labels = dl.split_dataset_even_odd(D, y)

accuracy = qda_accuracy(training, training_labels, testing, testing_labels,39)

print("Accuracy:", accuracy)
training_data_projected, testing_data_projected = kernel_pca_transform(training, testing)
accuracy = accuracy_knn(training_data_projected,training_labels,testing_data_projected,testing_labels,1)

print(accuracy)


