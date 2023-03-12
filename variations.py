# Import necessary libraries
from sklearn import metrics
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
import time
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import dataloading as dl
import pca
import pandas as pd
import LDA as lda

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


def pca_vs_kernel_pca(training, training_labels, testing, testing_labels):
    k=[1,3,5,7]
    kernel_accuracies =[]
    training_data_projected, testing_data_projected = kernel_pca_transform(training, testing)

    for i in k:
        accuracy = accuracy_knn(training_data_projected, training_labels, testing_data_projected, testing_labels, i)
        kernel_accuracies.append(accuracy)

    pca_accuracies =[]

    for i in k:
        accuracy =pca.accuracy(training, training_labels, testing, testing_labels, i)
        pca_accuracies.append(accuracy)

    plt.plot(k,pca_accuracies, label="PCA")
    plt.plot(k, kernel_accuracies, label="kernel PCA")
    plt.xlabel('k_values')
    plt.ylabel('accuracy')
    plt.title("PCA vs Kernel PCA")
    plt.legend()
    plt.show()

def lda_vs_flda(training, training_labels, testing, testing_labels ):
    projection_matrix = lda.get_projection_matrix(training,training_labels,39,5,0)
    training_data_projected = lda.project_data(training,projection_matrix)
    testing_data_projected = lda.project_data(testing,projection_matrix)
    k_values = [1, 3, 5, 7]
    lda_accuracies = []
    for k in k_values:
        lda_accuracies.append(lda.calculate_accuracy(k, training_data_projected, training_labels, testing_data_projected,
                                             testing_labels, testing))
    flda = LinearDiscriminantAnalysis(n_components=39, solver='svd')
    X_train_flda = flda.fit_transform(training, training_labels)

    # Use the FLDA model to transform the testing set
    X_test_flda = flda.transform(testing)

    # Train a logistic regression classifier on the transformed training set
    flda_accuracies=[]
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k).fit(X_train_flda, training_labels)
        # Compute the accuracy of the classifier on the testing set
        accuracy = clf.score(X_test_flda, testing_labels)
        flda_accuracies.append(accuracy)
    plt.plot(k_values,lda_accuracies, label= "LDA")
    plt.plot(k_values,flda_accuracies, label="FLDA")
    plt.xlabel('k_values')
    plt.ylabel('accuracy')
    plt.title("LDA vs FLDA")
    plt.legend()
    plt.show()


"""
D, y = dl.load_dataset('./data/archive')
training, training_labels, testing, testing_labels = dl.split_dataset_even_odd(D, y)
training = pd.DataFrame(training)
testing = pd.DataFrame(testing)
#pca.apply_pca(training, training=1)
pca.apply_pca(testing, training=0)
pca_vs_kernel_pca(training, training_labels, testing, testing_labels)
lda_vs_flda(training,training_labels,testing,testing_labels)
"""

