import numpy as np 
import pandas as pd 
import sklearn as sk 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
import random
import warnings
warnings.filterwarnings('ignore')
# two common anamoly detection packages
# SVM took more time  training
# one class SVM an unsupervised learning model
#here we have to think how different is anomaly from others
#Neural networks can be used
#ANN took less time than others
df = pd.read_csv('creditcard.csv')
data = df.sample(frac = 0.1, random_state = 1)

fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
X_data = data.drop(['Class'], axis = 1)
print(X_data.shape)
Y_data = data['Class']
print(Y_data.shape)
# print(len(fraud))
# print(len(valid))
random.seed(1)

classifiers = {
    'Isolation Forest' : IsolationForest(max_samples=len(X_data), contamination=(len(fraud)/len(valid))),
    'Local Outlier Factor' : LocalOutlierFactor(n_neighbors=20,contamination=(len(fraud)/len(valid)), novelty=True)
}

# n_outliers = len(fraud)

# for i, (clf_name , clf) in enumerate(classifiers.items()):
    
#     if clf_name is 'Local Outlier Factor':
#         print(clf_name)
#         start = time.time()
#         y_pred = clf.fit_predict(X_data)
#         scores_pred = clf.negative_outlier_factor_
#         end = time.time()
#         print('Time: {}'.format(end - start))

#     else:
#         print(clf_name)
#         start = time.time()
#         clf.fit(X_data)
#         scores_pred = clf.decision_function(X_data)
#         y_pred = clf.predict(X_data)
#         end = time.time()
#         print('Time: {}'.format(end - start))

#     y_pred[y_pred == 1] = 0
#     y_pred[y_pred == -1] = 1

#     n_errors = (y_pred != Y_data).sum()

#     print('{} {}'.format(clf_name, n_errors))
#     print(accuracy_score(Y_data,y_pred))
#     print(classification_report(Y_data, y_pred))

# X_train, X_test, y_train, y_test = train_test_split(X_data,Y_data, test_size = 0.2)


# svm = OneClassSVM(kernel='rbf')
# svm.fit(X_train)
# y_pred = svm.predict(X_test)
# print(y_pred)
# y_pred[y_pred == 1] = 0
# y_pred[y_pred == -1] = 1
# print(accuracy_score(Y_data,y_pred))
# print(classification_report(Y_data, y_pred))

X_arr = np.array(X_data)
Y_arr = np.array(Y_data)
unique, counts = np.unique(Y_arr, return_counts=True)
print(dict(zip(unique, counts)))
clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(300, 300,300,100), max_iter=300,activation = 'relu')
start = time.time()
clf.fit(X_arr,Y_arr)
y_pre = clf.predict(X_arr)
unique, counts = np.unique(y_pre, return_counts=True)
print(dict(zip(unique, counts)))
end = time.time()
print('Time: {}'.format(end - start))
print(accuracy_score(Y_arr,y_pre))
print(classification_report(Y_arr, y_pre))
print(confusion_matrix(Y_arr,y_pre))
