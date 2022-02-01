## In this section we will see the implementation of KNN Algorithm using Scikit-learn library
'''
To install scikit-learn library you can go to your cmd line :
     In pycharm: View -> Tool widows -> Terminal
And run this cmd:
    pip install -U scikit-learn
You can check the installation by running this cmd:
    python -m pip show scikit-learn
For more details : please check this website:
https://scikit-learn.org/stable/install.html
Link of the Tuto:
https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url= "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# Or you can download the file and read it directly
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)
print (dataset.head())
# Split data set
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
print (len(y))
# Take 80% for training and 20% for test
from sklearn.model_selection import train_test_split
X_train, X_Test, Y_train, Y_test = train_test_split(X, y, test_size=0.20)
# Feature Scaling
''' 
The scaling is the operation of normalisation of the range of independent variables
-> For more details check:
 https://en.wikipedia.org/wiki/Feature_scaling#Motivation
 '''
from sklearn.preprocessing import  StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train =  scaler.transform(X_train)
X_Test = scaler.transform(X_Test)

# Training and Predictions:
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_Test)

#Evaluating the algorithm
'''
  For this evaluation you have :
 -> Confusion matix
 -> Precision
 -> Recall
 -> F1 score
 '''
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
#Comparing the error rate with the K value

error = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i= knn.predict(X_Test)
    error.append(np.mean(pred_i != Y_test))

plt.figure(figsize=(12,6))
plt.plot(range(1,40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K value')
plt.xlabel('k Value')
plt.ylabel('mEAN Error')
plt.legend()
plt.show()




