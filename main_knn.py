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
from sklearn.model_selection import train_test_split
X_train, X_Test, Y_train, Y_test = train_test_split(X, y, test_size=0.20)



