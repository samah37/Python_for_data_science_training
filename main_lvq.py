## In this section we will see the implementation of LVQ Algorithm
'''
https://machinelearningmastery.com/implement-learning-vector-quantization-scratch-python/
'''
import numpy as np
from math import sqrt
from random import randrange
from random import seed
from csv import reader
#Calculate distance between two vectors function
def euclidean_distance(row1, row2):
    distance=0.0
    for i in range(len(row1)-1):
        distance+= (row1[i] - row2[i])**2
    return np.sqrt(distance)

# Get the best BMU
def get_best_matching_unit(codebooks, test_row):
    distances = list()
    for codebook in codebooks:
        dist= euclidean_distance(codebook, test_row)
        distances.append((codebook, dist))
    distances.sort(key=lambda tup: tup[1])
    return distances[0][0]
#test
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
test_row = dataset[0]
bmu = get_best_matching_unit(dataset, test_row)
print(bmu)
#Create a random codeBook vector
def random_codebook(train):
    n_records = len(train)
    n_features = len(train[0])
    codebook = [train[randrange(n_records)][i] for i in range(n_features)]
    return codebook
def train_codebooks(train, n_codebooks, lrate, epochs):
    codebooks = [random_codebook(train) for i in range(n_codebooks)]
    for epoch in range(epochs):
        rate = lrate* (1.0 -(epoch/float(epochs)))
        sum_error = 0.0
        for row in train:
            bmu = get_best_matching_unit(codebooks, row)
            for i in range(len(row)-1):
                error =  row[i] - bmu[i]
                sum_error += error**2
                if bmu[-1] == row[-1]:
                    bmu[i]+= rate * error
                else:
                    bmu[i]-= rate * error
        print ('>epoch=%d, lrate=%.3f, error=%.3f'% (epoch, rate, sum_error))
    return codebooks
# Test the training function

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
learn_rate = 0.3
n_epochs = 10
n_codebooks = 2
codebooks = train_codebooks(dataset, n_codebooks, learn_rate, n_epochs)
print('Codebooks: %s' % codebooks)
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
        return dataset
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column]= float(row[column].strip())







