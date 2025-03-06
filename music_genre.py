code = """\
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
import random
import operator
import math

# Function to compute distance between two feature vectors
def distance(instance1, instance2, k):
    mm1, cm1 = instance1[0], instance1[1]
    mm2, cm2 = instance2[0], instance2[1]
    dist = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    dist += np.dot(np.dot((mm2 - mm1).T, np.linalg.inv(cm2)), (mm2 - mm1))
    dist += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    dist -= k
    return dist

# Function to find k-nearest neighbors
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    return [distances[x][0] for x in range(k)]

# Function to determine the nearest class
def nearestClass(neighbors):
    classVote = {}
    for response in neighbors:
        classVote[response] = classVote.get(response, 0) + 1
    return sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)[0][0]

# Load dataset
dataset = []
def loadDataset(filename, split, trSet, teSet):
    with open(filename, "rb") as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])

# Training and Testing
trainingSet, testSet = [], []
loadDataset("my.dat", 0.66, trainingSet, testSet)

# Running KNN
predictions = [nearestClass(getNeighbors(trainingSet, testSet[x], 5)) for x in range(len(testSet))]

# Accuracy Calculation
accuracy = sum([1 for x in range(len(testSet)) if testSet[x][-1] == predictions[x]]) / len(testSet)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
"""