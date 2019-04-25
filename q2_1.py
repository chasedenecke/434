import sys
import numpy as np
import math
from statistics import mode

class Node():
    def __init__(self, dataIndices, data, depth=1, parent=None):
        self.dataIndices = dataIndices
        self.data = data
        self.depth = depth
        self.splitValue = None
        self.feature = None # The index of feature according to which this node splits data points
        self.parent = parent
        self.children = []

    def entropy(self, Xtrain):
        count = [0] * 2 # index 0 is number of 1 and index 2 is number of  -1
        for y_i in range(len(self.dataIndices[0])):
            if Xtrain[y_i][0] == -1:
                count[0] += 1 # count[0] = negative examples
            count[1] += 1 # count[1] = positive examples

            totalData = sum(count)

            H = 0
            if not self.Children:
                for c in count:
                    purity = c / len(self.dataIndices[0])
                    H -= purity * math.log2(purity)
                return H
            else:
               return  RemainingEntropyOfChildren(Xtrain, totalData)

    def RemaningEntropyofChildren(self, Xtrain, totalData):
        H = 0
        for node in self.children:
            H += self.len(node)/ (totalData * child.entropy(Xtrain))
        return H
    
        # sort dataIndices according to value of feature
        # outer loop: iterate through all features in remaining feature list
        # inner loop: iterate through possible split values to find the one that minimizes entropy for that feature
    def determineFeature():
        print("fuck python indentation")
        minFeatureEntropy = 1
        for feature in self.data.T:
            

class Tree():
    def __init__(self, node):
        self.root = node

def Normalize(X):
    return (X-X.min(0)) / X.ptp(0); # X.min retrieves the smallest value in the whole array. 
                                    # X - X.min(0) shifts all the values in X so that the smallest value is now 0. 
                                    # X.ptp(0) is X.max - X.min, 
                                    # so dividing by that value scales all values in X to between 0 and 1.

def GetNormalData(data):
    nd = Normalize(data)
    x = nd[:,1:] # First column is class labels, so don't include that in our X values.
    y = data[:,0]
    return x, y

def main():
    train = np.genfromtxt(sys.argv[1], delimiter=',')
    test = np.genfromtxt(sys.argv[2], delimiter=',')

    Xtrain, Ytrain = GetNormalData(train)
    Xtest, Ytest = GetNormalData(train)

    Xtrain = np.insert(Xtrain, 0, Ytrain, axis=1)
    root = Node([x for x in range(Xtrain.shape[0])], Xtrain, 1)
    
main()


'''
Each node has left and right child and a set of data points that flow to it
Parent nodes hand indices of data down to each child according to which side of the split value that data point falls on
Node figures out which feature it should represent
Need a set of discrete feature values

Finding threshold for continuous valued features:
    - sort indices of data features according to the value of that feature for each data point. Remove redundant values using collections library (see stack overflow post on slack)
    - for attribute in attributes: # iterate through possible attributes this node could represent
        - for(i = 1; i < last feature; i++)
            testThreshold = (sortedFeatureValues[i] + sortedFeatureValues[i+1])/2
            if entropy(threshold, feature) < minEntropy:
                bestThreshold = testThreshold
                minEntropy = currentEntropy
        if minEntropy < minEntropyForAllFeatures
        minEntropyForAllFeatures = minEntropy

    self.feature = feature that had minEntropyForAllFeatures
'''
