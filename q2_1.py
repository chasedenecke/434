import sys
import numpy as np
import math
from statistics import mode

class Node():
    def __init__(self, dataIndices, Xtrain, Ytrain, depth=1, parent=None):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.depth = depth
        self.splitValue = None
        self.feature = None # The index of feature according to which this node splits data points
        self.parent = parent
        self.children = []
        self.determineFeature()

    def entropy(self):
        count = [0,0] # index 0 is number of 1 and index 2 is number of  -1
        for y_i in range(len(self.Ytrain)):
            if self.Ytrain[y_i][0] == -1:
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
            return  RemainingEntropyOfChildren(self.Ytrain, totalData)

    def RemaningEntropyofChildren(self, Ytrain, totalData):
        H = 0
        for node in self.children:
            H += self.len(node)/ (totalData * child.entropy(Ytrain))
        return H
    
        # sort dataIndices according to value of feature
        # outer loop: iterate through all features in remaining feature list
        # inner loop: iterate through possible split values to find the one that minimizes entropy for that feature
    def determineFeature(self):
        print("fuck python indentation")
        minFeatureEntropy = 1
        for column, feature in enumerate(self.data.T): # Iterate through columns of data by first transposing data matrix
            print("feature = ", feature)
            sortedFeature = np.unique(feature)
            sortedFeature.sort()
            print("sortedFeature = ", sortedFeature)
            print("sortedFeature.size = ", sortedFeature.size)
            # for x in range(self.data.shape[1]):
            #     print(self.data[sortedIndices[x]][column], end =" ")
            # print("\n")

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

    root = Node( Xtrain, Ytrain, 1)
    
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
