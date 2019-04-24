import sys
import numpy as np
from statistics import mode

class Node():
    def __init__(self, dataIndices, splitValue=None, parent=None):
        self.parent = parent
        self.Children = []
        self.dataIndices = dataIndices
        self.splitValue = splitValue

    def GetData(self):
        print(self.dataIndices[0])
        print(self.dataIndices[1])
#    def entropy(self):
#        count = list()



class Tree():
    def __init__(self, node):
        self.root = node

def Normalize(X):
    return (X-X.min(0)) / X.ptp(0);

def GetNormalData(data):
    nd = Normalize(data)
    x = nd[:,1:]
    y = data[:,0]
    return x, y

def main():
    train = np.genfromtxt(sys.argv[1], delimiter=',')
    test = np.genfromtxt(sys.argv[2], delimiter=',')

    Xtrain, Ytrain = GetNormalData(train)
    Xtest, Ytest = GetNormalData(train)
    row, col = np.indices((train.shape[0], train.shape[1]))

    root = Node((row,col))
    root.GetData()

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
