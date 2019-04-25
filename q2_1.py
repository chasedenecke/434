import sys
import numpy as np
import math
from statistics import mode
from itertools import compress

class Node():
    # It would have made a dreadfully ugly child;
    # but it makes a rather handsome pig.
    def __init__(self, Xtrain, Ytrain, depth=1, parent=None):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        print("Xtrain = ", self.Xtrain)
        self.depth = depth
        self.splitValue = None
        self.feature = None # The index of feature according to which this node splits data points
        self.parent = parent
        self.children = []
        self.determineFeature()

    # Given a split value between 0 and 1 and an columnIndex
    def GetInfoGain(self, split, columnIndex):
        coloumn = self.Xtrain[:, columnIndex] # Get Column index
        leftSplit = [] # Hold data needed for left split. Holds (idex, value)
        rightSplit = [] # Hold data for right split. Holds (index, value)

        # Enumerate through the column we keep track of what index the
        # element less than the split was located at.
        for i, elem in enumerate(column):
            if elem < split:
                leftSplit.append(i)
            else:
                rightSplit.append(i)

        # Purity based off the splits splits
        col_size = len(column)
        left_p = len(leftSplit) / col_size
        right_p = len(rightSplit) / col_size

        # Need to have the current entropy of the
        # node to show the info gained from the split.
        # combining them so current is a list of key values for the class.
        current_entropy = entropy(leftSplit + rightSplit)

        # Return info gained by the given split value.
        return (current_entropy - left_p * entropy(leftSplit) -
                right_p * entropy(rightSplit))

    def entroypy(self, data):
        classe_count = [0, 0]
        for v_i in data:
            if self.Ytrain[v_i] == -1:
                class_count[0] += 1
            else:
                class_count[1] += 1

        # Zero case, this might be possible.
        if class_count[0] and class_count[1] == 0:
            return 0
        # We're all made here.
        neg_v = - class_count[0] * math.log2(class_count[0])
        pos_v = - class_count[1] * math.log2(class_count[1])
        return pos_v + neg_v

        # sort dataIndices according to value of feature
        # outer loop: iterate through all features in remaining feature list
        # inner loop: iterate through possible split values to find the one
        # that minimizes entropy for that feature
    def determineFeature(self):
        print("fuck python indentation")
        minFeatureEntropy = 1
        for column, feature in enumerate(self.Xtrain.T): # Iterate through columns of data by first transposing data matrix
            sortedFeature = np.unique(feature)
            sortedFeature.sort()
            print("sortedFeature = ", sortedFeature)
            for x, y in list(zip(sortedFeature, sortedFeature[1:])):
                threshold = (x + y)/2
                self.entropy(threshold, column)

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

    root = Node(Xtrain, Ytrain, 1)

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



    TODO:
    - Add leaf node functionality (if class labels are all the same, return)
    - Entropy function
        - Takes two arguments: split value and column index for feature in question
        - Calculate entropy based on the number of feature values less than the split value

'''
