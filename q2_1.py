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
        self.depth = depth
        self.splitValue = None
        self.feature = None # The index of feature according to which this node splits data points
        self.parent = parent
        self.children = []
        self.featureIndex = -1
        self.infoGain = -1
        self.col_size = -1
        self.determineFeature()
    # Given a split value between 0 and 1 and an columnIndex
    # returns the information gain based off the split.
    def GetInfoGain(self, split, columnIndex):
        column = self.Xtrain[:, columnIndex] # Get Column index
        leftSplit = [] # Hold data needed for left split. Holds Y keys for left split.
        rightSplit = [] # Hold data for right split. Holds Y keys for right split.

        # Enumerate through the column we keep track of what index the
        # element less than the split was located at.
        for i, elem in enumerate(column):
            if elem <= split:
                leftSplit.append(i)
            else:
                rightSplit.append(i)

        # Purity based off the splits splits
        self.col_size = len(column)
        left_p = len(leftSplit) / self.col_size
        right_p = len(rightSplit) / self.col_size

        # Need to have the current entropy of the
        # node to show the info gained from the split.
        # combining them so current is a list of key values for the class.
        current_entropy = self.entropy(leftSplit + rightSplit) 

        # Return info gained by the given split value.
        return (current_entropy - left_p * self.entropy(leftSplit) -
                right_p * self.entropy(rightSplit))

    def entropy(self, data):
        class_count = [0, 0]
        for v_i in data:
            if self.Ytrain[v_i] == -1:
                class_count[0] += 1
            else:
                class_count[1] += 1

        neg = class_count[0] / len(data)
        pos = class_count[1] / len(data)
        # Zero case, this might be possible.
        if neg == 0 or pos == 0:
            return 0
        # We're all made here.
        neg_v = - neg * math.log2(neg)
        pos_v = - pos * math.log2(pos)
        return pos_v + neg_v

        # sort dataIndices according to value of feature
        # outer loop: iterate through all features in remaining feature list
        # inner loop: iterate through possible split values to find the one
        # that minimizes entropy for that feature
    def determineFeature(self):
        for column, feature in enumerate(self.Xtrain.T): # Iterate through columns of data by first transposing data matrix
            sortedFeature = np.unique(feature)
            sortedFeature.sort()
            for x in sortedFeature[:-1]: # This should technically stop 1 spot before the last element of the array but I am lazy
                infoGain = self.GetInfoGain(x, column)
                if infoGain > self.infoGain:
                    self.featureIndex = column
                    self.infoGain = infoGain
                    self.splitValue = x
                #self.entropy(threshold, column)

    # returns list of all leaf-level descendant's information gain and the number of data points in those leaf nodes
    def reproduce(self):
        self.determineFeature()

        # Base case. If the data is purely one category or the other return.
        # Also return if the depth counter we set at the root node has reached zero.
        if self.infoGain == 0 or self.depth == 0:
            return [(self.infoGain, self.col_size)]
        else:
            column = self.Xtrain[:, self.featureIndex] # Get Column index
        leftSplitX = [] # Hold data needed for left split. Holds Y keys for left split.
        rightSplitX = [] # Hold data for right split. Holds Y keys for right split.
        leftSplitY = []
        rightSplitY = []

        # Enumerate through the column we keep track of what index the
        # element less than the split was located at.
        for i, elem in enumerate(self.Xtrain.T[self.featureIndex]):
            if elem <= self.splitValue:
                leftSplitX.append(self.Xtrain[i])
                leftSplitY.append(self.Ytrain[i])
            else:
                rightSplitX.append(self.Xtrain[i])
                rightSplitY.append(self.Ytrain[i])
        
        # Create child nodes and encourage them to give you grandkids
        leftChild = Node(leftSplitX, leftSplitY, self.depth - 1, None)
        rightChild = Node(rightSplitX, rightSplitY, self.depth - 1, None)
        leftList = leftChild.reproduce()
        rightList = rightChild.reproduce()
        return leftList + rightList
        

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
    print(root.reproduce())
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
        - How does entropy work for leaf nodes?
    - 
'''
