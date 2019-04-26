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
        self.parent = parent
        self.leftChild = None
        self.rightChild = None
        self.featureIndex = None
        self.infoGain = 0
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
            for x in sortedFeature[:-1]: 
                # print("column before if = ", column)
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
            self.infoGain = 0
            self.featureIndex = None
            print("BASE CASE")
            print("self.featureIndex = ", self.featureIndex)
            print("self.infoGain = ", self.infoGain)
            print("self.depth = ", self.depth)
            return [(self.infoGain, self.col_size)]
        else:
            leftSplitX = np.empty([0, self.Xtrain.shape[1]]) # Hold data needed for left split. Holds Y keys for left split.
            rightSplitX = np.empty([0, self.Xtrain.shape[1]]) # Hold data for right split. Holds Y keys for right split.
            leftSplitY = np.empty([0])
            rightSplitY = np.empty([0])

            # Enumerate through the column we keep track of what index the
            # element less than the split was located at.
            for i, elem in enumerate(self.Xtrain.T[self.featureIndex]):
                if elem <= self.splitValue:
                    leftSplitX = np.append(leftSplitX, self.Xtrain[i:i+1], axis=0)
                    leftSplitY = np.append(leftSplitY, self.Ytrain[i:i+1], axis=0)
                else:
                    rightSplitX = np.append(rightSplitX, self.Xtrain[i:i+1], axis=0)
                    rightSplitY = np.append(rightSplitY, self.Ytrain[i:i+1], axis=0)
            #yValues = {"left": [self.Ytrain[x[0]] for x in self.Ytrain[:self.featureIndex+1]], "right": [self.Ytrain[x[0]] for x in self.Ytrain[self.featureIndex+1:]]}
            # yLeft = [y for y in self.Ytrain[:self.featureIndex+1]]
            # yRight = [y for y in self.Ytrain[self.featureIndex+1:]]

            # Create child nodes and encourage them to give you grandkids
            self.leftChild = Node(leftSplitX, leftSplitY, self.depth - 1, None)
            self.rightChild = Node(rightSplitX, rightSplitY, self.depth - 1, None)
            leftList = self.leftChild.reproduce()
            rightList = self.rightChild.reproduce()
            return leftList + rightList
    
    # Pass data values and labels to this function
    # Figure out which data values go to each child based on the split value and featureIndex
    # Call child's test function with those data values if child != None
    # Base case: if both children are None, return # of errors
    def test(self, Xtest, Ytest):
        if self.leftChild == None and self.rightChild == None:
            negatives = 0
            positives = 0
            for i in Ytest:
                if i == -1:
                    negatives += 1
                if i == 1:
                    positives += 1
            return min(negatives, positives)

        leftSplitX = np.empty([0, self.Xtrain.shape[1]]) # Hold data needed for left split. Holds Y keys for left split.
        rightSplitX = np.empty([0, self.Xtrain.shape[1]]) # Hold data for right split. Holds Y keys for right split.
        leftSplitY = np.empty([0])
        rightSplitY = np.empty([0])
        error = 0

        # Enumerate through the column we keep track of what index the
        # element less than the split was located at.
        for i, elem in enumerate(self.Xtrain.T[self.featureIndex]):
            if elem <= self.splitValue:
                leftSplitX = np.append(leftSplitX, self.Xtrain[i:i+1], axis=0)
                leftSplitY = np.append(leftSplitY, self.Ytrain[i:i+1], axis=0)
            else:
                rightSplitX = np.append(rightSplitX, self.Xtrain[i:i+1], axis=0)
                rightSplitY = np.append(rightSplitY, self.Ytrain[i:i+1], axis=0)
        
        if self.leftChild != None:
            error += self.leftChild.test(leftSplitX, leftSplitY)
        if self.rightChild != None:
            error += self.rightChild.test(rightSplitX, rightSplitY)
        return error


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
    print(root.featureIndex)
    print(root.test(Xtrain, Ytrain))
    print(root.test(Xtest, Ytest))
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
