from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import sys
import os
from decimal import *

data = pd.read_csv(sys.argv[1], sep="\t", header=None, dtype=object, skiprows=[0])

test_data = pd.read_csv(sys.argv[2], sep="\t", header=None, dtype=object, skiprows=[0])

Y = data.loc[:,[1]]

# Y is a 2d dataframe. train_test_split requires it to have a single dimension, so we flatten it into 1d with this line.
Y = Y.unstack() 

# Prior to this line, Y's cells are of type string. This line converts the cells to ints
Y = Y.astype('int')

# column 0 of data are strings labelling the sequence and column 1 is the sequence label (pseudoknot or not), so we start indexing at 2
X = data.loc[:, [x for x in range(2, data.shape[1])]]

# X_test has no labels, so we start the indexing at 1 instead of 2
X_test = test_data.loc[:, [x for x in range(1, 1054)]] # This is currently set up to work with featuresall.
                                                        # If you want it to work with features103, replace 1054 with test_data.shape[1]

# print("X_test = ", X_test)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(X_test)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y.T, test_size=0.2)

# Create a random forest with n_estimators trees
clf = RandomForestClassifier(n_estimators=500, n_jobs=4) # class_weight=balanced causes the samples to be weighted inversely to the frequency of their class label

# Train the model
clf.fit(X, Y)

# Make predictions
Y_prob = clf.predict_proba(X_test)

for i, x in test_data.iterrows():
    print(x.iloc[0], ",", Y_prob[i][1], sep="")
'''
Sources: https://www.datacamp.com/community/tutorials/random-forests-classifier-python
Thiss file currently trains on some of the data uses to evaluate its performance.
This is obviously bad practice. It is only set up this way right now because we are
waiting for testing data to modify the code.
'''
