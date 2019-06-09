from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import sys
import os
from decimal import *

getcontext().prec = 18

data = pd.read_csv(sys.argv[1], sep="\t", header=None, dtype=object, skiprows=[0])
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

Y = data.loc[:,[1]]

# Y is a 2d dataframe. train_test_split requires it to have a single dimension, so we flatten it into 1d with this line.
Y = Y.unstack() 

# Prior to this line, Y's cells are of type string. This line converts the cells to ints
Y = Y.astype('int') 
X = data.loc[:, [x for x in range(2, data.shape[1])]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y.T, test_size=0.3)

# Create a random forest with n_estimators trees
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample") # class_weight=balanced causes the samples to be weighted inversely to the frequency of their class label

# Train the model
clf.fit(X_train, Y_train)

# Make predictions
Y_pred = clf.predict(X)

Y_prob = clf.predict_proba(X)
for index, row in data.iterrows():
    print(row[0] + "," + str(Y_prob[index][1]))