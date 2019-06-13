# Usage: python proportion_test.py [name of file with predictions]

# This program just checks to see how many outputs in a prediction file are
# positive and how many are negative
import sys

if __name__ == "__main__":
    lines = []
    positiveExamples = 0
    negativeExamples = 0
    with open(sys.argv[1], "r") as fp:
        for x in fp:
            lines.append(x.split(','))
        for x in lines:
            if float(x[1]) > 0.5:
                positiveExamples += 1
            else:
                negativeExamples += 1
    print("positive examples:", positiveExamples)
    print("negative examples:", negativeExamples)