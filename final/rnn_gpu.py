""" RNN """
import numpy as np
import math
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import model_from_json

# Used to limit accurecy for those trying to repoduce our results.
np.random.seed(7) 

def longest_word_len(filename):
    """ Find the length of the longest word  """
    with open(filename, 'r') as file_p:
        words = file_p.read().split()
    max_len = len(max(words, key=len))
    return max_len

def read_and_pad(filename, binary_class, max_len):
    """ Reads in rna and pads it."""
    rna_samples = []
    with open(filename) as file_p:
        for i, line in enumerate(file_p):
            if i % 2 != 0:
                line = [ord(char) for char in line]
                rna_pad = line + [0] * (max_len - len(line))
                rna_samples.append([rna_pad, binary_class])

    return np.asarray(rna_samples)

def process(samples):
     """ Used to split rna from their class."""
     sample_x = samples[:,[0][0]]
     sample_y = samples[:,[1]]
     
     return np.asarray(list(sample_x)), np.asarray(sample_y)

def RNN(train_x, train_y, test_x, test_y, max_len):
    """ LSTN """
    vector_length = 32
    model = Sequential()
    model.add(Embedding(5000, vector_length, input_length=max_len))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(train_x, train_y, epochs=1, batch_size=64)

    scores = model.evaluate(test_x, test_y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    print(test_x[0], test_y[0])
    yhat = model.predict(test_x[0])[0]
    print(yhat)
    model_to_json = model.to_json()
    with open("weights.json", "w") as json_file:
        json_file.write(model_to_json)
    model.save_weights("weights.h5")
    print("Saved model")

class LoadData:
    """ Load file data.  """
    def __init__(self, pks_file, pkfs_file):
        self.pks_file = pks_file
        self.pkfs_file = pkfs_file

    def find_longest_word(self):
        """ Find the longest word and set it. """
        pks = longest_word_len(self.pks_file)
        pkfs = longest_word_len(self.pkfs_file)
        return max(pks, pkfs)

if __name__ == "__main__":
    PKS = "pks_Train.fasta"
    PKFS = "pkfs_Train.fasta"
    DATA = LoadData(PKS, PKFS)
    MAXLEN = DATA.find_longest_word()

    PKS_RNA_SAMPLES = read_and_pad(PKS, 1, MAXLEN)
    PKFS_RNA_SAMPLES = read_and_pad(PKFS, 0, MAXLEN)
    RNA_SAMPLES = np.vstack((PKS_RNA_SAMPLES, PKFS_RNA_SAMPLES))

    np.random.shuffle(RNA_SAMPLES)
    RNA_LEN = len(RNA_SAMPLES)

    print("Total samples found: ", RNA_LEN)
    LAST_5PCT = math.ceil(RNA_LEN * 0.05)
    print("Using 95% for traning: ", RNA_LEN - LAST_5PCT)
    print("Using 5% for testing: ", LAST_5PCT)

    TRAIN = RNA_SAMPLES[:RNA_LEN - LAST_5PCT, :]
    TEST = RNA_SAMPLES[RNA_LEN - LAST_5PCT:, :]

    TRAIN = np.array(TEST)
    TEST = np.array(TEST)

    TRAIN_X, TRAIN_Y = process(TRAIN)
    TEST_X, TEST_Y = process(TEST)
    #print("TRAIN ", TRAIN_X)
    print(TRAIN_X)
    RNN(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, MAXLEN)

