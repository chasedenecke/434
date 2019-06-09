import pandas as pd
import random
import numpy as np
import tensorflow as tf
import time

from collections import deque
from sklearn import preprocessing 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Flatten, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint

SEQ_LEN = 4383 # From dataPad.py
RATIO_TO_PREDICT = "class"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{int(time.time())}"

def preprocess_df(df):
  df = df.drop("label", 1)

  sequential_data = []
  #rna_string = deque(maxlen=SEQ_LEN)

  for rna in df.values:
    #rna_string.append(i[0])
    #if len(rna_string) == SEQ_LEN:
    #print("rna[0][i]", rna[-2])
    #print("rna[-1]", rna[-1])
    #exit()
    sequential_data.append([np.array(rna[-2]), rna[-1]])

  random.shuffle(sequential_data)
  # You can tell I was strugling here....
  train_x = np.asarray([sequential_data[i][0] for i in
      range(len(sequential_data))])
  train_y = np.asarray([sequential_data[i][1] for i in
      range(len(sequential_data))])

  return train_x, train_y

"""
  for i in df.values:
    sequential_data.append([np.array(i[0]), i[-1]])
  
  random.shuffle(sequential_data)
  print(sequential_data[0])
  return sequential_data
"""

df = pd.read_csv("./sequences.csv", names=["label", "rna", "class"])

times = sorted(df.index.values)

print("\nHead before shuffle.")
print(df.head())

df = df.sample(frac=1).reset_index(drop=True)

print("\nHead after shuffle.")
print(df.head())

last_5pct = times[-int(0.05*len(times))]
print("\nUsin 5% of the data, total: ", last_5pct)

validation_main_df = df[(df.index >= last_5pct)]
main_df = df[(df.index < last_5pct)]

train_x, tran_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

train_x = np.reshape(train_x, (1, train_x.shape[0]))
validation_x = np.reshape(validation_x, (1, validation_x.shape[0]))
print(train_x.shape[1:])
print(train_x.shape)

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]),activation='relu', return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}"

#checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

#history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validation_x, validation_y), callbacks=[tensorboard, checkpoint])

score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save(filepath)

