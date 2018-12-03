import os
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import metrics
from tensorflow.python.keras import losses

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# load dataset
imdb = tf.keras.datasets.imdb
(xtrain, ytrain), (xtest, ytest) = imdb.load_data(num_words=10000)

# setup word index
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# pad to all the same length
xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

# setup validation split
xval = xtrain[:10000]
xtrain = xtrain[10000:]
yval = ytrain[:10000]
ytrain = ytrain[10000:]

# make model

def build_model(word_size=10000, out_neurons=16, number_hidden_layers=1):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(word_size, out_neurons))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    for _ in range(number_hidden_layers):
        model.add(tf.keras.layers.Dense(out_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    return model

for vocab_size in [10000,15000,20000,25000,30000]:
    for out_neurons in [8,16,24,32,64,128,256,512]:
        model = build_model(vocab_size, out_neurons, (out_neurons+1)//2)
        model.summary()
        model.compile(optimizer=tf.train.AdamOptimizer(), 
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        model.fit(xtrain,ytrain,batch_size=512,epochs=20,verbose=1,validation_data=(xval,yval))
        test_loss, test_acc = model.evaluate(xtest, ytest)
        print('Test accuracy with vocabulary size of {0} and output neurons of {1}: {2}'.format(vocab_size,out_neurons,test_acc))
