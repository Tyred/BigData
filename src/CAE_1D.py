import keras
from keras.layers import Input, Conv1D, Activation, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import timeit

# Constants 
query_length = 1024
batch_size = 16
epochs = 50
dataset_name = input("Dataset name: ")
experiment = input("Which experiment? One of pairwise_distance, similarity_search or matrix_profile ")
file_train = '../Data/{0}/{1}/{1}_train.txt'.format(experiment, dataset_name)

# Window Function
def window(data, window_size):  # With overlay
    windowed_data = []
    i = 0

    while(i + window_size-1 < len(data)):
        windowed_data.append(data[i:(i+window_size)])
        i += window_size//2

    if((i - (window_size//2) + window_size-1) != len(data)):
        i = len(data) - window_size
        windowed_data.append(data[i:len(data)]) # add the rest

    return windowed_data 

def windowNoOverlay(data, window_size):  # Without overlay
    windowed_data = []
    i = 0

    while(i + window_size-1 < len(data)):
        windowed_data.append(data[i:(i+window_size)])
        i += window_size

    if (i != len(data)):
        i = len(data) - window_size
        windowed_data.append(data[i:len(data)]) # add the rest

    return windowed_data

data = np.genfromtxt(file_train, delimiter='\n',)
print("Data shape:", data.shape)

# Getting rid of the NaNs and infs
data = np.array(pd.Series(data).interpolate())

# 'Windowing' 
data = np.array(window(data, query_length))
print("Window Data shape:", data.shape)

# Reshaping
data.shape = data.shape[0], query_length, 1
print("Final Data shape:", data.shape)

# Stacked Autoencoder
class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})


input_data = Input(shape=(query_length, 1))
#x = ZeroPadding1D(2)(input_data) # If the input series length isn't multiple of 8
x = Conv1D(16, (3), activation='swish', padding='same')(input_data)
x = MaxPooling1D((2), padding='same')(x)
x = Conv1D(8, (3), activation='swish', padding='same')(x)
x = MaxPooling1D((2), padding='same')(x)
x = Conv1D(1, (3), activation='swish', padding='same')(x)
encoded = MaxPooling1D((2), padding='same')(x)

x = Conv1D(8, (3), activation='swish', padding='same')(encoded)
x = UpSampling1D((2))(x)
x = Conv1D(8, (3), activation='swish', padding='same')(x)
x = UpSampling1D((2))(x)
x = Conv1D(16, (3), activation='swish', padding='same')(x)
x = UpSampling1D((2))(x)
decoded = Conv1D(1, (3), activation='swish', padding='same')(x)
#decoded = Cropping1D(cropping=(2,2))(decoded)# If the input series length isn't multiple of 8
encoder = Model(input_data, encoded)
encoder.summary()

autoencoder = Model(input_data, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=2)

history = autoencoder.fit(data, data, epochs= epochs, batch_size= batch_size, callbacks = [callback])

autoencoder.save('../models/' + experiment + '/' + dataset_name + '/autoencoder_{}.h5'.format(dataset_name))
encoder.save('../models/' + experiment + '/' + dataset_name + '/encoder_{}.h5'.format(dataset_name))


# Plot the loss by epoch
training_loss = history.history['loss']
epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss, 'r--')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('../models/' + experiment + '/' + dataset_name +  '/{}_trainLoss.png'.format(dataset_name))
plt.show()
