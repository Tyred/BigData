import keras
from keras.layers import Input, Conv1D, Activation, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import argparse

# Constants 
batch_size = 16
epochs = 50

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

def parser_args(cmd_args):

	parser = argparse.ArgumentParser(sys.argv[0], description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-e", "--exp", type=str, action="store", default="pairwise_distances", help="Experiment")
	parser.add_argument("-d", "--dataset", type=str, action="store", default="PigArtPressure", help="Dataset name")

	return parser.parse_args(cmd_args)

# obtaining arguments from command line
args = parser_args(sys.argv[1:])

dataset = args.dataset
exp = args.exp

if (exp == "pairwise_distances"):
	data = np.genfromtxt('../data/' + exp + '/' + dataset + '.txt', delimiter=' ',)
	print("Data shape:", data.shape)
elif (exp == "similarity_search"):
	data = np.genfromtxt('../data/' + exp + '/' + dataset + '/' + 'Data.txt', delimiter=' ',)
	print("Data shape:", data.shape)	

else:
	data = np.genfromtxt('../data/' + exp + '/' + dataset + '/' + dataset + '_train.txt', delimiter=' ',)
	print("Data shape:", data.shape)

# Getting rid of the NaNs and infs with interpolation
if (len(data.shape) == 1):
	data = np.array(pd.Series(data).interpolate())
	serie_length = 1024

	# 'Windowing' 
	data = np.array(window(data, serie_length))
	print("Window Data shape:", data.shape)
else:
	serie_length = data.shape[1]
	print("Serie length:", serie_length)

# Reshaping
data.shape = data.shape[0], serie_length, 1
print("Final Data shape:", data.shape)

# Swish Activation
def swish(x, beta = 1):
    return (x * K.sigmoid(beta * x))

get_custom_objects().update({'Swish': Activation(swish)})

# Convolutional Autoencoder
input_data = Input(shape=(serie_length, 1))
#x = ZeroPadding1D(2)(input_data) # If the input series length isn't multiple of 8 uncomment this
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
#decoded = Cropping1D(cropping=(2,2))(decoded)# If the input series length isn't multiple of 8 uncomment this
encoder = Model(input_data, encoded)
encoder.summary()

autoencoder = Model(input_data, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=2)

history = autoencoder.fit(data, data, epochs= epochs, batch_size= batch_size, callbacks = [callback])

encoder.save('../models/' + exp + '/' + '/new_train/encoder_{}.h5'.format(dataset))


# Plot the loss by epoch
training_loss = history.history['loss']
epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss, 'r--')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
