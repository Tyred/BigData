import keras
from keras.layers import Activation
from keras.models import load_model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

import tensorflow as tf


import numpy as np
import pandas as pd

import timeit

import sys
import argparse

# Constants
#window_size = 1024

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

def parser_args(cmd_args):

	parser = argparse.ArgumentParser(sys.argv[0], description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-e", "--exp", type=str, action="store", default="pairwise_distances", help="Experiment")
	parser.add_argument("-d", "--dataset", type=str, action="store", default="PigArtPressure", help="Dataset name")
	
	return parser.parse_args(cmd_args)

# obtaining arguments from command line
args = parser_args(sys.argv[1:])

dataset = args.dataset
exp = args.exp

def swish(x, beta = 1):
    return (x * K.sigmoid(beta * x))

get_custom_objects().update({'Swish': Activation(swish)})

# Swish Activation
#class Swish(Activation):
#	def __init__(self, activation, **kwargs):
#		super(Swish, self).__init__(activation, **kwargs)
#		self.__name__ = 'swish'

#def swish(x):
#	return (K.sigmoid(x) * x)

#get_custom_objects().update({'swish': Swish(swish)})

encoder = load_model('../models/' + exp + '/new_train/' + 'encoder_' + dataset + ".h5", compile = False)

if (exp == "pairwise_distances"):
	data = np.genfromtxt('../data/' + exp + '/' + dataset + '.txt', delimiter=' ',)
	print("Data shape:", data.shape)
elif (exp == "similarity_search"):
	data = np.genfromtxt('../data/' + exp + '/' + dataset + '/' + 'Data.txt', delimiter=' ',)
	print("Data shape:", data.shape)

	print("Encoding the queries as well")
	for i in range(1, 6):
		query = np.genfromtxt('../data/' + exp + '/' + dataset + '/' + 'Query' + str(i) + '.txt', delimiter=' ',)
		query.shape = 1, query.shape[0], 1
		query = encoder.predict(query)
		query.shape = query.shape[1]
		np.savetxt('../data/' + exp + '/' + dataset + '/coded_data/Query' + str (i) + '.txt', query)
		del query

else:
	data = np.genfromtxt('../data/' + exp + '/' + dataset + '/' + dataset + '_test.txt', delimiter=' ',)
	print("Data shape:", data.shape)

# Getting rid of the NaNs and infs with interpolation
if (len(data.shape) == 1):
	data = np.array(pd.Series(data).interpolate())
	serie_length = 1024

	# 'Windowing' 
	data = np.array(windowNoOverlay(data, serie_length))
	print("Window Data shape:", data.shape)
else:
	serie_length = data.shape[1]
	print("Serie length:", serie_length)

data.shape = data.shape[0], serie_length, 1

# Workaround to load the libraries so it doesn't count in the timer, 
# in production these libraries would be already loaded
coded_data = encoder.predict(data)

start = timeit.default_timer()

coded_data = encoder.predict(data)
print("Coded Data shape:", coded_data.shape)

stop = timeit.default_timer()

print("Time to code the serie:", stop - start)

coded_data.shape = coded_data.shape[0], coded_data.shape[1]
if (exp == "similarity_search"):
	np.savetxt('../data/' + exp + '/' + dataset + '/coded_data/' + 'Data.txt', coded_data)
elif(exp == "pairwise_distances"):
	np.savetxt('../data/' + exp + '/coded_data/' + dataset + '_coded.txt', coded_data)
else:
	np.savetxt('../data/' + exp + '/' + dataset + '/' + dataset + '_coded.txt', coded_data)