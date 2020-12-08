import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy.spatial.distance
import sys
import timeit

def parser_args(cmd_args):

	parser = argparse.ArgumentParser(sys.argv[0], description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-d", "--dataset", type=str, action="store", default="PigCVP", help="Dataset for evaluation")
	parser.add_argument("-m", "--model", type=str, action="store", default="PigArtPressure", help="Model name")

	return parser.parse_args(cmd_args)

# obtaining arguments from command line
args = parser_args(sys.argv[1:])

dataset = args.dataset

data = np.genfromtxt('../data/pairwise_distances/' + dataset + '.txt', delimiter = ' ',)

coded_data = np.genfromtxt('../data/pairwise_distances/coded_data/' + dataset + '_latent.tsv', delimiter = '\t',)

def pairwise_dist(data):

	start = timeit.default_timer()

	dist_m = scipy.spatial.distance.pdist(data)

	stop = timeit.default_timer()

	print("Time to calculate all pairwise distances: ", stop - start)

	#np.save("drive/My Drive/UFSCar/FAPESP/IC/Results/Task_1/" + "pdist_" + dataset_name + "_raw.npy", dist_m)

	plt.figure()
	plt.imshow(scipy.spatial.distance.squareform(dist_m))
	plt.colorbar()
	#plt.savefig("drive/My Drive/UFSCar/FAPESP/IC/Results/Task_1/" + "pairwise_distance_" + dataset_name + "_raw.pdf")
	plt.show()

	return 0

pairwise_dist(data)
pairwise_dist(coded_data)
