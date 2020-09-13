import numpy as np
import random
import argparse
import sys
import matplotlib.pyplot as plt

_EPS = 1e-14

def mpself(seq, subseq_len):
    
    # prerequisites
    exclusion_zone = int(np.round(subseq_len/2))
    ndim = seq.shape[0]
    seq_len = seq.shape[1]
    matrix_profile_len = seq_len - subseq_len + 1
    first_subseq = np.flip(seq[:,0:subseq_len],1) 

    # windowed cumulative sum of the sequence
    seq_cum_sum = np.hstack((np.zeros((ndim,1)), np.cumsum(seq,1)))
    seq_cum_sum = seq_cum_sum[:,subseq_len:]-seq_cum_sum[:,0:seq_len - subseq_len + 1]

    seq_cum_sum2 = np.hstack((np.zeros((ndim,1)), np.cumsum(np.square(seq),1)))
    seq_cum_sum2 = seq_cum_sum2[:,subseq_len:]-seq_cum_sum2[:,0:seq_len - subseq_len + 1]

    # mean and standard deviations (necessary for z-norm)
    mu_all = seq_cum_sum / subseq_len
    # TODO improve this equation
    sigma_all = np.sqrt((seq_cum_sum2 + seq_cum_sum*seq_cum_sum/subseq_len - 2 * seq_cum_sum * mu_all) / subseq_len)

    # sliding dot product
    prods = np.full([ndim,seq_len+subseq_len-1], np.inf)
    for i_dim in range(0,ndim):
        prods[i_dim,:] = np.convolve(first_subseq[i_dim,:],seq[i_dim,:])

    prods = prods[:, subseq_len-1:seq_len] # only the interesting products
    prods_inv = np.copy(prods)

    # first distance profile
    # DP^2 = 2m * {1 - [(QT - m*mu_q*mu_t) / (m*sigma_q*sigma_t)] }
    dist_profile = np.sum(2*subseq_len*(1-((prods - subseq_len*mu_all[:,0:1]*mu_all)/(subseq_len*sigma_all[:,0:1]*sigma_all))), axis=0)
    dist_profile[0:exclusion_zone] = np.inf
    
    matrix_profile = np.full(matrix_profile_len, np.inf)
    matrix_profile[0] = np.min(dist_profile)

    mp_index = -np.ones((matrix_profile_len), dtype=int)
    mp_index[0] = np.argmin(dist_profile)

    # for all the other values of the profile
    for i_subseq in range(1,matrix_profile_len):
        
        sub_value = seq[:,i_subseq-1, np.newaxis] * seq[:,0:prods.shape[1]-1]
        add_value = seq[:,i_subseq+subseq_len-1, np.newaxis] * seq[:, subseq_len:subseq_len+prods.shape[1]-1]

        prods[:,1:] = prods[:,0:prods.shape[1]-1] - sub_value + add_value
        prods[:,0] = prods_inv[:,i_subseq]
        
        # dist_profile
        dist_profile = np.sum(2*subseq_len*(1-((prods - subseq_len*mu_all[:,i_subseq:i_subseq+1]*mu_all)/(subseq_len*sigma_all[:,i_subseq:i_subseq+1]*sigma_all))), axis=0)
        
        # excluding trivial matches
        dist_profile[max(0,i_subseq-exclusion_zone+1):min(matrix_profile_len,i_subseq+exclusion_zone)] = np.inf
        
        matrix_profile[i_subseq] = np.min(dist_profile)
        mp_index[i_subseq] = np.argmin(dist_profile)
        
    return matrix_profile, mp_index

def parser_args(cmd_args):

	parser = argparse.ArgumentParser(sys.argv[0], description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-d", "--dataset", type=str, action="store", default="PigCVP", help="Dataset for evaluation")
	
	return parser.parse_args(cmd_args)

# obtaining arguments from command line
args = parser_args(sys.argv[1:])

dataset = args.dataset

coded_data = np.genfromtxt('../data/matrix_profile/' + dataset + '/' + dataset + '_coded.txt', delimiter=" ")
print(coded_data.shape)
coded_data = coded_data.flatten()
coded_data.shape = 1, coded_data.shape[0]
print(coded_data.shape)

mp, mpi = mpself(coded_data, 128)

print("Motif", np.min(mp))
print("Motif index", np.argmin(mp))


raw_data = np.genfromtxt('../data/matrix_profile/' + dataset + '/' + dataset +'_test.txt', delimiter=" ")
print(raw_data.shape)
raw_data.shape = 1, raw_data.shape[0]
print(raw_data.shape)

mp, mpi = mpself(raw_data, 1024)

print("Motif", np.min(mp))
print("Motif Index:", np.argmin(mp))