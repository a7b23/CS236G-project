import os
import argparse
import sys

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats


parser = argparse.ArgumentParser(description='w fitting')

parser.add_argument('--dataset', type=str, default='cats',
                        help='directory containing features')
parser.add_argument('--model_type', default='kde', type=str,
                    help='model type')
parser.add_argument('--bandwidth', default='scott', type=str,
                    help='bandwidth for KDE')
parser.add_argument('--mixtures', default=5, type=int,
                    help='number of mixtures')
parser.add_argument('--sample_size', default=10000, type=int,
                    help='size of selected samples')
parser.add_argument('--algorithm', default='random', type=str,
                    help='algorithm to use')

args = parser.parse_args()

dataset = args.dataset
model_type = args.model_type
mixtures = args.mixtures
bandwidth = args.bandwidth
sample_size = args.sample_size
algorithm = args.algorithm

orig_images = np.load('images_' + str(dataset) + '/images_50000.npy')

if algorithm == "random":
	indices = np.arange(len(orig_images))
	np.random.shuffle(indices)
	indices = indices[:sample_size]
	out_fname = os.path.join("images_" + str(dataset), 
			"images_random_" + str(sample_size) + ".npy") 

else:
	if model_type == "kde":
		fname = "feats_w/" + str(dataset) + "_" + model_type + "_" + bandwidth + "_scores.npy"
		out_fname = os.path.join("images_" + str(dataset), 
			"images_" + model_type + "_" + bandwidth + "_" + str(sample_size) + ".npy") 
	else:
		fname = "feats_w/" + str(dataset) + "_" + model_type + "_" + str(mixtures) + "_scores.npy"
		out_fname = os.path.join("images_" + str(dataset), 
			"images_" + model_type + "_" + str(mixtures) + "_" + str(sample_size) + ".npy") 

	scores = np.load(fname)
	indices = np.argsort(scores)[::-1]
	indices = indices[:sample_size]

sampled_images = orig_images[indices]
print(sampled_images.shape)
np.save(out_fname, sampled_images)


