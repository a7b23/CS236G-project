import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats

import argparse
import sys

parser = argparse.ArgumentParser(description='w fitting')

parser.add_argument('--dataset', type=str, default='cats',
                        help='directory containing features')
parser.add_argument('--model_type', default='kde', type=str,
                    help='model type')
parser.add_argument('--bandwidth', default='scott', type=str,
                    help='bandwidth for KDE')
parser.add_argument('--mixtures', default=5, type=int,
                    help='number of mixtures')

args = parser.parse_args()

dataset = args.dataset
model_type = args.model_type
mixtures = args.mixtures

data = np.load("feats_w/" + str(dataset) + ".npy")

print(data.shape)

if model_type == 'kde':
	bandwidth = args.bandwidth
	data = data.T
	if bandwidth.isdigit():
		bandwidth = int(bandwidth)

	kernel = stats.gaussian_kde(data, bandwidth)
	scores = kernel.evaluate(data)
	out_fname = "feats_w/" + str(dataset) + "_" + model_type + "_" + str(bandwidth) + "_scores.npy"
else:
	gm = GaussianMixture(n_components=mixtures, random_state=0).fit(data)
	scores = gm.score_samples(data)
	out_fname = "feats_w/" + str(dataset) + "_" + model_type + "_" + str(mixtures) + "_scores.npy"

print(scores.shape)


np.save(out_fname, scores)

