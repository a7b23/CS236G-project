import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import os
import sys
import argparse

join=os.path.join

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | svhn')
parser.add_argument('--feat_dir', default = "feats", help='path to dataset')

args = parser.parse_args()


dataset = args.dataset
feat_dir = args.feat_dir
# labels_dir = feat_dir

labels_dir = "feats"

# train_features = np.load(join(feat_dir, "train_feats_ae_5_moco_load.npy"))
# val_features = np.load(join(feat_dir, "val_feats_ae_5_moco_load.npy"))

# train_labels = np.load(join(feat_dir, "train_labels_ae_5_moco_load.npy"))
# val_labels = np.load(join(feat_dir, "val_labels_ae_5_moco_load.npy"))

train_features = np.load(join(feat_dir, dataset + "_train_feats.npy"))
val_features = np.load(join(feat_dir, dataset + "_test_feats.npy"))

# train_features = np.load(join(feat_dir, "train_feats_ae_6_moco.npy"))
# val_features = np.load(join(feat_dir, "val_feats_ae_6_moco.npy"))

# train_features = np.load(join(feat_dir, "moco_nvae_1_train_feats_intriguing.npy"))
# val_features = np.load(join(feat_dir, "moco_nvae_1_val_feats_intriguing.npy"))

train_labels = np.load(join(labels_dir, dataset + "_train_labels.npy"))
val_labels = np.load(join(labels_dir, dataset + "_test_labels.npy"))

train_features = np.reshape(train_features, [len(train_features), -1])
val_features = np.reshape(val_features, [len(val_features), -1])

print(train_features.shape, val_features.shape, train_labels.shape, val_labels.shape)
indices = np.arange(len(train_features))
np.random.shuffle(indices)

top_k = 100

indices = indices[:50000]
train_features = train_features[indices]
train_labels = train_labels[indices]

# distances = euclidean_distances(val_features, train_features)
distances = -cosine_similarity(val_features, train_features)

pred_indices = np.argsort(distances, axis=-1)[:,:top_k]

pred_labels = train_labels[pred_indices]

top_k = np.arange(0, 101, 5)
top_k[0] = 1
for k in top_k:

	out = stats.mode(pred_labels[:,:k],axis=1).mode

	labels = np.array([val[0] for val in out])

	# print(pred_labels.shape, val_labels.shape)
	print(k, np.mean(np.equal(labels, val_labels)))





