
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from nn_model import Net
import sys

import argparse
import sys
import torchvision.datasets as datasets
join=os.path.join

parser = argparse.ArgumentParser(description='linear classification')

parser.add_argument('--feat_dir', type=str, default='./feats',
                        help='directory containing features')
parser.add_argument('--dataset', required=True, choices = ["cifar10", "svhn", "cifar_mnist_cifar", "cifar_mnist_mnist"])
parser.add_argument('--model_type', default="logistic", choices = ["svm", "logistic", "nn"])

# parser.add_argument('--nvae', default=False, action="store_true",
#                     help='whether to do for nvae')
# parser.add_argument('--ddim', default=False, action="store_true",
#                     help='whether to do for ddim')


args = parser.parse_args()

feat_dir = args.feat_dir
model_type = args.model_type
dataset = args.dataset

def get_model(model_name, inp_size):
	if model_name == "svm":
		clf = svm.LinearSVC(random_state=0)

	elif model_name == "logistic":
		clf = LogisticRegression(random_state=0)

	elif model_name == "nn":
		clf = Net(inp_size = inp_size, out_size = 10)

	return clf

def get_train_data():
	
	
	train_data = np.load(join(feat_dir, dataset + "_train_feats.npy"))
	train_data = np.reshape(train_data, [len(train_data), -1])

	labels = np.load(join(feat_dir, dataset + "_train_labels.npy"))

	print(train_data.shape, labels.shape)

	return train_data, labels

def get_val_data():

	val_data = np.load(join(feat_dir, dataset + "_test_feats.npy"))
	val_data = np.reshape(val_data, [len(val_data), -1])

	labels = np.load(join(feat_dir, dataset + "_test_labels.npy"))
	
	print(val_data.shape, labels.shape)

	return val_data, labels


accs = []
train_data, train_labels = get_train_data()
val_data, val_labels = get_val_data()

model = get_model(model_type, train_data.shape[1])

print(train_data.shape, val_data.shape, train_labels.shape, val_labels.shape)

model.fit(train_data, train_labels)
acc = model.score(val_data, val_labels)

print("the acc is ", acc)




