import argparse
import os

import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from nn_model import Net

join = os.path.join


def get_model(model_name, inp_size):
    if model_name == "svm":
        clf = svm.LinearSVC(random_state=0)

    elif model_name == "logistic":
        clf = LogisticRegression(random_state=0, max_iter=200)

    elif model_name == "nn":
        clf = Net(inp_size=inp_size, out_size=10)

    return clf


def get_data(split='train'):
    data = np.load(join(feat_dir, dataset + f"_{split}_feats.npy"))
    data = np.reshape(data, [len(data), -1])

    labels = np.load(join(feat_dir, dataset + f"_{split}_labels.npy"))

    print(data.shape, labels.shape)

    return data, labels


def eval_linear(model_type, train_data, train_labels, val_data, val_labels):
    model = get_model(model_type, train_data.shape[1])

    print(train_data.shape, val_data.shape, train_labels.shape, val_labels.shape)

    model.fit(train_data, train_labels)
    acc = model.score(val_data, val_labels)

    print("the acc is ", acc)
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear classification')

    parser.add_argument('--feat_dir', type=str, default='./feats',
                        help='directory containing features')
    parser.add_argument('--dataset', required=True,
                        choices=["cifar10", "svhn", "cifar_mnist_cifar", "cifar_mnist_mnist", "timagenet"])
    parser.add_argument('--model_type', default="logistic", choices=["svm", "logistic", "nn"])

    args = parser.parse_args()

    feat_dir = args.feat_dir
    dataset = args.dataset

    eval_linear(args.model_type, *get_data(split='train'), *get_data(split='test'))
