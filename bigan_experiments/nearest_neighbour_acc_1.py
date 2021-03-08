import argparse
import os

import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

join = os.path.join

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        help='cifar10 | svhn | timagenet | cifar_mnist_cifar | cifar_mnist_mnist')
    parser.add_argument('--feat_dir', default="feats", help='path to dataset')

    args = parser.parse_args()

    dataset = args.dataset
    feat_dir = args.feat_dir
    labels_dir = args.feat_dir

    # Load train and test features
    train_features = np.load(join(feat_dir, dataset + "_train_feats.npy"))
    val_features = np.load(join(feat_dir, dataset + "_test_feats.npy"))


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

    pred_indices = np.argsort(distances, axis=-1)[:, :top_k]

    pred_labels = train_labels[pred_indices]

    # Generate classification accuracy for different values of k from 1 to 100
    top_k = np.arange(0, 101, 5)
    top_k[0] = 1
    for k in top_k:
        out = stats.mode(pred_labels[:, :k], axis=1).mode

        labels = np.array([val[0] for val in out])

        # print(pred_labels.shape, val_labels.shape)
        print(k, np.mean(np.equal(labels, val_labels)))
