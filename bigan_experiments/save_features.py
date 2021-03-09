import argparse
import os
from functools import partial

import numpy as np
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets, transforms

join = os.path.join

batch_size = 64
latent_size = 256
cuda_device = "0"


def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def get_random_uniform_batch(data, targets, num_classes=10, samples_per_class=100):
    random_batch = np.zeros((num_classes * samples_per_class, data.shape[1]))
    random_targets = np.zeros(num_classes * samples_per_class)
    indices = np.random.permutation(data.shape[0])
    batch_size = 0
    label_counts = np.zeros(num_classes)
    for i in indices:
        if label_counts[targets[i]] < samples_per_class:
            label_counts[targets[i]] += 1
            random_batch[batch_size, :] = data[i, :]
            random_targets[batch_size] = targets[i]
            batch_size += 1
        if batch_size >= num_classes * samples_per_class:
            break

    return random_batch, random_targets


def get_embeddings(loader, netE, fname):
    all_embeddings = []
    all_targets = []

    for idx, (data, target) in enumerate(loader):
        temp, h1, h2, h3 = netE.forward(Variable(tocuda(data)))

        temp = temp.view(temp.size(0), -1)
        all_embeddings.extend(temp.cpu().data.numpy())
        all_targets.extend(target.cpu().data.numpy())
        print(idx, len(loader))
        if len(all_embeddings) >= 100000:
            break

    all_embeddings = np.array(all_embeddings)[:100000]
    all_targets = np.array(all_targets)[:100000]

    print(all_embeddings.shape, all_targets.shape)

    if fname is not None:
        np.save(fname, all_embeddings)
        np.save(fname.replace("feats.npy", "labels.npy"), all_targets)

    return all_embeddings, all_targets


def get_data_loaders(args):
    root = val_root = args.dataroot
    if opt.dataset == 'svhn':
        train_dataset_cls = partial(datasets.SVHN, split='extra')
        val_dataset_cls = partial(datasets.SVHN, split='train')
    elif opt.dataset == 'cifar10':
        train_dataset_cls = partial(datasets.CIFAR10, train=True)
        val_dataset_cls = partial(datasets.CIFAR10, train=False)

    elif opt.dataset == "cifar_mnist_cifar":
        train_dataset_cls = partial(datasets.CIFAR10, aug_type=1, dataset="cifar", train=True)
        val_dataset_cls = partial(datasets.CIFAR10, aug_type=1, dataset="cifar", train=False)

    elif opt.dataset == "cifar_mnist_mnist":
        train_dataset_cls = partial(datasets.CIFAR10, aug_type=1, dataset="mnist", train=True)
        val_dataset_cls = partial(datasets.CIFAR10, aug_type=1, dataset="mnist", train=False)

    elif opt.dataset == "timagenet":
        train_dataset_cls = datasets.ImageFolder
        val_dataset_cls = datasets.ImageFolder
        root = "/atlas/u/tsong/data/timagenet/train/"
        val_root = "/atlas/u/a7b23/data/tiny-imagenet-200/val"
    else:
        raise NotImplementedError
    train_loader = torch.utils.data.DataLoader(
        train_dataset_cls(root=root, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor()
                          ])),
        batch_size=batch_size, shuffle=False, num_workers=16)

    test_loader = torch.utils.data.DataLoader(
        val_dataset_cls(root=val_root, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
        batch_size=batch_size, shuffle=False, num_workers=16)
    return train_loader, test_loader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        help='cifar10 | svhn | cifar_mnist_cifar | cifar_mnist_mnist | timagenet',
                        choices=['cifar10', 'svhn', 'cifar_mnist_mnist', 'cifar_mnist_cifar', 'timagenet'])
    parser.add_argument('--feat_dir', required=True, help='features directory')

    parser.add_argument('--dataroot', default="/atlas/u/a7b23/data", help='path to dataset')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--model_path', required=True)

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    if not opt.dataset == "timagenet":
        from model import *
    else:
        from model_timagenet import *

    encoder_state_dict = torch.load(opt.model_path)
    netE = Encoder(latent_size, True)
    netE.load_state_dict(encoder_state_dict)
    netE = tocuda(netE)

    print("Model restored")

    if not os.path.exists(opt.feat_dir):
        os.makedirs(opt.feat_dir)

    train_loader, test_loader = get_data_loaders(opt)

    get_embeddings(train_loader, netE, join(opt.feat_dir, opt.dataset + "_train_feats.npy"))
    get_embeddings(test_loader, netE, join(opt.feat_dir, opt.dataset + "_test_feats.npy"))
