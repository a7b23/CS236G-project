import argparse
import os

from bigan_experiments.classify_linear import eval_linear
from bigan_experiments.nearest_neighbour_acc_1 import eval_knn
from save_features import get_data_loaders, get_embeddings

join = os.path.join

batch_size = 64
latent_size = 256
cuda_device = "0"


def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        help='cifar10 | svhn | cifar_mnist_cifar | cifar_mnist_mnist | timagenet',
                        choices=['cifar10', 'svhn', 'cifar_mnist_mnist', 'cifar_mnist_cifar', 'timagenet'])

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

    train_features, train_labels = get_embeddings(train_loader, netE, None)
    test_features, test_labels = get_embeddings(test_loader, netE, None)

    knn_acc = eval_knn(train_features, train_labels, test_features, test_labels)

    logistic_acc = eval_linear('logistic', train_features, train_labels, test_features, test_labels)

    print(f"KNN={knn_acc * 100:.2f}, Linear={logistic_acc * 100:.2f}")
