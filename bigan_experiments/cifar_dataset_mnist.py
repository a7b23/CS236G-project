import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image


class CIFAR10_MNIST(data.Dataset):
    def __init__(self, root, aug_type=1, train=True, transform=None, target_transform=None, download=False,
                 indices=None):
        super(data.Dataset).__init__()
        print("present here")
        self.cifar10 = datasets.CIFAR10(root, train, download=download)
        self.mnist = datasets.MNIST(root, train=True, download=download)
        if indices:
            indices_val = np.load(indices)
            self.cifar10.data = self.cifar10.data[indices_val]
            self.cifar10.targets = [self.cifar10.targets[idx] for idx in indices_val]

        self.data = torch.tensor(self.cifar10.data.transpose((0, 3, 1, 2)))
        self.targets = torch.tensor(self.cifar10.targets)
        self.tensor_dataset = data.TensorDataset(self.data, self.targets)
        self.transform = transform
        self.target_transform = target_transform
        self.aug_type = aug_type

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        img, target = self.tensor_dataset[index]
        mnist_size = 8
        if self.aug_type == 2:
            mnist_size = 16

        img = transforms.ToPILImage()(img)
        mnist_img = self.mnist[np.random.randint(len(self.mnist))][0]

        mnist_img = np.array(mnist_img)
        mnist_img = (mnist_img > 1.0) * 255.0
        mnist_img = Image.fromarray(mnist_img.astype(np.uint8))

        mnist_img_resized = mnist_img.resize((mnist_size, mnist_size), Image.LANCZOS)
        mnist_img_resized_np = np.repeat(np.expand_dims(np.array(mnist_img_resized), axis=-1), 3, axis=-1)
        mnist_img_resized_np = (mnist_img_resized_np > 150.0) * 255.0
        img_np = np.array(img)
        new_img = np.zeros(img_np.shape)

        if self.aug_type == 1:
            new_img[2:2 + mnist_size, 2:2 + mnist_size, :] = mnist_img_resized_np
            new_img[2:2 + mnist_size, -10:-10 + mnist_size, :] = mnist_img_resized_np

            new_img[-10:-10 + mnist_size, 2:2 + mnist_size, :] = mnist_img_resized_np
            new_img[-10:-10 + mnist_size, -10:-10 + mnist_size, :] = mnist_img_resized_np

            new_img[12:12 + mnist_size, 12:12 + mnist_size, :] = mnist_img_resized_np
        else:
            new_img[8:8 + mnist_size, 8:8 + mnist_size, :] = mnist_img_resized_np
        new_img = Image.fromarray(new_img.astype(np.uint8))

        img_np = np.stack([img_np, new_img], axis=0)

        img_np = np.max(img_np, axis=0)

        img = Image.fromarray(img_np.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(data.Dataset).__init__()
        self.cifar10 = datasets.CIFAR100(root, train, download=download)
        self.data = torch.tensor(self.cifar10.data.transpose((0, 3, 1, 2)))
        self.targets = torch.tensor(self.cifar10.targets)
        self.tensor_dataset = data.TensorDataset(self.data, self.targets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        img, target = self.tensor_dataset[index]

        img = transforms.ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


if __name__ == "__main__":
    import os

    out_dir = "images_mnist_cifar"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    augmentation = [
        transforms.ToTensor(),
        # normalize
    ]

    dataroot = "/atlas/u/a7b23/data"
    batch_size = 64

    train_loader = torch.utils.data.DataLoader(
        CIFAR10_MNIST(root=dataroot, aug_type=1, train=True, download=False,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ])),
        batch_size=batch_size, shuffle=False)

    x, _ = next(iter(train_loader))

    vutils.save_image(x, "dataset_images.png")
