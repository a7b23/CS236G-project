import argparse
import os

import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import datasets, transforms

from cifar_dataset_mnist import CIFAR10_MNIST
from model import *

torch.autograd.set_detect_anomaly(True)

batch_size = 100
lr = 1e-4
latent_size = 256
num_epochs = 100
cuda_device = "0"


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=["cifar10", "svhn", "cifar_mnist"],
                    help='cifar10 | svhn | cifar_mnist')
parser.add_argument('--alpha', default=0.25, type=float, help='the weightage of the fake loss')

parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=boolean_string, default=True)
parser.add_argument('--save_model_dir', required=True)
parser.add_argument('--save_image_dir', required=True)

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
print(opt)

if not os.path.exists(opt.save_image_dir):
    os.makedirs(opt.save_image_dir)

if not os.path.exists(opt.save_model_dir):
    os.makedirs(opt.save_model_dir)


def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def log_sum_exp(input):
    m, _ = torch.max(input, dim=1, keepdim=True)
    input0 = input - m
    m.squeeze()
    return m + torch.log(torch.sum(torch.exp(input0), dim=1))


def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))


class TwoCropsTransformClean:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        k = self.base_transform(x)
        q_clean = transforms.ToTensor()(x)
        return [q_clean, k]


augs = transforms.Compose([transforms.RandomApply([
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

if opt.dataset == 'svhn':
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='extra', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ])),
        batch_size=batch_size, shuffle=True)
elif opt.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
                         transform=TwoCropsTransformClean(augs)),
        batch_size=batch_size, shuffle=True)
elif opt.dataset == 'cifar_mnist':
    train_loader = torch.utils.data.DataLoader(
        CIFAR10_MNIST(root=opt.dataroot, aug_type=1, train=True, download=False,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ])),
        batch_size=batch_size, shuffle=True)
else:
    raise NotImplementedError

netE = tocuda(Encoder(latent_size, True))
netG = tocuda(Generator(latent_size))
netD = tocuda(Discriminator(latent_size, 0.2, 1))

netE.apply(weights_init)
netG.apply(weights_init)
netD.apply(weights_init)

optimizerG = optim.Adam([{'params': netE.parameters()},
                         {'params': netG.parameters()}], lr=lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()


def get_perm(l):
    perm = torch.randperm(l)
    while torch.all(torch.eq(perm, torch.arange(l))):
        perm = torch.randperm(l)
    return perm


for epoch in range(num_epochs):

    i = 0
    for (data, target) in train_loader:

        real_label = Variable(tocuda(torch.ones(batch_size)))
        fake_label = Variable(tocuda(torch.zeros(batch_size)))

        noise1 = Variable(tocuda(torch.Tensor(data[0].size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        noise2 = Variable(tocuda(torch.Tensor(data[0].size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))

        if epoch == 0 and i == 0:
            netG.output_bias.data = get_log_odds(tocuda(data[0]))

        if data[0].size()[0] != batch_size:
            continue

        # Real images (x)
        d_real = Variable(tocuda(data[0]))
        # Augmented Real images (x_aug)
        d_real_aug = Variable(tocuda(data[1]))

        # Fake images (G(z))
        z_fake = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))
        d_fake = netG(z_fake)

        # Sample E(x) with mean and std learned from encoder
        z_real, _, _, _ = netE(d_real)
        z_real = z_real.view(batch_size, -1)

        mu, log_sigma = z_real[:, :latent_size], z_real[:, latent_size:]
        sigma = torch.exp(log_sigma)
        epsilon = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z = mu + epsilon * sigma

        # Sample E(x_aug) with mean and std learned from encoder
        z_real_aug, _, _, _ = netE(d_real_aug)
        z_real_aug = z_real_aug.view(batch_size, -1)

        mu_aug, log_sigma_aug = z_real_aug[:, :latent_size], z_real_aug[:, latent_size:]
        sigma_aug = torch.exp(log_sigma_aug)
        epsilon_aug = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z_aug = mu_aug + epsilon_aug * sigma_aug

        # Discriminator output for reals D(E(x), x)
        output_real, _ = netD(d_real + noise1, output_z.view(batch_size, latent_size, 1, 1))

        # Discriminator output for augmented reals D(E(x_aug), x)
        output_real_aug, _ = netD(d_real + noise1, output_z_aug.view(batch_size, latent_size, 1, 1))

        # Discriminator output for fakes D(z, G(z))
        output_fake, _ = netD(d_fake + noise2, z_fake)

        # Discriminator loss: -(alpha * log(D(E(x), x)) - (1-alpha) * log(D(E(x_aug), x_aug))) - log(1-D(z, G(z)))
        loss_d = (opt.alpha * criterion(output_real, real_label) + (1.0 - opt.alpha) * criterion(output_real_aug,
                                                                                                 real_label))
        loss_d += criterion(output_fake, fake_label)

        # Generator loss: -log(D(z, G(z))) - log(1-D(E(x), x))
        loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label)

        if loss_g.item() < 3.5:
            optimizerD.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizerD.step()

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

        if i % 1 == 0:
            print("Epoch :", epoch, "Iter :", i, "D Loss :", loss_d.item(), "G loss :", loss_g.item(),
                  "D(x) :", output_real.mean().item(), "D(G(x)) :", output_fake.mean().item())

        if i % 50 == 0:
            vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake.png' % (opt.save_image_dir))
            vutils.save_image(d_real.cpu().data[:16, ], './%s/real.png' % (opt.save_image_dir))

        i += 1

    if epoch % 25 == 0 or epoch == num_epochs - 1:
        torch.save(netG.state_dict(), './%s/netG_epoch_%d.pth' % (opt.save_model_dir, epoch))
        torch.save(netE.state_dict(), './%s/netE_epoch_%d.pth' % (opt.save_model_dir, epoch))
        torch.save(netD.state_dict(), './%s/netD_epoch_%d.pth' % (opt.save_model_dir, epoch))

        vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake_%d.png' % (opt.save_image_dir, epoch))
