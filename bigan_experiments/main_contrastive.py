import argparse
# from model import *
import os

import torch.optim as optim
import torchvision.utils as vutils
import wandb
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms

from cifar_dataset_mnist import CIFAR10_MNIST
from save_features import get_data_loaders, get_embeddings
import save_features
from classify_linear import eval_linear
from nearest_neighbour_acc_1 import eval_knn

lr = 1e-4
latent_size = 256
cuda_device = "0"


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x

save_features.tocuda = tocuda

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | svhn | cifar_mnist_mnist | timagenet | cifar_mnist_cifar',
                    choices=["cifar10", "svhn", "cifar_mnist_mnist", "timagenet", "cifar_mnist_cifar"])
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=boolean_string, default=True)
parser.add_argument('--cuda_device', type=str, default="0")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--projection_dim', type=int, default=256)
parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--beta', type=float, default=0.25)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--save_model_dir', required=True)
parser.add_argument('--save_image_dir', required=True)
parser.add_argument('--baseline', default=False, action="store_true")
parser.add_argument('--contrastive', default=False, action="store_true")
parser.add_argument('--only_pos', default=False, action="store_true")
parser.add_argument('--only_neg', default=False, action="store_true")


opt = parser.parse_args()
cuda_device = opt.cuda_device
batch_size = opt.batch_size
num_epochs = opt.num_epochs
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

if not opt.dataset == "timagenet":
    from model_contrastive import *
else:
    from model_timagenet import *

wandb.init(project="cs236g-bigan", entity="a7b23", dir='./wandb', config=opt)
print(opt)

if not os.path.exists(opt.save_image_dir):
    os.makedirs(opt.save_image_dir)

if not os.path.exists(opt.save_model_dir):
    os.makedirs(opt.save_model_dir)





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
        batch_size=batch_size, shuffle=True, num_workers=8)
elif opt.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
                         transform=TwoCropsTransformClean(augs)),
        batch_size=batch_size, shuffle=True, num_workers=8)
elif opt.dataset == 'cifar_mnist_cifar' or opt.dataset == 'cifar_mnist_mnist':
    train_loader = torch.utils.data.DataLoader(
        CIFAR10_MNIST(root=opt.dataroot, aug_type=1, train=True, download=False,
                      transform=TwoCropsTransformClean(augs)),
        batch_size=batch_size, shuffle=True, num_workers=8)
elif opt.dataset == "timagenet":
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=opt.dataroot,
                             transform=TwoCropsTransformClean(augs)),
        batch_size=batch_size, shuffle=True, num_workers=8)
else:
    raise NotImplementedError

netE = tocuda(Encoder(latent_size, opt.projection_dim, True))
netG = tocuda(Generator(latent_size))
netD = tocuda(Discriminator(latent_size, 0.2, 1))

netE.apply(weights_init)
netG.apply(weights_init)
netD.apply(weights_init)

optimizerG = optim.Adam([{'params': netE.parameters()},
                         {'params': netG.parameters()}], lr=lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

criterion_contrastive = nn.CrossEntropyLoss().cuda()

def get_perm(l):
    perm = torch.randperm(l)
    while torch.all(torch.eq(perm, torch.arange(l))):
        perm = torch.randperm(l)
    return perm

def add_contrastive_loss(feats, feats_aug, temperature):
    LARGE_NUM = 1e9
    batch_size = feats.size(0)

    feats = F.normalize(feats, dim=1)
    feats_aug = F.normalize(feats_aug, dim=1)

    masks = F.one_hot(torch.arange(batch_size), num_classes=batch_size).float().cuda()
    labels = torch.arange(batch_size).long().cuda()

    logits_aa = torch.matmul(feats, feats.T) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(feats_aug, feats_aug.T) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM

    logits_ab = torch.matmul(feats, feats_aug.T) / temperature
    logits_ba = torch.matmul(feats_aug, feats.T) / temperature

    loss_a = criterion_contrastive(torch.cat([logits_ab, logits_aa], axis=-1), labels)

    loss_b = criterion_contrastive(torch.cat([logits_ba, logits_bb], axis=-1), labels)

    loss = loss_a + loss_b

    return loss



for epoch in range(num_epochs):

    i = 0
    print(len(train_loader))
    for (data, target) in train_loader:
        step = epoch * len(train_loader) + i
        real_label = Variable(tocuda(torch.ones(batch_size)))
        fake_label = Variable(tocuda(torch.zeros(batch_size)))

        noise1 = Variable(tocuda(torch.Tensor(data[0].size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        noise2 = Variable(tocuda(torch.Tensor(data[0].size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        noise3 = Variable(tocuda(torch.Tensor(data[0].size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))

        if epoch == 0 and i == 0:
            netG.output_bias.data = get_log_odds(tocuda(data[0]))

        # print(data[0].size(), batch_size)
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
        z_real, z_real_projection = netE(d_real)
        z_real = z_real.view(batch_size, -1)

        mu, log_sigma = z_real[:, :latent_size], z_real[:, latent_size:]
        sigma = torch.exp(log_sigma)
        epsilon = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z = mu + epsilon * sigma

        # Sample E(x_aug) with mean and std learned from encoder
        z_real_aug, z_real_aug_projection = netE(d_real_aug)
        z_real_aug = z_real_aug.view(batch_size, -1)

        mu_aug, log_sigma_aug = z_real_aug[:, :latent_size], z_real_aug[:, latent_size:]
        sigma_aug = torch.exp(log_sigma_aug)
        epsilon_aug = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z_aug = mu_aug + epsilon_aug * sigma_aug

        # Discriminator output for reals D(E(x), x)
        output_real, _ = netD(d_real + noise1, output_z.view(batch_size, latent_size, 1, 1))
        # Discriminator output for augmented reals D(E(x_aug), x)
        output_real_aug, _ = netD(d_real + noise1, output_z_aug.view(batch_size, latent_size, 1, 1))

        # Discriminator output for fakes generated by shuffling reals reals D(E(x_1), x_2)
        shuff_indices = get_perm(d_real.size(0))
        d_real_fake = d_real.clone()[shuff_indices]
        output_real_fake, _ = netD(d_real_fake + noise3, output_z.view(batch_size, latent_size, 1, 1))

        # Discriminator output for fakes D(z, G(z))
        output_fake, _ = netD(d_fake + noise2, z_fake)

        # Discriminator loss:
        # -(1-alpha) * log(D(E(x), x)
        if opt.baseline:
            loss_d = criterion(output_real, real_label) + criterion(output_fake, fake_label)
        else:

            loss_d_1 = (1.0 - opt.alpha) * criterion(output_real, real_label)
            # -alpha * log(D(E(x_aug), x_aug)))
            loss_d_2 = opt.alpha * criterion(output_real_aug, real_label)
            # -beta * log(1-D(E(x_2), x_1)))
            loss_d_3 = opt.beta * criterion(output_real_fake, fake_label)
            # -(1-beta) * log(1-D(z, G(z)))
            loss_d_4 = (1.0 - opt.beta) * criterion(output_fake, fake_label)

            if opt.only_pos:
                loss_d = loss_d_1 + loss_d_2 + criterion(output_fake, fake_label)
            elif opt.only_neg:
                loss_d = loss_d_3 + loss_d_4 + criterion(output_real, real_label)
            else:
                loss_d = loss_d_1 + loss_d_2 + loss_d_3 + loss_d_4

        # Generator loss: -log(D(z, G(z))) - log(1-D(E(x), x))
        loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label)
        
        

        if loss_g.item() < 3.5:
            optimizerD.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizerD.step()

        if opt.contrastive:
            loss_contrastive = add_contrastive_loss(z_real_projection, z_real_aug_projection, opt.temperature)
            loss_g += loss_contrastive

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

        if i % 50 == 0:
            print("Epoch :", epoch, "Iter :", i, "D Loss :", loss_d.item(), "G loss :", loss_g.item(), 
                  "contrastive loss :", loss_contrastive.item(),
                  "D(x) :", output_real.mean().item(), "D(G(x)) :", output_fake.mean().item())
            wandb.log({"Epoch": epoch, "Iter": i, "D Loss": loss_d.item(), "G loss": loss_g.item(),
                       "contrastive loss": loss_contrastive.item(),
                       "D(x)": output_real.mean().item(), "D(G(x))": output_fake.mean().item()},
                      step=step)

        if i % 50 == 0:
            vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake.png' % (opt.save_image_dir))
            vutils.save_image(d_real.cpu().data[:16, ], './%s/real.png' % (opt.save_image_dir))
            wandb.log({'fakes': [wandb.Image(i) for i in d_fake.cpu().data[:16, ]],
                       'reals': [wandb.Image(i) for i in d_real.cpu().data[:16, ]]}, step=step)
        i += 1
    if epoch % 25 == 0 or epoch == num_epochs - 1:
        torch.save(netG.state_dict(), './%s/netG_epoch_%d.pth' % (opt.save_model_dir, epoch))
        torch.save(netE.state_dict(), './%s/netE_epoch_%d.pth' % (opt.save_model_dir, epoch))
        torch.save(netD.state_dict(), './%s/netD_epoch_%d.pth' % (opt.save_model_dir, epoch))

        vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake_%d.png' % (opt.save_image_dir, epoch))
        wandb.log({'fakes': [wandb.Image(i) for i in d_fake.cpu().data[:16, ]]}, step=step)


train_loader, test_loader = get_data_loaders(opt)

train_features, train_labels = get_embeddings(train_loader, netE, None)
test_features, test_labels = get_embeddings(test_loader, netE, None)

print("features inferred")

knn_acc = eval_knn(train_features, train_labels, test_features, test_labels)

logistic_acc = eval_linear('logistic', train_features, train_labels, test_features, test_labels)

print(f"KNN={knn_acc * 100:.2f}, Linear={logistic_acc * 100:.2f}")

wandb.log({"KNN acc ": knn_acc*100.0, "Linear acc ": logistic_acc * 100.0}, step=step+1)

