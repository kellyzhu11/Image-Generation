#reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'lsun', help='cifar10 | lsun | mnist')
parser.add_argument('--dataroot', default='./lsun', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--img_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--dim_z', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--dim_d', type=int, default=64, help='Size of feature maps in discriminator')
parser.add_argument('--dim_g', type=int, default=64, help='Size of feature maps in generator')
parser.add_argument('--num_epoch', type=int, default=20, help='number of epochs to train')
parser.add_argument('--netD_lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--netG_lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./results/lsun/bedroom', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', default = 123, type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--print_every', default=10, help='step of printing losses during training')
parser.add_argument('--save_img_every', default=100, help='step of saving images during training')


args = parser.parse_args()

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True


if args.dataset == 'lsun':
    classes = [ c + '_train' for c in args.classes.split(',')]
    print(classes)
    dataset = dset.LSUN(root=args.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(args.img_size),
                            transforms.CenterCrop(args.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif args.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(args.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif args.dataset == 'mnist':
    dataset = dset.MNIST(root=args.dataroot, download=True,
                        transform=transforms.Compose([
                            transforms.Resize(args.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))
    nc=1

# def show_images(images, real = True):
#     plt.figure(figsize=(8,8))
#     plt.axis("off")
#     if real:
#         plt.title("Training Images")
#     else:
#         plt.title("Generated Images")
#     plt.imshow(np.transpose(vutils.make_grid(images.to(device), padding=2, normalize=True).cpu(),(1,2,0)))  
#     plt.show()

# #plt.imshow(np.transpose(vutils.make_grid(fake_imgs.detach().cpu(), padding=2, normalize=True),(1,2,0)))
print('Finished reading data')
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.workers))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Plot some training images
# real_batch = next(iter(dataloader))
# show_images(real_batch[0])

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

dim_z = int(args.dim_z)
dim_d = int(args.dim_d)
dim_g = int(args.dim_g)

img_size = int(args.img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( dim_z, dim_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(dim_g * 8),
            nn.ReLU(True),
            # state size. (dim_g*8) x 4 x 4
            nn.ConvTranspose2d(dim_g * 8, dim_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_g * 4),
            nn.ReLU(True),
            # state size. (dim_g*4) x 8 x 8
            nn.ConvTranspose2d(dim_g * 4, dim_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_g * 2),
            nn.ReLU(True),
            # state size. (dim_g*2) x 16 x 16
            nn.ConvTranspose2d( dim_g * 2, dim_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_g),
            nn.ReLU(True),
            # state size. (dim_g) x 32 x 32
            nn.ConvTranspose2d( dim_g, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.model(input)
        return output


netG = Generator().to(device)
netG.apply(weights_init)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, dim_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim_d) x 32 x 32
            nn.Conv2d(dim_d, dim_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim_d*2) x 16 x 16
            nn.Conv2d(dim_d * 2, dim_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim_d*4) x 8 x 8
            nn.Conv2d(dim_d * 4, dim_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim_d*8) x 4 x 4
            nn.Conv2d(dim_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.model(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator().to(device)
netD.apply(weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))

criterion = nn.BCELoss()

fixed_noise = torch.randn(args.batch_size, dim_z, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.netD_lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.netG_lr, betas=(0.5, 0.999))

# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerG,
#                                                     milestones=[500, 750], 
#                                                     gamma=0.1,
#                                                     last_epoch=-1)

# Commented out IPython magic to ensure Python compatibility.
for epoch in range(args.num_epoch):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_imgs = data[0].to(device)
        batch_size = real_imgs.size(0)
        label = torch.full((batch_size,), real_label, dtype=real_imgs.dtype, device=device)

        output = netD(real_imgs)
        D_loss_real = criterion(output, label)
        D_loss_real.backward()
        D_score_real = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, dim_z, 1, 1, device=device)
        fake_imgs = netG(noise)
        #label.fill_(fake_label)
        label.fill_(fake_label)
        output = netD(fake_imgs.detach())
        
        D_loss_fake = criterion(output, label)
        D_loss_fake.backward()
        D_score_fake1 = output.mean().item()

        D_loss = D_loss_real + D_loss_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake_imgs)
        D_score_fake2 = output.mean().item()

        G_loss = criterion(output, label)
        G_loss.backward()
        
        optimizerG.step()
        # lr_scheduler.step()

        if i % args.print_every == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D_real: %.4f D_fake: %.4f/%.4f'
                  % (epoch+1, args.num_epoch, i, len(dataloader),
                    D_loss.item(), G_loss.item(), D_score_real, D_score_fake1, D_score_fake2))
    print('Saving images...')
    vutils.save_image(real_imgs,
            '%s/real_samples.png' % args.outf,
            normalize=True)
    with torch.no_grad():
        fake_imgs = netG(fixed_noise)
    # show_images(fake_imgs.detach(), real = False)
    vutils.save_image(fake_imgs.detach(),
            '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))



