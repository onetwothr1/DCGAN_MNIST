import torch
import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, nc, nz, ngf):
    super(Generator, self).__init__()

    # out_size = (in_size − 1) × stride − 2 × padding + kernel_size

    self.layers = nn.Sequential(
        # nz X 1 X 1
        nn.ConvTranspose2d(nz, ngf * 4, kernel_size=4, stride=1, padding=0),
        nn.BatchNorm2d(ngf * 4, affine=False),
        nn.ReLU(),
        # (ngf*4) X 4 X 4

        nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(ngf * 2, affine=False),
        nn.ReLU(),
        # (ngf*2) X 7 X 7

        nn.ConvTranspose2d(ngf * 2, ngf,  kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(ngf, affine=False),
        nn.ReLU(),
        # (ngf) X 14 X 14

        nn.ConvTranspose2d(ngf, nc,  kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
        # nc X 28 X 28
    )
  
  def forward(self, x):
    return self.layers(x)

  def load(self, model_path):
      self.load_state_dict(torch.load(model_path))


class Discriminator(nn.Module):
  def __init__(self,nc, ndf):
    super(Discriminator, self).__init__()

    # out_size = (in_size - kernel_size + 2 * padding) / stride + 1

    self.layers = nn.Sequential(
        # nc X 28 X 28
        nn.Conv2d(nc, ndf, 4, 2, 1),
        nn.LeakyReLU(0.2),
        # ndf X 14 X 14

        nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
        nn.BatchNorm2d(ndf * 2, affine=False),
        nn.LeakyReLU(0.2),
        # (ndf*2) X 7 X 7

        nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1),
        nn.BatchNorm2d(ndf * 4, affine=False),
        nn.LeakyReLU(0.2),
        # (ndf*4) X 4 X 4
    
        nn.Conv2d(ndf * 4, 1, 4, 1, 0),
        nn.Sigmoid()
        # 1 X 1 X 1
    )

  def forward(self, x):
    return self.layers(x).squeeze()

  def load(self, model_path):
      self.load_state_dict(torch.load(model_path))



      