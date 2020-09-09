import torch.nn as nn

from utils import get_conf


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.cfg = get_conf("conf/model/generator")
        self.gen = nn.Sequential(
            # input is Z, going into a convolution
            nn.utils.spectral_norm(nn.ConvTranspose2d(self.cfg.nz, self.cfg.ngf * 8, 4, 1, 0, bias=False)),
            nn.ReLU(True),
            nn.BatchNorm2d(self.cfg.ngf * 8),
            # state size. (ngf*8) x 4 x 4
            nn.utils.spectral_norm(nn.ConvTranspose2d(self.cfg.ngf * 8, self.cfg.ngf * 4, 4, 2, 1, bias=False)),
            nn.ReLU(True),
            nn.BatchNorm2d(self.cfg.ngf * 4),
            # state size. (ngf*4) x 8 x 8
            nn.utils.spectral_norm(nn.ConvTranspose2d(self.cfg.ngf * 4, self.cfg.ngf * 2, 4, 2, 1, bias=False)),
            nn.ReLU(True),
            nn.BatchNorm2d(self.cfg.ngf * 2),
            # state size. (ngf*2) x 16 x 16
            nn.utils.spectral_norm(nn.ConvTranspose2d(self.cfg.ngf * 2, self.cfg.ngf, 4, 2, 1, bias=False)),
            nn.ReLU(True),
            nn.BatchNorm2d(self.cfg.ngf),
            # state size. (ngf) x 32 x 32
            nn.utils.spectral_norm(nn.ConvTranspose2d(self.cfg.ngf, self.cfg.nc, 4, 2, 1, bias=False)),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.gen(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cfg = get_conf("conf/model/discriminator")
        self.dis = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.utils.spectral_norm(nn.Conv2d(self.cfg.nc, self.cfg.ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.utils.spectral_norm(nn.Conv2d(self.cfg.ndf, self.cfg.ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.cfg.ndf * 2),
            # state size. (ndf*2) x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(self.cfg.ndf * 2, self.cfg.ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.cfg.ndf * 4),
            # state size. (ndf*4) x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(self.cfg.ndf * 4, self.cfg.ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.cfg.ndf * 8),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.cfg.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.dis(input)