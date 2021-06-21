import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AdaInstanceNorm1d(nn.Module):
    def __init__(self, num_features, c_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(c_dim, num_features * 2)

    def forward(self, x, c):
        h = self.fc(c)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, c_dim):
        super(AdaResBlk, self).__init__()
        self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, kernel_size=3, padding=1)
        self.in1 = AdaInstanceNorm1d(dim_out, c_dim)
        self.in2 = AdaInstanceNorm1d(dim_out, c_dim)
        self.actv1 = nn.ReLU()
        self.actv2 = nn.ReLU()

        if dim_in != dim_out:
            self.shortcut = nn.Conv1d(dim_in, dim_out, 1)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, input, label):
        out = self.conv1(input)
        out = self.in1(out, label)
        out = self.actv1(out)
        out = self.conv2(out)
        out = self.in2(out, label)
        out = self.actv2(out)
        out = (self.shortcut(input) + out) / math.sqrt(2)  # unit variance
        return out


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResBlk, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size=3, padding=1),
            nn.InstanceNorm1d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(dim_out, dim_out, kernel_size=3, padding=1),
            nn.InstanceNorm1d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        if dim_in != dim_out:
            self.shortcut = nn.Conv1d(dim_in, dim_out, 1)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, input):
        out = self.main(input)
        out = (self.shortcut(input) + out) / math.sqrt(2)  # unit variance
        return out


class Discriminator(nn.Module):
    """Discriminator network"""

    def __init__(self, c_dim, code_dim):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(code_dim, 128, 3, 1, 1, groups=2),
            nn.LeakyReLU(0.01),
            nn.Conv1d(128, 256, 2, 2, 0, groups=2),
            nn.LeakyReLU(0.01),
            nn.Conv1d(256, 512, 2, 2, 0, groups=1),
            nn.LeakyReLU(0.01),
            nn.Conv1d(512, 1024, 2, 2, 0, groups=1),
            nn.LeakyReLU(0.01),
            nn.Conv1d(1024, 2048, 2, 2, 0, groups=1),
            nn.LeakyReLU(0.01),
        )

        self.cls = nn.Conv1d(2048, c_dim, 1)
        self.src = nn.Conv1d(2048, 1, 1)

    def forward(self, codes):

        out_main = self.main(codes.transpose(2, 1))

        out_cls = self.cls(out_main)
        out_src = self.src(out_main)

        return out_src.view(out_src.size(0), -1), out_cls.view(out_cls.size(0), -1)


class Generator(nn.Module):
    """Generator network"""

    def __init__(self, c_dim, code_dim):
        super(Generator, self).__init__()
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv1d(code_dim, 128, 7, 1, 3),
                nn.InstanceNorm1d(128, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01),
                nn.Conv1d(128, 256, 4, 2, 1),
                nn.InstanceNorm1d(256, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01),
                nn.Conv1d(256, 512, 4, 2, 1),
                nn.InstanceNorm1d(512, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01),
            )
        )

        for i in range(3):
            layers.append(AdaResBlk(512, 512, c_dim))

        layers.append(
            nn.Sequential(
                nn.ConvTranspose1d(512, 512, 4, 2, 1),
                nn.InstanceNorm1d(512, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose1d(512, 256, 4, 2, 1),
                nn.InstanceNorm1d(256, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01),
                nn.Conv1d(256, code_dim, 1),
            )
        )

        self.main = nn.ModuleList(layers)

    def forward(self, codes, c_trg):

        out_f0 = codes.transpose(2, 1)
        out_f0 = self.main[0](out_f0)

        for layer in self.main[1:-1]:
            out_f0 = layer(out_f0, c_trg)

        out_f0 = self.main[-1](out_f0)
        out_f0 = out_f0.transpose(2, 1)

        return out_f0
