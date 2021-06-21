import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Module):
    """Discriminator network"""

    def __init__(self, dim_h=1024, n_h=1):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Linear(256, dim_h))
        layers.append(nn.LeakyReLU(0.02))
        for i in range(n_h):
            layers.append(nn.Linear(dim_h, dim_h))
            layers.append(nn.LeakyReLU(0.02))

        self.main = nn.Sequential(*layers)
        self.src = nn.Linear(dim_h, 1)
        self.cls = nn.Linear(dim_h, 2)

    def forward(self, x):
        out = self.main(x)
        out_src = self.src(out)
        out_cls = self.cls(out)
        return out_src, out_cls


class Generator(nn.Module):
    """Generator network"""

    def __init__(self, dim_z=64, dim_h=1024, n_h=1):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Linear(dim_z + 2, dim_h))
        layers.append(nn.LeakyReLU(0.02))
        for i in range(n_h):
            layers.append(nn.Linear(dim_h, dim_h))
            layers.append(nn.LeakyReLU(0.02))
        layers.append(nn.Linear(dim_h, 256))

        self.main = nn.Sequential(*layers)

    def label2onehot(self, labels):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, 2)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def forward(self, z, c):
        input = torch.cat([z, self.label2onehot(c).to(z.dtype).to(z.device)], dim=-1)
        x = torch.relu(self.main(input))
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x
