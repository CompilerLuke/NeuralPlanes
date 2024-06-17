from dataclasses import dataclass
from math import ceil

from NeuralPlanes.plane import Planes
import torch
from torch import nn

@dataclass
class NeuralMapConf:
    num_components: int = 4
    num_features_backbone: int = 712
    num_features: int = 32
    depth_sigma: float = 0.05


class EncoderMultiHead(nn.Module):
    def __init__(self, in_dim, out_dim, per_head=32, reduction=8):
        super().__init__()

        n_heads = in_dim // per_head

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(per_head, per_head // reduction),
                nn.LeakyReLU(),
            ) for i in range(n_heads)
        ])

        self.n_heads = n_heads
        self.per_head = per_head
        self.mlp = nn.Sequential(nn.Linear(in_dim // reduction, out_dim), nn.LeakyReLU())
    def forward(self, x):
        x_block = x.reshape((-1,self.n_heads,self.per_head))

        reduce = torch.cat([self.heads[i](x_block[:,i]) for i in range(self.n_heads)], dim=1)
        return self.mlp(reduce)

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        n_heads = int(ceil(in_dim / 64))

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.mlp(x)
class NeuralMap(nn.Module):
    conf: NeuralMapConf
    planes: Planes

    def __init__(self, planes: Planes, atlas_alpha: torch.Tensor, atlas_mu: torch.Tensor, atlas_var: torch.Tensor, atlas_weight: torch.Tensor, conf: NeuralMapConf):
        super().__init__()

        device = atlas_alpha.device

        self.planes = planes
        self.atlas_alpha = torch.nn.Parameter(atlas_alpha)
        self.atlas_mu = torch.nn.Parameter(atlas_mu)
        self.atlas_var = torch.nn.Parameter(atlas_var)
        self.atlas_weight = torch.nn.Parameter(atlas_weight)
        self.conf = conf
        self.encoder = Encoder(conf.num_features_backbone, conf.num_features).to(device)

    def n_components(self):
        return self.atlas_mu.shape[0]

    def n_features(self):
        return self.atlas_mu.shape[1]