from dataclasses import dataclass
from NeuralPlanes.plane import Planes
import torch
from torch import nn

@dataclass
class NeuralMapConf:
    num_components: int = 4
    num_features_backbone: int = 32
    num_features: int = 32
    depth_sigma: float = 0.05
    importance_alpha: float = 0.3

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=0.7):
        super().__init__()

        self.alpha = alpha
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, (1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(out_dim, out_dim+1, (1,1)),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = x["image"]
        x = self.mlp(x)
        return { "image": x}

def normalize_features(map_conf: NeuralMapConf, x):
    alpha = map_conf.importance_alpha
    score, x = x[:,0], x[:,1:]
    importance = (1-alpha) + alpha*torch.sigmoid(score)[:,None]
    return torch.cat([importance, x / torch.linalg.norm(x+1e-9, dim=1, keepdim=True)], dim=1)

class NeuralMap(nn.Module):
    conf: NeuralMapConf
    planes: Planes

    def __init__(self, planes: Planes, atlas_alpha: torch.Tensor, atlas_mu: torch.Tensor, atlas_var: torch.Tensor, atlas_weight: torch.Tensor, conf: NeuralMapConf, encoder=None):
        super().__init__()

        if encoder is None:
            encoder = Encoder(conf.num_features_backbone, conf.num_features)

        device = atlas_alpha.device

        self.planes = planes
        self.atlas_alpha = torch.nn.Parameter(atlas_alpha)
        self.atlas_mu = torch.nn.Parameter(atlas_mu)
        self.atlas_var = torch.nn.Parameter(atlas_var)
        self.atlas_weight = torch.nn.Parameter(atlas_weight)
        self.conf = conf
        self.encoder = encoder 

    def n_components(self):
        return self.atlas_mu.shape[0]

    def n_features(self):
        return self.atlas_mu.shape[1]