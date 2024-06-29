from dataclasses import dataclass
from math import ceil

from NeuralPlanes.plane import Planes
import torch
from torch import nn

from dataclasses import dataclass
from NeuralPlanes.plane import Planes
import torch
from torch import nn


from NeuralPlanes.clustering.splat import GaussianCluster

@dataclass
class NeuralMapConf:
    num_components: int = 4
    num_features: int = 32 
    num_cluster_init: int = 4
    depth_sigma: float = 0.05
    importance_alpha: float = 0.3

    @property
    def cell_dim(self):
        cluster = self.make_cluster()
        return cluster.as_vec_dim()

    def make_cluster(self):
        return GaussianCluster(num_components=self.num_components, num_features=self.num_features)

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.alpha = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, z, model_first=False):
        return x # todo: re-enable positional encoding
        """
        Args:
            x: Tensor of shape (batch_size, d_model), or (d_model,batch_size)
            z: Tensor of shape (batch_size), or (x,y)
        """

        if self.d_model%2 != 0: 
            return x # Model dim must be divisible by 2

        d_model = self.d_model
        div_term = torch.pi / 100.0 * torch.exp(torch.arange(0, d_model, 2, device=x.device)).float()

        if model_first:
            a = torch.sin(z[None,:] * div_term[:,None]) # (d_model//2,batch_size)
            b = torch.cos(z[None,:] * div_term[:,None]) # (d_model//2,batch_size)
            pe = torch.stack((a,b), dim=1) # (d_model//2,2,batch_size)
            pe = pe.reshape(d_model, pe.shape[2])
        else:
            a = torch.sin(z[:,None] * div_term[None, :])  # (batch_size, d_model//2)
            b = torch.cos(z[:,None] * div_term[None, :])  # (batch_size, d_model//2)
            pe = torch.stack((a, b), dim=2)  # (batch_size, d_model//2, 2)
            pe = pe.reshape(x.shape[0], d_model)
            
        x = x + torch.sigmoid(self.alpha) * pe
        return x

class AttentionMask(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.BatchNorm2d(num_features+1),
            nn.Conv2d(num_features+1, 2*num_features, (3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*num_features),
            nn.Conv2d(2*num_features, num_features, (3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(num_features, 1, (3,3), padding=(1,1))
        )

    def forward(self, x):
        return 0.3 + 0.7*torch.sigmoid(self.mlp(x))[:,0]


class ClusterWeighter(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.num_features = num_features

        in_dim = 2*num_features + 2

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 4*num_features),
            nn.LeakyReLU(),
            nn.Linear(4*num_features, num_features),
            nn.LeakyReLU(),
            nn.Linear(num_features, num_features),
            nn.LeakyReLU(),
            nn.Linear(num_features,1),
        )

    def forward(self, mu, var, occupancy, up):
        flatten_components = False

        if len(mu.shape) == 3: # (height, num_components, num_features)
            height, num_components, num_features = mu.shape
            assert num_features == self.num_features
            mu = mu.reshape(height*num_components, num_features)
            var = var.reshape(height*num_components, num_features)
            occupancy = occupancy.reshape(height*num_components, 1)
            up = up.reshape(height*num_components, 1)
            flatten_components = True

        # (batch_size, num_features)
        x = torch.cat([mu, var, occupancy, up], dim=1)
        x = self.mlp(x)[:,0]
        x = torch.sigmoid(x + occupancy[:,0]) # bias the network to assign high occupancy, high weights

        if flatten_components:
            x = x.reshape((height, num_components))

        return x

class NeuralMap(nn.Module):
    conf: NeuralMapConf
    planes: Planes

    def __init__(self, planes: Planes, conf: NeuralMapConf, atlas: torch.Tensor = None, atlas_weight: torch.Tensor = None, encoder=None):
        super().__init__()

        if encoder is None:
            encoder = Encoder(3, conf.num_features)

        if atlas is None:
            width, height = planes.atlas_size
            atlas = torch.zeros((conf.cell_dim, height,width))
            atlas_weight = torch.zeros((height, width))

        self.planes = planes
        self.register_buffer('atlas', atlas)
        self.register_buffer('atlas_weight', atlas_weight)
        self.conf = conf
        self.encoder = encoder

        self.positional = PositionalEncoding(d_model=conf.num_features)
        self.attention_mask = AttentionMask(conf.num_features)
        self.cluster_weighter = ClusterWeighter(conf.num_features)

    @property
    def atlas_mu(self):
        def get_mu(feature):
            cluster = self.conf.make_cluster()
            cluster.from_vector(feature)
            return cluster.features

        def vmap1d(fn):
            return torch.vmap(fn, in_dims=(1,), out_dims=(2,))
        
        result = vmap1d(vmap1d(get_mu))(self.atlas) 
        return result 
    
    def encode(self, image, depth):
        features = self.encoder(image)["image"]

        attention, features = features[:,0], features[:,1:]
        attention = torch.sigmoid(attention)
        features = features / (1e-9 + torch.linalg.norm(features, dim=1, keepdim=True))

        #joint_features = torch.cat([depth[:,None] / 20.0, features], dim=1)
        #attention = self.attention_mask(joint_features)

        return torch.cat([attention[:,None], features], dim=1)

    def n_components(self):
        return self.atlas_mu.shape[0]

    def n_features(self):
        return self.atlas_mu.shape[1]