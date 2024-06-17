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
    cluster_ratio: float = 2.0

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, z, model_first=False):
        """
        Args:
            x: Tensor of shape (batch_size, d_model), or (d_model,batch_size)
            z: Tensor of shape (batch_size), or (x,y)
        """

        if self.d_model%2 != 0: 
            return x # Model dim must be divisible by 2

        d_model = self.d_model
        div_term = torch.exp(torch.arange(0, d_model, 2)).float() 
        # * (-torch.log(torch.tensor(10000.0)) / d_model)).to(x.device)

        if model_first:
            pe = torch.zeros((d_model, x.shape[1]), device=x.device)
            
            a = torch.sin(z * div_term[:,None]) # (d_model//2,batch_size)
            b = torch.cos(z * div_term[:,None]) # (d_model//2,batch_size)
            pe = torch.stack((a,b), dim=1) # (d_model//2,2,batch_size)
            pe = pe.reshape(d_model, pe.shape[2])
        else: 
            pe = torch.zeros((len(x), d_model), device=x.device)
            pe[:, 0::2] = torch.sin(z * div_term[None,:])
            pe[:, 1::2] = torch.cos(z * div_term[None,:])
            
        x = x + 0.1*torch.sigmoid(self.alpha) * pe
        return x

class ClusterWeighter(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.num_features = num_features

        in_dim = 2*num_features + 1 

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 4*num_features),
            nn.LeakyReLU(),
            nn.Linear(4*num_features, num_features),
            nn.LeakyReLU(),
            nn.Linear(num_features, num_features),
            nn.LeakyReLU(),
            nn.Linear(num_features,1),
        )

    def forward(self, mu, var, occupancy):
        flatten_components = False

        if len(mu.shape) == 3: # (height, num_components, num_features) 
            height, num_components, num_features = mu.shape
            assert num_features == self.num_features
            mu = mu.reshape(height*num_components, num_features)
            var = var.reshape(height*num_components, num_features)
            occupancy = occupancy.reshape(height*num_components, 1)
            flatten_components = True

        # (batch_size, num_features)
        x = torch.cat([mu, var, occupancy], dim=1)
        x = self.mlp(x)[:,0]
        x = torch.sigmoid(x + occupancy[:,0]) # bias the network to assign high occupancy, high weights

        if flatten_components:
            x = x.reshape((height, num_components))

        return x

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

        self.positional = PositionalEncoding(d_model=conf.num_features)
        self.cluster_weighter = ClusterWeighter(conf.num_features)
     

    def n_components(self):
        return self.atlas_mu.shape[0]

    def n_features(self):
        return self.atlas_mu.shape[1]