from NeuralPlanes.localization.map import NeuralMap
import torch
from NeuralPlanes.clustering.splat import GaussianCluster
from jaxtyping import Float, jaxtyped
from torch import Tensor 

@jaxtyped
def pool_images(map: NeuralMap, 
               masks: Float[Tensor, "height width length"], 
               images: Float[Tensor, "c height width length"], 
               occupancy: Float[Tensor, "height width length"], 
               z: Float[Tensor, "height width length"], 
               up):
    n_components, n_features = map.conf.num_components, map.conf.num_features
    device = images.device
    importance, values = images[0], images[1:]

    zero = torch.tensor(0., device=device)
    values = torch.where(masks[None,:,:], values, zero)
    z = torch.where(masks, z, zero)
    sample_weight = torch.where(masks, importance * occupancy, zero)

    def cluster_column(z, sample_weight, features):
        features = features.transpose(0,1)

        cluster = map.conf.make_cluster()
        cluster.fit(z, sample_weight, features)
        return sample_weight.max(), cluster.as_vector()
    
    def map1d(func):
        return torch.vmap(func, in_dims=(0,0,1), out_dims=(0,1,), randomness='different')

    weight, vec = map1d(map1d(cluster_column))(z, sample_weight, values)
    return weight, vec