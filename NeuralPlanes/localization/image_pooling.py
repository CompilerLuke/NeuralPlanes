from NeuralPlanes import gmm
from NeuralPlanes.localization.map import NeuralMap, NeuralMapConf
import torch
from torch_kmeans import SoftKMeans

def pool_images(map: NeuralMap, masks, images, occupancy, t, weight_init=None, mu_init=None, var_init=None):
    map_conf = map.conf
    n_components = map_conf.num_components
    n_features = map_conf.num_features
    c,n_height,n_width,n_len = images.shape
    device = images.device

    assert c == n_features+1

    assert map_conf.cluster_ratio >= 1.0

    images = images.permute(1,2,3,0) # Soft-kmeans expects feature last, whereas torch.grid_sample expects it first
    images = images.reshape(n_height*n_width,n_len,c)
    occupancy = occupancy.reshape((n_height*n_width,n_len))
    masks = masks.reshape((n_height*n_width,n_len))

    importance, values = images[:, :, 0], images[:, :, 1:]

    n_clusters = int(map_conf.cluster_ratio * map_conf.num_components)
    if n_clusters > 1:
        soft_k_means = SoftKMeans(normalize=None, n_clusters= n_clusters) # features are already unit-normed
        mu = soft_k_means(values).centers # (n,k,d)
    else:
        mu = values.mean(dim=1, keepdim=True)
    
    importance_norm = importance / (1e-9 + importance.sum(dim=1, keepdim=True))
    var = (importance_norm[:,None,:,None] * (values[:,None,:] - mu[:,:,None])**2).sum(dim=2) # todo: use importance, or just 1/n

    cluster_weight = torch.maximum(torch.cosine_similarity(values[:,None,:], mu[:,:,None], dim=3), torch.tensor(0.,device=device))
    cluster_weight = cluster_weight / (1e-7 + torch.sum(cluster_weight,dim=2,keepdim=True))
    occupancy_cluster = torch.sum(cluster_weight * occupancy[:,None,:], dim=2)

    alpha =  map.cluster_weighter(mu, var, occupancy_cluster[:,:,None])
    indices = torch.topk(alpha, k=map_conf.num_components, dim=1)[1]

    cell_weight = torch.max(alpha, dim=1)[0]

    indices_repeat = indices[:,:,None].repeat(1,1,n_features)

    return cell_weight.reshape(n_height,n_width),\
        torch.gather(alpha,1,indices).permute(1,0).reshape((n_components,n_height,n_width)),\
        torch.gather(mu,1,indices_repeat).permute(1,2,0).reshape((n_components,n_features,n_height,n_width)),\
        torch.gather(var,1,indices_repeat).permute(1,2,0).reshape((n_components,n_features,n_height,n_width))