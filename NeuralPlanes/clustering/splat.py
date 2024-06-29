import torch
from typing import Callable
import functorch
from torch import nn

class GaussianCluster:
    mu: torch.Tensor
    var: torch.Tensor
    features: torch.Tensor
    dist_fn: Callable[[torch.Tensor], torch.Tensor]

    reduction: str
    initialized: bool
    num_components: int

    eps: float

    def __init__(self, num_components, num_features, dist: str = 'l2', reduction='avg', eps=1e-6):
        l2_dist = lambda x, y, dim: torch.linalg.norm(x - y, dim=dim)
        cos_dist = lambda x, y, dim: 1 - torch.maximum(torch.cosine_similarity(x, y, dim=dim),
                                                       torch.tensor(0., device=x.device))

        if dist == 'l2':
            dist_fn = l2_dist
        elif dist == 'cos':
            dist_fn = cos_dist
        else:
            raise NotImplemented()

        self.num_components = num_components
        self.num_features = num_features
        self.reduction = reduction
        self.dist_fn = dist_fn
        self.initialized = False
        self.eps = eps

    def _sample(self, mu, var, features, z_in):
        eps = self.eps

        kernel = torch.exp(-(z_in[None, :] - mu[:, None]) ** 2 / var[:, None])
        weight = kernel.sum(dim=0)
        kernel_norm = kernel / (eps + weight[None, :])
        features_interp = (features[:, None] * kernel_norm[:, :, None]).sum(dim=0)
        return features_interp

    def sample(self, z_in):
        return self._sample(self.mu, self.var, self.features, z_in)

    def _loss(self, mu, var, features, z_in, importance_in, features_in, dist_weight = 1e-2):
        dist, eps = self.dist_fn, self.eps

        kernel = torch.exp(-(z_in[None, :] - mu[:, None]) ** 2 / var[:, None])
        weight = kernel.sum(dim=0)
        kernel_norm = kernel / (eps + weight[None, :])
        weight_norm = weight / (eps + weight.sum())

        features_interp = (features[:, None] * kernel_norm[:, :, None]).sum(dim=0)
        importance_norm = importance_in / (eps + importance_in.sum())

        loss_feature = (importance_norm * dist(features_in, features_interp, dim=1)).sum()
        loss_density = -dist_weight * (importance_norm * torch.log(weight_norm)).sum()

        return loss_feature + loss_density

    def loss(self, z_in, importance_in, features_in, dist_weight = 1e-2):
        mu, var, features = self.mu, self.var, self.features
        return self._loss(mu, var, features, z_in, importance_in, features_in)

    def init(self, z_in, importance_in, features_in):
        num_components = self.num_components

        min_z, max_z = z_in.min(), z_in.max()
        n_bins = num_components*4
        bins = torch.linspace(min_z, max_z, n_bins+1, device=z_in.device)
        hist = torch.histc(z_in, bins=n_bins) # min=min_z, max=max_z)

        initial = torch.multinomial(hist, num_samples=num_components)
        mu = 0.5+(bins[initial]+bins[initial+1]) 

        var = torch.full_like(mu, 0.5)
        self.mu = nn.Parameter(mu)
        self.var = nn.Parameter(var)

        self.initialized = True

    def assign_features(self, z_in, importance_in, features_in, ):
        mu, var, reduction = self.mu, self.var, self.reduction

        kernel = torch.exp(-((z_in[None, :] - mu[:, None])) ** 2 / var[:, None])

        if reduction == 'max':
            assignment = torch.softmax(5 * kernel, dim=0)
            mask = assignment > 0.8
            features_masked = torch.where(mask[:, :, None], features_in[None], torch.tensor(0., device=mu.device))
            features = features_masked.max(dim=1)[0]
        elif reduction == 'avg':
            kernel = kernel / (1e-9 + kernel.sum(dim=1, keepdim=True))
            features = (kernel[:, :, None] * features_in[None]).sum(dim=1)
        elif reduction == 'none':
            features = None
        else:
            raise NotImplemented()

        self.features = features

    def fit(self, z_in, importance_in, features_in, n_steps=30, lr=1e-1, grad=False):
        if not self.initialized:
            self.init(z_in, importance_in, features_in)

        mu, var = self.mu, self.var

        loss_fn = functorch.grad_and_value(self._loss, argnums=(0, 1))

        with torch.enable_grad() if grad else torch.no_grad():
            for i in range(n_steps):
                self.assign_features(z_in, importance_in, features_in)
                grad, loss = loss_fn(mu, var, self.features, z_in, importance_in, features_in)
                mu = mu - lr * grad[0]
                var = var - lr * grad[1]

        self.assign_features(z_in, importance_in, features_in)

        return self

    def as_vec_dim(self):
        return self.num_components * (self.num_features + 2) 

    def as_vector(self):
        mu, var, features = self.mu, self.var, self.features
        return torch.cat([mu[:,None], var[:,None], features], dim=1).flatten()
    
    def from_vector(self, vec):
        num_components, num_features = self.num_components, self.num_features
        vec = vec.reshape((num_components, num_features+2))
        self.mu = vec[:,0]
        self.var = vec[:,1]
        self.features = vec[:,2:]