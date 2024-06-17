from NeuralPlanes import gmm
from NeuralPlanes.localization.map import NeuralMapConf
import torch


def pool_images(map_conf: NeuralMapConf, masks, images, occupancy, t, weight_init=None, mu_init=None, var_init=None):
    n_components = map_conf.num_components
    n_features = map_conf.num_features
    c,n_height,n_width,n_len = images.shape
    device = images.device

    assert c == n_features+1

    def gaussian_mixture(mask, values, occupancy, mu_init, var_init):
        g = gmm.GaussianMixture(n_components=n_components, n_features=n_features, mu_init=mu_init[None], var_init=var_init[None], covariance_type="diag")

        weight, values = values[:,0], values[:,1:]
        
        image_weight = weight * occupancy
        image_weight_norm = image_weight / (1e-9 + torch.sum(image_weight))
        
        g.fit(image_weight_norm.unsqueeze(1), values, n_iter=10)

        cell_weight = torch.max(occupancy)
        
        #x: torch.Tensor(n, 1, d)
        #mu: torch.Tensor(1, k, d)
        #var: torch.Tensor(1, k, d) or (1, k, d, d)
        #pi: torch.Tensor(1, k, 1)

        zero = torch.tensor(0.,device=device)
        one = torch.tensor(1.,device=device)
        mask = cell_weight > 0
        return cell_weight, torch.where(mask,  g.pi.squeeze(0).squeeze(1), one/n_components), torch.where(mask, g.mu.squeeze(0), zero), torch.where(mask, g.var.squeeze(0), one)

    def initial_params(mask, values, occupancy):
        indices = torch.multinomial(occupancy+1e-9, n_components, replacement= n_components >= len(occupancy))
        mu_init = values[indices][:, 1:]

        weight, values = values[:,0], values[:,1:]
        image_weight = weight * occupancy
        image_weight_norm = image_weight / (1e-9 + torch.sum(image_weight))

        var_init = torch.sum(image_weight_norm[None,:,None] * (values[None,:] - mu_init[:,None])**2, dim=1)
        var_init = var_init + 1e-1

        return mu_init, var_init

    def update_params(mask, values, occupancy, weight_prev, mu_prev, var_prev):
        init = weight_prev < 1e-3 # todo: more sophisticated reinitialization decision
        mu_init, var_init = initial_params(mask, values, occupancy)
        return mu_init, var_init

        init_mask = init[None, None]
        return torch.where(init_mask, mu_init, mu_prev), torch.where(init_mask, var_init, var_prev)

    images = images.permute(1,2,3,0)
    images = images.reshape(n_height*n_width,n_len,c)
    occupancy = occupancy.reshape((n_height*n_width,n_len))
    masks = masks.reshape((n_height*n_width,n_len))

    if mu_init is None or var_init is None:
        mu_init, var_init = torch.vmap(initial_params, randomness='different')(masks, images, occupancy)
    else:
        weight_init = weight_init.reshape((n_height*n_width,))
        mu_init = mu_init.permute(2,3,0,1).reshape((n_height*n_width,n_components,n_features))
        var_init = var_init.permute(2,3,0,1).reshape((n_height*n_width,n_components,n_features))

        mu_init, var_init = torch.vmap(update_params, randomness='different')(masks, images, occupancy, weight_init, mu_init, var_init)

    cell_weight, alpha, mu, var = torch.vmap(gaussian_mixture, randomness='error')(masks, images, occupancy, mu_init, var_init)

    return cell_weight.reshape(n_height,n_width), \
           alpha.permute(1,0).reshape((n_components,n_height,n_width)), \
           mu.permute(1,2,0).reshape((n_components,n_features,n_height,n_width)), \
           var.permute(1,2,0).reshape((n_components,n_features,n_height,n_width))
