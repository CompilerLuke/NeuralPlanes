from NeuralPlanes.plane import single_frustum_multiple_plane_visibility
from NeuralPlanes.camera import Camera, frustum_points, compute_ray
from NeuralPlanes.localization.map import NeuralMap
from NeuralPlanes import gmm
import torch

#@torch.compile
def score_cells(map: NeuralMap, plane_indices: torch.Tensor, coords: torch.Tensor, values: torch.Tensor, occupancy: torch.Tensor):
    """
    Computes the log-likelihood p_ij = Pr[ o_j, f_j | u_i, mu_i], where j = 0..n-1 is the feature index, i = 0..p-1 is the plane index
    :param map: NeuralMap
    :param plane_indices: (p, n)
    :param coords: (p, n, 2)
    :param values: (c, n)
    :param occupancy: n
    :return: (p, n)
    """

    n_components, n_features, atlas_height, atlas_width = map.atlas_mu.shape
    device = coords.device

    assert len(coords.shape) == 3 and coords.shape[0] == len(plane_indices), f"Expecting coords (p,n,2), got {coords.shape}, p={len(plane_indices)}"
    n = coords.shape[1]

    assert len(values.shape) == 2 and (values.shape[1]==1 or n == values.shape[1]) and values.shape[0] == map.conf.num_features, f"Expecting values (c,n), got {values.shape},c={map.conf.num_features}, n={n}"
    assert len(occupancy.shape) == 1 and (occupancy.shape[0]==1 or n == occupancy.shape[0]), f"Expecting occupancy (n), got {occupancy.shape}"

    planes = map.planes

    mask = torch.all((0 <= coords) & (coords <= 1), dim=2)
    coords = planes.coord_x0.to(device)[plane_indices][:,None].to(device) + planes.coord_size.to(device)[plane_indices][:,None] * coords
    coords = 2*(coords/ torch.tensor(map.planes.atlas_size,device=coords.device)[None,:]) - 1

    mu_cells = torch.nn.functional.grid_sample(map.atlas_mu.reshape((1,n_components*n_features, atlas_height, atlas_width)), coords[None])[0]
    var_cells = torch.nn.functional.grid_sample(map.atlas_var.reshape((1,n_components*n_features, atlas_height, atlas_width)), coords[None])[0]
    alpha_cells = torch.nn.functional.grid_sample(map.atlas_alpha.reshape((1,n_components, atlas_height, atlas_width)), coords[None])[0]
    weight_cells = torch.nn.functional.grid_sample(map.atlas_weight.reshape((1,1, atlas_height, atlas_width)), coords[None])[0,0]
    weight_cells = torch.where(mask, weight_cells, torch.tensor(0.))

    print("mask ", mask.type(torch.float).mean(), weight_cells.mean())

    def score_cell(weight, alpha, mu, var, value, occupancy):
        mixture = gmm.GaussianMixture(n_components=n_components, n_features=n_features, pi_init=alpha.reshape((1,n_components,1)), mu_init=mu.reshape((1,n_components, n_features)), var_init=var.reshape((1,n_components,n_features)), covariance_type='diag')
        log_prob = mixture.score_samples(occupancy[None], value[None], normalize=False)[0]
        return torch.where(weight > 0, log_prob, torch.tensor(0)), weight

    def score_plane(weight, alpha, mu, var, value, occupancy):
        return torch.vmap(score_cell, in_dims=(0,1,1,1,1,0))(weight, alpha, mu, var, value, occupancy)

    return torch.vmap(score_plane, in_dims=(0,1,1,1,None,None))(weight_cells, alpha_cells, mu_cells, var_cells, values, occupancy)

#@torch.compile
def score_pose(map: NeuralMap, cam: Camera, image: torch.Tensor, depths: torch.Tensor, samples: int = 5, dense=False):
    planes = map.planes

    device = image.device
    c,height,width = image.shape

    x0s = torch.zeros((len(planes), 2))
    x1s = torch.ones((len(planes), 2))
    visibility_mask = single_frustum_multiple_plane_visibility(map.planes, torch.arange(len(planes)), frustum_points(cam), x0s, x1s)

    plane_indices = torch.arange(len(planes),device=device)[visibility_mask]

    depth_sigma = map.conf.depth_sigma

    t_samples = torch.normal(depths[:,:,None].repeat(1,1,samples), depths[:,:,None].repeat(1,1,samples) * depth_sigma)

    orig, dir = compute_ray(width,height,cam,device)
    #print("Orig ", orig.shape, "dir", dir.shape, "t samples", t_samples.shape)
    pos = orig[:,:,None] + dir[:,:,None] * t_samples[:,:,:,None]

    x0s = planes.x0s.to(device)[plane_indices]
    us = planes.us.to(device)[plane_indices]
    vs = planes.vs.to(device)[plane_indices]
    us = us / torch.linalg.norm(us,dim=1)[:,None]**2
    vs = vs / torch.linalg.norm(vs,dim=1)[:,None]**2

    pos = pos.reshape((height*width*samples,3))

    coords = torch.stack([torch.einsum("ik,ijk->ij", us, pos[None] - x0s[:,None,:]), torch.einsum("ik,ijk->ij",vs,pos[None]-x0s[:,None,:])], dim=2)

    # don't weight by occupancy since samples are already normally distributed,
    # hence when computing cross-entropy p(x) is captured implicitly by the discrete set distribution
    occupancy = torch.full((height,width,samples,), 1.0, device=device) # / (height * width * samples))

    feature_samples = image[:,:,:,None].repeat(1,1,1,samples).reshape((c,height*width*samples))
    occupancy_samples = occupancy.reshape(height*width*samples)

    if len(plane_indices) == 0:
        if dense:
            return torch.zeros((height,width,samples), device=image.device), torch.zeros((height,width,samples), device=image.device),  feature_samples.reshape((c,height,width,samples))
        else:
            return torch.tensor(0., device=device)

    #print("Scoring cells planes= ", len(plane_indices))
    log_prob, weight = score_cells(map, plane_indices, coords, feature_samples, occupancy_samples)
    log_prob = log_prob.reshape((len(plane_indices),height,width,samples))
    weight = weight.reshape((len(plane_indices),height,width,samples))

    print("log prob mean ", log_prob.mean(), "weight mean", weight.mean())

    prob = (torch.exp(log_prob) * weight).sum(dim=0) / (1e-9 + weight.sum(dim=0))
    weight = weight.sum(dim=0)

    #prob = torch.exp(log_prob)

    if dense:
        return prob, weight, pos.reshape((height,width,samples,3)), feature_samples.reshape((c,height,width,samples))
    else:
        return torch.sum(prob * weight) / (1e-9 + torch.sum(weight))