from NeuralPlanes.plane import single_frustum_multiple_plane_visibility
from NeuralPlanes.camera import Camera, frustum_points, compute_ray
from NeuralPlanes.localization.map import NeuralMap
from NeuralPlanes import gmm
import torch

#@torch.compile
def score_cells(map: NeuralMap, plane_indices: torch.Tensor, coords: torch.Tensor, values: torch.Tensor, occupancy: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
    """
    Computes the log-likelihood p_ij = Pr[ o_j, f_j | u_i, mu_i], where j = 0..n-1 is the feature index, i = 0..p-1 is the plane index
    :param map: NeuralMap
    :param plane_indices: (p,)
    :param coords: (p, n, 3)
    :param values: (c, n)
    :param occupancy: n
    :param x0: (p,2)
    :param x1: (p,2)
    :return: (p, n)
    """
    n_planes = len(plane_indices)
    n_components, n_features, atlas_height, atlas_width = map.atlas_mu.shape
    device = coords.device

    if x0 is None:
        x0 = torch.zeros((n_planes,2), device=device)
    else:
        x0 = x0.to(device)
    if x1 is None:
        x1 = torch.ones((n_planes,2), device=device)
    else:
        x1 = x1.to(device)
        
    assert len(coords.shape) == 3 and coords.shape[0] == len(plane_indices) and coords.shape[2]==3, f"Expecting coords (p,n,3), got {coords.shape}, p={len(plane_indices)}"
    n = coords.shape[1]

    assert len(values.shape) == 2 and (values.shape[1]==1 or n == values.shape[1]) and values.shape[0] == map.conf.num_features+1, f"Expecting values (c+1,n), got {values.shape},c={map.conf.num_features}, n={n}"
    assert len(occupancy.shape) == 1 and (occupancy.shape[0]==1 or n == occupancy.shape[0]), f"Expecting occupancy (n), got {occupancy.shape}"

    planes = map.planes

    mask = torch.all((x0[:,None,:] <= coords[:,:,0:2]) & (coords[:,:,0:2] <= x1[:,None,:]), dim=2)

    coords2 = coords[:,:,0:2]
    z = coords[:,:,2]
    coords2 = planes.coord_x0.to(device)[plane_indices][:,None].to(device) + planes.coord_size.to(device)[plane_indices][:,None] * coords2
    coords2 = 2*(coords2/ torch.tensor(map.planes.atlas_size,device=coords2.device)[None,:]) - 1

    #values = map.positional(values[:,None].reshape(n_features, ), z, model_first=False)

    mu_cells = torch.nn.functional.grid_sample(map.atlas_mu.reshape((1,n_components*n_features, atlas_height, atlas_width)), coords2[None])[0]
    var_cells = torch.nn.functional.grid_sample(map.atlas_var.reshape((1,n_components*n_features, atlas_height, atlas_width)), coords2[None])[0]
    alpha_cells = torch.nn.functional.grid_sample(map.atlas_alpha.reshape((1,n_components, atlas_height, atlas_width)), coords2[None])[0]
    weight_cells = torch.nn.functional.grid_sample(map.atlas_weight.reshape((1,1, atlas_height, atlas_width)), coords2[None])[0,0]
    weight_cells = torch.where(mask, weight_cells, torch.tensor(0.))

    mu_cells = torch.where(mask, mu_cells, torch.tensor(0., device=device))
    var_cells = torch.where(mask, var_cells, torch.tensor(1., device=device))
    alpha_cells = torch.where(mask, alpha_cells, torch.tensor(1. / n_components, device=device))
    weight_cells = torch.where(mask, weight_cells, torch.tensor(0., device=device))

    def score_plane(z, weight, alpha, mu, var, value, occupancy):
        n = value.shape[1]
        mu = mu.reshape((n_components,n_features,n))
        var = var.reshape((n_components,n_features,n))
        alpha = alpha.reshape((n_components,n))
        weight_cells = weight.reshape((n,))

        feature_weight, value = value[0], value[1:] 

        value = map.positional(value, z, model_first=True)
        scores = torch.maximum(torch.cosine_similarity(mu[:,:], value[None,:,:], dim=1), torch.tensor(0.,device=device)) # (mu=(n_c,n_f,n), value=(c,n))
        score, top_index = scores.max(dim=0)

        score = alpha.gather(0, top_index[None,:])[:,0] * score
        return score, weight

    return torch.vmap(score_plane, in_dims=(0,0,1,1,1,None,None))(z, weight_cells, alpha_cells, mu_cells, var_cells, values, occupancy)

#@torch.compile
def score_pose(map: NeuralMap, cam: Camera, image: torch.Tensor, depths: torch.Tensor, samples: int = 5, plane_indices: torch.Tensor = None, rel_x0=None, rel_x1=None, dense=False):
    planes = map.planes

    device = image.device
    c,height,width = image.shape

    if plane_indices is None:
        plane_indices = torch.arange(len(planes))

    if rel_x0 is None:
        rel_x0 = torch.zeros((len(plane_indices), 2))
    if rel_x1 is None:
        rel_x1 = torch.ones((len(plane_indices), 2))
    plane_indices = plane_indices.to(device)
    visibility_mask = single_frustum_multiple_plane_visibility(map.planes, plane_indices, frustum_points(cam), rel_x0, rel_x1)

    plane_indices = plane_indices.clone().to(device)[visibility_mask]

    depth_sigma = map.conf.depth_sigma

    t_samples = torch.normal(depths[:,:,None].repeat(1,1,samples), depths[:,:,None].repeat(1,1,samples) * depth_sigma)

    orig, dir = compute_ray(width,height,cam,device)
    #print("Orig ", orig.shape, "dir", dir.shape, "t samples", t_samples.shape)
    pos = orig[:,:,None] + dir[:,:,None] * t_samples[:,:,:,None]

    x0s = planes.x0s.to(device)[plane_indices]
    normal = planes.planes.to(device)[plane_indices, 0:3]
    us = planes.us.to(device)[plane_indices]
    vs = planes.vs.to(device)[plane_indices]
    us = us / torch.linalg.norm(us,dim=1)[:,None]**2
    vs = vs / torch.linalg.norm(vs,dim=1)[:,None]**2

    pos = pos.reshape((height*width*samples,3))

    offset = pos[None]-x0s[:,None,:]
    coords = torch.stack([torch.einsum("ik,ijk->ij", us, pos[None] - x0s[:,None,:]), torch.einsum("ik,ijk->ij",vs,offset), torch.einsum("ik,ijk->ij", normal, offset)], dim=2)

    # don't weight by occupancy since samples are already normally distributed,
    # hence when computing cross-entropy p(x) is captured implicitly by the discrete set distribution
    occupancy = torch.full((height,width,samples,), 1.0, device=device) # / (height * width * samples))

    feature_samples = image[:,:,:,None].repeat(1,1,1,samples).reshape((c,height*width*samples))
    occupancy_samples = occupancy.reshape(height*width*samples)

    if len(plane_indices) == 0:
        if dense:
            return torch.zeros((height,width,samples), device=image.device), torch.zeros((height,width,samples), device=image.device),  feature_samples.reshape((c,height,width,samples))
        else:
            return torch.tensor(0., device=device), torch.tensor(0., device=device)

    #print("Scoring cells planes= ", len(plane_indices))
    scores, weight = score_cells(map, plane_indices, coords, feature_samples, occupancy_samples, x0=rel_x0, x1=rel_x1)
    scores = scores.reshape((len(plane_indices),height,width,samples))
    weight = weight.reshape((len(plane_indices),height,width,samples))

    #print("log prob mean ", log_prob.mean(), "weight mean", weight.mean())

    scores = torch.where(weight>0, scores, torch.tensor(0., device=device))
    score = (scores * weight).sum(dim=0) / (1e-9 + weight.sum(dim=0))
    weight = torch.ones_like(score) # weight: already included in the alphas, for compatibility reason set it to one

    if dense:
        return (weight*score), weight, pos.reshape((height,width,samples,3)), feature_samples.reshape((c,height,width,samples))
    else:
        return torch.sum(weight*score) / (1e-9 + torch.sum(weight)), weight.mean()