import torch
from torch import nn
from NeuralPlanes.plane import project_to_planes_sparse
from typing import Union, List
from NeuralPlanes.camera import Camera, compute_ray
import torch_scatter

def sample_linear_pdf(x0, x1, d, n):
    u = torch.rand(n, device=x0.device)
    return d*(-x0 + torch.sqrt(x0**2 + (x1**2 - x0**2)*u)) / (x1 - x0 + 1e-9)


def sample_ray(origin, dir, min_t, max_t, samples, xs=None, ys=None): # Returns pos, t, dt
    batch_size,height,width,_ = origin.shape

    device = origin.device

    if True: #xs is None or ys is None:
        dt = (max_t - min_t) / samples
        base = torch.linspace(min_t, max_t, (samples+1), device=device)[None,:,None,None]
        t = base + dt*torch.rand((batch_size,samples+1,height,width), device=device)
    else:
        xs = xs.permute(0,2,3,1) # (b,h,w,d)
        ys = ys.permute(0,2,3,1) # (b,h,w,d)

        x1, y1 = xs[:,:,:,1:], ys[:,:,:,1:]
        x0, y0 = xs[:,:,:,:-1], ys[:,:,:,:-1]

        depth = x0.shape[-1]

        weights = (x1-x0)*(y0 + y1) # (b,h,w,d)
        assert weights.min() >= 0

        weights = weights.reshape((-1,depth))
        i = (torch.multinomial(weights, samples+1, True).reshape((batch_size*height*width,samples+1))) # (bxhxw, d)

        i = (depth*torch.arange(batch_size*height*width, device=device)[:,None] + i).flatten()
        t = x0.flatten()[i] + sample_linear_pdf(y0.flatten()[i], y1.flatten()[i], x1.flatten()[i]-x0.flatten()[i], len(i))
        t = t.reshape((batch_size,height,width,samples+1))
        t = torch.cat([xs,t], dim=3).sort(dim=0)[0]
        t = t.permute(0,3,1,2) # (b,d,h,w)

    return origin[:,None] + t[:,:-1,:,:,None]*dir[:,None], t[:, :-1, :, :], t[:, 1:,:,:] - t[:, :-1,:,:]

def raytrace_volume(planes, min_depth, max_depth, res, camera: Union[List[Camera], Camera], atlas, decoder, density_decoder, alpha_xs, alpha_ys):
    width,height,depth_steps = res
    device = atlas.device
    
    if isinstance(camera, list):
        batch_size = len(camera)
    else:
        batch_size = 1
        camera = [camera]
        
    origin, dir = compute_ray(width,height,camera,device)

    pos, t, dt = sample_ray(origin, dir, min_depth, max_depth, depth_steps, alpha_xs, alpha_ys)
    depth = pos.shape[1]

    pixel_ids, plane_ids, coord = project_to_planes_sparse(planes, pos)
    
    coord = 2*(coord / torch.tensor(planes.atlas_size,device=device))

    features = nn.functional.grid_sample(atlas.unsqueeze(0), coord.unsqueeze(0).unsqueeze(2))
    features = features[0, :, :, 0]
    map_dim = features.shape[0]

    combined_features = torch_scatter.scatter_add(dim=1, index=pixel_ids, src=features, dim_size=batch_size*depth*height*width)
    #torch.zeros((map_dim, batch_size*depth*height*width), device=device).scatter_add(dim=1, index=pixel_ids[None,:].repeat(3,1), src=features)
    #torch.ones((map_dim, batch_size*depth*height*width), device=device).scatter_reduce(dim=1, index=pixel_ids[None,:], src=features, include_self=True, reduce='sum')

    values = decoder(combined_features.transpose(0,1)) # (bxdxhxw, c)
    values = values.reshape((batch_size, depth, height, width, -1)) # (b,d,h,w, c)
    values = values.permute((0,4,1,2,3))

    density = density_decoder(combined_features.transpose(0,1)) # (bxdxhxw)
    density = density.reshape((batch_size, depth, height, width))
    
    alpha = dt * density 

    return values, alpha, t


@torch.compile
def alpha_weight(values, alpha):
    batch,depth,height,width = alpha.shape
    alpha = alpha / (1e-9 + alpha.sum(dim=1)[:,None,:,:] )
    weighted_values = values * alpha[:,None]
    return weighted_values.sum(dim=2)

@torch.compile
def alpha_blend(values, alpha):
    batch,depth,height,width = alpha.shape
    device = alpha.device

    alpha = 1 - torch.exp(-alpha)
    alpha = alpha * torch.cumprod(torch.cat([torch.ones((batch,1,height,width), device=device), 1-alpha[:,:-1]], dim=1),dim=1)

    weighted_values = values * alpha[:,None]
    return weighted_values.sum(dim=2)