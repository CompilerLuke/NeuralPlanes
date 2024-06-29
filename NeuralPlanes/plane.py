import torch
from torch import nn
from typing import Tuple
import rectangle_packing_solver as rps
from NeuralPlanes.camera import frustum_normals
from matplotlib import pyplot as plt
from math import ceil

class Planes(nn.Module):
    x0s: torch.Tensor # (n,3)
    us: torch.Tensor # (n,3)
    vs: torch.Tensor # (n,3)
    planes: torch.Tensor # (n,4), todo: seperate normal and offset
    coord_x0: torch.Tensor # (n,2)
    coord_size: torch.Tensor # (n,2) 

    atlas_size: Tuple[float,float]

    def __init__(self, x0s,us,vs,planes,coord_x0,coord_size,atlas_size):
        super().__init__()
        
        self.register_buffer('x0s', x0s)
        self.register_buffer('us', us)
        self.register_buffer('vs', vs)
        self.register_buffer('planes', planes)
        self.register_buffer('coord_x0', coord_x0)
        self.register_buffer('coord_size', coord_size)

        self.atlas_size = atlas_size

    def __len__(self):
        return self.x0s.shape[0]


def ray_trace(planes: Planes, origin: torch.Tensor, dir: torch.Tensor, indices: torch.Tensor = None, mask: torch.Tensor = None, eps=1e-9, max_dist=300) -> torch.Tensor:
    if not indices is None and len(indices) == 0:
        return torch.full((origin.shape[0],), torch.inf), torch.zeros((origin.shape[0],))

    dir = dir / torch.linalg.norm(dir, dim=1).unsqueeze(1)
    normal = planes.planes[:,0:3]
    thr = planes.planes[:,3]

    u = planes.us
    v = planes.vs

    u = u / torch.linalg.norm(u, dim=1).unsqueeze(1)**2
    v = v / torch.linalg.norm(v, dim=1).unsqueeze(1)**2

    device = origin.device

    t = (thr.unsqueeze(0) - torch.einsum("jk,ik->ij", normal, origin)) / (torch.einsum("jk,ik->ij", normal, dir) + eps)

    offset = t[:,:,None]*origin[:,None,:] - planes.x0s.unsqueeze(0)

    u = torch.einsum("jk,ijk->ij", u, offset)
    v = torch.einsum("jk,ijk->ij", v, offset)

    _mask = (0 <= u) & (u <= 1) & (0 <= v) & (v <= 1) & (0 <= t) & (t <= max_dist)
    t = torch.where(_mask, t, torch.tensor(torch.inf, device=device))

    if not indices is None:
        t = t[:, indices] # todo: index earlier
    if not mask is None:
        t = torch.where(mask, t, torch.tensor(torch.inf))

    return torch.min(t, dim=1)

@torch.compile
def project_to_planes(planes, pos, max_dist=300): # Input (b,h,w,3), output pixel_ids: (#hits), plane_ids: (#hits), proj: (#hits, 3)
    u_dir = planes.us / torch.linalg.norm(planes.us, dim=1).unsqueeze(1)**2
    v_dir = planes.vs / torch.linalg.norm(planes.vs, dim=1).unsqueeze(1)**2

    device = pos.device

    batch_size,depth,height,width,_ = pos.shape

    offset = pos[:,:,:,:,None] - planes.x0s[None,None,None,None,:]

    u = torch.einsum("pl,bzyxpl->bzyxp", u_dir, offset)
    v = torch.einsum("pl,bzyxpl->bzyxp", v_dir, offset)
    mask = (0 <= u) & (u <= 1) & (0 <= v) & (v <= 1)

    pixel_ids = torch.arange((batch_size * depth * height * width), device=device).unsqueeze(1)
    pixel_ids = pixel_ids.repeat_interleave(len(planes), 1)
    pixel_ids = pixel_ids.reshape((batch_size,depth,height,width,len(planes)))[mask]

    plane_ids = torch.arange(len(planes), device=device)[None,None].repeat((batch_size,depth,height,width,1))[mask]
    
    u = u[mask]
    v = v[mask]
    
    proj = planes.x0s[plane_ids] + u[:,None]*planes.us[plane_ids] + v[:,None]*planes.vs[plane_ids]

    eps_self = 1e-6
    dir = pos.reshape((-1,3))[pixel_ids,:] - proj
    origin =  proj + dir*eps_self

    indices = 1+torch.arange(len(planes)-1)
    hit = ray_trace(planes, origin, dir, indices= indices, max_dist=max_dist)

    miss = torch.isinf(hit[0])
    pixel_ids = pixel_ids[miss]
    plane_ids = plane_ids[miss]
    proj = proj[miss]
    u = u[miss]
    v = v[miss]

    coord = torch.stack([u,v], dim=1) 
    coord = planes.coord_x0[plane_ids] + planes.coord_size[plane_ids]*coord
    
    return pixel_ids, plane_ids, proj, coord

@torch.compile
def project_to_planes_sparse(planes, pos, stride=8):
    pos = pos.detach()

    batch_size,depth,height,width,_ = pos.shape
    
    pos_block = pos[:, :, ::stride, ::stride]
    pixel_ids, plane_ids, proj, coord = project_to_planes(planes, pos_block)

    _,depth_block,height_block,width_block,_ = pos_block.shape
    
    x = pixel_ids % width_block
    y = (pixel_ids // width_block) % height_block
    z = (pixel_ids // (width_block * height_block)) % depth_block
    b = (pixel_ids // (width_block * height_block * depth_block))

    stride_width = width // width_block
    stride_height = height // height_block
    stride_depth = depth // depth_block

    block_offset = (
       width * height * torch.arange(stride_depth, device=pos.device)[:, None, None]
       + width*torch.arange(stride_height,device=pos.device)[None,:,None]
       + torch.arange(stride_width,device=pos.device)[None,None,:]
    )

    pixel_ids = (b*depth*height*width + z*height*width*stride_depth + y*width*stride_height + x*stride_width)[:,None,None,None] + block_offset[None,:,:,:]
    pixel_ids = pixel_ids.flatten()
    plane_ids = plane_ids.repeat_interleave(stride_width*stride_height*stride_depth,0)

    u_dir = planes.us / torch.linalg.norm(planes.us, dim=1).unsqueeze(1)**2
    v_dir = planes.vs / torch.linalg.norm(planes.vs, dim=1).unsqueeze(1)**2

    offset = pos.reshape((-1,3))[pixel_ids] - planes.x0s[plane_ids]

    u = torch.einsum("ij,ij->i", u_dir[plane_ids], offset)
    v = torch.einsum("ij,ij->i", v_dir[plane_ids], offset)

    coord = torch.stack([u,v], dim=1) 
    coord = planes.coord_x0[plane_ids] + planes.coord_size[plane_ids]*coord

    return pixel_ids, plane_ids, coord


def draw_planes(ax, planes, stride=None, indices=None, color=None, vmin=None, vmax=None):
    if stride is None:
        stride = int(ceil(max(planes.atlas_size) / 512))

    xs = []
    ys = []

    if indices==None:
        indices = range(len(planes))

    xs = []
    for i in indices:
        width, height = planes.coord_size[i]
        v, u = torch.meshgrid(torch.linspace(0,1,int(height)//stride), torch.linspace(0,1,int(width)//stride))

        x0,u_dir,v_dir = planes.x0s[i], planes.us[i], planes.vs[i]

        x = x0.reshape((1,1,3)).cpu() + u_dir.reshape((1,1,3)).cpu()*u.unsqueeze(2) + v_dir.cpu().reshape((1,1,3))*v.unsqueeze(2)
        x = x.cpu()

        plane_color = None
        if not color is None:
            coord = int(planes.coord_x0[i][0]), int(planes.coord_x0[i][1])
            size = int(planes.coord_size[i][0]), int(planes.coord_size[i][1])
            #draw_planes(ax, planes, indices=[i], color= self.map.atlas_weight[coord[1]:coord[1]+size[1], coord[0]:coord[0]+size[0]]

            if len(color.shape) == 2:
                #color = torch.nn.functional.grid_sample(color[None,None], coords[None])[0,0]
                plane_color = color[coord[1]:coord[1]+size[1]:stride, coord[0]:coord[0]+size[0]:stride]
                plane_color = plane_color.clone().detach().cpu()
                if vmin and vmax:
                    plane_color = (plane_color - vmin) / (vmax - vmin)
                plane_color = plt.cm.viridis(plane_color)
            else:
                plane_color = color[:, coord[1]:coord[1]+size[1]:stride, coord[0]:coord[0]+size[0]:stride]
                plane_color = plane_color.clone().detach().cpu()
                plane_color = plane_color.permute(1,2,0)

        if isinstance(plane_color, torch.Tensor):
            plane_color = plane_color.numpy()
        ax.plot_surface(x[:,:,0].numpy(), x[:,:,1].numpy(), x[:,:,2].numpy(), facecolors= plane_color)


def make_planes(plane_points, resolution=1):
    few_planes = 5

    x0,x1,x2 = plane_points
    n = x0.shape[0]

    normal = torch.cross(x1-x0, x2-x1, dim=1)
    normal = normal / torch.linalg.norm(normal, dim=1).unsqueeze(1)

    len0 = torch.linalg.norm(x1-x0, dim=1)
    len1 = torch.linalg.norm(x2-x1, dim=1)

    map_size = torch.stack([torch.round(len0 * resolution), torch.round(len1 * resolution)], dim=1) 
    #torch.nested.nested_tensor([torch.zeros((int(height), int(width), map_dim)) for width, height in map_size])

    plane = torch.cat([normal, torch.einsum("ij,ij->i", normal, x0).unsqueeze(1)], dim=1)
    us = x1 - x0 
    vs = x2 - x1

    if resolution == 0:
        map_coords_x0 = None
        map_coords_size = None
        map_size = None
    elif n < few_planes:
        y = torch.cat([torch.zeros(1), torch.cumsum(map_size[:,1],0)], dim=0)
        x = torch.zeros(n)

        width = torch.max(map_size[:,0])
        height = y[-1]
        map_coords_x0 = torch.stack([x, y[:-1]], dim=1).type(torch.int)
        map_coords_size = map_size.type(torch.int)
        map_size = (int(width), int(height))
    else:
        problem = rps.Problem(rectangles=map_size.tolist())
        floor_map_size = map_size[0]
        solution = rps.Solver().solve(problem=problem, show_progress=True, width_limit=2*floor_map_size[0])

        map_coords_x0 = torch.tensor([[int(rect['x']), int(rect['y'])] for rect in solution.floorplan.positions], requires_grad=False)
        map_coords_size = torch.tensor([[int(rect['width']), int(rect['height'])] for rect in solution.floorplan.positions], requires_grad=False)
        map_size = [int(x) for x in solution.floorplan.bounding_box]

    planes = Planes(
        x0s= x0,
        us= us,
        vs= vs,
        planes= plane,
        coord_x0=map_coords_x0,
        coord_size=map_coords_size,
        atlas_size=map_size
    )
    return planes


"""
Work-around for current limitation of vmap, calling .item() on a Tensor is not supported
"""
def index_1d(vec, id):
    if type(id) is int:
        return vec[id]
    return torch.gather(vec, 0, id.unsqueeze(0).repeat(1, vec.shape[1]), sparse_grad=True).squeeze(0)


def plane_box_points(planes: Planes, plane_id: int, rel_x0: torch.Tensor = None, rel_x1: torch.Tensor = None, max_dist=20.0, perp_thr=0.1):
    plane_normal = index_1d(planes.planes, plane_id)[0:3]

    rel_x0 = rel_x0 if not rel_x0 is None else (0,0)
    rel_x1 = rel_x1 if not rel_x1 is None else (1,1)

    x0, u, v = index_1d(planes.x0s, plane_id), index_1d(planes.us, plane_id), index_1d(planes.vs, plane_id)

    plane_points_base = torch.stack([
        x0 + u*rel_x0[0] + v*rel_x0[1],
        x0 + u*rel_x1[0] + v*rel_x0[1],
        x0 + u*rel_x1[0] + v*rel_x1[1],
        x0 + u*rel_x0[0] + v*rel_x1[1]
    ])

    perp_mask = torch.abs(torch.einsum("j,ij->i", plane_normal, planes.planes[:,0:3])) < perp_thr

    mask = (~perp_mask) & (torch.arange(len(planes),device=perp_mask.device) != plane_id)

    # note: current limitation of vmap prevents us from skipping calculations using indices, have to use mask instead
    dist_to_intersec, inter_indices = ray_trace(planes, plane_points_base + plane_normal[None], plane_normal[None].repeat(4,1), mask=mask)

    #print("intersection indices", dist_to_intersec, inter_indices)
    dist = torch.minimum(torch.tensor(max_dist), dist_to_intersec)

    plane_points_top = plane_points_base + plane_normal[None] * dist[:,None]
    return torch.cat([plane_points_base, plane_points_top], dim=0)

def single_frustum_single_plane_visibility(planes: Planes, plane_id: int, frustum_points: torch.Tensor, rel_x0, rel_x1):
    device = frustum_points.device
    plane_normal = index_1d(planes.planes, plane_id)[0:3].to(device)

        #planes.planes[plane_id, 0:3]
    plane_points = plane_box_points(planes, plane_id, rel_x0, rel_x1).to(device)

    # Frustum normals
    axes_frustum = frustum_normals(frustum_points)

    # Box extruded from plane
    axes_box = torch.stack([index_1d(planes.us, plane_id).to(device), index_1d(planes.vs, plane_id).to(device), plane_normal])

    axes = torch.cat([axes_frustum, axes_box], dim=0)

    frustum_proj = torch.einsum("ik,jk->ij", axes, frustum_points.reshape((8, 3)))
    frustum_proj0 = torch.min(frustum_proj, dim=1)[0]
    frustum_proj1 = torch.max(frustum_proj, dim=1)[0]

    plane_proj = torch.einsum("ik,jk->ij", axes, plane_points)
    plane_proj0 = torch.min(plane_proj, dim=1)[0]
    plane_proj1 = torch.max(plane_proj, dim=1)[0]

    line_intersect = ((frustum_proj0 <= plane_proj0) & (plane_proj0 <= frustum_proj1)) \
                     | ((plane_proj0 <= frustum_proj0) & (frustum_proj0 <= plane_proj1))

    intersect = torch.all(line_intersect, dim=0)
    return intersect

frustum_plane_visibility = torch.vmap(single_frustum_single_plane_visibility, in_dims=(None,None,0,None,None), out_dims=0)

single_frustum_multiple_plane_visibility = torch.vmap(single_frustum_single_plane_visibility, in_dims=(None,0,None,0,0), out_dims=0)