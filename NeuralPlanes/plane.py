import torch
from torch import nn
from typing import Tuple
import rectangle_packing_solver as rps

class Planes(nn.Module):
    x0s: torch.Tensor # (n,3)
    us: torch.Tensor # (n,3)
    vs: torch.Tensor # (n,3)
    planes: torch.Tensor # (n,4)
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


def ray_trace(planes: Planes, origin: torch.Tensor, dir: torch.Tensor, indices: torch.Tensor = None, eps=1e-9, max_dist=300) -> torch.Tensor:
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

    mask = (0 <= u) & (u <= 1) & (0 <= v) & (v <= 1) & (0 <= t) & (t <= max_dist)
    t = torch.where(mask, t, torch.tensor(torch.inf, device=device))

    if not indices is None:
        t = t[:, indices]

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


def draw_planes(ax, planes, res=10, indices=None):
    xs = []
    ys = []

    if indices==None:
        indices = range(len(planes))

    for i in indices:
        width, height = planes.coord_size[i]
        u, v = torch.meshgrid(torch.linspace(0,1,int(width)), torch.linspace(0,1,int(height)))

        x0,u_dir,v_dir = planes.x0s[i], planes.us[i], planes.vs[i]

        x = x0.reshape((1,1,3)) + u_dir.reshape((1,1,3))*u.unsqueeze(2) + v_dir.reshape((1,1,3))*v.unsqueeze(2)
    
        ax.plot_surface(x[:,:,0], x[:,:,1], x[:,:,2])

def make_planes(plane_points, resolution=1):
    x0,x1,x2 = plane_points

    normal = torch.cross(x1-x0, x2-x0, dim=1)
    normal = normal / torch.linalg.norm(normal, dim=1).unsqueeze(1)

    len0 = torch.linalg.norm(x1-x0, dim=1)
    len1 = torch.linalg.norm(x2-x1, dim=1)

    map_size = torch.stack([torch.round(len0 * resolution), torch.round(len1 * resolution)], dim=1) 
    #torch.nested.nested_tensor([torch.zeros((int(height), int(width), map_dim)) for width, height in map_size])

    plane = torch.cat([normal, torch.einsum("ij,ij->i", normal, x0).unsqueeze(1)], dim=1)
    us = x1 - x0 
    vs = x2 - x1

    problem = rps.Problem(rectangles=map_size.tolist())
    floor_map_size = map_size[0]
    solution = rps.Solver().solve(problem=problem, show_progress=True, width_limit=floor_map_size[0])

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