from dataclasses import dataclass
import torch
from NeuralPlanes import utils

@dataclass 
class Camera:
    size: torch.Tensor
    f: torch.Tensor 
    c: torch.Tensor

    R: torch.Tensor
    t: torch.Tensor

    near: float = 0.1
    far: float = 100

    def to(self, device):
        return Camera(
            size=self.size.to(device),
            f=self.f.to(device),
            c=self.c.to(device),
            R=self.R.to(device),
            t=self.t.to(device),
            near=self.near,
            far=self.far
        )

    def scale(self, s):
        return Camera(
            size=torch.tensor([int(self.size[0]*s), int(self.size[1]*s)]),
            f=torch.tensor([self.f[0]*s, self.f[1]*s]),
            c=torch.tensor([self.c[0]*s, self.c[1]*s]),
            R=self.R,
            t=self.t,
            near=self.near,
            far=self.far
        )
    def intrinsic_matrix(cam):
        K = torch.tensor([
            [cam.f[0],0,cam.c[0]],
            [0,cam.f[1],cam.c[1]],
            [0,0,1]
        ], device=cam.R.device)

        return K

    def extrinsic_matrix(cam):
        R = cam.R.transpose(0,1)
        t = R @ -cam.t
        return torch.cat([R, t[:,None]], dim=1)

    def to_image_space(cam, p3d: torch.Tensor):
        # 3 x 4
        proj = cam.intrinsic_matrix() @ cam.extrinsic_matrix()
        homo3d = torch.cat([p3d, torch.ones((p3d.shape[0],1),device=p3d.device)], dim=1)
        homo = (homo3d @ proj.T)
        p2d = homo[:,:2] / homo[:,2:3]
        return torch.cat([p2d, homo[:,2:3]], dim=1)


def compute_ray(width, height, camera, device, same_z: bool = True):
    if isinstance(camera, list):
        batch_size = len(camera)
        origin = []
        dir = []
        for cam in camera:
            o,d = compute_ray(width,height,cam,device)
            origin.append(o)
            dir.append(d)

        origin = torch.stack(origin, dim=0)
        dir = torch.stack(dir, dim=0)
        return origin, dir        

    y,x = torch.meshgrid(torch.linspace(0,camera.size[1],height,device=device), torch.linspace(0,camera.size[0],width,device=device))
    x = torch.stack([x,y], dim=2)

    cam_c = camera.c.to(device)
    cam_f = camera.f.to(device)

    origin = camera.t.to(device)[None,None].repeat((height,width,1))
    dir = ((x - cam_c.reshape(1,1,2)) / cam_f)
    dir = torch.cat([((x - cam_c.reshape(1,1,2)) / cam_f), torch.ones((height,width,1),device=device)], dim=2) @ camera.R.T.to(device)
    dir = dir / torch.linalg.norm(dir, dim=2).unsqueeze(2)

    if same_z:
        fwd = camera.R @ torch.tensor([0, 0, 1.], device=camera.R.device)
        scale = 1.0/torch.einsum("ijk,k->ij",dir,fwd.to(device))
        dir = dir*scale[:,:,None]

    return origin, dir


def frustum_points(camera: Camera, near=None, far=None):
    near = camera.near if near is None else near
    far = camera.far if far is None else far
    origin, dir = compute_ray(2, 2, camera, camera.c.device, same_z=True)

    points_near = origin + dir*near
    points_far = origin + dir*far

    return torch.stack([points_near, points_far]) # [2,2,2, 3]

def frustum_normals(frustum_points: torch.Tensor):
    assert len(frustum_points.shape) == 4 and list(frustum_points.shape) == [2, 2, 2, 3]

    bottom_normal = torch.cross(frustum_points[0,0,1] - frustum_points[0,0,0], frustum_points[1,0,0] - frustum_points[0,0,0], dim=0)
    right_normal = torch.cross(frustum_points[0,1,1] - frustum_points[0,0,1], frustum_points[1,0,1] - frustum_points[0,0,1], dim=0)
    top_normal = torch.cross(frustum_points[1,1,0] - frustum_points[0,1,0], frustum_points[0,1,1] - frustum_points[0,1,0], dim=0)
    left_normal = torch.cross(frustum_points[1,0,0] - frustum_points[0,0,0], frustum_points[0,1,0] - frustum_points[0,0,0], dim=0)
    front_normal = torch.cross(frustum_points[0,1,0] - frustum_points[0,0,0], frustum_points[0,0,1] - frustum_points[0,0,0], dim=0)
    back_normal = -front_normal

    normals = torch.stack([bottom_normal, right_normal, top_normal, left_normal, front_normal, back_normal])
    normals = normals / torch.linalg.norm(normals,dim=1)[:,None]

    return normals

def frustum_ray_intersection(frustum_points: torch.Tensor, origin: torch.Tensor, dir: torch.Tensor, max_dist: float = 500, eps = 1e-9):
    assert len(frustum_points.shape) == 4 and list(frustum_points.shape) == [2, 2, 2, 3]
    assert len(origin.shape)==2 and len(dir.shape)==2 and origin.shape[1] == 3 and dir.shape[1] == 3

    normals = frustum_normals(frustum_points)

    # bottom,right,top,left,front,back
    plane_x00 = torch.stack([
        frustum_points[0, 0, 0],
        frustum_points[0, 0, 1],
        frustum_points[0, 1, 1],
        frustum_points[0, 1, 0],
        frustum_points[0, 0, 0],
        frustum_points[1, 0, 0]
    ])
    plane_x01 = torch.stack([
        frustum_points[0, 0, 1],
        frustum_points[0, 1, 1],
        frustum_points[0, 1, 0],
        frustum_points[0, 0, 0],
        frustum_points[0, 0, 1],
        frustum_points[1, 0, 1]
    ])
    plane_x10 = torch.stack([
        frustum_points[1, 0, 0],
        frustum_points[1, 0, 1],
        frustum_points[1, 1, 1],
        frustum_points[1, 1, 0],
        frustum_points[0, 1, 0],
        frustum_points[1, 1, 0]
    ])
    plane_x11 = torch.stack([
        frustum_points[1, 0, 1],
        frustum_points[1, 1, 1],
        frustum_points[1, 1, 0],
        frustum_points[1, 0, 0],
        frustum_points[0, 1, 1],
        frustum_points[1, 1, 1]
    ])

    u = ((plane_x10+plane_x11)-(plane_x00+plane_x01))/2
    v = plane_x11-plane_x10
    s = plane_x01-plane_x00

    """
    todo: investigate numerics
    assert torch.linalg.norm(torch.einsum("ij,ij->i", normals, u)) < 1e-3
    assert torch.linalg.norm(torch.einsum("ij,ij->i", normals, v)) < 1e-3
    assert torch.linalg.norm(torch.einsum("ij,ij->i", normals, s)) < 1e-3
    assert torch.linalg.norm(torch.einsum("ij,ij->i", u, v)) < 1e-3
    assert torch.linalg.norm(torch.einsum("ij,ij->i", u, s)) < 1e-3
    """

    un = torch.linalg.norm(u, dim=1)
    vn = torch.linalg.norm(v, dim=1)
    sn = torch.linalg.norm(s, dim=1)

    t = (torch.einsum("ij,ij->i", normals, plane_x00)[:,None] - torch.einsum("ijk,ik->ij", origin[None], normals)) / (eps + torch.einsum("ijk,ik->ij",dir[None],normals))
    inter = origin[None] + t[:,:,None]*dir[None]

    off = inter - 0.5*(plane_x00[:,None]+plane_x01[:,None])
    lam = torch.einsum("ijk,ik->ij", off, u/un[:,None])
    mu = torch.einsum("ijk,ik->ij", off, v/vn[:,None])

    mask = (0<t) & (t<max_dist) & (2*torch.abs(mu) < (1-lam/un[:,None])*sn[:,None] + lam/un[:,None]*vn[:,None]) & (0 < lam) & (lam < un[:,None])

    t_min = torch.min(torch.where(mask, t, torch.tensor(torch.inf)), dim=0)[0]
    t_max = torch.max(torch.where(mask, t, torch.tensor(-torch.inf)), dim=0)[0]

    return t_min, t_max


