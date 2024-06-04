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

def compute_ray(width, height, camera, device):
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

    return origin, dir