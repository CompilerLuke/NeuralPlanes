import torch 
from torch import nn
from NeuralPlanes.camera import Camera
from NeuralPlanes.nerf.volume import raytrace_volume

class ColorDecoder(nn.Module):
    def __init__(self, model_offset, model_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(model_dim // 2, 3),
        )
        self.model_offset = model_offset
        self.model_dim = model_dim
        
    def forward(self, x):
        x = x[:,self.model_offset:self.model_offset+self.model_dim]
        x = self.mlp(x) #x[:,1:4]
        return x #torch.sigmoid(x)
    
class DensityDecoder(nn.Module):
    def __init__(self, model_offset, model_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(model_dim // 2, 1),
        )

        self.model_offset = model_offset
        self.model_dim = model_dim
        
    def forward(self, x):
        x = x[:,self.model_offset:self.model_offset+self.model_dim]
        x = self.mlp(x)[:,0] 
        return torch.sigmoid(x) #*x


class MapModule(nn.Module):
    def __init__(self, planes, model_dim, num_levels=3):
        super().__init__()
        self.planes = planes
        
        self.color_decoder = ColorDecoder(model_dim//2, model_dim//2)
        self.density_decoder = DensityDecoder(0, model_dim//2)
        
        width,height = planes.atlas_size
        
        if True:
            levels = []
            for i in range(num_levels):
                size = (model_dim, int(height)*2**i, int(width)*2**i)
                atlas = torch.rand(size)
                levels.append(atlas)

            self.atlases = nn.ParameterList(levels)
        else:
            self.atlas = nn.Parameter(torch.rand((model_dim, height, width)))
    def raytrace_volume(self, min_depth, max_depth, res, camera: Camera, alpha_xs : torch.Tensor = None, alpha_ys : torch.Tensor = None):
        if True:
            num_levels = len(self.atlases)
            _,height_fine,width_fine = self.atlases[-1].shape
            atlas = self.atlases[-1]
            for i in range(0,len(self.atlases)-1,-1):
                _,height,width = self.atlases[i].shape
                atlas = 0.5*atlas + self.atlases[i].repeat_interleave(width_fine//width, 2).repeat_interleave(height_fine//height, 1)
        else:
            atlas = self.atlas
        return raytrace_volume(self.planes, min_depth, max_depth, res, camera, atlas, self.color_decoder, self.density_decoder, alpha_xs, alpha_ys)

