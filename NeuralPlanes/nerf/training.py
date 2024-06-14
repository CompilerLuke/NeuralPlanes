import torch
from torch import nn
import tqdm
from matplotlib import pyplot as plt
from NeuralPlanes.utils import select_device
from NeuralPlanes.nerf.model import MapModule
from NeuralPlanes.nerf.volume import alpha_blend

device = select_device()

def train_map(map_module: MapModule, image_loader, output_dir):
    map_module = map_module.to(device)

    min_depth = 0.1
    max_depth = 20
    z_res = 30
    depth_sigma = 1
    depth_weight = 0.05
    smoothing = 0.01

    loss_fn = nn.HuberLoss()
    num_epochs = 1000

    optim = torch.optim.Adam([
        {'params': map_module.color_decoder.parameters(), 'lr': 3e-3},
        {'params': map_module.density_decoder.parameters(), 'lr': 3e-3},
        {'params': map_module.atlases.parameters(), 'lr': 3e-2}
    ])

    loss_sum = 0
    count = 0
    for epoch in range(num_epochs):
        for images, cams, corrs in tqdm.tqdm(image_loader):
            images = images.to(device)

            batch,c,height,width = images.shape

            #volume_coarse, density_coarse, t_coarse = map_module.raytrace_volume(min_depth, max_depth, (width,height,z_res), cams)
            #pred_coarse = alpha_weight(volume_coarse, density_coarse)

            volume_fine, density_fine, t_fine = map_module.raytrace_volume(min_depth, max_depth, (width,height,z_res), cams) # t_coarse, density_coarse)
            pred_fine = alpha_blend(volume_fine, density_fine)

            optim.zero_grad()
            loss = loss_fn(10*pred_fine, 10*images) 
            #atlas = map_module.atlas
            #loss = loss + smoothing*torch.mean((atlas[:,0:-2,1:-1] + atlas[:,2:,1:-1] + atlas[:,1:-1,0:-2] + atlas[:,1:-1,2:] - 4*atlas[:,1:-1,1:-1])**2) #+ 1/3*loss_fn(pred_coarse, images)
            
            for i, (cam, (p2d, p3d)) in enumerate(zip(cams, corrs)):
                p2d = p2d.to(device)
                p3d = p3d.to(device)

                depth = torch.linalg.norm(p3d - cam.t.to(device)[None,:], dim=1)

                assert (p2d[:,0] < width).all() and (p2d[:,1] < height).all()

                eps = 1e-9

                depth_based_density = torch.exp(-((t_fine[i, :, p2d[:,1], p2d[:,0]] - depth[None])/depth_sigma)**2)
                depth_based_density = depth_based_density / (eps + torch.linalg.norm(depth_based_density, dim=0)[None])
                
                density_fine_p2 = density_fine[i, :, p2d[:,1], p2d[:,0]]
                density_fine_p2 = density_fine_p2 / (eps + torch.linalg.norm(density_fine_p2, dim=0)[None])
                
                cross_entropy = -torch.mean(torch.einsum("ij,ij->j", torch.log(density_fine_p2), depth_based_density)) #density_fine_p2 * depth_based_density)

                loss = loss + depth_weight * cross_entropy

            loss.backward()

            optim.step()

            loss_sum += loss
            count += 1

            if count % 100 == 0:
                pred = pred_fine[0].permute(1, 2, 0).cpu().detach().numpy()
                img = images[0].permute(1, 2, 0).cpu().detach().numpy()

                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(img)
                axes[1].imshow(pred)
                plt.savefig(f"{output_dir}/plot_{epoch}.png")
                print(f"[{epoch}] Loss - {loss_sum/count} {cross_entropy}")

            del images

    torch.save(map_module.state_dict(), output_dir + "model.pt")