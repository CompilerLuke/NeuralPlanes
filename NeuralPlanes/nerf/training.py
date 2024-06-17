import torch
from torch import nn
import tqdm
from matplotlib import pyplot as plt
from NeuralPlanes.utils import select_device
<<<<<<< Updated upstream:NeuralPlanes/nerf/training.py
from NeuralPlanes.nerf.model import MapModule
from NeuralPlanes.nerf.volume import alpha_blend

device = select_device()

=======
from NeuralPlanes.model import MapModule
from NeuralPlanes.volume import alpha_blend, norm_alpha_weight

device = select_device()

class PosedImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, trajectories, keypoints, transform=None, max_images=0):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        if max_images:
            self.image_paths = self.image_paths[:max_images]
        
        self.trajectories = trajectories
        self.keypoints = keypoints

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]
        
        if self.transform:
            image = self.transform(image)
            height, width = image.shape[1:3]
        else:
            height, width = orig_height, orig_width

        img_name = Path(img_path).stem.replace("__", "_")
        cam = self.trajectories[img_name]

        if self.keypoints:
            idx = self.keypoints.db_names.index(img_name)
            
            p2d = (self.keypoints.keypoints[idx] * torch.tensor([width/orig_width, height/orig_height])).type(torch.int)
            p3d = self.keypoints.p3d[idx]

            #print("Loaded image p2d x max y max", p2d[:,0].max(), p2d[:,1].max(), "width,height", width, height)

            return image, cam, (p2d, p3d)  
        else:
            return image, cam


def create_image_loader(images_dir, trajectories, keypoints):
    image_dataset = PosedImageDataset(images_dir, trajectories=trajectories, keypoints=keypoints, transform= v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=192),
    ]), max_images=200)

    def collate_fn(data):
        images = []
        cams = []
        corrs = []

        for image, cam, corr in data:
            images.append(image)
            cams.append(cam)
            corrs.append(corr)

        return torch.stack(images), cams, corrs

    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=True, num_workers=4,
                                               collate_fn=collate_fn)

    return image_loader


>>>>>>> Stashed changes:NeuralPlanes/train.py
def train_map(map_module: MapModule, image_loader, output_dir):
    map_module = map_module.to(device)

    min_depth = 0.1
    max_depth = 10
    z_res = 30
    depth_sigma = 1
    depth_weight = 1
    smoothing = 0.01

    loss_fn = nn.L1Loss()
    num_epochs = 1000

    optim = torch.optim.Adam([
        {'params': map_module.color_decoder.parameters(), 'lr': 5e-3},
        {'params': map_module.density_decoder.parameters(), 'lr': 5e-3},
        {'params': map_module.atlases.parameters(), 'lr': 5e-2}
    ])

    loss_sum = 0
    count = 0
    for epoch in range(num_epochs):
        for images, cams, corrs in tqdm.tqdm(image_loader):
            images = images.to(device)

            batch,c,height,width = images.shape

            #volume_coarse, density_coarse, t_coarse = map_module.raytrace_volume(min_depth, max_depth, (width,height,z_res), cams)
            #pred_coarse = alpha_weight(volume_coarse, density_coarse)

            volume_fine, density_fine, t_fine, dt_fine = map_module.raytrace_volume(min_depth, max_depth, (width,height,z_res), cams) # t_coarse, density_coarse)
            pred_fine = alpha_blend(volume_fine, density_fine, dt_fine)

            optim.zero_grad()
            loss = loss_fn(pred_fine, images) 
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

                depth_based_density = torch.exp(-((t_kps - depth[None])/depth_sigma)**2)
                density_fine_kps = density_fine[i, :, p2d[:,1], p2d[:,0]]
                termination_prob = density_fine_kps * torch.exp(-torch.cumsum(torch.cat([torch.zeros((1,p2d.shape[0]), device=device), density_fine_kps[:-1]], dim=0),dim=0)) 
                
                cross_entropy = -torch.mean(torch.sum(torch.log(termination_prob) * depth_based_density * dt, dim=0)) 
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