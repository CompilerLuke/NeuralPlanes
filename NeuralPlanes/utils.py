import torch
from torchvision.transforms import v2
import os
import cv2
from pathlib import Path

torch.set_float32_matmul_precision('high')

def select_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

class PosedImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, trajectories, keypoints, transform=None, max_images=0):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if
                            fname.endswith(('.png', '.jpg', '.jpeg'))]
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

            p2d = (self.keypoints.keypoints[idx] * torch.tensor([width / orig_width, height / orig_height])).type(
                torch.int)
            p3d = self.keypoints.p3d[idx]

            # print("Loaded image p2d x max y max", p2d[:,0].max(), p2d[:,1].max(), "width,height", width, height)

            return image, cam, (p2d, p3d)
        else:
            return image, cam

def create_image_loader(images_dir, trajectories, keypoints):
    image_dataset = PosedImageDataset(images_dir, trajectories=trajectories, keypoints=keypoints, transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=128),
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

