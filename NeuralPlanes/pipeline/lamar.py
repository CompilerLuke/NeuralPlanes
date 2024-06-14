from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
from pathlib import Path
from NeuralPlanes.camera import Camera
from NeuralPlanes.plane import make_planes
from NeuralPlanes.utils import select_device
from NeuralPlanes.pipeline.dataloader import *
from NeuralPlanes.localization.negative_pose_mining import RansacMiningConf
from NeuralPlanes.localization.training import NeuralMapBuilder, NeuralMapBuilderConf
import logging
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import tqdm
import numpy as np
import os
import copy
from skimage import io

logger = logging.getLogger(__name__)

@dataclass
class SensorMeta:
    sensor_id: str
    name: str

@dataclass
class WifiMeta(SensorMeta):
    pass


@dataclass
class BluetoothMeta(SensorMeta):
    pass


@dataclass
class CameraMeta(SensorMeta):
    img: Tuple[int,int]
    focal: Tuple[float,float]
    center: Tuple[float,float]

@dataclass
class Sensors:
    by_id: Dict[str,SensorMeta]

@dataclass
class RigMeta:
    rig_id: str
    sensor_id: str
    R: torch.Tensor # 3x3 matrix
    t: torch.Tensor # 3-vector

@dataclass
class Rigs:
    by_id: Dict[str, RigMeta]

@dataclass
class TrajectoryMeta:
    timestamp: int
    device_id: str
    R: torch.Tensor # 3x3 matrix
    t: torch.Tensor # 3-vector
    covar: float

@dataclass
class Trajectories:
    by_timestamp: Dict[int, TrajectoryMeta]

@dataclass
class ImageMeta:
    timestamp: int
    sensor_id: str
    camera: Camera
    path: str

class Images(Dataset):
    base_path: str
    by_path: Dict[str, ImageMeta]
    images: List[ImageMeta]

    def __init__(self, base_path, by_path, images):
        super().__init__()
        self.base_path = base_path
        self.by_path = by_path
        self.images = images
    
    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return self.images.__iter__()

    def __getitem__(self, index):
        meta = self.images[index]
        image = io.imread(str(self.base_path) + "/" + meta.path)
        assert not image is None, f"Could not read {path}"
        return {
                "image": image,
                "camera": meta.camera,
                "path": meta.path
            }
def parse_sensors(sensor_path: Path) -> Sensors:
    with open(sensor_path, 'r') as f:
        sensors = Sensors(by_id={})

        for i, line in enumerate(f.read().split("\n")[1:-1]):
            tokens = line.replace(" ", "").split(",")

            sensor_id, name, sensor_type = tokens[0:3]

            if sensor_type == "bluetooth":
                sensor = BluetoothMeta(sensor_id=sensor_id, name=name)
            elif sensor_type == "wifi":
                sensor = WifiMeta(sensor_id=sensor_id, name=name)
            elif sensor_type == "camera":
                camera_type = tokens[3]
                if camera_type != "PINHOLE":
                    logger.warn(f"Ignoring non-pihole camera {sensor_id}")
                    continue

                img = (int(tokens[4]), int(tokens[5]))
                focal = (float(tokens[6]),float(tokens[7]))
                center = (float(tokens[8]),float(tokens[9]))

                sensor = CameraMeta(sensor_id=sensor_id, name=name, img=img, focal=focal, center=center)
            else:
                logger.warn(f"Ignoring sensor {sensor_id} {name}, unknown sensor type {sensor_type}. Expecting bluetooth or camera")
                continue

            sensors.by_id[sensor.sensor_id] = sensor

        return sensors


def map_float(xs):
    return [float(x) for x in xs]


def quat_to_matrix(rot_quat):
    return torch.tensor(Rotation.from_quat([rot_quat[1],rot_quat[2],rot_quat[3],rot_quat[0]]).as_matrix(), dtype=torch.float)


def parse_rigs(rig_path: Path) -> Rigs:
    with open(rig_path, "r") as f:
        rigs = Rigs(by_id={})

        for i, line in enumerate(f.read().split("\n")[1:-1]):
            tokens = line.replace(" ", "").split(",")

            rot_quat = map_float(tokens[2:6])
            rot_matrix = quat_to_matrix(rot_quat)
            trans = torch.tensor(map_float(tokens[6:9]), dtype=torch.float)

            rig = RigMeta(rig_id=tokens[0], sensor_id=tokens[1], R=rot_matrix, t=trans)
            rigs.by_id[rig.rig_id] = rig

        return rigs


def parse_trajectories(trajectory_path: Path) -> Trajectories:
    with open(trajectory_path) as f:
        trajectories = Trajectories(by_timestamp={})

        for i, line in enumerate(f.read().split("\n")[1:-1]):
            tokens = line.replace(" ", "").split(",")

            timestamp = int(tokens[0])
            device_id = tokens[1]
            rot_quat = map_float(tokens[2:6])
            rot_matrix = quat_to_matrix(rot_quat)
            trans = torch.tensor(map_float(tokens[6:9]), dtype=torch.float)
            covar = float(tokens[9]) if len(tokens) >= 10 else 0

            trajectory = TrajectoryMeta(timestamp=timestamp, device_id=device_id, R=rot_matrix, t=trans, covar=covar)

            assert not timestamp in trajectories.by_timestamp
            trajectories.by_timestamp[timestamp] = trajectory

        return trajectories


def parse_images(base_path: Path, image_path: Path, trajectories: Trajectories, rigs: Rigs, sensors: Sensors, filter: Callable[[str], bool]) -> Images:
    with open(image_path) as f:
        images = Images(base_path=str(base_path), by_path={}, images=[])

        for i, line in enumerate(f.read().split("\n")[1:-1]):
            tokens = line.replace(" ", "").split(",")

            timestamp = int(tokens[0])
            sensor_id = tokens[1]
            image_path = tokens[2]

            if not filter(image_path):
                continue

            trajectory = trajectories.by_timestamp[timestamp]

            sensor = sensors.by_id[sensor_id]

            if trajectory.device_id in rigs.by_id:
                rig = rigs.by_id[trajectory.device_id]
                R = trajectory.R @ rig.R
                t = trajectory.t + (trajectory.R @ rig.t)
            else:
                if trajectory.device_id.startswith("hl"):
                    logging.warn(f"Could not find hololens rig for device id {trajectory.device_id}")
                    continue

                R = trajectory.R
                t = trajectory.t

            camera = Camera(
                size=sensor.img,
                f=sensor.focal,
                c=sensor.center,
                R=R,
                t=t,
                near=0.1,
                far=50
            )

            image = ImageMeta(timestamp=timestamp, sensor_id=sensor_id, camera=camera, path=image_path)
            images.by_path[image.path] = image
            images.images.append(image)

        return images


def plot_image_poses(fig, ax, images: Images, device=None, cam_size = 0.5, stride = 1, linewidth = 0.5):
    color_per_sensor : Dict[str, torch.Tensor] = {}

    segments = []
    colors = []

    for image in list(images.by_timestamp.values())[::stride]:
        if device and not image.sensor_id.startswith(device):
            continue

        if image.sensor_id in color_per_sensor:
            color = color_per_sensor[image.sensor_id]
        else:
            color = torch.rand(3)
            color_per_sensor[image.sensor_id] = color

        camera = image.camera

        pos = camera.t[0:2]
        forward = (camera.R @ torch.tensor([0.,0.,1.]))[0:2]
        forward = forward / torch.linalg.norm(forward)[None]
        side = (camera.R @ torch.tensor([1.,0.,0.]))[0:2]
        side = side / torch.linalg.norm(side)[None]

        v0, v1, v2 = pos + cam_size*(-0.3*side), pos + cam_size*(0.3*side), pos + cam_size*(forward)

        segments.append(torch.stack([v0,v1]))
        segments.append(torch.stack([v1,v2]))
        segments.append(torch.stack([v2,v0]))

        colors.append(color)
        colors.append(color)
        colors.append(color)

    segments = torch.stack(segments)
    colors = torch.stack(colors)

    lc = LineCollection(segments.numpy(), colors=colors.numpy(), linewidths=linewidth)
    ax.add_collection(lc)
    ax.autoscale()


def parse_floorplan(floor_plan_path):
    floor_plan = cv2.imread(str(floor_plan_path / "hg_floor_plan_e.png"))

    with open(floor_plan_path / "transforms.json") as f:
        transforms = json.load(f)
    with open(floor_plan_path / "hg_floor_plan_e.json") as f:
        planes_json = json.load(f)

    floor_plan_height, floor_plan_width, _ = floor_plan.shape
    img_rel_to_abs = torch.tensor(transforms["floormap_to_world"])
    ground = -5
    height = 20

    x0s, x1s, x2s = [], [], []

    # Add floor
    x2, x1, x0 = torch.tensor([
        [0., 0., 1.],
        [1., 0., 1.],
        [1., 1., 1.]
    ]) @ img_rel_to_abs.T

    normal = torch.cross(x1-x0, x2-x1)

    x0s.append(torch.tensor([x0[0], x0[1], ground])[None])
    x1s.append(torch.tensor([x1[0], x1[1], ground])[None])
    x2s.append(torch.tensor([x2[0], x2[1], ground])[None])

    for polygon in planes_json:
        contour = torch.stack([torch.tensor([p["x"], p["y"]]) for p in polygon["content"]])
        contour = torch.cat([contour, contour[0].unsqueeze(0)], dim=0)
        contour = torch.flip(contour, dims=(0,))

        n = contour.shape[0]

        contour = contour / torch.tensor([floor_plan_width, floor_plan_height])  # To relative image coordinates 0-1
        contour = torch.cat([contour, torch.ones((n, 1))], dim=1)
        contour = (contour @ img_rel_to_abs.T)[:, :2]  # To absolute world-space

        x0 = torch.cat([contour[:-1], torch.full((n - 1, 1), ground)], dim=1)
        x1 = torch.cat([contour[1:], torch.full((n - 1, 1), ground)], dim=1)
        x2 = x1 + torch.tensor([0, 0, height])

        x0s.append(x0)
        x1s.append(x1)
        x2s.append(x2)

    return floor_plan, (torch.cat(x0s, dim=0), torch.cat(x1s, dim=0), torch.cat(x2s, dim=0)), img_rel_to_abs

def plot_floor_planes(fig, ax, plane_points):
    x0s,x1s,x2s = plane_points

    n = x0s.shape[0]

    norm = torch.cross(x1s-x0s, x2s-x1s, dim=1)
    norm = norm / torch.linalg.norm(norm,dim=1)[:,None]

    x = torch.zeros((6*n,3))
    x[0::6] = x0s
    x[1::6] = (x0s+x1s)/2
    x[2::6] = (x0s+x1s)/2 + norm 
    x[3::6] = (x0s+x1s)/2
    x[4::6] = x1s
    x[5::6] = torch.inf

    ax.plot(x[:,0], x[:,1])

def parse_session(session_map_path, filter: Callable[[str],str]):
    sensors = parse_sensors(session_map_path / "sensors.txt")
    rigs = parse_rigs(session_map_path / "rigs.txt")
    trajectories = parse_trajectories(session_map_path / "trajectories.txt")
    images = parse_images(session_map_path / "raw_data", session_map_path / "images.txt", trajectories, rigs, sensors, filter=filter)

    return images

def collate_fn(preprocess):
    def apply(data):
        result = {}
        for data in data:
            data = preprocess(data)
            for key in data:
                if not key in result:
                    result[key] = [data[key]]
                else:
                    result[key].append(data[key])
        result = {k: torch.stack(v) if isinstance(v[0],torch.Tensor) else v for k,v in result.items()}
        return result
    return apply

def create_dataloader(dataset: Images, preprocess, conf: DataloaderConf):
    logger.info("Preprocessing on ", conf.device)
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=collate_fn(preprocess))
    return dataloader
class Model(nn.Module):
    def __init__(self):
        super().__init__()    
    
    def preprocess(self, data):
        pass 

    def forward(self):
        pass 

class MonocularDepthModel(Model):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)

    """
    Copied from https://github.com/YvanYin/Metric3D/blob/main/hubconf.py#L145
    """
    def preprocess(self, data):
        rgb_origin = data["image"]
        camera = data["camera"]
        
        # input_size = (544, 1216) # for convnext model
        input_size = (616, 1064) # for vit model

        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        # remember to scale intrinsic, hold depth
        intrinsic = [camera.f[0] * scale, camera.f[1] * scale, camera.c[0] * scale, camera.c[1] * scale]
        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        #### normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)

        return { "rgb": rgb, "intrinsic": intrinsic, "origin_size": rgb_origin.shape, "pad_info": pad_info, "path": data["path"] }
    def forward(self, data):
        rgb = data["rgb"]

        with torch.no_grad():
            pred_depths, confidence, output_dict = self.model.inference({'input': rgb})

        result = []
        # todo: batch
        for pred_depth, intrinsic, origin_size, pad_info in zip(pred_depths, data["intrinsic"], data["origin_size"], data["pad_info"]):
            pred_depth = pred_depth[:, pad_info[0] : pred_depth.shape[1] - pad_info[1], pad_info[2] : pred_depth.shape[2] - pad_info[3]]
        
            # upsample to original size
            pred_depth = torch.nn.functional.interpolate(pred_depth[None, :, :, :], (origin_size[0],origin_size[1]), mode='bilinear')[0,0]
            ###################### canonical camera space ######################

            #### de-canonical transform
            canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
            pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
            pred_depth = torch.clamp(pred_depth, 0, 300)

            result.append(pred_depth)

        return { "image": result }

class Backbone(Model):
    def __init__(self, conf):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        #self.model.classifier[4] = nn.Identity()

        print(self.model)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = select_device()
        self.model = self.model.eval().to(self.device)
    def preprocess(self, data):
        return { **data, "image": self.transform(data["image"]) }
    def forward(self, data):
        return {"image": self.model(data["image"].to(self.device))["out"] }

def precompute_depth(model, out_dir, images: Images, max_size, dataloader_conf):
    preprocess = compose([resize(max_size=max_size), model.preprocess])
    model = model.to(dataloader_conf.device)
    dataloader = create_dataloader(images, preprocess, conf= dataloader_conf)

    for i, data in enumerate(tqdm.tqdm(dataloader)):
        path = data["path"]
        data= {k: v.to(dataloader_conf.device) if isinstance(v,torch.Tensor) else v for k,v in data.items()}
        output = model(data)

        for depth, path in zip(output["image"], path):
            depth = depth.detach().cpu().numpy().astype(np.half)
            depth = depth[:,:,None]
            path = (out_dir / Path(path)).with_suffix(".tif")

            path.parent.mkdir(parents=True, exist_ok=True)
            error = io.imsave(str(path), depth)

def precompute_embedding(model, out_dir, images: Images, dataloader_conf):
    preprocess = compose([resize(max_size=max_size), model.preprocess])
    model = model.to(dataloader_conf.device)
    dataloader = create_dataloader(images, preprocess, conf= dataloader_conf)

    for i, data in enumerate(tqdm.tqdm(dataloader)):
        path = data["path"]
        data = {k: v.to(dataloader_conf.device) if isinstance(v,torch.Tensor) else v for k,v in data.items()}
        output = model(data)

        for image, path in zip(output["image"], path):
            image = image.detach().permute(1,2,0).cpu().numpy().astype(np.half)
            image = image.reshape((image.shape[0], image.shape[1]*7, 3)) # 7*3 = 21 channels from embedding

            path = (out_dir / Path(path)).with_suffix(".tif")

            path.parent.mkdir(parents=True, exist_ok=True)
            error = io.imsave(str(path), image)

if __name__ == "__main__":
    Backbone({})

    import argparse
    parser = argparse.ArgumentParser(
                    prog='NeuralPlanesLamarTrain',
                    description='',
                    epilog='')
    parser.add_argument('directory')
    args = parser.parse_args()
    
    lamar_path = Path(args.directory)

    session_map_path = lamar_path / "sessions" / "map"
    session_map_image_path = session_map_path / "raw_data"
    session_map_path_precompute = lamar_path / "sessions" / "map_precompute_256"
    session_map_path_depth = session_map_path_precompute / "depth"
    session_map_path_embedding = session_map_path_precompute / "embedding"

    images = parse_session(session_map_path, lambda path: path.startswith("ios"))

    print("Map images ", len(images))

    floor_plan_img, plane_points, floorplan_to_world = parse_floorplan(lamar_path / "floor_plan")

    fig, ax = plt.subplots()

    #plot_image_poses(fig, ax, images)
    #plot_floor_planes(fig, ax, plane_points)
    #plt.show()

    dataloader_conf = DataloaderConf(
        batch_size=16,
        cache_size=64,
        num_workers=8
    )

    max_size = 256

    if False and os.path.exists(session_map_path_depth):
        print("Already pre-computed depth:", session_map_path_depth)
    else:
        precompute_depth(MonocularDepthModel(), session_map_path_depth, images, max_size, dataloader_conf)

    if os.path.exists(session_map_path_embedding):
        print("Already pre-computed embedding:", session_map_path_embedding)
    else:
        precompute_embedding(Backbone({}), session_map_path_embedding, images, max_size, dataloader_conf)

    planes = make_planes(plane_points, resolution=4)

    pre = Images(images=[], by_path={}, base_path=session_map_path_depth)

    cameras = []
    for image in tqdm.tqdm(images):
        path = image.path
        new_path = str(Path(path).with_suffix(".tif"))

        depth = io.imread(str(session_map_path_depth / new_path))
        max_depth = depth.max()

        camera = image.camera.scale(min(1.0, max_size / max(image.camera.size)))
        camera.far = min(max_depth, 15)

        #print("Max depth ", max_depth, camera.far, new_path, "dtype= ", depth.dtype, "depth= ", depth.shape, "camera=", camera.size)
        image.camera = camera

        meta = copy.copy(images.by_path[path])
        meta.path = new_path

        cameras.append(camera)
        pre.images.append(meta)
        pre.by_path[path] = meta

    images = copy.copy(pre)
    images.base_path = session_map_path_embedding

    depths = copy.copy(pre)
    depths.base_path = session_map_path_depth

    map_builder_conf = NeuralMapBuilderConf(num_components=4, num_features_backbone=21, num_features=16, max_views_per_chunk=20, depth_sigma=0.2, chunk_size=32, ransac=RansacMiningConf(num_ref_kps=20, kp_per_ref=10, ransac_it=15, ransac_sample=10, top_k=5))
    builder = NeuralMapBuilder(planes, cameras=cameras, conf=map_builder_conf)

    accesses = []
    for i, j, k in builder.chunks():
        accesses.extend(list(builder.visible_camera_idx[i][j][k]))
    backbone = Backbone(conf={})

    image_loader = SequentialLoader(images, accesses=accesses, conf=dataloader_conf, collate_fn=collate_fn(lambda x: {"image": torch.tensor(x["image"].reshape(x["image"].shape[0], x["image"].shape[1]//7, 21), dtype=torch.float).permute(2,0,1)}))
    depth_loader = SequentialLoader(depths, accesses=accesses, conf=dataloader_conf, collate_fn=collate_fn(lambda x: {"image": torch.tensor(x["image"][:,:,0], dtype=torch.float)}))

    torch.autograd.set_detect_anomaly(True)
    builder.train(image_loader, depth_loader)
