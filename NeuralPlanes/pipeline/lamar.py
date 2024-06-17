from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
from pathlib import Path
from NeuralPlanes.camera import Camera
from NeuralPlanes.plane import make_planes, Planes
from NeuralPlanes.utils import select_device
from NeuralPlanes.pipeline.dataloader import *
from NeuralPlanes.localization.negative_pose_mining import RansacMiningConf
from NeuralPlanes.localization.training import NeuralMapBuilder, NeuralMapBuilderConf
from NeuralPlanes.localization.map import NeuralMap, NeuralMapConf
from NeuralPlanes.localization.unet import Unet
from NeuralPlanes.localization.monocular import MonocularDepthModel
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

        if polygon["labels"]["labelName"] == "in":
            print("flip")
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

def get_preprocessed_images(images, session_map_path_small, session_map_path_depth, max_size):
    images_small = copy.deepcopy(images)
    images_depth = copy.deepcopy(images)

    images_small.base_path = session_map_path_small
    images_depth.base_path = session_map_path_depth

    cameras = []
    for i in tqdm.tqdm(range(len(images))):
        image = images.images[i]
        path = image.path
        
        depth_path = str(Path(path).with_suffix(".tif"))
        images_depth.images[i].path = depth_path
    
        depth = io.imread(str(session_map_path_depth / depth_path))
        max_depth = depth.max()

        camera = image.camera.scale(min(1.0, max_size / max(image.camera.size)))
        camera.far = min(max_depth, 15)

        images_small.images[i].camera = camera 
        images_depth.images[i].camera = camera
        cameras.append(camera)

    return cameras, images_small, images_depth


def downsize_images(out_dir, images: Images, max_size, dataloader_conf):
    preprocess = compose([resize(max_size=max_size)])
    dataloader = create_dataloader(images, preprocess, conf= dataloader_conf)

    for i, data in enumerate(tqdm.tqdm(dataloader)):
        path = data["path"]
        for image, path in zip(data["image"], path):
            path = out_dir / Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            error = io.imsave(str(path), image)

def load_map(model_checkpoint):
    device = select_device()
    conf = NeuralMapConf(num_components=4, num_features_backbone=21, num_features=8, depth_sigma=0.2)
    encoder = Unet(encoder_freeze=True, classes= conf.num_features+1)

    height, width =  1736, 845
    n_planes = 73
    n_comp = conf.num_components
    n_feat = conf.num_features
    map = NeuralMap(planes=Planes(x0s=torch.zeros((n_planes,3)),
                                    us=torch.zeros((n_planes,3)), 
                                    vs=torch.zeros((n_planes,3)), 
                                    planes=torch.zeros((n_planes,4)), 
                                    coord_x0=torch.zeros((n_planes,2)), 
                                    coord_size=torch.zeros((n_planes,2)), 
                                    atlas_size=[width, height]), 
                        atlas_alpha=torch.zeros((n_comp,height,width), device=device), 
                        atlas_mu=torch.zeros((n_comp, n_feat, height, width),device=device), 
                        atlas_var=torch.zeros((n_comp, n_feat, height, width),device=device), 
                        atlas_weight=torch.zeros((height, width),device=device), encoder=encoder, conf=conf)
    map.load_state_dict(torch.load(model_checkpoint), strict=False)
    encoder = encoder.to(device)
    return map

def gen_planes_from_floorplan(lamar_path):
    floor_plan_img, plane_points, floorplan_to_world = parse_floorplan(lamar_path / "floor_plan")

    plot_planes = False 
    if plot_planes:
        fig, ax = plt.subplots()
        plot_image_poses(fig, ax, images)
        plot_floor_planes(fig, ax, plane_points)
        plt.show()

    planes = make_planes(plane_points, resolution=4)
    return planes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='NeuralPlanesLamarTrain',
                    description='',
                    epilog='')
    parser.add_argument('directory')

    args = parser.parse_args()
    
    lamar_path = Path(args.directory)

    max_size = 256

    session_map_path = lamar_path / "sessions" / "map"
    session_map_image_path = session_map_path / "raw_data"
    session_map_path_precompute = lamar_path / "sessions" / ("map_precompute_"+str(max_size))
    session_map_path_depth = session_map_path_precompute / "depth"
    session_map_path_small = session_map_path_precompute / "small_images"
    model_checkpoint = "checkpoint/model.pt"

    images = parse_session(session_map_path, lambda path: path.startswith("ios"))

    print("Map images ", len(images))

    dataloader_conf = DataloaderConf(
        batch_size=16,
        cache_size=64,
        num_workers=0
    )

    if os.path.exists(session_map_path_depth):
        print("Already pre-computed depth:", session_map_path_depth)
    else:
        precompute_depth(MonocularDepthModel(), session_map_path_depth, images, max_size, dataloader_conf)

    if os.path.exists(session_map_path_small):
        print("Already down-sampled images:", session_map_path_small)
    else:
        downsize_images(session_map_path_small, images, max_size, dataloader_conf)

    cameras, images_small, images_depth = get_preprocessed_images(images, session_map_path_small, session_map_path_depth, max_size)

    map_builder_conf = NeuralMapBuilderConf(num_components=4, num_features_backbone=0, num_features=8, num_epochs=1, max_views_per_chunk=20, depth_sigma=0.2, chunk_size=32, ransac=RansacMiningConf(num_ref_kps=20, kp_per_ref=10, ransac_it=15, ransac_sample=10, top_k=5))

    load_checkpoint = os.path.exists(model_checkpoint)
    if load_checkpoint:
        map = load_map(model_checkpoint)
        encoder = map.encoder
        builder = NeuralMapBuilder(map=map, cameras=cameras, conf=map_builder_conf)
    else:
        planes = gen_planes_from_floorplan(lamar_path)
        encoder = Unet(encoder_freeze=True, classes= map_builder_conf.num_features+1)
        builder = NeuralMapBuilder(planes=planes, cameras=cameras, encoder=encoder, conf=map_builder_conf)

    accesses = []
    for i, j, k in builder.chunks():
        accesses.extend(list(builder.visible_camera_idx[i][j][k]))
    
    image_loader = SequentialLoader(images_small, accesses=accesses, conf=dataloader_conf, collate_fn=collate_fn(lambda x: encoder.preprocess(x)))
    depth_loader = SequentialLoader(images_depth, accesses=accesses, conf=dataloader_conf, collate_fn=collate_fn(lambda x: {"image": torch.tensor(x["image"][:,:,0], dtype=torch.float)}))

    #      # lambda x: {"image": torch.tensor(x["image"].reshape(x["image"].shape[0], x["image"].shape[1]//7, 21), dtype=torch.float).permute(2,0,1)}))

    torch.autograd.set_detect_anomaly(True)
    builder.train(image_loader, depth_loader, "logs/fit/run", "checkpoint/model.pt")
