from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
from NeuralPlanes.camera import Camera
from NeuralPlanes.plane import make_planes
import logging
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import torch
import json
import cv2
import sys

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

@dataclass
class Images:
    by_timestamp: Dict[int, ImageMeta]


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


def parse_images(image_path, trajectories: Trajectories, rigs: Rigs, sensors: Sensors) -> Images:
    with open(image_path) as f:
        images = Images(by_timestamp={})

        for i, line in enumerate(f.read().split("\n")[1:-1]):
            tokens = line.replace(" ", "").split(",")

            timestamp = int(tokens[0])
            sensor_id = tokens[1]
            image_path = tokens[2]

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
                t=t
            )

            image = ImageMeta(timestamp=timestamp, sensor_id=sensor_id, camera=camera, path=image_path)
            images.by_timestamp[image.timestamp] = image

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
    x0, x1, x2 = torch.tensor([
        [0., 0., 1.],
        [1., 0., 1.],
        [1., 1., 1.]
    ]) @ img_rel_to_abs.T

    x0s.append(torch.tensor([x0[0], x0[1], ground])[None])
    x1s.append(torch.tensor([x1[0], x1[1], ground])[None])
    x2s.append(torch.tensor([x2[0], x2[1], ground])[None])

    for polygon in planes_json:
        contour = torch.stack([torch.tensor([p["x"], p["y"]]) for p in polygon["content"]])
        contour = torch.cat([contour, contour[-1].unsqueeze(0)], dim=0)

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
    x = torch.zeros((3*n,3))

    x[0::3] = x0s
    x[1::3] = x1s
    x[2::3] = torch.inf

    ax.plot(x[:,0], x[:,1])

def parse_session(session_map_path):
    sensors = parse_sensors(session_map_path / "sensors.txt")
    rigs = parse_rigs(session_map_path / "rigs.txt")
    trajectories = parse_trajectories(session_map_path / "trajectories.txt")
    images = parse_images(session_map_path / "images.txt", trajectories, rigs, sensors)

    return images


if __name__ == "__main__":
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)

    lamar_path = Path("../../data/HGE")
    session_map_path = lamar_path / "sessions" / "map"
    images = parse_session(session_map_path)

    floor_plan_img, plane_points, floorplan_to_world = parse_floorplan(lamar_path / "floor_plan")

    fig, ax = plt.subplots()

    plot_image_poses(fig, ax, images)
    plot_floor_planes(fig, ax, plane_points)
    plt.show()

    #pred_depth, confidence, output_dict = model.inference({'input': rgb})

    planes = make_planes(plane_points)


