<<<<<<< HEAD
import cv2
import torch
from scipy.spatial.transform import Rotation
import json
import h5py
from dataclasses import dataclass
from typing import List
from pathlib import Path
import itertools
from matplotlib import pyplot as plt
from NeuralPlanes.camera import Camera
from NeuralPlanes.plane import make_planes, draw_planes
from NeuralPlanes.nerf.model import MapModule
from NeuralPlanes.utils import create_image_loader
from NeuralPlanes import nerf


def parse_trajectories(ref_trajectories):
    with open(ref_trajectories, 'r') as f:
        trajectories = {}
        for i, lines in enumerate(f.read().split("\n")[1:-1]):
            tokens = lines.replace(" ", "").split(",")
            cam = tokens[1]
            image = str(i // 4).zfill(5) + "-" + tokens[1]

            # todo: read intrinsics from sensors.txt
            quat = torch.tensor([float(x) for x in tokens[2:6]])

            pose = Camera(
                size=torch.tensor([1280, 1920]),
                f=torch.tensor([960, 960]),
                c=torch.tensor([639.8245614035088, 959.8245614035088]),
                t=torch.tensor([float(x) for x in tokens[6:9]]),
                R=torch.tensor(Rotation.from_quat(quat).as_matrix(), dtype=torch.float)
            )

            trajectories[image] = pose

        return trajectories
    raise "Could not load reference trajectories"


def parse_floorplan(floor_plan_path):
    floor_plan = cv2.imread(floor_plan_path + "hg_floor_plan_e.png")

    with open(floor_plan_path + "transforms.json") as f:
        transforms = json.load(f)
    with open(floor_plan_path + "hg_floor_plan_e.json") as f:
        planes_json = json.load(f)

    floor_plan_height, floor_plan_width, _ = floor_plan.shape
    img_rel_to_abs = torch.tensor(transforms["relative_to_absolute"]) @ torch.tensor(transforms["floor_to_relative"])
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

    return floor_plan, (torch.cat(x0s, dim=0), torch.cat(x1s, dim=0), torch.cat(x2s, dim=0))


@dataclass
class LocalFeatures:
    keypoints: List[torch.Tensor]
    scores: List[torch.Tensor]
    descriptors: List[torch.Tensor]
    db_names: List[str]
    p3d: List[torch.Tensor]
    uncertainty: float = 0


def list_h5_names(path):
    names = []
    with h5py.File(str(path), "r", libver="latest") as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip("/"))

        fd.visititems(visit_fn)
    return list(set(names))


def db_iterator(paths):
    if isinstance(paths, (Path, str)):
        paths = [paths]

    name2db = {n: i for i, p in enumerate(paths) for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())
    db_names = db_names_h5

    names_by_db = itertools.groupby(db_names, key=lambda name: name2db[name])
    for idx, group in names_by_db:
        with h5py.File(str(paths[idx]), "r", libver="latest") as fd:
            for name in group:
                if name.startswith("floor_plan"):
                    continue
                yield Path(name).stem.replace("__", "_"), fd[name]


def load_keypoint_database(paths) -> LocalFeatures:
    if isinstance(paths, (Path, str)):
        paths = [paths]

    db_names = []
    desc = []
    keypoints = []
    scores = []
    p3d = []
    for name, data in db_iterator(paths):
        db_names.append(name)
        desc.append(torch.tensor(data['descriptors'].__array__()))
        keypoints.append(torch.tensor(data['keypoints'].__array__()).squeeze(0))
        scores.append(torch.tensor(data['scores'].__array__()))
        p3d.append(torch.tensor(data['p3d'].__array__()))

    print(db_names)

    return LocalFeatures(
        keypoints=keypoints,
        scores=scores,
        descriptors=desc,
        p3d=p3d,
        db_names=db_names
    )


def train_nerf():
    building_path = "../../data/HG_navviz/"
    floor_plan_path = building_path + "raw_data/floor_plan/"
    train_path = "../../data/train_HG/"

    trajectories = parse_trajectories(building_path + "trajectories.txt")
    floor_plan_img, plane_points = parse_floorplan(floor_plan_path)

    local_features = load_keypoint_database(building_path + "features.h5")

    # plane_points = [x[0:6] for x in plane_points]

    map_dim = 24
    planes = make_planes(plane_points, resolution=0.25)

    ax = plt.figure().add_subplot(projection='3d')
    draw_planes(ax, planes=planes)
    plt.savefig(train_path + "plane_plot.png")

    map_module = MapModule(planes, map_dim)

    model_file = train_path + "model.pt"

    map_module = torch.compile(map_module)
    nerf.training.train_map(map_module, create_image_loader(building_path + "raw_data/images_undistr_center",
                                                            trajectories=trajectories, keypoints=local_features),
                            output_dir=train_path)


if __name__ == "__main__":
=======
import cv2
import torch
from scipy.spatial.transform import Rotation
import json
import h5py
from dataclasses import dataclass
from typing import List
from pathlib import Path
import itertools
from matplotlib import pyplot as plt
from NeuralPlanes.camera import Camera
from NeuralPlanes.plane import make_planes, draw_planes
from NeuralPlanes.nerf.model import MapModule
from NeuralPlanes.utils import create_image_loader
from NeuralPlanes import nerf


def parse_trajectories(ref_trajectories):
    with open(ref_trajectories, 'r') as f:
        trajectories = {}
        for i, lines in enumerate(f.read().split("\n")[1:-1]):
            tokens = lines.replace(" ", "").split(",")
            cam = tokens[1]
            image = str(i // 4).zfill(5) + "-" + tokens[1]

            # todo: read intrinsics from sensors.txt
            quat = torch.tensor([float(x) for x in tokens[2:6]])

            pose = Camera(
                size=torch.tensor([1280, 1920]),
                f=torch.tensor([960, 960]),
                c=torch.tensor([639.8245614035088, 959.8245614035088]),
                t=torch.tensor([float(x) for x in tokens[6:9]]),
                R=torch.tensor(Rotation.from_quat(quat).as_matrix(), dtype=torch.float)
            )

            trajectories[image] = pose

        return trajectories
    raise "Could not load reference trajectories"


def parse_floorplan(floor_plan_path):
    floor_plan = cv2.imread(floor_plan_path + "hg_floor_plan_e.png")

    with open(floor_plan_path + "transforms.json") as f:
        transforms = json.load(f)
    with open(floor_plan_path + "hg_floor_plan_e.json") as f:
        planes_json = json.load(f)

    floor_plan_height, floor_plan_width, _ = floor_plan.shape
    img_rel_to_abs = torch.tensor(transforms["relative_to_absolute"]) @ torch.tensor(transforms["floor_to_relative"])
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

    return floor_plan, (torch.cat(x0s, dim=0), torch.cat(x1s, dim=0), torch.cat(x2s, dim=0))


@dataclass
class LocalFeatures:
    keypoints: List[torch.Tensor]
    scores: List[torch.Tensor]
    descriptors: List[torch.Tensor]
    db_names: List[str]
    p3d: List[torch.Tensor]
    uncertainty: float = 0


def list_h5_names(path):
    names = []
    with h5py.File(str(path), "r", libver="latest") as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip("/"))

        fd.visititems(visit_fn)
    return list(set(names))


def db_iterator(paths):
    if isinstance(paths, (Path, str)):
        paths = [paths]

    name2db = {n: i for i, p in enumerate(paths) for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())
    db_names = db_names_h5

    names_by_db = itertools.groupby(db_names, key=lambda name: name2db[name])
    for idx, group in names_by_db:
        with h5py.File(str(paths[idx]), "r", libver="latest") as fd:
            for name in group:
                if name.startswith("floor_plan"):
                    continue
                yield Path(name).stem.replace("__", "_"), fd[name]


def load_keypoint_database(paths) -> LocalFeatures:
    if isinstance(paths, (Path, str)):
        paths = [paths]

    db_names = []
    desc = []
    keypoints = []
    scores = []
    p3d = []
    for name, data in db_iterator(paths):
        db_names.append(name)
        desc.append(torch.tensor(data['descriptors'].__array__()))
        keypoints.append(torch.tensor(data['keypoints'].__array__()).squeeze(0))
        scores.append(torch.tensor(data['scores'].__array__()))
        p3d.append(torch.tensor(data['p3d'].__array__()))

    print(db_names)

    return LocalFeatures(
        keypoints=keypoints,
        scores=scores,
        descriptors=desc,
        p3d=p3d,
        db_names=db_names
    )


def train_nerf():
    building_path = "../../data/HG_navviz/"
    floor_plan_path = building_path + "raw_data/floor_plan/"
    train_path = "../../data/train_HG/"

    trajectories = parse_trajectories(building_path + "trajectories.txt")
    floor_plan_img, plane_points = parse_floorplan(floor_plan_path)

    local_features = load_keypoint_database(building_path + "features.h5")

    # plane_points = [x[0:6] for x in plane_points]

    map_dim = 24
    planes = make_planes(plane_points, resolution=0.25)

    ax = plt.figure().add_subplot(projection='3d')
    draw_planes(ax, planes=planes)
    plt.savefig(train_path + "plane_plot.png")

    map_module = MapModule(planes, map_dim)

    model_file = train_path + "model.pt"

    map_module = torch.compile(map_module)
    nerf.training.train_map(map_module, create_image_loader(building_path + "raw_data/images_undistr_center",
                                                            trajectories=trajectories, keypoints=local_features),
                            output_dir=train_path)


if __name__ == "__main__":
>>>>>>> 94705e72e8c0385fabc93a5581eb9b6101735c88
    train_nerf()