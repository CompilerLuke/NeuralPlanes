from NeuralPlanes.utils import PosedImageDataset
from NeuralPlanes.plane import Planes, frustum_plane_visibility, plane_box_points
from NeuralPlanes.camera import Camera, frustum_points, frustum_ray_intersection
import torch
from torch.utils.data import Dataset
from math import ceil
from typing import List
from dataclasses import dataclass, field
from NeuralPlanes.utils import select_device
from NeuralPlanes.localization.image_pooling import pool_images
from NeuralPlanes.localization.negative_pose_mining import mine_negatives, RansacMiningConf
from NeuralPlanes.localization.map import NeuralMap, NeuralMapConf
import math
import gc
import tqdm
from matplotlib import pyplot as plt


@dataclass
class NeuralMapBuilderConf(NeuralMapConf):
    ransac: RansacMiningConf = field(default_factory=lambda: RansacMiningConf())
    chunk_size: int = 5
    depth_sigma: float = 0.5
    num_epochs: int = 10
    max_views_per_chunk: int = 15
    column_sampling_density: float = 2.0
    depth_distribution_samples: int = 10


class NeuralMapBuilder:
    planes: Planes
    image_ids: List[str]
    cameras: List[Camera]
    conf: NeuralMapBuilderConf
    map: NeuralMap

    chunk_coord: List[torch.Tensor]  # (h,w,2)
    chunk_size: List[torch.Tensor]  # (h,w,2)
    visible_camera_idx: List[List[List[torch.Tensor]]]
    inside_camera_idx: List[List[List[
        torch.Tensor]]]  # indices refer to elements of visible_camera_idx, so absolute id = visible_camera_idx[i][j][k][inside_camera_idx[i][j][k]]

    def __init__(self, planes, cameras: List[Camera], conf: NeuralMapBuilderConf):
        self.planes = planes

        self.cameras = cameras
        self.conf = conf
        self.device = select_device()

        height, width = planes.atlas_size

        self.map = NeuralMap(
            planes=planes,
            atlas_alpha=torch.zeros((conf.num_components, height, width), device=self.device),
            atlas_mu=torch.zeros((conf.num_components, conf.num_features, height, width), device=self.device),
            atlas_var=torch.zeros((conf.num_components, conf.num_features, height, width), device=self.device),
            atlas_weight=torch.zeros((height, width), device=self.device),
            conf=conf
        )

        self._assign_cameras_to_planes()

    def _chunk_neural_map(self, plane_id: int):
        chunk_size = self.conf.chunk_size

        planes = self.planes
        height, width = planes.coord_size[plane_id]

        v0, u0 = torch.meshgrid(chunk_size * torch.arange(0, int(ceil(height / chunk_size))),
                                chunk_size * torch.arange(0, int(ceil(width / chunk_size))))

        u1 = torch.minimum(u0 + chunk_size, width)
        v1 = torch.minimum(v0 + chunk_size, height)

        return torch.stack([u0, v0], dim=2), torch.stack([u1 - u0, v1 - v0], dim=2)

    def _assign_cameras_to_planes(self):
        planes = self.planes
        cameras = self.cameras

        fps = torch.stack([frustum_points(cam, cam.near, cam.far) for cam in cameras])

        self.chunk_coord = []
        self.chunk_size = []
        self.visible_camera_idx = []
        self.inside_camera_idx = []

        for plane_id in range(len(planes)):
            chunk_coord, chunk_size = self._chunk_neural_map(plane_id)

            plane_size_unsqueezed = torch.tensor(planes.coord_size[plane_id])[None, None]
            rel_x0 = chunk_coord.type(torch.float) / plane_size_unsqueezed
            rel_x1 = (chunk_coord + chunk_size).type(torch.float) / plane_size_unsqueezed

            visibility = frustum_plane_visibility(planes, plane_id, fps, (0., 0.), (1., 1.))
            indices = torch.arange(len(fps))[visibility]
            fps_visible = fps[visibility]

            indices_chunk_grid = []
            indices_inside_chunk_grid = []

            x0 = planes.x0s[plane_id]
            us = planes.us[plane_id]
            vs = planes.vs[plane_id]
            us = us / torch.linalg.norm(us) ** 2
            vs = vs / torch.linalg.norm(vs) ** 2

            for i in range(chunk_coord.shape[0]):
                indices_per_row = []
                indices_inside_per_row = []

                for j in range(chunk_coord.shape[1]):
                    visibility_chunk = frustum_plane_visibility(planes, plane_id, fps_visible, tuple(rel_x0[i, j]),
                                                                tuple(rel_x1[i, j]))
                    indices_chunk = indices[visibility_chunk]

                    chunk_center = planes.x0s[plane_id] + 0.5 * planes.us[plane_id] * (
                                rel_x0[i, j, 0] + rel_x1[i, j, 0]) + 0.5 * planes.vs[plane_id] * (
                                               rel_x0[i, j, 1] + rel_x1[i, j][1])

                    dist_to_chunk = torch.tensor(
                        [torch.linalg.norm(cameras[id].t - chunk_center, dim=0) for id in indices_chunk])
                    best = \
                    torch.topk(dist_to_chunk, k=min(len(dist_to_chunk), self.conf.max_views_per_chunk), largest=False)[
                        1]

                    indices_chunk = indices_chunk[best]

                    inside_chunk_mask = torch.tensor([rel_x0[i, j, 0] <= torch.dot(us, cameras[x].t - x0) <= rel_x1[
                        i, j, 0] and rel_x0[i, j, 1] <= torch.dot(vs, cameras[x].t - x0) <= rel_x1[i, j, 1] for x in
                                                      indices_chunk], dtype=torch.bool)
                    inside_chunk = torch.arange(len(best))[inside_chunk_mask]
                    # frustum_plane_visibility(planes, plane_id, [frustum_points(camera.) for i in best])

                    indices_per_row.append(indices_chunk)
                    indices_inside_per_row.append(inside_chunk)

                indices_chunk_grid.append(indices_per_row)
                indices_inside_chunk_grid.append(indices_inside_per_row)

            self.chunk_coord.append(chunk_coord)
            self.chunk_size.append(chunk_size)
            self.visible_camera_idx.append(indices_chunk_grid)
            self.inside_camera_idx.append(indices_inside_chunk_grid)

            for i in range(chunk_coord.shape[0]):
                for j in range(chunk_coord.shape[1]):
                    self.plot_cameras_on_chunk((plane_id, i, j))

    # @torch.compile
    def sample_images(self, chunk_id, images, depths, cameras):
        planes = self.planes
        device = self.device

        plane_id, i, j = chunk_id

        chunk_coord = self.chunk_coord[plane_id][i][j]
        chunk_size = self.chunk_size[plane_id][i][j]

        width, height = chunk_size
        width = int(width)
        height = int(height)

        rel_x0 = chunk_coord.type(torch.float) / planes.coord_size[plane_id]
        rel_x1 = (chunk_coord + chunk_size).type(torch.float) / planes.coord_size[plane_id]

        v, u = torch.meshgrid(torch.linspace(rel_x0[1], rel_x1[1], height, device=self.device),
                              torch.linspace(rel_x0[0], rel_x1[0], width, device=self.device))

        out_masks = []
        out_image_values = []
        out_occupancy_values = []
        out_tsamples = []
        out_pos = []

        planes_x0s = planes.x0s[plane_id][None, None].to(self.device)
        planes_us = planes.us[plane_id][None, None].to(self.device)
        planes_vs = planes.vs[plane_id][None, None].to(self.device)
        planes_planes = planes.planes[plane_id, 0:3][None, None].to(self.device)

        for input_image, input_depth, cam in zip(images, depths, cameras):
            # print(input_image.shape, input_depth.shape, cam.size)
            # assert input_image.shape[1:] == input_depth.shape

            c, img_height, img_width = input_image.shape
            # assert [img_width, img_height] == list(cam.size)

            cam = cam.to(device)
            fp = frustum_points(cam)

            origin = planes_x0s + u[:, :, None] * planes_us + v[:, :, None] * planes_vs
            dir = planes_planes.repeat((height, width, 1))
            origin, dir = origin.reshape((-1, 3)), dir.reshape((-1, 3))

            tmin, tmax = frustum_ray_intersection(fp, origin, dir)
            mask = torch.isfinite(tmin)

            #
            start = origin.reshape((-1, 3)) + tmin[:, None] * dir.reshape((-1, 3))
            end = origin.reshape((-1, 3)) + tmax[:, None] * dir.reshape((-1, 3))

            length = self.conf.column_sampling_density * torch.max(
                torch.cat([torch.linalg.norm(end - start, dim=1)[mask], torch.zeros(1, device=device)]))
            length = int(ceil(length))

            if length == 0:
                continue

            def get_sample_coords(tsample):
                tsample = tmin[:, None] * (1 - tsample) + tmax[:, None] * tsample
                pos = (origin[:, None] + tsample[:, :, None] * dir[:, None])
                proj = cam.to_image_space(pos.reshape(-1, 3)).reshape((height * width, tsample.shape[1], 3))
                sample_coords = 2 * (proj[:, :, 0:2] / torch.tensor([img_width + 1, img_height + 1],
                                                                    device=device)) - img_width / (img_width + 1)
                return sample_coords, proj[:, :, 2], pos

            def sample_linear_pdf(u, x0, x1, d):
                return d * (-x0 + torch.sqrt(x0 ** 2 + (x1 ** 2 - x0 ** 2) * u)) / (x1 - x0 + 1e-9)

            def depth_to_occupancy(depth, depth_values):
                depth_sigma = depth_values * self.conf.depth_sigma  # 1.0 / (math.sqrt(2 * math.pi) * depth_sigma) *

                occupancy = torch.exp(-0.5 * ((
                                                          depth - depth_values) / depth_sigma) ** 2)  # 1.0 / (math.sqrt(2 * math.pi) * depth_sigma) *
                return torch.where(mask[:, None], occupancy, torch.tensor(0, device=device))

            depth_distribution_samples = self.conf.depth_distribution_samples
            depth_sample_coords, depth, _ = get_sample_coords(
                torch.linspace(0, 1, depth_distribution_samples, device=device)[None, :])
            depth_values = \
            torch.nn.functional.grid_sample(input_depth[None, None], depth_sample_coords[None, :, :, 0:2])[0, 0]
            occupancy = depth_to_occupancy(depth, depth_values)

            # sample based on linear pdf
            tsample_index = torch.multinomial(occupancy + 1e-9, num_samples=length,
                                              replacement=length > self.conf.depth_distribution_samples)
            tsample = torch.rand_like(tsample_index, dtype=torch.float)

            x0 = torch.gather(depth_values, 1, tsample_index)
            limit = torch.tensor(self.conf.depth_distribution_samples - 1, device=device).unsqueeze(0)
            x1 = torch.gather(depth_values, 1, torch.minimum(tsample_index + 1, limit))

            delta = 1.0 / (depth_distribution_samples - 1)
            tsample = tsample_index * delta + sample_linear_pdf(tsample, x0, x1, delta)
            # tsample = tsample.reshape((height,width,length))

            sample_coords, depth, pos = get_sample_coords(tsample)
            image_values = torch.nn.functional.grid_sample(input_image[None], sample_coords[None])[0]
            depth_values = torch.nn.functional.grid_sample(input_depth[None, None], sample_coords[None])[0, 0]
            occupancy = depth_to_occupancy(depth, depth_values)

            # depth = depth.reshape((height,width,length))
            occupancy = occupancy.reshape((height, width, length))
            image_values = image_values.reshape((input_image.shape[0], height, width, length))
            depth_values = depth_values.reshape((height, width, length))
            tsample = tsample.reshape((height, width, length))
            pos = pos.reshape((height, width, length, 3))
            mask = torch.isfinite(tsample)

            out_masks.append(mask)
            out_pos.append(pos)
            out_image_values.append(image_values)
            out_occupancy_values.append(occupancy)
            out_tsamples.append(tsample)

        if len(out_masks) == 0:
            return None, None, None, None, None

        return torch.cat(out_masks, dim=2), \
            torch.cat(out_image_values, dim=3), \
            torch.cat(out_occupancy_values, dim=2), \
            torch.cat(out_tsamples, dim=2), \
            torch.cat(out_pos, dim=2)

    def chunks(self):
        result = []
        for i in range(len(self.visible_camera_idx)):
            height, width, _ = self.chunk_size[i].shape
            for j in range(height):
                for k in range(width):
                    result.append((i, j, k))
        return result

    # @torch.compile
    def update_chunk(self, chunk_id, features, depths, cameras):
        plane_idx, i, j = chunk_id

        b, c, height, width = features.shape
        assert c == self.conf.num_features

        print("Sampling images")
        masks, values, occupancy, tsamples, pos = self.sample_images(chunk_id, features, depths, cameras)
        if masks is None:
            return False

        map = self.map
        x0, y0 = self.chunk_coord[plane_idx][i, j]
        sx, sy = self.chunk_size[plane_idx][i, j]
        x1, y1 = x0 + sx, y0 + sy

        print("Pooling images")
        weight, alpha, mu, var = pool_images(masks, values, occupancy, tsamples,
                                             weight_init=map.atlas_weight[y0:y1, x0:x1],
                                             mu_init=map.atlas_mu[:, :, y0:y1, x0:x1],
                                             var_init=map.atlas_var[:, :, y0:y1, x0:x1]
                                             )

        atlas_weight, atlas_alpha, atlas_mu, atlas_var = map.atlas_weight.detach(), map.atlas_alpha.detach(), map.atlas_mu.detach(), map.atlas_var.detach()

        atlas_weight[y0:y1, x0:x1] = weight
        atlas_alpha[:, y0:y1, x0:x1] = alpha
        atlas_mu[:, :, y0:y1, x0:x1] = mu
        atlas_var[:, :, y0:y1, x0:x1] = var

        map.atlas_weight.value = atlas_weight
        map.atlas_alpha.value = atlas_alpha
        map.atlas_mu.value = atlas_mu
        map.atlas_var.value = atlas_var

        return True

    def plot_cameras_on_chunk(self, chunk):
        return
        plane, i, j = chunk

        visible_camera_ids = self.visible_camera_idx[plane][i][j]
        if len(visible_camera_ids) == 0:
            return
        max_cameras = 1

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        fp = torch.stack([frustum_points(self.cameras[idx]) for idx in
                          visible_camera_ids[:min(len(visible_camera_ids), max_cameras)]])
        fp = fp.reshape((-1, 3))

        planes = self.planes

        x0 = self.chunk_coord[plane][i][j] / planes.coord_size[plane]
        x1 = (self.chunk_coord[plane][i][j] + self.chunk_size[plane][i][j]) / planes.coord_size[plane]

        points = plane_box_points(self.planes, plane, x0, x1)
        points = points.reshape((-1, 3))

        ax.scatter(fp[:, 0], fp[:, 1], fp[:, 2])
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

        ax.set_aspect('equal')

        plt.savefig(f"out/chunk_{plane}_{i}_{j}_cameras.png")
        # plt.show()

    def train(self, image_db, depth_db):
        conf = self.conf

        device = select_device()
        optim = torch.optim.Adam(params=self.map.encoder.parameters(), lr=1e-3)

        first = True
        for epoch in range(conf.num_epochs):
            for plane, i, j in tqdm.tqdm(self.chunks()):
                ids = self.visible_camera_idx[plane][i][j]
                if len(ids) == 0:
                    continue

                cameras = [self.cameras[id].to(device) for id in ids]

                print(f"Updating chunk {plane} {i} {j} - {len(cameras)}")

                features = torch.stack([image_db[id]["image"] for id in ids]).to(device)
                depths = torch.stack([depth_db[id]["image"] for id in ids]).to(device)

                b, c, height, width = features.shape

                print(f"Encoding chunk {plane} {i} {j} - {len(cameras)}")

                features = self.map.encoder(features.permute(0, 2, 3, 1).reshape(b * height * width, c))
                features = features.reshape(b, height, width, conf.num_features).permute(0, 3, 1, 2)

                print("Generated feature")

                if not self.update_chunk((plane, i, j), features=features, depths=depths, cameras=cameras):
                    continue
                print("Updated chunk")
                if first:
                    continue

                if len(self.inside_camera_idx[plane][i][j]) == 0:
                    continue
                optim.zero_grad()

                loss = 0.0
                for id in self.inside_camera_idx[plane][i][j]:
                    cam, image, depth = cameras[id], features[id], depths[id]
                    loss = loss + mine_negatives(self.map, cam, image, depth, self.conf.ransac)
                    print("Mined negative")
                loss.backward()

                optim.step()
                optim.zero_grad()

                print("Loss loss = ", loss)
                gc.collect()
                # further zeroeing required?, self.map.parameters()

            first = False


