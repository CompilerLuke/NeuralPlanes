from NeuralPlanes.plane import Planes, frustum_plane_visibility, plane_box_points, draw_planes
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
from torch.utils.tensorboard import SummaryWriter
import datetime

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
    column_sampling_density: float = 10
    depth_distribution_samples: int = 10
    min_weight_thresh: float = 0.05

class NeuralMapBuilder:
    planes: Planes
    image_ids: List[str]
    cameras: List[Camera]
    conf: NeuralMapBuilderConf
    map: NeuralMap

    chunk_coord: List[torch.Tensor]  # (h,w,2)
    chunk_size: List[torch.Tensor]  # (h,w,2)
    visible_camera_idx: List[List[List[torch.Tensor]]]
    inside_camera_idx: List[List[List[torch.Tensor]]] # indices refer to elements of visible_camera_idx, so absolute id = visible_camera_idx[i][j][k][inside_camera_idx[i][j][k]]

    def __init__(self, cameras: List[Camera], planes=None, map: NeuralMap = None, encoder=None, conf: NeuralMapBuilderConf = NeuralMapBuilderConf()):
        self.cameras = cameras
        self.conf = conf
        self.device = select_device()
    
        if map:
            self.planes = map.planes
            self.map = map
        else:
            self.planes = planes
            width, height = planes.atlas_size

            self.map = NeuralMap(
                encoder = encoder,
                planes=planes,
                conf=conf
            )

        self.planes = self.planes.to('cpu')
        self._assign_cameras_to_planes()
        self._select_chunk_order()
        self.map = self.map.to(self.device)
        self.cameras = [cam.to(self.device) for cam in self.cameras]

    def _select_chunk_order(self):
        result = []
        for i in range(len(self.visible_camera_idx)):
            height, width, _ = self.chunk_size[i].shape
            for j in range(height):
                for k in range(width):
                    result.append((i, j, k))
            
        result = torch.tensor(result)[torch.randperm(len(result))[:len(result)]]
        self.chunk_order = result

    def _chunk_neural_map(self, plane_id: int):
        chunk_size = self.conf.chunk_size

        planes = self.planes
        width, height = planes.coord_size[plane_id]
        width, height = int(width), int(height)

        v0, u0 = torch.meshgrid(chunk_size * torch.arange(0, int(ceil(float(height) / chunk_size))),
                                chunk_size * torch.arange(0, int(ceil(float(width) / chunk_size))))

        u1 = torch.minimum(u0 + chunk_size, torch.tensor(width))
        v1 = torch.minimum(v0 + chunk_size, torch.tensor(height))

        return torch.stack([u0, v0], dim=2), torch.stack([u1 - u0, v1 - v0], dim=2)
    
    def _chunk_relative_extent(self,plane_id,i,j):
        chunk_coord = self.chunk_coord[plane_id][i][j]
        chunk_size = self.chunk_size[plane_id][i][j]
        size = self.planes.coord_size[plane_id]
        rel_x0 = chunk_coord.type(torch.float) / size
        rel_x1 = (chunk_coord + chunk_size).type(torch.float) / size
        return rel_x0, rel_x1

    def _assign_cameras_to_planes(self):
        print("Assigning cameras to planes : ", len(self.planes))
        
        planes = self.planes
        cameras = self.cameras

        fps = torch.stack([frustum_points(cam, cam.near, cam.far) for cam in cameras])

        self.chunk_coord = []
        self.chunk_size = []
        self.visible_camera_idx = []
        self.inside_camera_idx = []

        planes = self.planes.to('cpu')
        cameras = [cam.to('cpu') for cam in self.cameras]

        for plane_id in tqdm.tqdm(range(len(planes))):
            chunk_coord, chunk_size = self._chunk_neural_map(plane_id)

            plane_size_unsqueezed = torch.tensor(planes.coord_size[plane_id])[None, None]
            rel_x0 = chunk_coord.type(torch.float) / plane_size_unsqueezed
            rel_x1 = (chunk_coord + chunk_size).type(torch.float) / plane_size_unsqueezed

            visibility = frustum_plane_visibility(planes, plane_id, fps, (0., 0.), (1., 1.))
            indices = torch.arange(len(fps), device=self.device)[visibility]
            fps_visible = fps[visibility]

            indices_chunk_grid = []
            indices_inside_chunk_grid = []

            x0 = planes.x0s[plane_id]
            us = planes.us[plane_id]
            vs = planes.vs[plane_id]
            us = us / torch.linalg.norm(us)**2
            vs = vs / torch.linalg.norm(vs)**2

            for i in range(chunk_coord.shape[0]):
                indices_per_row = []
                indices_inside_per_row = []

                for j in range(chunk_coord.shape[1]):
                    visibility_chunk = frustum_plane_visibility(planes, plane_id, fps_visible, tuple(rel_x0[i, j]),
                                                                tuple(rel_x1[i, j]))
                    indices_chunk = indices[visibility_chunk]
                    
                    chunk_center = planes.x0s[plane_id] + 0.5*planes.us[plane_id]*(rel_x0[i,j,0]+rel_x1[i,j,0]) + 0.5*planes.vs[plane_id]*(rel_x0[i,j,1]+rel_x1[i,j][1])

                    focal = 4.

                    cams = [cameras[x] for x in indices_chunk]
                    dist_to_chunk = torch.tensor([torch.linalg.norm(cam.t + cam.R @ torch.tensor([0.,0.,focal],dtype=torch.float) - chunk_center, dim=0) for cam in cams])

                    indices_chunk = indices_chunk[torch.argsort(dist_to_chunk)]

                    best_indices = []
                    for _, k in zip(range(self.conf.max_views_per_chunk), indices_chunk):
                        tooclose = False
                        for l in best_indices:
                            if torch.linalg.norm(cameras[k].t - cameras[l].t) < 0.5:
                                tooclose = True
                                break
                        if not tooclose:
                            best_indices.append(k)

                    indices_chunk = best_indices
                    def inside_chunk(t):
                        return rel_x0[i, j, 0] <= torch.dot(us, t - x0) <= rel_x1[i, j, 0] and rel_x0[i, j, 1] <= torch.dot(vs, t - x0) <= rel_x1[i, j, 1]

                    inside_chunk_indices = []
                    for k, idx in enumerate(indices_chunk):
                        cam = cameras[idx]
                        if inside_chunk(cam.t) and inside_chunk(cam.t + cam.R @ torch.tensor([0.,0.,focal],dtype=torch.float)): # todo: use mean depth to compute overlap
                            inside_chunk_indices.append(k)

                    indices_per_row.append(indices_chunk)
                    indices_inside_per_row.append(torch.tensor(inside_chunk_indices, dtype=torch.int))

                indices_chunk_grid.append(indices_per_row)
                indices_inside_chunk_grid.append(indices_inside_per_row)

            self.chunk_coord.append(chunk_coord.to(self.device))
            self.chunk_size.append(chunk_size.to(self.device))
            self.visible_camera_idx.append(indices_chunk_grid)
            self.inside_camera_idx.append(indices_inside_chunk_grid)

            for i in range(chunk_coord.shape[0]):
                for j in range(chunk_coord.shape[1]):
                    self.plot_cameras_on_chunk((plane_id, i, j))

    #@torch.compile
    def sample_images(self, chunk_id, images, depths, cameras):
        planes = self.planes
        device = self.device

        plane_id, i, j = chunk_id

        conf = self.conf

        chunk_coord = self.chunk_coord[plane_id][i][j]
        chunk_size = self.chunk_size[plane_id][i][j]

        width, height = chunk_size
        width = int(width)
        height = int(height)

        rel_x0, rel_x1 = self._chunk_relative_extent(plane_id, i, j)

        delta = (rel_x1 - rel_x0) / torch.tensor([width, height], device=rel_x0.device)

        v, u = torch.meshgrid(rel_x0[None,1] + delta[None,1]*torch.arange(height,device=device), rel_x0[None,0] + delta[None,0]*torch.arange(width,device=device))

        out_masks = []
        out_image_values = []
        out_occupancy_values = []
        out_z = []
        out_pos = []

        planes_x0s = planes.x0s[plane_id][None, None].to(self.device)
        planes_us =  planes.us[plane_id][None, None].to(self.device)
        planes_vs = planes.vs[plane_id][None, None].to(self.device)
        planes_planes = planes.planes[plane_id, 0:3][None, None].to(self.device)

        for input_image, input_depth, cam in zip(images, depths, cameras):
            #print(input_image.shape, input_depth.shape, cam.size)
            #assert input_image.shape[1:] == input_depth.shape

            c, img_height, img_width = input_image.shape
            #assert [img_width, img_height] == list(cam.size)

            fp = frustum_points(cam)

            origin = planes_x0s + u[:, :, None] * planes_us + v[:, :, None] * planes_vs
            dir = planes_planes.repeat((height, width, 1))
            origin, dir = origin.reshape((-1, 3)), dir.reshape((-1, 3))

            tmin, tmax = frustum_ray_intersection(fp, origin, dir)
            mask = torch.isfinite(tmin)

            #
            start = origin.reshape((-1, 3)) + tmin[:, None] * dir.reshape((-1, 3))
            end = origin.reshape((-1, 3)) + tmax[:, None] * dir.reshape((-1, 3))


            length = self.conf.column_sampling_density #* torch.max(torch.cat([torch.linalg.norm(end - start, dim=1)[mask], torch.zeros(1,device=device)]))
            length = int(ceil(length))

            if length == 0:
                continue

            def get_sample_coords(tsample):
                tsample = tmin[:,None] * (1 - tsample) + tmax[:,None] * tsample
                pos = (origin[:,None] + tsample[:,:,None] * dir[:,None])
                proj = cam.to_image_space(pos.reshape(-1,3)).reshape((height*width,tsample.shape[1],3))
                sample_coords = 2 * (proj[:,:,0:2] / torch.tensor([img_width+1, img_height+1],device=device)) - img_width/(img_width+1)
                return sample_coords, proj[:,:,2], pos

            def sample_linear_pdf(u, x0, x1, d):
                x0,x1 = torch.minimum(x0,x1), torch.maximum(x0,x1)
                return d * (-x0 + torch.sqrt(x0 ** 2 + (x1 ** 2 - x0 ** 2) * u)) / (x1 - x0 + 1e-9)

            def depth_to_occupancy(depth, depth_values):
                log_depth = torch.log(2 + depth_values)
                depth_sigma = depth_values * self.conf.depth_sigma
                occupancy = 2.4/log_depth * torch.exp(-0.5 * ((depth - depth_values) / depth_sigma) ** 2)
                return torch.where(mask[:,None], occupancy, torch.tensor(0,device=device))

            depth_distribution_samples = self.conf.depth_distribution_samples
            depth_sample_coords, depth, _ = get_sample_coords(torch.linspace(0,1,depth_distribution_samples,device=device)[None,:])
            depth_values = torch.nn.functional.grid_sample(input_depth[None, None], depth_sample_coords[None,:,:,0:2])[0, 0]

            occupancy = depth_to_occupancy(depth, depth_values)

            # sample based on linear pdf
            enable_importance_sampling = True
            if enable_importance_sampling:
                tsample_index = torch.multinomial(occupancy+1e-9, num_samples=length, replacement= length > self.conf.depth_distribution_samples)

                x0 = torch.gather(occupancy, 1, tsample_index)
                limit = torch.tensor(self.conf.depth_distribution_samples-1, device=device).unsqueeze(0)
                x1 = torch.gather(occupancy, 1, torch.minimum(tsample_index+1, limit))

                delta = 1.0/(depth_distribution_samples-1)
                offset = torch.rand((height*width, length),device=device)
                #sample_linear_pdf(tsample, x0, x1, delta)
                tsample = tsample_index*delta + offset*delta
                #tsample = tsample.reshape((height,width,length))
            else:
                tsample = torch.linspace(0, 1, length, device=device)[None].repeat(height*width, 1)

            sample_coords, depth, pos = get_sample_coords(tsample)
            image_values = torch.nn.functional.grid_sample(input_image[None], sample_coords[None])[0]
            depth_values = torch.nn.functional.grid_sample(input_depth[None,None], sample_coords[None])[0,0]

            z = torch.einsum("k,ijk->ij", planes.planes[plane_id,0:3].to(device), pos - planes.x0s.to(device)[plane_id][None,None,:])
            z = torch.where(mask[:,None], z, torch.tensor(0,device=device))

            occupancy = depth_to_occupancy(depth, depth_values)

            #depth = depth.reshape((height,width,length))
            occupancy = occupancy.reshape((height,width,length))
            image_values = image_values.reshape((c, height, width, length))
            depth_values = depth_values.reshape((height, width, length))
            #tsample = tsample.reshape((height, width, length))
            pos = pos.reshape((height, width, length, 3))
            z = z.reshape((height, width, length))
            mask = mask.reshape((height,width,1)).repeat(1,1,length)

            out_masks.append(mask)
            out_pos.append(pos)
            out_image_values.append(image_values)
            out_occupancy_values.append(occupancy)
            out_z.append(z)

        if len(out_masks) == 0:
            return None, None, None, None, None, None

        occupancy_per_image = torch.tensor([x.mean() for x in out_occupancy_values], device=self.device)

        return torch.cat(out_masks, dim=2), \
               torch.cat(out_image_values, dim=3), \
               torch.cat(out_occupancy_values, dim=2), \
               torch.cat(out_z, dim=2), \
               torch.cat(out_pos, dim=2), \
               occupancy_per_image

    def chunks(self):
        return self.chunk_order

    #@torch.compile
    def update_chunk(self, chunk_id, features, depths, cameras):
        plane_idx, i, j = chunk_id

        b,c,height,width = features.shape
        assert c == self.conf.num_features+1

        print("Sampling images")
        masks, values, occupancy, z, pos, occupancy_per_image = self.sample_images(chunk_id, features, depths, cameras)
        if masks is None:
            return False, None

        map = self.map
        normal = self.planes.planes[plane_idx,0:3]
        x0, y0 = self.planes.coord_x0[plane_idx] + self.chunk_coord[plane_idx][i, j]
        sx, sy = self.chunk_size[plane_idx][i, j]
        x1, y1 = x0 + sx, y0 + sy

        x0,x1 = int(x0),int(x1)
        y0,y1 = int(y0),int(y1)

        print("Pooling images")
        weight, values = pool_images(map, masks, values, occupancy, z, up= torch.dot(normal, torch.tensor([0.,0.,1.], device=normal.device)))

        atlas = map.atlas.detach()
        atlas_weight = map.atlas_weight.detach()
        atlas_weight[y0:y1, x0:x1] = weight
        atlas[:, y0:y1, x0:x1] = values
        map.atlas = atlas
        map.atlas_weight = atlas_weight

        if plane_idx==0:
            self.plot_map(pos, features=values, occupancy=occupancy, cameras=cameras, indices=[0])

        return True, occupancy_per_image #weight.mean() > self.conf.min_weight_thresh

    def plot_cameras_on_chunk(self, chunk):
        return
        plane, i, j = chunk

        visible_camera_ids = self.visible_camera_idx[plane][i][j]
        if len(visible_camera_ids) == 0:
            return
        max_cameras = 1

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        fp = torch.stack([frustum_points(self.cameras[idx]) for idx in visible_camera_ids[:min(len(visible_camera_ids), max_cameras)]])
        fp = fp.reshape((-1,3))

        planes = self.planes

        x0 = self.chunk_coord[plane][i][j] / planes.coord_size[plane]
        x1 = (self.chunk_coord[plane][i][j]+self.chunk_size[plane][i][j]) / planes.coord_size[plane]

        points = plane_box_points(self.planes, plane, x0, x1)
        points = points.reshape((-1,3))

        ax.scatter(fp[:,0],fp[:,1],fp[:,2])
        ax.scatter(points[:,0], points[:,1], points[:,2])

        ax.set_aspect('equal')

        plt.savefig(f"out/chunk_{plane}_{i}_{j}_cameras.png")
        #plt.show()

    def plot_map(self, pos, features, occupancy, cameras, stride=4, indices=None):
        return

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        fps = [frustum_points(cam,cam.near,cam.far).cpu() for cam in cameras]

        import numpy as np

        xs = []
        for fp in fps:
            for j in range(2):
                for i in range(4):
                    table0 = [0,1,1,0]
                    table1 = [0,0,1,1]
                    xs.append(fp[j][table1[i]][table0[i]])
                xs.append([np.inf,np.inf,np.inf])
            for i in range(4):
                xs.append(fp[0,i//2,i%2])
                xs.append(fp[1,i//2,i%2])
                xs.append([np.inf,np.inf,np.inf])

        xs = np.array(xs)
        ax.plot(xs[:,0], xs[:,1], xs[:,2])
        ax.set_aspect('equal')
        plt.show()

        planes = self.planes
        features = features[::stride, ::stride]
        features = features[1:,].reshape((self.conf.num_features,-1)).transpose(0,1)
        occupancy = occupancy[::stride, ::stride].flatten()

        pos_flat = pos[::stride, ::stride].reshape((-1,3))
        pos_flat = pos_flat.detach().cpu()

        if self.conf.num_features==3:
            color_flat = features.detach().cpu()
            #draw_planes(ax, planes, color=self.map.atlas_mu[0], indices=indices)
            ax.scatter(pos_flat[:, 0], pos_flat[:, 1], pos_flat[:, 2], c=color_flat, s= 1/4)
        else:
            color_flat = occupancy.detach().cpu()
            color_flat = plt.cm.viridis(color_flat)
            #draw_planes(ax, planes, color=self.map.atlas_weight, indices=indices)
            ax.scatter(pos_flat[:,0], pos_flat[:,1], pos_flat[:,2], c=color_flat, s= 1/4)

        #ax.set_aspect('equal')
        plt.show()

    def plot_images(self, images):
        return

        num_images = len(images)
        if num_images <= 4:
            fig, axes = plt.subplots(num_images)
            for ax, image in zip(axes, images):
                ax.imshow(image["image"].permute(1,2,0).cpu())
        else:
            grid = int(num_images**0.5)
            fig, axes = plt.subplots(grid, grid)
            for i in range(grid):
                for j in range(grid):
                    axes[i,j].imshow(images[grid*i + j]["image"].permute(1,2,0).cpu())

    def extract_features(self, ids, image_db, depth_db):
        device = self.device
        cameras =  [self.cameras[id].to(device) for id in ids]
        device = self.device

        images = [ image_db[id] for id in ids ]
        features_orig = torch.stack([ image["image"] for image in images]).to(device)
        depths = torch.stack([depth_db[id]["image"] for id in ids]).to(device)

        self.plot_images(images)

        assert all([torch.isclose(cam.t.cpu(), image["camera"].t.cpu()).all() for cam, image in zip(cameras, images)])
        print("Encoding chunk")
        b, c, height, width = features_orig.shape

        features = self.map.encode({"image":  features_orig}, depths)
        return cameras, features, depths

    def gen(self, image_db, depth_db, save_checkpoint):
        map = self.map
        map.atlas_weight.zero_()
        map.atlas.zero_()

        with torch.no_grad():
            conf = self.conf

            device = select_device()

            for plane, i, j in tqdm.tqdm(self.chunks()):
                ids = self.visible_camera_idx[plane][i][j]
                print("visible camera idx ", ids)
                if len(ids) == 0:
                    continue

                cameras, features, depths = self.extract_features(ids, image_db, depth_db)
                print(f"Updating chunk {plane} {i} {j} - {len(cameras)}")
                self.update_chunk((plane, i, j), features=features, depths=depths, cameras=cameras)
                print("Updated chunk")

            if save_checkpoint:
                torch.save(self.map.state_dict(), save_checkpoint)

    def train(self, image_db, depth_db, log_dir, save_checkpoint):
        conf = self.conf

        writer = SummaryWriter(log_dir)

        device = select_device()
        optim = torch.optim.Adam(params= self.map.parameters(), lr=3e-2)

        first = True
        iter = 0
        for epoch in range(conf.num_epochs):
            for plane, i, j in tqdm.tqdm(self.chunks()):
                ids = self.visible_camera_idx[plane][i][j]
                if len(ids) == 0:
                    continue

                cameras, features, depths = self.extract_features(ids, image_db, depth_db)
                print(f"Updating chunk {plane} {i} {j} - {len(cameras)}", features.shape)

                updated, occupancy_per_image = self.update_chunk((plane, i, j), features=features, depths=depths, cameras=cameras)
                if not updated:
                    continue

                iter += 1

                print("Updated chunk")
                #if first:
                #    continue

                inside_camera_idx = self.inside_camera_idx[plane][i][j]
                inside_camera_idx = [idx for idx in inside_camera_idx if occupancy_per_image[idx] > self.conf.min_weight_thresh]

                if len(inside_camera_idx) == 0:
                    continue

                rel_x0, rel_x1 = self._chunk_relative_extent(plane,i,j)

                losses = torch.zeros(len(inside_camera_idx), device=self.device)
                scores = torch.zeros(len(inside_camera_idx), device=self.device)
                neg_scores = torch.zeros(len(inside_camera_idx), device=self.device)
                weights = torch.zeros(len(inside_camera_idx), device=self.device)

                has_loss = False
                for i, cam_id in enumerate(inside_camera_idx):
                    cam, image, depth = cameras[cam_id], features[cam_id], depths[cam_id]
                    negative_cam = [cameras[j] for j in inside_camera_idx if j != cam_id and torch.linalg.norm(cameras[j].t - cam.t) > 0.5]

                    result = mine_negatives(self.map, plane, rel_x0, rel_x1, cam, negative_cam, image, depth, self.conf.ransac)
                    if not result:
                        weights[i] = 0
                        continue
                    l, score, neg_score = result
                    losses[i] = l
                    weights[i] = 1.0 #occupancy_per_image[cam_id]
                    scores[i] = score
                    neg_scores[i] = neg_score
                    has_loss = True
                    print("Mined negative")

                if not has_loss:
                    continue
                loss = losses.sum() / weights.sum()
                loss.backward()

                if not torch.isfinite(loss):
                    print("Loss is inf")
                    raise "NaN value"

                optim.step()
                optim.zero_grad()

                writer.add_scalar('training loss', loss.item(), iter)

                if neg_scores.sum() > 0:
                    writer.add_scalars('contrastive', { 'score': scores.mean().item(), 'neg_score':  neg_scores[neg_scores>0].mean().item()}, iter)

                print("Loss loss = ", loss)


            first = False

        if save_checkpoint:
            torch.save(self.map.state_dict(), save_checkpoint)
        print("Generating FINAL MAP")
        self.gen(image_db, depth_db, save_checkpoint)
        writer.close()
