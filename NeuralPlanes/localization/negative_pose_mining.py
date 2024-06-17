from NeuralPlanes.localization.map import NeuralMap
from NeuralPlanes.camera import Camera
from NeuralPlanes.localization.pose_scoring import score_pose, score_cells
from dataclasses import dataclass
import torch
import cv2
import numpy as np
import copy
import logging
from math import ceil, floor

log = logging.getLogger(__name__)

@dataclass
class RansacMiningConf:
    num_ref_kps: int = 8
    kp_per_ref: int = 20
    scale_thresh: float = 1.5
    ransac_it: int = 10
    ransac_sample: int = 8
    top_k: int = 5
    area: float = 5
    scale_factor: float = 1.0 / 8


def mine_negatives(map: NeuralMap, camera: Camera, image, depths, conf: RansacMiningConf):
    device = image.device

    # Down-sample
    camera = camera.scale(conf.scale_factor).to(device)
    image = torch.nn.functional.interpolate(image[None], scale_factor=conf.scale_factor)[0]
    depths = torch.nn.functional.interpolate(depths[None, None], scale_factor=conf.scale_factor)[0,0]

    num_ref_kps, kp_per_ref = conf.num_ref_kps, conf.kp_per_ref
    assert conf.ransac_sample <= conf.num_ref_kps


    prob, weight, pos, features = score_pose(map, camera, image, depths, dense=True)
    height,width,samples = prob.shape

    #avg_dist_ref = torch.mean(torch.linalg.norm(pos[None,:] - pos[:,None], dim=2))
    ref_score = (prob * weight).sum() / (1e-9 + weight.sum())

    features = features.reshape((features.shape[0],-1))

    index = torch.max(prob * weight, dim=2)[1]

    prob = prob.take_along_dim(index[:,:,None],2).flatten()
    weight = weight.take_along_dim(index[:,:,None],2).flatten()
    pos = pos.take_along_dim(index[:,:,None,None],2).reshape((-1,3))

    indices_ref = torch.topk(prob * weight, k=num_ref_kps)[1]

    pos_ref_kp = pos[indices_ref]
    features_ref_kp = features.reshape((map.conf.num_features,-1))[:,indices_ref]

    planes = map.planes

    floor_plane = 0 # todo: get actual floor planes

    x0 = planes.x0s[floor_plane]
    us = planes.us[floor_plane]
    vs = planes.vs[floor_plane]
    us = us / torch.linalg.norm(us)**2
    vs = vs / torch.linalg.norm(vs)**2

    u0 = min(max(0, torch.dot(camera.t.cpu()-conf.area-x0, us)), 1)
    u1 = min(max(0, torch.dot(camera.t.cpu()+conf.area-x0, us)), 1)
    v0 = min(max(0, torch.dot(camera.t.cpu()-conf.area-x0, vs)), 1)
    v1 = min(max(0, torch.dot(camera.t.cpu()+conf.area-x0, vs)), 1)

    u0,u1 = min(u0,u1), max(u0,u1)
    v0,v1 = min(v0,v1), max(v0,v1)

    plane_height, plane_width = planes.coord_size[floor_plane]

    height,width = int(ceil(u1*plane_width)-floor(u0*plane_width)), int(ceil(v1*plane_height)-floor(v0*plane_height))
    print("Map size searched ", height, width, u0, u1, v0, v1)
    v,u = torch.meshgrid(torch.linspace(u0,u1,height,device=device), torch.linspace(v0,v1,width,device=device))
    coords = torch.stack([u.flatten(),v.flatten()], dim=1)
    n_coords = coords.shape[0]

    def score_cell_per_feature(feature):
        # todo: avoid repeating feature and broadcast directly
        log_prob, weight = score_cells(map, torch.tensor([floor_plane], device=device), coords[None], feature[:,None].repeat(1,n_coords), torch.ones((n_coords), device=device))
        return torch.exp(log_prob[0]) * weight[0]

    score_matrix = torch.vmap(score_cell_per_feature, in_dims=1)(features_ref_kp)
    indices_match = torch.topk(score_matrix, k=kp_per_ref, dim=1)[1]

    top_k_scores = torch.zeros(conf.top_k, device=device)

    for i in range(conf.ransac_it):
        sample = conf.ransac_sample

        sample_ref_kp = torch.randperm(num_ref_kps)[:sample]
        sample_matching_kp = torch.randint(0, kp_per_ref, (sample,))

        match_coords = coords[indices_match[sample_ref_kp, sample_matching_kp]]

        match_pos = planes.x0s[None,0].to(device) + planes.us[None,0].to(device)*match_coords[:,0,None] + planes.vs[None,0].to(device)*match_coords[:,1,None]
        ref_pos = pos_ref_kp[sample_ref_kp]

        # todo: this assumes plane is floor plane
        match_pos = match_pos[:,0:2].cpu().detach().numpy().astype(float)
        ref_pos = ref_pos[:,0:2].cpu().detach().numpy().astype(float)

        #print("Computing affine transformation")
        matrix, inliers = cv2.estimateAffinePartial2D(ref_pos, match_pos, ransacReprojThreshold=0.5)

        if matrix is None:
            print("Failed to solve for pose", ref_pos, match_pos)
            continue

        a, b, tx = matrix[0]
        c, d, ty = matrix[1]

        theta = np.arctan2(-b, a)
        s = a / np.cos(theta)

        delta_R = torch.tensor([
            [a/s, b/s, 0],
            [c/s, d/s, 0],
            [0,   0,   1]
        ], dtype=torch.float, device=device)

        delta_t = torch.tensor([tx,ty,0], dtype=torch.float, device=device)

        new_cam : Camera = copy.copy(camera)
        new_cam.R = delta_R @ camera.R
        new_cam.t = delta_R @ camera.t + delta_t

        #print("Scoring pose")
        score = score_pose(map, new_cam, image, depths * s)
        print("Pose ", score, delta_R, delta_t)
        top_k_scores = torch.topk(torch.cat([top_k_scores, score.unsqueeze(0)]), k=conf.top_k, dim=0)[0]

    print("Total weight ", map.atlas_weight.sum())
    print("Top matches ", ref_score, top_k_scores, "mean depth= ", depths.mean())

    if top_k_scores.mean() < 1e-2:
        loss = -ref_score
    else:
        loss = -ref_score / top_k_scores.mean()
    return loss