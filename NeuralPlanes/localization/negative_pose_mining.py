from NeuralPlanes.localization.map import NeuralMap
from NeuralPlanes.camera import Camera
from NeuralPlanes.localization.pose_scoring import score_pose, score_cells
from dataclasses import dataclass
import torch
import cv2
import numpy as np
import copy
import logging

log = logging.getLogger(__name__)

@dataclass
class RansacMiningConf:
    num_ref_kps: int = 8
    kp_per_ref: int = 20
    scale_thresh: float = 1.5
    ransac_it: int = 10
    ransac_sample: int = 8
    top_k: int = 5


def mine_negatives(map: NeuralMap, camera: Camera, image, depths, conf: RansacMiningConf):
    num_ref_kps, kp_per_ref = conf.num_ref_kps, conf.kp_per_ref
    assert conf.ransac_sample <= conf.num_ref_kps

    device = image.device

    prob, weight, pos, features = score_pose(map, camera, image, depths, dense=True)
    height,width,samples = prob.shape

    #avg_dist_ref = torch.mean(torch.linalg.norm(pos[None,:] - pos[:,None], dim=2))
    ref_score = (prob * weight).sum() / (1e-9 + weight.sum())

    features = features.reshape((features.shape[0],-1))

    index = torch.max(prob * weight, dim=2)[1]
    prob = prob[index[None,None]].flatten()
    weight = weight[index[None,None]].flatten()
    pos = pos[index[None,None]].reshape((-1,3))

    indices_ref = torch.topk(prob * weight, k=num_ref_kps)[1]

    pos_ref_kp = pos[indices_ref]
    features_ref_kp = features.reshape((map.conf.num_features,-1))[:,indices_ref]

    planes = map.planes
    height,width = planes.coord_size[0]
    v,u = torch.meshgrid(torch.linspace(0,1,height), torch.linspace(0,1,width))
    coords = torch.stack([u.flatten(),v.flatten()], dim=1)
    n_coords = coords.shape[0]

    floor_plane = 0 # todo: get actual floor planes

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

        match_pos = planes.x0s[None,0] + planes.us[None,0]*match_coords[:,0,None] + planes.vs[None,0]*match_coords[:,1,None]
        ref_pos = pos_ref_kp[sample_ref_kp]

        # todo: this assumes plane is floor plane
        match_pos = match_pos[:,0:2].detach().numpy().astype(float)
        ref_pos = ref_pos[:,0:2].detach().numpy().astype(float)

        print("Computing affine transformation")
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
        ], dtype=torch.float)

        delta_t = torch.tensor([tx,ty,0], dtype=torch.float)

        new_cam : Camera = copy.copy(camera)
        new_cam.R = delta_R @ camera.R
        new_cam.t = delta_R @ camera.t + delta_t

        score = score_pose(map, new_cam, image, depths * s)

        top_k_scores = torch.topk(torch.cat([top_k_scores, score.unsqueeze(0)]), k=conf.top_k, dim=0)[0]

    if top_k_scores.mean() < 1e-2:
        loss = -ref_score
    else:
        loss = -ref_score / top_k_scores.mean()
    return loss