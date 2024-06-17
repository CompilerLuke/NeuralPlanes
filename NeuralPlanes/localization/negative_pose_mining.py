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


def mine_negatives(map: NeuralMap, floor_plane: int, rel_x0: torch.Tensor, rel_x1: torch.Tensor, camera: Camera, image, depths, conf: RansacMiningConf):
    device = image.device
    plane_indices = torch.tensor([floor_plane])

    # Down-sample
    camera = camera.scale(conf.scale_factor).to(device)
    image = torch.nn.functional.interpolate(image[None], scale_factor=conf.scale_factor)[0]
    depths = torch.nn.functional.interpolate(depths[None, None], scale_factor=conf.scale_factor)[0,0]

    num_ref_kps, kp_per_ref = conf.num_ref_kps, conf.kp_per_ref
    assert conf.ransac_sample <= conf.num_ref_kps


    prob, weight, pos, features = score_pose(map, camera, image, depths, dense=True, plane_indices=plane_indices, rel_x0=rel_x0[None], rel_x1=rel_x1[None])
    height,width,samples = prob.shape

    #avg_dist_ref = torch.mean(torch.linalg.norm(pos[None,:] - pos[:,None], dim=2))
    ref_score = (prob * weight).sum() / (1e-9 + weight.sum())

    if not torch.isfinite(ref_score):
        score_pose(map, camera, image, depths, dense=True, plane_indices=plane_indices, rel_x0=rel_x0[None], rel_x1=rel_x1[None])
        return torch.tensor(0.0, device=device)

    features = features.reshape((features.shape[0],-1))

    index = torch.max(prob * weight, dim=2)[1]

    prob = prob.take_along_dim(index[:,:,None],2).flatten()
    weight = weight.take_along_dim(index[:,:,None],2).flatten()
    pos = pos.take_along_dim(index[:,:,None,None],2).reshape((-1,3))

    indices_ref = torch.topk(prob * weight, k=num_ref_kps)[1]

    pos_ref_kp = pos[indices_ref]
    features_ref_kp = features[:,indices_ref]

    planes = map.planes

    x0 = planes.x0s[floor_plane]
    normal = planes.planes[floor_plane,0:3]
    us = planes.us[floor_plane]
    vs = planes.vs[floor_plane]

    us_len = torch.linalg.norm(us)
    vs_len = torch.linalg.norm(vs)
    un = us / us_len
    vn = vs / vs_len

    u0,v0 = rel_x0 
    u1,v1 = rel_x1
    plane_width, plane_height = planes.coord_size[floor_plane]

    height,width = int(ceil(u1*plane_width)-floor(u0*plane_width)), int(ceil(v1*plane_height)-floor(v0*plane_height))
    print("Map size searched ", height, width, u0, u1, v0, v1)
    v,u = torch.meshgrid(torch.linspace(v0,v1,height,device=device), torch.linspace(u0,u1,width,device=device))
    coords = torch.stack([u.flatten(),v.flatten()], dim=1)
    n_coords = coords.shape[0]

    def score_cell_per_feature(feature):
        # todo: avoid repeating feature and broadcast directly
        log_prob, weight = score_cells(map, plane_indices.to(device), coords[None], feature[:,None].repeat(1,n_coords), torch.ones((n_coords), device=device), x0= rel_x0[None], x1= rel_x1[None])
        return torch.exp(log_prob[0]) * weight[0]

    score_matrix = torch.vmap(score_cell_per_feature, in_dims=1)(features_ref_kp)
    indices_match = torch.topk(score_matrix, k=kp_per_ref, dim=1)[1]

    negatives = torch.zeros(conf.ransac_it, device=device)
    negatives_count = 0

    x0 = x0.to(device)
    un = un.to(device)
    vn = vn.to(device)
    rel_to_abs_coord = torch.stack([us_len, vs_len]).to(device)
    camera_pos_plane_basis = torch.stack([ torch.dot(un, camera.t - x0), torch.dot(vn, camera.t-x0) ])
    #, camera_v = camera.t - torch.linalg.norm(camera.t,normal)*normal 

    for i in range(conf.ransac_it):
        sample = conf.ransac_sample

        sample_ref_kp = torch.randperm(num_ref_kps)[:sample]
        sample_matching_kp = torch.randint(0, kp_per_ref, (sample,))

        match_coords = coords[indices_match[sample_ref_kp, sample_matching_kp]]

        # Coordinate in plane basis, with origin at camera
        match_pos = match_coords * rel_to_abs_coord[None,:] - camera_pos_plane_basis
        
        ref_pos = pos_ref_kp[sample_ref_kp]
        ref_pos =  torch.stack([torch.einsum("j,ij->i", un, ref_pos-x0), torch.einsum("j,ij->i", vn, ref_pos-x0)],dim=1) - camera_pos_plane_basis

        """
        if floor_plane==0:
            match_pos2 = planes.x0s[None,0].to(device) + planes.us[None,floor_plane].to(device)*match_coords[:,0,None] + planes.vs[None,floor_plane].to(device)*match_coords[:,1,None] - camera.t
            ref_pos2 = pos_ref_kp[sample_ref_kp] - camera.t
            match_pos2 = match_pos2[:,0:2]
            ref_pos2 = ref_pos2[:,0:2]
            assert torch.isclose(match_pos, match_pos2).all() and torch.isclose(ref_pos, ref_pos2).all()
        """

        match_pos = match_pos.cpu().detach().numpy().astype(float)
        ref_pos = ref_pos.cpu().detach().numpy().astype(float)

        #print("Computing affine transformation")
        matrix, inliers = cv2.estimateAffinePartial2D(ref_pos, match_pos, ransacReprojThreshold=0.2)

        if matrix is None:
            print("Failed to solve for pose", ref_pos, match_pos)
            continue

        a, b, tx = matrix[0]
        c, d, ty = matrix[1]

        theta = np.arctan2(-b, a)
        s = a / np.cos(theta)

        if not (0.5 < s and s < 1.5):
            continue

        plane_to_xy = torch.stack([un.cpu(),vn.cpu(),normal.cpu()],dim=1)
        xy_to_plane = torch.linalg.inv(plane_to_xy)

        delta_R = torch.tensor([
            [a/s, b/s, 0],
            [c/s, d/s, 0],
            [0,   0,   1]
        ], dtype=torch.float)

        delta_R = (plane_to_xy @ delta_R @ xy_to_plane).to(device)

        delta_t = torch.tensor([tx,ty,0], dtype=torch.float)
        delta_t = (plane_to_xy @ delta_t).to(device)

        new_cam : Camera = copy.copy(camera)
        new_cam.R = delta_R @ camera.R
        new_cam.t = camera.t + delta_t

        #print("Scoring pose")
        score, weight = score_pose(map, new_cam, image, depths * s)
        if not torch.isfinite(score):
            score_pose(map, camera, image, depths, dense=True, plane_indices=plane_indices, x0=rel_x0, x1=rel_x1)
            print("Mined pose is not finite ", score)
            continue

        if score < 0.05:
            continue

        negatives[negatives_count] = score
        negatives_count += 1
        #print("Pose ", score, delta_R, delta_t)
    
    negatives = negatives[:negatives_count]
    top_k_scores = torch.topk(negatives, k=min(negatives_count, conf.top_k), dim=0)[0]

    print("Total weight ", map.atlas_weight.sum())
    print("Top matches ", ref_score, top_k_scores, "mean depth= ", depths.mean())

    loss_fn = torch.nn.BCELoss(reduction='mean')

    loss = loss_fn(1e-9 + (1-2e-9)*ref_score, torch.tensor(1., device=device))
    if len(top_k_scores) > 0:
        loss = loss + loss_fn(1e-9 + (1-2e-9)*top_k_scores, torch.zeros(len(top_k_scores), device=device))
        return loss, ref_score, top_k_scores.mean()
    else:
        return loss, ref_score, 0.0 