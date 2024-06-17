import torch
from torch import nn
import cv2

"""
Modified from https://github.com/YvanYin/Metric3D/blob/main/hubconf.py#L145
"""
class MonocularDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)

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
