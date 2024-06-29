from NeuralPlanes.localization.map import NeuralMap, NeuralMapConf
from NeuralPlanes.localization.pose_scoring import score_pose
from NeuralPlanes.localization.unet import Unet
from NeuralPlanes.pipeline.lamar import *
from NeuralPlanes.pipeline.dataloader import *
from NeuralPlanes.plane import Planes 
from NeuralPlanes.utils import select_device
from NeuralPlanes.camera import compute_ray
from matplotlib import pyplot as plt 
import copy

device = select_device()

lamar_path = Path("../../data/HGE/HGE")
model_checkpoint = "checkpoint/model.pt"
max_size = 256

session_map_path = lamar_path / "sessions" / "map"
session_map_image_path = session_map_path / "raw_data"
session_map_path_precompute = lamar_path / "sessions" / ("map_precompute_"+str(max_size))
session_map_path_depth = session_map_path_precompute / "depth"
session_map_path_small = session_map_path_precompute / "small_images"

images_raw = parse_session(session_map_path, lambda path: path.startswith("ios"))

floor_plan_img, plane_points, floorplan_to_world = parse_floorplan(lamar_path / "floor_plan")


cameras, images_small, images_depth = get_preprocessed_images(images_raw, session_map_path_small, session_map_path_depth, max_size)

conf = NeuralMapConf(num_components=4, num_features_backbone=21, num_features=16, depth_sigma=0.2)
encoder = Encoder()
#Unet(encoder_freeze=True, classes= conf.num_features+1)

height, width =   1032, 672
n_planes = 130
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
map = map.to(device)

image_id = 500
image = images_small[image_id]
image_torch = encoder.preprocess(image)["image"]
depth = images_depth[image_id]["image"][:,:,0]
depth_torch = torch.tensor(depth, dtype=torch.float, device=device)

print(image_torch.shape)
features = map.encode({"image": image_torch[None].to(device) }, depth)[0]

weight, rem_features = features[0,:,:], features[1:,:,:]
print(weight.min(), weight.max(), rem_features.min(), rem_features.max())

print("Image=", rem_features.mean(), "Max depth=", depth.max())

planes = map.planes
map = map.to(device)
map = map.eval()

x00 = planes.x0s[0]
x01 = planes.x0s[0] + planes.us[0]
x10 = planes.x0s[0] + planes.vs[0]
x11 = planes.x0s[0] + planes.us[0] + planes.vs[0]

div = 60
div_rot = 1
cam = image["camera"]
xs, ys, thetas = torch.meshgrid(torch.linspace(x00[0], x01[0], div), torch.linspace(x01[1], x11[1], div),
                                torch.linspace(0, 2 * torch.pi, div_rot))
xs, ys, thetas = xs.flatten(), ys.flatten(), thetas.flatten()

print(cam.size, image["image"].shape)

scores = torch.zeros(len(xs))
weights = torch.zeros(len(xs))

print("Features=", features.shape)

scale_factor = 1.0 / 2
cam = cam.scale(scale_factor)
feature_small = torch.nn.functional.interpolate(features[None], scale_factor=scale_factor)[0]
depth_small = torch.nn.functional.interpolate(depth_torch[None, None], scale_factor=scale_factor)[0, 0]

R_canon, t_canon = cam.R, torch.tensor([0, 0, cam.t[2]])

with torch.no_grad():
    for i, (x, y, theta) in enumerate(zip(tqdm.tqdm(xs), ys, thetas)):
        new_cam = copy.copy(cam)
        new_cam.t = t_canon + torch.tensor([x, y, 0])

        R = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0, ],
            [0, 0, 1]
        ])

        new_cam.R = R @ R_canon @ R.T
        new_cam.t = t_canon + torch.tensor([x, y, 0])
        new_cam = new_cam.to(device)
        # print(new_cam.size.device,new_cam.R.device, map.planes.x0s.device, features.device, depth_small.device)
        plane_indices = torch.tensor([0])
        score, weight = score_pose(map, new_cam, feature_small, depth_small, plane_indices=plane_indices,
                                   rel_x0=torch.zeros((len(plane_indices), 2), device=device),
                                   rel_x1=torch.ones((len(plane_indices), 2), device=device))
        print(score, feature_small.mean(), depth_small.mean())
        # plane_indices=torch.tensor([0])) # plane_indices=torch.tensor([0]))
        scores[i] = score
        weights[i] = weight
    scores = scores.reshape((div, div, div_rot))
    weights = weights.reshape((div, div, div_rot))
    xs = xs.reshape((div, div, div_rot))
    ys = ys.reshape((div, div, div_rot))
    theta = thetas.reshape((div, div, div_rot))

print(scores.mean(), scores.max())

