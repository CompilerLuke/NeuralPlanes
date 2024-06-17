import unittest
import torch
from NeuralPlanes.plane import make_planes, draw_planes
from NeuralPlanes.camera import Camera, frustum_points, compute_ray
from NeuralPlanes.localization.image_pooling import pool_images
from NeuralPlanes.localization.pose_scoring import score_pose
from NeuralPlanes.localization.negative_pose_mining import mine_negatives, RansacMiningConf
from NeuralPlanes.localization.map import NeuralMap, NeuralMapConf
from NeuralPlanes.localization.training import NeuralMapBuilder, NeuralMapBuilderConf
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

class MapBuilderTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Plane xy, yz, xz
        x0 = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=torch.float)
        x1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float)
        x2 = torch.tensor([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=torch.float)

        self.basic_planes = make_planes((x0, x1, x2), resolution=10)

        R = torch.diag(torch.ones(3))

        self.cameras = [
            Camera(
                size=torch.tensor([10.0, 10.0]),
                c=torch.tensor([5., 5.]),
                f=torch.tensor([10., 10.]),
                t=torch.tensor([0.5, 0.5, 0.5]),
                R=torch.tensor(R)
            ),
            Camera(
                size=torch.tensor([10.0, 10.0]),
                c=torch.tensor([5, 5]),
                f=torch.tensor([1000., 1000.]),
                t=torch.tensor([1.5, 0.5, 0.5]),
                R=torch.tensor(R)
            ),
        ]

        self.image_ids = ["cam_A", "cam_B"]
        self.builder_conf = NeuralMapBuilderConf(chunk_size=100, depth_sigma=0.2)

        R = Rotation.from_rotvec([-90.0,0.0,0.0], degrees=True).as_matrix()
        R = torch.tensor(R,dtype=torch.float)

        camera = Camera(
                size=torch.tensor([10.0, 10.0]),
                c=torch.tensor([5, 5]),
                f=torch.tensor([10., 10.]),
                t=torch.tensor([0.5, 0.1, 0.5]),
                R=R,
                near=0.05,
                far=1
            )

        self.forward_cam = camera

    def test_chunk(self):
        builder = NeuralMapBuilder(planes=self.basic_planes, image_ids=self.image_ids, cameras=self.cameras,
                                   conf=NeuralMapBuilderConf(chunk_size=3))

        print(self.basic_planes.coord_size[0])
        print(self.builder_conf.chunk_size)

        coord, size = builder._chunk_neural_map(0)

        self.assertTrue(torch.allclose(coord, torch.tensor([[[0, 0],
                                                         [3, 0],
                                                         [6, 0],
                                                         [9, 0]],
                                                        [[0, 3],
                                                         [3, 3],
                                                         [6, 3],
                                                         [9, 3]],
                                                        [[0, 6],
                                                         [3, 6],
                                                         [6, 6],
                                                         [9, 6]],
                                                        [[0, 9],
                                                         [3, 9],
                                                         [6, 9],
                                                         [9, 9]]])))
        self.assertTrue(torch.allclose(size, torch.tensor([[[3, 3],
                                                         [3, 3],
                                                         [3, 3],
                                                         [1, 3]],
                                                        [[3, 3],
                                                         [3, 3],
                                                         [3, 3],
                                                         [1, 3]],
                                                        [[3, 3],
                                                         [3, 3],
                                                         [3, 3],
                                                         [1, 3]],
                                                        [[3, 1],
                                                         [3, 1],
                                                         [3, 1],
                                                         [1, 1]]])))

    def test_assign_cameras_to_planes(self):
        builder = NeuralMapBuilder(planes=self.basic_planes, cameras=self.cameras,
                                   conf=self.builder_conf)
        builder._assign_cameras_to_planes()

        print("ASSIGN CAMERAS TO PLANES")
        print(builder.chunk_coord)
        print(builder.chunk_size)
        print(builder.visible_camera_idx)

        print("Cameras")
        print(self.cameras[0].t, self.cameras[1].t)
        print(self.basic_planes.planes[0,0:3])
        print(self.basic_planes.planes[1,0:3])
        print(self.basic_planes.planes[2,0:3])

        print("VISIBLE")
        print(builder.visible_camera_idx)

        self.assertTrue(all([len(grid)==1 and len(grid[0])==1 for grid in builder.visible_camera_idx]))

        self.assertEqual([0], builder.visible_camera_idx[0][0][0].tolist())
        self.assertEqual([0,1], builder.visible_camera_idx[1][0][0].tolist(), [])
        self.assertEqual([0], builder.visible_camera_idx[2][0][0].tolist())

    def gen_cos_image(self, height=10, width=10):
        v, u = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
        image_cos = torch.stack([0.5*torch.cos(torch.pi * u)+0.5]) # * torch.cos(torch.pi * v)
        depth = torch.full((height, width,), 0.5)

        return image_cos, depth

    def test_sample_images(self):
        builder = NeuralMapBuilder(planes=self.basic_planes, cameras=self.cameras,
                                   conf=self.builder_conf)

        width,height = 10, 10
        image, depth = self.gen_cos_image()

        print("GET CHUNKS")
        chunk_id = builder.chunks()[0]
        print(chunk_id)

        camera = self.forward_cam

        masks, image_values, occupancy, tsamples, out_pos = builder.sample_images(chunk_id, [image.to(builder.device)], [depth.to(builder.device)], [camera.to(builder.device)])

        weight,alpha,mu,var = pool_images(masks, image_values, occupancy, tsamples)


        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        orig, dir = compute_ray(width, height, camera, device='cpu')
        x = orig + dir * depth[:, :, None]
        ax.scatter(x[:,:,0], x[:,:,1], x[:,:,2], c=image[0])

        print(weight)
        print("Weight range", )

        draw_planes(ax, self.basic_planes, indices=[0], color= weight) #torch.where(weight > 0.9, mu[0,0], torch.zeros((height,width))))

        print(masks.shape, image_values.shape, occupancy.shape, out_pos.shape)

        pos = out_pos[masks].cpu()
        ax.scatter(pos[:,0], pos[:,1], pos[:,2])

        fp = frustum_points(camera).reshape((8,3)).cpu()
        ax.scatter(fp[:,0],fp[:,1],fp[:,2])

        plt.show()



    def test_scoring(self):
        planes = self.basic_planes

        n_components = 1
        n_features = 1
        atlas_height, atlas_width = planes.atlas_size
        atlas_mu = torch.zeros((n_components, n_features, atlas_height, atlas_width))
        atlas_var = torch.ones((n_components, n_features, atlas_height, atlas_width))
        atlas_alpha = torch.ones((n_components, atlas_height, atlas_width))
        atlas_weight = torch.ones((n_components, atlas_height, atlas_width))

        height,width = [10,10]
        v,u = torch.meshgrid(torch.linspace(0,1,height), torch.linspace(0,1,width))
        image_zero = torch.zeros((1,height,width))
        image_outlier = torch.full((1, height, width), 1.0)
        image_cos = torch.stack([torch.cos(torch.pi*u)*torch.cos(torch.pi*v)])
        depth = torch.full((height,width,), 0.5)

        camera = self.forward_cam

        map = NeuralMap(planes, atlas_alpha, atlas_mu, atlas_var, atlas_weight, NeuralMapConf(depth_sigma=0.05,num_features=1))

        score_zero = score_pose(map, camera, image_zero, depth)
        score_cos = score_pose(map, camera, image_cos, depth)
        score_one = score_pose(map, camera, image_outlier, depth)

        self.assertGreater(score_zero, score_cos)

        print(score_zero, score_cos, score_one)

        score_cos_dense, weight, pos, features = score_pose(map, camera, image_cos, depth, dense=True)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        pos = pos.reshape((-1,3))
        score_cos_dense = (score_cos_dense * weight).flatten()
        ax.scatter(pos[:,0].detach(),pos[:,1].detach(),pos[:,2].detach(),c=score_cos_dense.detach())
        ax.set_aspect('equal')

        plt.show()

    def test_training_dryrun(self):
        builder = NeuralMapBuilder(planes=self.basic_planes,cameras=self.cameras,
                                   conf=NeuralMapBuilderConf(
                                       num_features=1,
                                       num_features_backbone=1,
                                       ransac=RansacMiningConf(
                                           num_ref_kps=4,
                                           kp_per_ref=2,
                                           ransac_it=1,
                                           ransac_sample=4,
                                           scale_factor=1,
                                       )
                                   ))

        width,height = 10, 10
        v,u = torch.meshgrid(torch.linspace(0,1,height), torch.linspace(0,1,width))
        image_cos = torch.stack([torch.cos(torch.pi*u)*torch.cos(torch.pi*v)])

        depth = torch.full((height, width,), 0.5)


        builder.train([{"image": image_cos},{"image": image_cos}], [{"image": depth}, {"image": depth}])

