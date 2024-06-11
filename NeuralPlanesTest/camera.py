from NeuralPlanes.camera import frustum_ray_intersection, frustum_points, frustum_normals, Camera
import unittest
import torch
from math import sqrt


class CameraTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_frustum_points(self):
        f = 2
        cam = Camera(
            size=torch.tensor([1.0, 1.0]),
            c=torch.tensor([0.5, 0.5]),
            f=torch.tensor([f, f]),
            t=torch.tensor([0.0, 0.0, 0.0]),
            R=torch.diag(torch.ones(3))
        )

        near = 0.1
        far = 1.0
        points = frustum_points(cam, near, far)

        ref = torch.tensor([
            [
                [
                    [-near / f / 2, -near / f / 2, near],
                    [near / f / 2, -near / f / 2, near],
                ],
                [
                    [-near / f / 2, near / f / 2, near],
                    [near / f / 2, near / f / 2, near]
                ]
            ],
            [
                [
                    [-far / f / 2, -far / f / 2, far],
                    [far / f / 2, -far / f / 2, far],
                ],
                [
                    [-far / f / 2, far / f / 2, far],
                    [far / f / 2, far / f / 2, far]
                ]
            ]
        ])

        self.assertTrue(torch.allclose(points, ref))

    def test_intrinsic(self):
        R = torch.tensor([
            [-1., 0, 0],
            [0, -1., 0],
            [0, 0, 1.]
        ])

        cam = Camera(
            size=torch.tensor([1.0, 1.0]),
            c=torch.tensor([0.5, 0.5]),
            f=torch.tensor([5, 3]),
            t=torch.tensor([1.0, 0.0, 0.0]),
            R=R
        )

        self.assertTrue(torch.allclose(cam.intrinsic_matrix(), torch.tensor([
            [5, 0, 0.5],
            [0, 3, 0.5],
            [0, 0, 1]
        ], dtype=torch.float)))

        self.assertTrue(torch.allclose(cam.extrinsic_matrix(), torch.tensor([[-1., 0., 0., 1.],
                                                                            [0., -1., 0., 0.],
                                                                            [0., 0., 1., 0.]])))

        cam = Camera(
            size=torch.tensor([1.0, 1.0]),
            c=torch.tensor([0.0, 0.0]),
            f=torch.tensor([1, 1]),
            t=torch.tensor([0.0, 0.0, -1.0]),
            R=torch.tensor(R)
        )
        proj = cam.to_image_space(torch.tensor([[1.0,0.0,0.0], [2.0,0.0,1.0]]))
        self.assertTrue(torch.allclose(proj, torch.tensor([[-1.,0.],[-1.,0.]])))

    def test_frustum_normals(self):
        f = 1 / 2
        cam = Camera(
            size=torch.tensor([1.0, 1.0]),
            c=torch.tensor([0.5, 0.5]),
            f=torch.tensor([f, f]),
            t=torch.tensor([0.0, 0.0, 0.0]),
            R=torch.diag(torch.ones(3))
        )

        normals = frustum_normals(frustum_points(cam, 0.1, 1.0)[None])

        self.assertTrue(torch.allclose(normals, torch.tensor([
            [[0.0000, -0.7071, -0.7071]],  # bottom
            [[0.7071, 0.0000, -0.7071]],  # right
            [[0.0000, 0.7071, -0.7071]],  # top
            [[-0.7071, -0.0000, -0.7071]],  # left
            [[0.0000, 0.0000, -1.0000]],  # front
            [[0.0000, 0.0000, 1.0000]]  # back
        ])))  # front

    def test_ray_frustum(self):
        frustum_points = torch.tensor([
            [0., 0., 0.],
            [1.0, 0.0, 0.],
            [0., 1.0, 0.0],
            [1., 1.0, 0.0],
            [0., 0., 1.],
            [1.0, 0.0, 1.],
            [0., 1.0, 1.0],
            [1., 1.0, 1.0],
        ]).reshape((2, 2, 2, 3))

        origin = torch.tensor([
            [0.5, 0.5, -1.],
            [0.5, 0.5, 2.],
            [-1., 0.5, 0.5],
            [-1., 0.5, 1.5],
        ])

        dir = torch.tensor([
            [0., 0., 1],
            [0., 0., -1],
            [1., 0., 0.],
            [1., 0., 0.],
        ])

        tmin, tmax = frustum_ray_intersection(frustum_points, origin, dir)

        print(tmin, tmax)

        self.assertTrue(torch.allclose(tmin, torch.tensor([[
            1.0,
            1.0,
            1.0,
            torch.inf
        ]])))

        self.assertTrue(torch.allclose(tmax, torch.tensor([
            2.0,
            2.0,
            2.0,
            -torch.inf
        ])))
