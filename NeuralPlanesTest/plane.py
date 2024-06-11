import unittest
from NeuralPlanes.plane import *
from NeuralPlanes.camera import Camera, frustum_points
from matplotlib import pyplot as plt
import torch

class PlanesTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PlanesTest, self).__init__(*args, **kwargs)

        # Plane xy, yz, xz
        x0 = torch.tensor([[0,0,0], [0,0,0], [0,0,0]], dtype=torch.float)
        x1 = torch.tensor([[1,0,0], [0,1,0], [1,0,0]], dtype=torch.float)
        x2 = torch.tensor([[1,1,0], [0,1,1], [1,0,1]], dtype=torch.float)

        self.basic_planes = make_planes((x0,x1,x2), resolution=0)

        self.basic_point = torch.tensor([0.5,0.5,0.5])

    def test_project_sparse(self):
        """
        planes = self.basic_planes
        pos = self.basic_point[None,None,None]
        pixel_ids, plane_ids, proj, coord = project_to_planes(planes, pos)

        pixel_ids2, plane_ids2, coord2 = project_to_planes_sparse(planes, pos.repeat([1,1,2,2,1]), stride=2)

        assert ((pixel_ids2[3::4] == pixel_ids+3).all())
        assert (coord2[0::4] == coord).all()
        assert (coord2[1::4] == coord).all()
        assert (coord2[2::4] == coord).all()
        assert (coord2[3::4] == coord).all()

        print(planes.atlas_size)
        print(pixel_ids)
        print(coord)

        ax = plt.figure().add_subplot(projection='3d')

        x = torch.stack([pos.reshape((-1,3))[pixel_ids], proj, torch.full((len(pixel_ids),3), torch.inf)], dim=1).reshape((-1,3))
        ax.plot(x[:,0], x[:,1], x[:,2])

        ax.set_aspect('equal')
        indices = torch.unique(plane_ids)
        draw_planes(ax, self.basic_planes, indices=indices[indices != 0])
        """

    def test_plane_box_points(self):
        box_points = plane_box_points(self.basic_planes, 1, max_dist=100)
        self.assertTrue(torch.allclose(box_points, torch.tensor([[   0.,    0.,    0.],
        [   0.,    1.,    0.],
        [   0.,    1.,    1.],
        [   0.,    0.,    1.],
        [100.,    0.,    0.],
        [100.,    1.,    0.],
        [100.,    1.,    1.],
        [100.,    0.,    1.]])))

    def test_plane_frustum_intersection(self):
        fps = frustum_points(Camera(
            size=torch.tensor([1.,1.]),
            f=torch.tensor([0.5,0.5]),
            c=torch.tensor([0.5,0.5]),
            t=torch.tensor([0.5,0.5,0.]),
            R=torch.diag(torch.ones(3))
        ), 0.1, 0.4)

        visible = frustum_plane_visibility(self.basic_planes, 1, fps[None], [0.,0.], [1.,1.])
        self.assertTrue((visible == torch.tensor([True])).all())

        fps = frustum_points(Camera(
            size=torch.tensor([1., 1.]),
            f=torch.tensor([0.5, 0.5]),
            c=torch.tensor([0.5, 0.5]),
            t=torch.tensor([0.5, 0.5, 2.0]),
            R=torch.diag(torch.ones(3))
        ), 0.1, 0.4)

        visible = frustum_plane_visibility(self.basic_planes, 1, fps[None], [0.,0.], [1.,1.])
        self.assertTrue((visible == torch.tensor([False])).all())


