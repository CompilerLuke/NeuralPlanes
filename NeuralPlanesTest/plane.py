import unittest
from NeuralPlanes.plane import project_to_planes, draw_planes, project_to_planes_sparse
from NeuralPlanes.hg_pipeline import make_planes, parse_floorplan, parse_trajectories
from matplotlib import pyplot as plt
import torch

class PlanesTest():
    def test_project_sparse(self):
        pos = trajectories[list(trajectories.keys())[50]].t[None,None,None,None]
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
        draw_planes(ax, planes, indices=indices[indices != 0])

if __name__ == "__main__":
    building_path = "../data/HG_navviz/"
    floor_plan_path = building_path + "raw_data/floor_plan/"
    train_path = "../data/train_HG/"

    trajectories = parse_trajectories(building_path + "trajectories.txt")
    floor_plan_img, plane_points = parse_floorplan(floor_plan_path)

    #plane_points = [x[0:6] for x in plane_points]

    map_dim = 32
    planes = make_planes(plane_points, resolution=0.5)


    PlanesTest().test_project_sparse()
    plt.show()