"""
This module includes a class which streamlines the current ecomodel.
"""
import CSF
import numpy as np
from Utils.Utils import load_point_cloud
from SegmentRGI.SegmentRGI import classify_wood_leaf
from pathlib import Path
from tempfile import TemporaryDirectory
from plyfile import PlyData, PlyElement
from Utils.plot_tools import SimplePlotter
import open3d as o3d
import networkx as nx
import cc3d
import matplotlib.pyplot as plt
import os
from pc_skeletor import LBC, SLBC
from pc_skeletor.utility import simplify_graph

import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import mistree as mist
from Utils.Utils import load_point_cloud
from Utils.plot_tools import  ResultsPlotter
import numpy as np
from sklearn.neighbors import NearestNeighbors
from Utils.RobustCylinderFitting import RobustCylinderFitter
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
import pyvista as pv
from circle_fit import taubinSVD
from pathlib import Path

# John
import time
from TreeQSMSteps.cover_sets import cover_sets
from e_segmenters import SegmenterScanline
from Utils.define_input import define_input
from treeqsm import treeqsm
from TreeQSMSteps.cover_sets import cover_sets
from TreeQSMSteps.segments import segments
from TreeQSMSteps.correct_segments import correct_segments
from TreeQSMSteps.tree_sets import tree_sets
from TreeQSMSteps.relative_size import relative_size
from TreeQSMSteps.cylinders import cylinders





class EcomodelLite:
    def __init__(self):
        super().__init__()
        self.results_folder = ""
        self.segmenter = SegmenterScanline()
        self.plotter = SimplePlotter()
        self.ground_z = 0

    def remove_ground(self, point_cloud, remove_under_ground = True):
        """
        Uses CSF To remove ground in point cloud. Stores the ground level in self.ground_z for the tile.
        
        Args: 
            point_cloud (Nx3 Array): Point cloud representing tile.
            remove_under_ground (bool): Whether to points below the ground level or not. 

        Returns:
            point_cloud: Point cloud without a ground. 
        """
        csf = CSF.CSF()
        new_min_z = float('inf')

        # prameter settings
        csf.params.cloth_resolution = 2
        csf.params.class_threshold = 0.5
        csf.params.interations = 500

        csf.setPointCloud(point_cloud)
        ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
        non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
        csf.do_filtering(ground, non_ground, exportCloth=False)
        ground_mask = np.array(ground)
        non_ground_mask = np.array(non_ground)
        ground_points = point_cloud[ground_mask]
        mean_ground_height = np.mean(ground_points[:,2]) 
        print(mean_ground_height)

        point_cloud = point_cloud[non_ground_mask]
        self.ground_z = mean_ground_height

        if remove_under_ground:
            above_ground_mask = point_cloud[:,2] > mean_ground_height
            point_cloud = point_cloud[above_ground_mask]
        
        return point_cloud


    def classify_wood_leaf_on_array(self,tree_cloud, input_params=None):
        """
        Helper method

        Run classify_wood_leaf() directly on an in-memory NumPy point cloud array.
        Saves the temporary segment, classifies it, and rebuilds boolean masks.
        """
        with TemporaryDirectory() as tmpdir:
            tmp_ply = Path(tmpdir) / "segment.ply"
            tmp_results = Path(tmpdir) / "results"
            tmp_results.mkdir(exist_ok=True)

            # Save current segment to temporary PLY
            vertex_dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('Intensity', 'f4')]
            structured = np.zeros(tree_cloud.shape[0], dtype=vertex_dtype)
            structured['x'] = tree_cloud[:, 0]
            structured['y'] = tree_cloud[:, 1]
            structured['z'] = tree_cloud[:, 2]
            structured['Intensity'] = tree_cloud[:, 3]

            ply_elements = PlyElement.describe(structured, 'vertex')
    
            PlyData([ply_elements]).write(str(tmp_ply))

            # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tree_cloud[:, :3]))
            # o3d.io.write_point_cloud(str(tmp_ply), pcd)


            # Run existing classification pipeline
            classify_wood_leaf(str(tmp_ply), save_dir=str(tmp_results), show_plots=False, **input_params)

            # Read back the classified clouds
            wood_file = tmp_results / "segment_wood.ply"
            leaf_file = tmp_results / "segment_leaves.ply"
            if not wood_file.exists() or not leaf_file.exists():
                raise FileNotFoundError("Wood/Leaf outputs not found after classification.")

            wood_pcd = o3d.io.read_point_cloud(str(wood_file))
            leaf_pcd = o3d.io.read_point_cloud(str(leaf_file))
            tree_coords = np.asarray(tree_cloud[:, :3])
            wood_coords = np.asarray(wood_pcd.points)
            leaf_coords = np.asarray(leaf_pcd.points)

            def build_mask(sub_coords):
                mask = np.isin(
                    tree_coords.view([('', tree_coords.dtype)] * 3),
                    sub_coords.view([('', sub_coords.dtype)] * 3)
                )
                return mask.squeeze()

            wood_mask = build_mask(wood_coords)
            leaf_mask = build_mask(leaf_coords)

            return wood_mask, leaf_mask
        


    def remove_leaves_rgi(self, point_cloud):
        """
        Removes leaf points from a point cloud.

        Args:
            point_cloud (Nx4): point_cloud representing a tile with trees. 

        Return:
            only_wood (Nx4): Point cloud with no points. 
        """
        input_params = {
            "noise_percentile": 0,
            "angle_deg":7, 
            "curv_thresh":0.07, 
            "resid_thresh":0.05, 
            "k":100,
            "minClusterSize" : 40,
            "maxClusterSize" : 100000,
            "smoothMode" : True,
            "useResidualTest" : True,
            "useCurvatureTest" : True,
        }

        wood_mask, leaf_mask = self.classify_wood_leaf_on_array(point_cloud, input_params)

        only_wood = point_cloud[wood_mask]

        return only_wood

    def filter_intensity(self, point_cloud, intensity):
        """
        Filter by intensity
        """
        intensity_mask = point_cloud[:,3] > intensity
        return point_cloud[intensity_mask]



    
    def perform_instance_segmentation(self, point_cloud):
        """"""
        point_cloud, labels = self.segmenter.process(point_cloud)

        return point_cloud, labels

    def normalize_point_cloud(self, point_cloud):
        self.mean = np.mean(point_cloud[:,0:3], axis=0)

        point_cloud[:, :3] = point_cloud[:, :3] - self.mean

        return point_cloud

    def get_cylinders(self, point_cloud, instance_labels):
        """
        Returns a set of cylinders.
        
        Args: 

        """
        labeled_point_cloud = np.concatenate((point_cloud[:,:3], instance_labels[:, np.newaxis]), axis=1)

        cylinder_starts = np.empty((0,3))
        cylinder_radii = np.array([])
        cylinder_axes = np.empty((0,3))
        cylinder_lengths = np.array([])

        for segment in np.unique(instance_labels):
            if segment == -1:
                continue
            
            segment_mask = (labeled_point_cloud[:,3] == segment)
            tree_cloud = labeled_point_cloud[segment_mask, :3]

            if len(tree_cloud) < 100:
                continue
            try:
                qsm_input = define_input(labeled_point_cloud[:,:3], 1, 1, 1)[0]
            except np.linalg.LinAlgError as e:
                print(f"Failed to define input {e}")
                continue

            qsm_input['PatchDiam1'] = 0.025
            qsm_input['PatchDiam2Min'] = 0.05
            qsm_input['PatchDiam2Max'] = 0.08
            qsm_input['BallRad1'] = 0.03
            qsm_input['BallRad2'] = 0.09
            qsm_input['nmin1'] = 5

            try: 
                cover1 = cover_sets(tree_cloud, qsm_input)
                cover1, Base, Forb = tree_sets(tree_cloud, cover1, qsm_input)
                segment1 = segments(cover1, Base, Forb, qsm=True)
                segment1 = correct_segments(tree_cloud, cover1, segment1, qsm_input, 0, 1, 1)
                RS = relative_size(tree_cloud, cover1, segment1)
                cover1 = cover_sets(tree_cloud, qsm_input, RS)
                cover1, Base, Forb = tree_sets(tree_cloud, cover1, qsm_input, segment1)
                segment1 = segments(cover1, Base, Forb)
                segment1 = correct_segments(tree_cloud, cover1, segment1, qsm_input,1,1,0)
                cylinder = cylinders(tree_cloud,cover1,segment1,qsm_input)
            except Exception as e: 
                print(f"Failed to get cylinders {e}")
                continue


            cylinder_starts = np.concatenate([cylinder_starts,cylinder["start"]])
            cylinder_radii = np.append(cylinder_radii,cylinder["radius"])
            cylinder_axes = np.concatenate([cylinder_axes,cylinder["axis"]])
            cylinder_lengths = np.append(cylinder_lengths,cylinder["length"])

        cylinder_data = np.concatenate((cylinder_starts, cylinder_radii.reshape(-1, 1), cylinder_axes, cylinder_lengths.reshape(-1, 1)), axis=1)

        return cylinder_data

    def view_cylinders(self, point_cloud, cylinder_data):
        """
        
        """
        print(cylinder_data.shape)
        for cylinder_row in range(0, cylinder_data.shape[0]):
            cylinder = cylinder_data[cylinder_row]

            start = cylinder[0:3]
            radius = cylinder[3:4]
            axis = cylinder[4:7]
            length = cylinder[7:8]

            self.plotter.add_cylinder(start, axis, radius, length)
        
        self.plotter.add_point_cloud(point_cloud)

        self.plotter.show()

    def process_tile(self, tile_path):
        """
        Process a point cloud tile. 
        """
        path = Path(tile_path)
        xyz_data, full_data = load_point_cloud(str(path), full_data=True)
        full_data = model.remove_ground(full_data)
        full_data = model.filter_intensity(full_data, 42000)
        full_data = model.remove_leaves_rgi(full_data)
        model.save_point_cloud(f"{path.parent}\{path.stem}_leaves_removed.xyz", full_data)
        full_data = model.normalize_point_cloud(full_data)
        full_data, instance_labels = model.perform_instance_segmentation(full_data)
        cylinder_data = model.get_cylinders(full_data, instance_labels)

        np.savetxt()

if __name__ == "__main__":
    model = EcomodelLite()
    # path = Path(r"G:\Projects\TreeCanopyLidar\Datasets\2025_1cm\tile_00005_2_0.laz")
    path = Path(r"G:\Projects\TreeCanopyLidar\Datasets\2025_1cm\tile_00005_2_0_leaves_removed.xyz")
    folder_path = path.parent
    basename = path.stem
    tile_data, full_data = load_point_cloud(str(path), full_data=True)
    # full_data = model.remove_ground(full_data)
    # full_data = model.filter_intensity(full_data, 42000)
    # full_data = model.remove_leaves_rgi(full_data)
    # model.save_point_cloud(f"{folder_path}\{path.stem}_leaves_removed.xyz", full_data)
    full_data = model.normalize_point_cloud(full_data)
    full_data, instance_labels = model.perform_instance_segmentation(full_data)
    cylinder_data = model.get_cylinders_individual(full_data, instance_labels)

    # view_data = np.concatenate((full_data,instance_labels[:,np.newaxis] ), axis=1)
    # model.view_point_cloud(view_data)
    # model.save_point_cloud(f"{folder_path}\{path.stem}_labeled.xyz", view_data)
    model.view_cylinders(full_data, cylinder_data)
    exit()


    # tile_data, full_data = load_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\_experimentation\testing_my_method\leaves_intensity_removed.xyz", full_data=True)
    # # tile_data, full_data = load_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\_experimentation\testing_my_method\two_trees.las", full_data=True)
    # model.perform_instance_segmentation(full_data)

    # # Connected components test:
    # folder_path = r"G:\Projects\TreeCanopyLidar\PyTLidar\_experimentation\testing_my_method\components_cc\one"
    # all_cylinders = []
    # for file in os.listdir(folder_path):
    #     root, ext = os.path.splitext(file)

    #     pc, pcdata = load_point_cloud(f"{folder_path}\\{file}", full_data=True)
    #     model.simple_plotter.add_point_cloud(pcdata)
    #     # pcdata = model.normalize_point_cloud(pcdata)
        
    #     model.segmenter.process(pcdata)



    #     # cylinders = model.segmenter.create_skeleton_and_cylinders(pcdata)
    #     # if cylinders is None:
    #     #     continue


    #     # for index in range(cylinders.shape[0]):
    #     #     start = cylinders[index, :3]
    #     #     radius = cylinders[index, 3]
    #     #     axis = cylinders[index, 4:7]
    #     #     length = cylinders[index, 7]

    #     #     print(start)
    #     #     print(radius)
    #     #     print(axis)
    #     #     print(length)

    #     #     model.simple_plotter.add_cylinder(start, axis, radius, length)

    # # model.segmenter.simple_plotter.plotter.show()
    # model.simple_plotter.show()
        


















    #     # Connected components test:
    # folder_path = r"G:\Projects\TreeCanopyLidar\PyTLidar\_experimentation\testing_my_method\components_cc\one"
    # all_cylinders = []
    # for file in os.listdir(folder_path):
    #     root, ext = os.path.splitext(file)
    #     if ext == '.txt':
    #         pc, pcdata = load_point_cloud(f"{folder_path}\\{file}", full_data=True)
    #         model.simple_plotter.add_point_cloud(pcdata)
    #         # pcdata = model.normalize_point_cloud(pcdata)
            
    #         cylinders = model.segmenter.create_skeleton_and_cylinders(pcdata)
    #         if cylinders is None:
    #             continue


    #         for index in range(cylinders.shape[0]):
    #             start = cylinders[index, :3]
    #             radius = cylinders[index, 3]
    #             axis = cylinders[index, 4:7]
    #             length = cylinders[index, 7]

    #             print(start)
    #             print(radius)
    #             print(axis)
    #             print(length)

    #             model.simple_plotter.add_cylinder(start, axis, radius, length)

    # # model.segmenter.simple_plotter.plotter.show()
    # model.simple_plotter.show()
    