"""
This module includes a class which streamlines the current ecomodel.
"""
import os 
import shutil
import CSF
import numpy as np
from Utils.Utils import load_point_cloud
from SegmentRGI.SegmentRGI import classify_wood_leaf
from pathlib import Path
from tempfile import TemporaryDirectory
from plyfile import PlyData, PlyElement
from Utils.plot_tools import SimplePlotter
import open3d as o3d
import cc3d
from copy import deepcopy
from Utils.Utils import load_point_cloud
import numpy as np
from pathlib import Path
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
import logging

logger = logging.getLogger("Ecomodel")
logger.setLevel(logging.INFO)
f_handler = logging.FileHandler('ecomodel_log.log')
c_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(formatter)
c_handler.setFormatter(formatter)
logger.addHandler(f_handler)
logger.addHandler(c_handler)

class DistanceBasedNoiseRemoval:
    def __init__(self, voxel_size=1, min_points=100):
        self.voxel_size = voxel_size
        self.min_points = min_points


    def remove_distant_small_clusters(self, point_cloud):
        """
        Remove small clusters.

        Step . 
        """
        data_xyz = point_cloud[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data_xyz)

        # print("Voxel Grid")
        # Create a Voxel Grid for the point cloud. Obtain the grid shape 
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        voxels = voxel_grid.get_voxels()

        # Create dense occupancy grid using the coordinates of the voxel grid. 
        coordinates = np.array([v.grid_index for v in voxels], dtype=int)
        max_idx = coordinates.max(axis=0)
        grid_shape = max_idx + 1 
        grid = np.zeros(grid_shape, dtype=np.uint8)
        grid[coordinates[:,0], coordinates[:,1], coordinates[:,2]] = 1

        # Perform classification. 
        # print("Classification")
        classified = cc3d.connected_components(grid)

        # 1. Determine the indices at for each voxel grid. 
        # 2. Determine the points that are within the voxel coordinates (point_mask)
        # 3. Remove the points using the mask.  
        point_cloud_with_classification = np.concatenate((point_cloud[:,:3], np.zeros((point_cloud.shape[0], 1))), axis=1)
        # print("Iterations")
        for classification in np.unique(classified):
            mask = classified == classification
            # print(mask.shape)
            x_idx, y_idx, z_idx = np.where(mask)
            indices = np.concatenate((x_idx[:, np.newaxis], y_idx[:, np.newaxis], z_idx[:, np.newaxis]), axis=1)
            # print("Amount of indices", indices.shape)

            for index in indices:
                point_mask = self.get_mask_for_voxel(point_cloud, index, voxel_grid.origin, self.voxel_size)
                point_cloud_with_classification[point_mask, 3] = classification

        point_cloud_cleaned = self.remove_points(point_cloud_with_classification)
        # print("Done")
        return point_cloud_cleaned[:, :3]

    def get_mask_for_voxel(self, point_cloud, voxel_indices, origin, voxel_size):
        voxel_min = origin + voxel_indices * voxel_size
        voxel_max = voxel_min + voxel_size
        
        # print(voxel_min, voxel_max)
        mask = np.all(
            (point_cloud[:, :3] >= voxel_min) &
            (point_cloud[:, :3] <  voxel_max),   # strict upper bound
            axis=1
        )
        return mask

       
    def remove_points(self, classified_point_cloud, min_points = 500):
        """
        Removes clusters that have less than 20 points (sparse)
        """
        classification_copy = deepcopy(classified_point_cloud)
        for classification in np.unique(classified_point_cloud[:,3]):

            if np.count_nonzero(classified_point_cloud[:, 3] == classification) < min_points:
                mask = classification_copy[:, 3] != classification
                classification_copy = classification_copy[mask]
                # print("REMOVED", classification)
        
        return classification_copy








class EcomodelLite:
    def __init__(self, results_folder="results"):
        super().__init__()
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)
        self.results_folder = results_folder
        self.segmenter = SegmenterScanline()
        self.plotter = SimplePlotter()
        self.noise_remover = DistanceBasedNoiseRemoval(0.25, 100)
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
        print(non_ground_mask)

        if non_ground_mask.size == 0:
            return None

        point_cloud = point_cloud[non_ground_mask]
        self.ground_z = mean_ground_height

        if remove_under_ground:
            above_ground_mask = point_cloud[:,2] > mean_ground_height
            point_cloud = point_cloud[above_ground_mask]
        
        return point_cloud

    def normalize_point_cloud(self, point_cloud):
        self.mean = np.mean(point_cloud[:,0:3], axis=0)
        point_cloud[:, :3] = point_cloud[:, :3] - self.mean
        return point_cloud

    def unnormalize_point_cloud(self, point_cloud):
        point_cloud[:, :3] = point_cloud[:, :3] + self.mean
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

            # If tile is too small, skip tile. 
            if tree_cloud.shape[0] < 100:
                return None, None

            tree_coords = np.asarray(tree_cloud[:, :3])

            print(tree_cloud.shape[0])
            # Run existing classification pipeline
            classify_wood_leaf(str(tmp_ply), save_dir=str(tmp_results), show_plots=False, **input_params)

            # Read back the classified clouds
            wood_file = tmp_results / "segment_wood.ply"
            leaf_file = tmp_results / "segment_leaves.ply"
            
            # If there appears to be no wood in the file, skip tile. 
            if not wood_file.exists():
                return None, None
            
            def build_mask(sub_coords):
                mask = np.isin(
                    tree_coords.view([('', tree_coords.dtype)] * 3),
                    sub_coords.view([('', sub_coords.dtype)] * 3)
                )
                return mask.squeeze()
            if not leaf_file.exists():
                leaf_mask = np.zeros(tree_cloud.shape[0], dtype=bool)
            else:
                leaf_pcd = o3d.io.read_point_cloud(str(leaf_file))
                leaf_coords = np.asarray(leaf_pcd.points)
                leaf_mask = build_mask(leaf_coords)

            wood_pcd = o3d.io.read_point_cloud(str(wood_file))
            wood_coords = np.asarray(wood_pcd.points)
            wood_mask = build_mask(wood_coords)

            return wood_mask, leaf_mask
        
    def filter_intensity(self, point_cloud, intensity):
        """
        Filters points based on intensity.

        Args:
            point_cloud (Nx4): point_cloud representing a tile with trees.
            intensity (float): Removes points below this intensity threshold.
        """
        intensity_mask = point_cloud[:,3] > intensity
        return point_cloud[intensity_mask]

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

        if point_cloud.shape[0] < 100:
            return None

        wood_mask, leaf_mask = self.classify_wood_leaf_on_array(point_cloud, input_params)

        if wood_mask is None or leaf_mask is None:
            return None

        only_wood = point_cloud[wood_mask]

        return only_wood

    def perform_instance_segmentation(self, point_cloud):
        """
        Performs instance segmentation using. 
        """
        print("Performing Instance Segmentation....")
        point_cloud, labels = self.segmenter.process(point_cloud)

        return point_cloud, labels
    
    def get_cylinders(self, point_cloud, instance_labels):
        """
        Returns Cx8 numpy array representing the cylinders found in the tile. 

        Note: 
            Array is formatted as [start_x, start_y, start_z, radius, axis_x, axis_y, axis_z, length]
        
        Args: 
            point_cloud (Nx4): point cloud representing a tile with trees.
            instance_labels (N): Array of instance labels for each point in the point cloud.
        """
        labeled_point_cloud = np.concatenate((point_cloud[:,:3], instance_labels[:, np.newaxis]), axis=1)

        cylinder_starts = np.empty((0,3))
        cylinder_radii = np.array([])
        cylinder_axes = np.empty((0,3))
        cylinder_lengths = np.array([])

        for tree_instance in np.unique(instance_labels):
            if tree_instance == -1:
                continue
            
            segment_mask = (labeled_point_cloud[:,3] == tree_instance)
            tree_cloud = labeled_point_cloud[segment_mask, :3]
            print(tree_instance)

            np.savetxt(f"segment_{tree_instance}.xyz", tree_cloud)
            print("Removing Small clusters...")
            tree_cloud = self.noise_remover.remove_distant_small_clusters(tree_cloud)
            print("Done.")

            if len(tree_cloud) < 100:
                continue
            try:
                qsm_input = define_input(labeled_point_cloud[:,:3], 1, 1, 1)[0]
            except np.linalg.LinAlgError as e:
                print(f"Failed to define input {e}")
                continue

            np.savetxt("troubled_segment.xyz", tree_cloud)

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
        View cylinders in PyVista.
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

    def process_tile(self, tile_path, save_data=False, show_plots=False):
        """
        Process a point cloud tile. 
        """
        path = Path(tile_path)
        os.makedirs(f"{self.results_folder}/{path.stem}",exist_ok=True )
        _, full_data = load_point_cloud(str(path), full_data=True)
        full_data = model.normalize_point_cloud(full_data)
        full_data = model.remove_ground(full_data)

        if full_data is None: 
            print("Low data tile after removing ground.")
            return

        full_data = model.filter_intensity(full_data, 42000)
        if full_data.size < 100: 
            print("Empty array after filtering intensity.")
            return
        
        full_data = model.remove_leaves_rgi(full_data)
        if full_data is None:
            print("Unable to remove leaves on segment.")
            return 

        full_data, instance_labels = model.perform_instance_segmentation(full_data)
        if full_data is None or instance_labels is None:
            print("Unable to perform instance segmentation.")
            return

        cylinder_data = model.get_cylinders(full_data, instance_labels)
        print("Cylinder data shape", cylinder_data.shape)


        # Save results.
        if save_data:
            cylinder_data = self.unnormalize_point_cloud(cylinder_data)
            np.savetxt(f"{self.results_folder}/{path.stem}/{path.stem}_cylinders.txt", cylinder_data)
            unnormalized = self.unnormalize_point_cloud(full_data)
            with_labels = np.concatenate((unnormalized[:,:3], instance_labels[:,np.newaxis]), axis=1)
            np.savetxt(f"{self.results_folder}/{path.stem}/{path.stem}_leavesremoved.xyz", with_labels)
            with open(f"{self.results_folder}/{path.stem}/{path.stem}_data.txt", "w") as f:
                f.writelines(f"{self.mean[0]} {self.mean[1]} {self.mean[2]}")
                f.writelines("\n")
                f.writelines(str(self.ground_z))

        if show_plots:
            self.view_cylinders(full_data, cylinder_data)

    def process_tile_no_leaf_removal(self, tile_path, save_data=False, show_plots=False):
        """
        Perform only the instance segmentation and TreeQSM on the tile. Assumes leaf removal already complete. 
        """
        path = Path(tile_path)
        os.makedirs(f"{self.results_folder}/{path.stem}",exist_ok=True )
        xyz_data, full_data = load_point_cloud(str(path), full_data=True)
        full_data = model.normalize_point_cloud(full_data)

        full_data, instance_labels = model.perform_instance_segmentation(full_data)
        if full_data is None or instance_labels is None:
            print("Unable to perform instance segmentation.")
            return

        cylinder_data = model.get_cylinders(full_data, instance_labels)
        print("Cylinder data shape", cylinder_data.shape)
        cylinder_data = self.unnormalize_point_cloud(cylinder_data)

        # Save results. 
        if save_data:
            shutil.copy(str(path), f"{self.results_folder}/{path.stem}/{path.stem}.xyz")
            np.savetxt(f"{self.results_folder}/{path.stem}/{path.stem}_cylinders.txt", cylinder_data)
            with open(f"{self.results_folder}/{path.stem}/{path.stem}_data.txt", "w") as f:
                f.writelines(str(self.mean))
                f.writelines("\n")
                f.writelines(str(self.ground_z))

        if show_plots:
            self.view_cylinders(full_data, cylinder_data)



if __name__ == "__main__":
    model = EcomodelLite(results_folder="results_lite")
    # model.process_tile_no_leaf_removal(r"G:\Projects\TreeCanopyLidar\Datasets\2025_10x10")

    folder = r"G:\Projects\TreeCanopyLidar\Datasets\2025_10x10"
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.las', '.laz'))]
    for tile in files:
        logger.info("------------- Processing Tile %s -------------", tile)
        full_tile_path = os.path.join(folder, tile)
        model.process_tile(full_tile_path, save_data=True, show_plots=False)
        # break


    # Testing: 
    # model = EcomodelLite()
    # # path = Path(r"G:\Projects\TreeCanopyLidar\Datasets\2025_1cm\tile_00005_2_0.laz")
    # path = Path(r"G:\Projects\TreeCanopyLidar\Datasets\2025_1cm\tile_00005_2_0_leaves_removed.xyz")
    # # path = Path(r"G:\Projects\TreeCanopyLidar\Datasets\oblong_tile\trunks.laz")
    # folder_path = path.parent
    # basename = path.stem
    # tile_data, full_data = load_point_cloud(str(path), full_data=True)
    # # full_data = model.remove_ground(full_data)
    # # full_data = model.filter_intensity(full_data, 42000)
    # # full_data = model.remove_leaves_rgi(full_data)
    # # model.save_point_cloud(f"{folder_path}\{path.stem}_leaves_removed.xyz", full_data)
    # full_data = model.normalize_point_cloud(full_data)
    # # np.savetxt("Fulldata.xyz", full_data)
    # # model.remove_noise(full_data)
    # full_data, instance_labels = model.perform_instance_segmentation(full_data)
    # cylinder_data = model.get_cylinders(full_data, instance_labels)

    # # view_data = np.concatenate((full_data,instance_labels[:,np.newaxis] ), axis=1)
    # # model.view_point_cloud(view_data)
    # # model.save_point_cloud(f"{folder_path}\{path.stem}_labeled.xyz", view_data)
    # # model.view_cylinders(full_data, cylinder_data)
    

















