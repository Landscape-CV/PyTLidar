"""
This module includes a class which streamlines the current ecomodel.
"""
import os
import shutil
import CSF
import numpy as np
from Utils.Utils import load_point_cloud
from SegmentRGI.SegmentRGI import classify_wood_leaf,classify_wood_leaf_point_cloud
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
#from ecomodel_segmenters import SegmenterScanline, SegmenterTreeX, SegmenterTreeLearn
from ecomodel_segmenters import SegmenterScanline, SegmenterTreeLearn
from Utils.define_input import define_input
from treeqsm import treeqsm
from TreeQSMSteps.cover_sets import cover_sets
from TreeQSMSteps.segments import segments
from TreeQSMSteps.correct_segments import correct_segments
from TreeQSMSteps.tree_sets import tree_sets
from TreeQSMSteps.relative_size import relative_size
from TreeQSMSteps.cylinders import cylinders
import logging
from GBSeparation.remove_leaves import GBSeperationWoodLeafClassifier

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
    """
    This class removes small and distant clusters to improve the TreeQSM results. 

    It achieves this by:
    1. Voxelizing the point cloud to create a 3D grid representation.
    2. Performing connected component analysis on the voxel grid to identify clusters.
    3. Creating a mask for each cluster in the connected components, applying to the original point cloud and labeling appropriately. 
    4. Using the amount of points in clusters (which are seperated by voxel_size spaces in the voxel grid) and removing the smaller clusters. 
    
    Note: 
        Main function is 'remove_distant_small_clusters' 
    """
    def __init__(self, voxel_size=1, min_points=100):
        self.voxel_size = voxel_size
        self.min_points = min_points


    def clean(self, point_cloud):
        """
        Remove small clusters.

        Args: 
            point_cloud (array): Represents size of cluster. 
        """
        data_xyz = point_cloud[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data_xyz)

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
        classified = cc3d.connected_components(grid)

        # 1. Determine the indices at for each voxel grid. 
        # 2. Determine the points that are within the voxel coordinates (point_mask)
        # 3. Remove the points using the mask.  
        point_cloud_with_classification = np.concatenate((point_cloud[:,:3], np.zeros((point_cloud.shape[0], 1))), axis=1)
        for classification in np.unique(classified):
            mask = classified == classification
            x_idx, y_idx, z_idx = np.where(mask)
            indices = np.concatenate((x_idx[:, np.newaxis], y_idx[:, np.newaxis], z_idx[:, np.newaxis]), axis=1)

            for index in indices:
                point_mask = self.get_mask_for_voxel(point_cloud, index, voxel_grid.origin, self.voxel_size)
                point_cloud_with_classification[point_mask, 3] = classification

        point_cloud_cleaned = self.remove_points(point_cloud_with_classification)
        return point_cloud_cleaned[:, :3]

    def get_mask_for_voxel(self, point_cloud, voxel_indices, origin, voxel_size):
        """
        Uses the point cloud, voxel indices, and location of voxel to 
        obtain the mask within the normal point cloud space. 
        """
        voxel_min = origin + voxel_indices * voxel_size
        voxel_max = voxel_min + voxel_size
        
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
        
        return classification_copy

class TreeQSMCylinderFitting:
    """
    Class which implements the TreeQSM algorithm by using only the specific function calls 
    necessary to produce the cylinders. 
    """
    def __init__(self):
        pass

    def get_cylinders(self, point_cloud, instance_labels, noise_remover = None):
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
        cylinder_tree_instance = np.array([])

        for tree_instance in np.unique(instance_labels):
            if tree_instance == -1:
                continue
            
            segment_mask = (labeled_point_cloud[:,3] == tree_instance)
            tree_cloud = labeled_point_cloud[segment_mask, :3]
            print(tree_instance)

            np.savetxt(f"segment_{tree_instance}.xyz", tree_cloud)
            if noise_remover:
                print("Removing Small clusters...")
                tree_cloud = noise_remover.clean(tree_cloud)
                print("Done.")

            if len(tree_cloud) < 100:
                continue
            try:
                qsm_input = define_input(tree_cloud, 1, 1, 1)[0]
            except np.linalg.LinAlgError as e:
                print(f"Failed to define input {e}")
                import traceback
                traceback.print_exc()
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
                print(f"Failed to get cylinders. {e}")
                import traceback
                traceback.print_exc()
                continue

            cylinder_starts = np.concatenate([cylinder_starts,cylinder["start"]])
            cylinder_radii = np.append(cylinder_radii,cylinder["radius"])
            cylinder_axes = np.concatenate([cylinder_axes,cylinder["axis"]])
            cylinder_lengths = np.append(cylinder_lengths,cylinder["length"])
            cylinder_tree_instance = np.append(cylinder_tree_instance, np.full(cylinder["start"].shape[0], tree_instance))
            
        cylinder_data = np.concatenate((cylinder_starts, cylinder_radii.reshape(-1, 1), cylinder_axes, cylinder_lengths.reshape(-1, 1), cylinder_tree_instance.reshape(-1, 1)), axis=1)

        return cylinder_data

class TreeQSMFull:
    """
    Class that uses the actual TreeQSM function call to produce the cylinder results. 
    Has additional overhead but may be more robust. 
    """
    def __init__(self):
        pass

    def get_cylinders(self, point_cloud, instance_labels, noise_remover = None):
        """
        Uses actual TreeQSM.
        """
        labeled_point_cloud = np.concatenate((point_cloud[:,:3], instance_labels[:, np.newaxis]), axis=1)

        cylinder_starts = np.empty((0,3))
        cylinder_radii = np.array([])
        cylinder_axes = np.empty((0,3))
        cylinder_lengths = np.array([])
        cylinder_tree_instance = np.array([])

        for tree_instance in np.unique(instance_labels):
            if tree_instance == -1:
                continue


            segment_mask = (labeled_point_cloud[:,3] == tree_instance)
            tree_cloud = labeled_point_cloud[segment_mask, :3]
            print(tree_instance)



            def add_noise_to_uniform_cloud(point_cloud, noise_scale=0.001):
                """
                Add small random noise to uniform point clouds to prevent numerical degeneracies.
                noise_scale: standard deviation relative to point spacing
                """
                noise = np.random.normal(0, noise_scale, point_cloud.shape)
                return point_cloud + noise

            # In your main.py or TreeQSM call:
            tree_cloud = add_noise_to_uniform_cloud(tree_cloud, noise_scale=0.0005)


            try:
                qsm_input = define_input(tree_cloud,1,1,1)[0]
                qsm_input['plot'] = 0
                qsm_input['savepdf'] = 0
                qsm_input['savetxt'] = 0
            except np.linalg.LinAlgError as e:
                logger.warning(f"Unable to find axis for segment {tree_instance}")
            except Exception as e:
                logger.warning(f"Error defining initial params for segment {tree_instance}")
                continue
            models, _ = treeqsm(tree_cloud, qsm_input)
            if models == "ERROR":
                logger.info(f"Skipping Segment {tree_instance} (TreeQSM Failed)")
                continue

            qsm = models[0]
            cylinder = qsm['cylinder']

            cylinder_starts = np.concatenate([cylinder_starts,cylinder["start"]])
            cylinder_radii = np.append(cylinder_radii,cylinder["radius"])
            cylinder_axes = np.concatenate([cylinder_axes,cylinder["axis"]])
            cylinder_lengths = np.append(cylinder_lengths,cylinder["length"])
            cylinder_tree_instance = np.append(cylinder_tree_instance, np.full(cylinder["start"].shape[0], tree_instance))
            
        cylinder_data = np.concatenate((cylinder_starts, cylinder_radii.reshape(-1, 1), cylinder_axes, cylinder_lengths.reshape(-1, 1), cylinder_tree_instance.reshape(-1, 1)), axis=1)
        return cylinder_data



class CSFGroundRemoval:
    """
    Algorithm which removes ground from a point cloud. 
    """
    def __init__(self):
        pass

    def remove_ground(self, point_cloud, remove_under_ground = True):
        """
        Uses CSF To remove ground in point cloud. Stores the ground level in self.ground_z for the tile.
        
        Args: 
            point_cloud (Nx3 Array): Point cloud representing tile.
            remove_under_ground (bool): Whether to points below the ground level or not. 

        Returns:
            point_cloud: Point cloud without a ground. 
            ground_z: Ground level of the tile.
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
        ground_z = mean_ground_height

        if remove_under_ground:
            above_ground_mask = point_cloud[:,2] > mean_ground_height
            point_cloud = point_cloud[above_ground_mask]
        
        return point_cloud, ground_z


class RGIWoodLeafClassifier:
    def __init__(self, noise_percentile=0, angle_deg=7, curv_thresh=0.07, 
                 resid_thresh=0.05, k=30, minClusterSize=40, maxClusterSize=100000,
                 smoothMode=True, useResidualTest=True, useCurvatureTest=True):
        self.input_params = {
            "noise_percentile": noise_percentile,
            "angle_deg": angle_deg,
            "curv_thresh": curv_thresh,
            "resid_thresh": resid_thresh,
            "k": k,
            "minClusterSize": minClusterSize,
            "maxClusterSize": maxClusterSize,
            "smoothMode": smoothMode,
            "useResidualTest": useResidualTest,
            "useCurvatureTest": useCurvatureTest,
        }
        
    def classify(self, point_cloud):
        """
        Removes leaf points from a point cloud.

        Args:
            point_cloud (Nx4): point_cloud representing a tile with trees. 

        Return:
            only_wood (Nx4): Point cloud with no points. 
        """
        if point_cloud.shape[0] < 100:
            return None

        wood_mask, leaf_mask = classify_wood_leaf_point_cloud(point_cloud, **self.input_params)
        print(wood_mask, leaf_mask)
        if wood_mask is None or leaf_mask is None:
            return None
        if wood_mask is None or leaf_mask is None:
            return None

        # only_wood = point_cloud[wood_mask]
        # only_leaves = point_cloud[leaf_mask]

        return wood_mask, leaf_mask


class EcomodelFunctions:
    """
    Contains useful generic function used in ecomodel pipeline.
    """
    def __init__(self, results_folder="results", intensity_threshold=0,segmenter=None):
        super().__init__()
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)
        self.results_folder = results_folder
        self.intensity_threshold = intensity_threshold
        self.plotter = SimplePlotter()
        self.ground_z = 0

    @staticmethod
    def filter_intensity(point_cloud, intensity):
        """
        Filters points based on intensity.

        Args:
            point_cloud (Nx4): point_cloud representing a tile with trees.
            intensity (float): Removes points below this intensity threshold.

        Returns:
            point_cloud: Point cloud with intensity mask applied
            intensity_mask: mask containing points above the threshold. 
        """
        intensity_mask = point_cloud[:,3] > intensity
        return point_cloud[intensity_mask], intensity_mask

    @staticmethod
    def normalize_point_cloud(point_cloud):
        """
        Subtracts mean from point cloud

        Args: 
            point_cloud (Nx3 Array): Point cloud representing tile.

        Returns:
            point_cloud (Nx3 Array): Point cloud representing tile after normalization.
        """
        mean = np.mean(point_cloud[:,0:3], axis=0)
        point_cloud[:, :3] = point_cloud[:, :3] - mean
        return point_cloud, mean

    @staticmethod
    def unnormalize_point_cloud(point_cloud, mean):
        """
        Adds mean to point cloud

        Args: 
            point_cloud (Nx3 Array): Point cloud representing tile.

        Returns:
            point_cloud (Nx3 Array): Point cloud representing tile before normalization.

        """
        point_cloud[:, :3] = point_cloud[:, :3] + mean
        return point_cloud

    def view_cylinders(self, point_cloud, cylinder_data):
        """
        View cylinders in PyVista.

        Args:
            point_cloud (Nx4): point cloud representing a tile with trees.
            cylinder_data (Cx9): Cx9 numpy array representing the cylinders found in the tile.
        """
        print(cylinder_data.shape)
        for cylinder_row in range(0, cylinder_data.shape[0]):
            cylinder = cylinder_data[cylinder_row]

            start = cylinder[0:3]
            radius = cylinder[3]
            axis = cylinder[4:7]
            length = cylinder[7]

            # print(start, axis, radius, length)

            self.plotter.add_cylinder(start, axis, radius, length)
        
        self.plotter.add_point_cloud(point_cloud)

        self.plotter.show()

    def save_data(self, path, full_data, instance_labels, cylinder_data, mean, ground_z):
        """
        Saves the data. 

        Note: Unnormalizes the data back to original coordinates.

        Args:
            path: Path object representing original tile.

        """
        cylinder_data_copy = self.unnormalize_point_cloud(deepcopy(cylinder_data), mean)
        np.savetxt(f"{self.results_folder}/{path.stem}/{path.stem}_cylinders.txt", cylinder_data_copy)

        unnormalized = self.unnormalize_point_cloud(deepcopy(full_data), mean)
        with_labels = np.concatenate((unnormalized[:,:3], instance_labels[:,np.newaxis]), axis=1)
        np.savetxt(f"{self.results_folder}/{path.stem}/{path.stem}_leavesremoved.xyz", with_labels)
        
        with open(f"{self.results_folder}/{path.stem}/{path.stem}_data.txt", "w") as f:
            f.writelines(f"{mean[0]} {mean[1]} {mean[2]}")
            f.writelines("\n")
            f.writelines(str(ground_z))


class EcomodelScanline(EcomodelFunctions):
    """
    This Ecomodel uses the 
    """
    def __init__(self, results_folder, intensity_threshold):
        super().__init__(results_folder, intensity_threshold)
        self.instance_segmenter = SegmenterScanline()
        self.leaf_wood_classifier = RGIWoodLeafClassifier(noise_percentile=0, 
                                                          angle_deg=6, 
                                                          curv_thresh=0.07, 
                                                          resid_thresh=0.05, 
                                                          k=100,
                                                          minClusterSize=40, 
                                                          maxClusterSize=100000,
                                                          smoothMode=True, 
                                                          useResidualTest=True, 
                                                          useCurvatureTest=True)
        self.qsm = TreeQSMCylinderFitting()
        self.ground_removal = CSFGroundRemoval()

    def process_tile(self, tile_path, save_data = True, show_plots = True):
        """
        Process Tile Scanline Segmentation.
        """
        path = Path(tile_path)
        os.makedirs(f"{self.results_folder}/{path.stem}",exist_ok=True )
        _, full_data = load_point_cloud(str(path), full_data=True)
        full_data, mean = self.normalize_point_cloud(full_data)

        full_data, ground_z = self.ground_removal.remove_ground(full_data, True)

        full_data, intensity_mask = self.filter_intensity(full_data, 42000)
        # full_data = full_data[intensity_mask]

        wood_mask, leaf_mask = self.leaf_wood_classifier.classify(full_data)
        full_data = full_data[wood_mask]

        if full_data is None:
            logger.warning("Unable to remove leaves on segment.")
            return 
        # np.savetxt(f"{path.stem}_wood_only.txt", full_data)

        try:
            full_data, instance_labels = self.instance_segmenter.segment(full_data)
        except Exception as e:
            logger.warning(f"Error during instance segmentation: {e}")
            return
        if full_data is None or instance_labels is None:
            logger.warning("Unable to perform instance segmentation.")
            return

        # Visualualize intermediate step:
        # with_labels = np.concatenate((full_data[:,:3], instance_labels[:,np.newaxis]), axis=1)
        # np.savetxt(f"{path.stem}_labeled.txt", with_labels)

        cylinder_data = self.qsm.get_cylinders(full_data, instance_labels)
        print("Cylinder data shape", cylinder_data.shape)

        # Save results.
        if save_data:
            self.save_data(path, full_data, instance_labels, cylinder_data, mean, ground_z)

        if show_plots:
            self.view_cylinders(full_data, cylinder_data)




class EcomodelTreeX(EcomodelFunctions):
    def __init__(self, results_folder="results", intensity_threshold=0):
        super().__init__(results_folder, intensity_threshold)

        self.instance_segmenter = SegmenterTreeX()
        self.leaf_wood_classifier = RGIWoodLeafClassifier(noise_percentile=0, 
                                                          angle_deg=6, 
                                                          curv_thresh=0.07, 
                                                          resid_thresh=0.05, 
                                                          k=100,
                                                          minClusterSize=40, 
                                                          maxClusterSize=100000,
                                                          smoothMode=True, 
                                                          useResidualTest=True, 
                                                          useCurvatureTest=True)
        self.qsm = TreeQSMCylinderFitting()

    def process_tile(self, tile_path, save_data = True, show_plots = True):

        path = Path(tile_path)
        os.makedirs(f"{self.results_folder}/{path.stem}",exist_ok=True )
        _, full_data = load_point_cloud(str(path), full_data=True)
        full_data, mean = self.normalize_point_cloud(full_data)

        try:
            full_data, instance_labels = self.instance_segmenter.segment(full_data)
        except Exception as e:
            logger.warning(f"Error during instance segmentation: {e}")
            return
        if full_data is None or instance_labels is None:
            logger.warning("Unable to perform instance segmentation.")
            return

        # Filter out ground and extra noise manually.
        try: 
            ground_mask = instance_labels == -1
            full_data = full_data[~ground_mask]
            instance_labels = instance_labels[~ground_mask]
            ground_z = np.min(full_data[:,2])
        except ValueError as e:
            logger.warning(f"Error {e} - Couldnt get z value")
            ground_z = 0


        wood_mask, leaf_mask = self.leaf_wood_classifier.classify(full_data)
        full_data = full_data[wood_mask]
        instance_labels = instance_labels[wood_mask]
        full_data, intensity_mask = self.filter_intensity(full_data, 42000)
        instance_labels = instance_labels[intensity_mask]
        if full_data is None:
            logger.warning("Unable to remove leaves on segment.")
            return 
        # np.savetxt(f"{path.stem}_wood_only.txt", full_data)


        # with_labels = np.concatenate((full_data[:,:3], instance_labels[:,np.newaxis]), axis=1)
        # np.savetxt(f"{path.stem}_labeled.txt", with_labels)

        cylinder_data = self.qsm.get_cylinders(full_data, instance_labels)
        print("Cylinder data shape", cylinder_data.shape)

        # Save results.
        if save_data:
            self.save_data(path, full_data, instance_labels, cylinder_data, mean, ground_z)

        if show_plots:
            self.view_cylinders(full_data, cylinder_data)

class EcomodelTreeLearn(EcomodelFunctions):
    """
    Pipeline that uses TreeLearn for instance segmentation.

    Order of operations
    -------------------
    1. Load point cloud from disk
    2. Remove ground (CSF)
    3. Filter by intensity threshold
    4. Run TreeLearn instance segmentation
        - TreeLearn internally voxelises, tiles, runs the neural network,
          clusters via offset-shifted coordinates, and assigns remaining
          points via KNN.
        - Returns voxelised coords + per-point tree labels (0 = non-tree).
    5. Strip non-tree points (label == -1)
    6. Classify wood vs leaf (RGI)
    7. Keep wood points only
    8. Fit TreeQSM cylinders per tree instance
    9. Save / visualise results
    """

    def __init__(self, treelearn_config_path: str,
                 results_folder: str = "results",
                 intensity_threshold: int = 0):
        """
        Args:
            treelearn_config_path: Path to the TreeLearn YAML config file.
            results_folder:        Directory where per-tile result folders are written.
            intensity_threshold:   Points with intensity <= this value are dropped
                                   before segmentation.
        """
        from ecomodel_segmenters import SegmenterTreeLearn
        segmenter = SegmenterTreeLearn(treelearn_config_path,logger)
        super().__init__(results_folder, intensity_threshold, segmenter=segmenter)
        self.qsm = TreeQSMCylinderFitting()
        self.leaf_wood_classifier = RGIWoodLeafClassifier(
            noise_percentile=0,
            angle_deg=6,
            curv_thresh=0.07,
            resid_thresh=0.05,
            k=30,
            minClusterSize=40,
            maxClusterSize=100000,
            smoothMode=True,
            useResidualTest=True,
            useCurvatureTest=True,
        )
        self.ground_remover = CSFGroundRemoval()

    def process_tile(self, tile_path: str,
                     save_data: bool = True,
                     show_plots: bool = False):
        """
        Full TreeLearn-based processing pipeline for a single LAS/LAZ/XYZ tile.

        Args:
            tile_path:   Path to the input point cloud file.
            save_data:   Write cylinders, labeled cloud, and metadata to disk.
            show_plots:  Open the PyVista cylinder viewer after processing.
        """
        path = Path(tile_path)
        out_dir = os.path.join(self.results_folder, path.stem)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_folder, "EcomodelTreeLearn_Cylinders"), exist_ok=True)

        # ------------------------------------------------------------------
        # 1. Load
        # ------------------------------------------------------------------
        logger.info("[TreeLearn] Loading %s", path.name)
        _, full_data = load_point_cloud(str(path), full_data=True)

        # ------------------------------------------------------------------
        # 2. Ground removal
        # ------------------------------------------------------------------
        # logger.info("[TreeLearn] Removing ground")
        # result = self.ground_remover.remove_ground(full_data)
        # if result is None:
        #     logger.warning("[TreeLearn] Ground removal returned nothing — skipping tile.")
        #     return
        # full_data, ground_z = result

        # ------------------------------------------------------------------
        # 3. Intensity filter
        # ------------------------------------------------------------------
        if self.intensity_threshold > 0:
            full_data = self.filter_intensity(full_data, self.intensity_threshold)
            if full_data.size < 100:
                logger.warning("[TreeLearn] Too few points after intensity filter — skipping.")
                return "Too few points after intensity filter"

        # ------------------------------------------------------------------
        # 4. Normalise (centre coords for numerical stability)
        # ------------------------------------------------------------------
        full_data, mean = self.normalize_point_cloud(full_data)


        # ------------------------------------------------------------------
        # 5. TreeLearn instance segmentation
        # ------------------------------------------------------------------
        logger.info("[TreeLearn] Running TreeLearn segmentation")
        coords, instance_ids = self.segmenter.segment(full_data,out_dir)
        if coords is None or instance_ids is None:
            logger.warning("[TreeLearn] Segmentation failed — skipping tile.")
            return "TreeLearn Segmentation failed"

        # ------------------------------------------------------------------
        # 6. Drop non-tree points  (label == -1)
        # ------------------------------------------------------------------
        tree_mask = instance_ids != -1
        coords       = coords[tree_mask]

        instance_ids = instance_ids[tree_mask]

        if coords.shape[0] == 0:
            logger.warning("[TreeLearn] No tree points found — skipping tile.")
            return "TreeLearn No tree points found"

        # ------------------------------------------------------------------
        # 7. Leaf / wood classification (operates on xyz + intensity)
        #    coords is the original full-density array returned by the segmenter
        # ------------------------------------------------------------------
        logger.info("[TreeLearn] Classifying wood / leaf")
        wood_mask, leaf_mask = self.leaf_wood_classifier.classify(coords)
        if wood_mask is None:
            logger.warning("[TreeLearn] Leaf/wood classification failed — using all tree points.")
            wood_coords = coords
            wood_ids    = instance_ids
        else:
            wood_coords = coords[wood_mask]
            wood_ids    = instance_ids[wood_mask]

        if wood_coords.shape[0] == 0:
            logger.warning("[TreeLearn] No wood points remain — skipping tile.")
            return "No wood points remain after Tree-Leaf Segmentation"
        os.makedirs(os.path.join(out_dir,"Tree-Leaf segmentation"), exist_ok=True)
        np.savetxt(os.path.join(out_dir,"Tree-Leaf segmentation", f"{path.stem}_wood_only.txt"), wood_coords)

        # ------------------------------------------------------------------
        # 8. TreeQSM cylinder fitting
        # ------------------------------------------------------------------
        logger.info("[TreeLearn] Fitting QSM cylinders")
        cylinder_data = self.qsm.get_cylinders(wood_coords, wood_ids,
                                                noise_remover=self.noise_remover)
        if cylinder_data is None or cylinder_data.shape[0] == 0:
            logger.warning("[TreeLearn] No cylinders produced — skipping save.")
            return "No cylinders produced by TreeQSM"

        logger.info("[TreeLearn] Cylinder data shape: %s", cylinder_data.shape)

        # ------------------------------------------------------------------
        # 9. Save
        # ------------------------------------------------------------------
        if save_data:
            cylinder_data_out = self.unnormalize_point_cloud(cylinder_data.copy(), mean)
            os.makedirs(os.path.join(out_dir, "TreeQSM Output"), exist_ok=True)
            np.savetxt(os.path.join(out_dir,'TreeQSM Output', f"{path.stem}_cylinders.txt"), cylinder_data_out)
            np.savetxt(os.path.join(self.results_folder, 'EcomodelTreeLearn_Cylinders', f"{path.stem}_cylinders.txt"), cylinder_data_out)


            unnorm_coords = self.unnormalize_point_cloud(wood_coords[:, :3].copy(), mean)
            labeled = np.hstack([unnorm_coords, wood_ids[:, np.newaxis]])
            np.savetxt(os.path.join(out_dir,'Tree-Leaf segmentation', f"{path.stem}_labeled.txt"), labeled)

            with open(os.path.join(out_dir,'TreeQSM Output', f"{path.stem}_data.txt"), "w") as f:
                f.write(f"{mean[0]} {mean[1]} {mean[2]}\n")
                #f.write(f"{ground_z}\n")

            logger.info("[TreeLearn] Results saved to %s", out_dir)

        # ------------------------------------------------------------------
        # 10. Visualise
        # ------------------------------------------------------------------
        if show_plots:
            self.view_cylinders(wood_coords, cylinder_data)

        return "Completed"




if __name__ == "__main__":
    #### This block is for testing the TreeX Ecomodel on a single tile.
    model = EcomodelScanline(results_folder="results", intensity_threshold=42000)
    model.process_tile(r"G:\Projects\TreeCanopyLidar\Datasets\2025_10x10\retile_573088_2840085_0_1.laz", save_data=True, show_plots=True)

    #### Just segmentation:
    # model = EcomodelScanline(results_folder="results", intensity_threshold=42000)
    # path = Path(r"G:\Projects\TreeCanopyLidar\Datasets\2025_10x10\retile_573088_2840085_0_1.laz")
    # _, full_data = load_point_cloud(str(path), full_data=True)
    # full_data, instance_labels = model.instance_segmenter.segment(full_data)
    # with_labels = np.concatenate((full_data[:,:3], instance_labels[:,np.newaxis]), axis=1)
    # np.savetxt(f"{path.stem}_TreeX_Classification.txt", with_labels)


    #### Uncomment this block to run on all tiles in a folder.
    # model = EcomodelTreeX(results_folder="results", intensity_threshold=42000)
    # folder = r"G:\Projects\TreeCanopyLidar\Datasets\2025_10x10"
    # files = [f for f in os.listdir(folder) if f.lower().endswith(('.las', '.laz'))]
    # for tile in files:
    #     logger.info("------------- Processing Tile %s -------------", tile)
    #     full_tile_path = os.path.join(folder, tile)
    #     model.process_tile(full_tile_path, save_data=True, show_plots=True)
    #     # break

















