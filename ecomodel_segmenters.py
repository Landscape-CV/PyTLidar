"""
This module includes a class which streamlines the current ecomodel.
"""

# Scanline
import numpy as np
from TreeQSMSteps.cover_sets import cover_sets
from Utils.TreeSegmentation import segment_point_cloud
from ecomodel import Tile

# TreeX
import numpy as np
from pointtree.instance_segmentation import TreeXAlgorithm, TreeXPresetTLS, TreeXPresetULS, TreeXPresetOriginal
from dataclasses import replace
from pointtorch import read


class SegmenterScanline:
    """
    Returns labels for each Tree. 

    point_cloud --> labels. 
    """
    def __init__(self):
        pass

    def process(self, point_cloud, intensity_threshold = 0):
        """
        Segments the point cloud into groups from a single tree
        Parameters: 
                min_points (int): Minimum number of points in a cluster.
        Returns:
                numpy.ndarray: Clustered point cloud, shape (n_points, 3).
        """         

        inputs = {'PatchDiam1': 0.15, 'BallRad1':.15, 'nmin1': 25}
        tile = Tile(point_cloud[:, :3], point_cloud)

        cover = cover_sets(tile.get_cloud_as_array(), inputs, qsm =False, device = 'cpu', full_point_data = tile.point_data)
        if len(cover['sets']) == 0:
            return None, None
        
        labels = cover['sets']
        
        noise_mask = labels >-1
        tile.cloud = tile.cloud[noise_mask]
        tile.point_data = tile.point_data[noise_mask]
        labels = labels[noise_mask]
        tile.cover_sets=labels

        if len(labels) == 0:
            return None, None


        # Default parameters for the tree instance segmentation.
        default_arguments = {
            "max_dist": 0.16,
            "min_height" :.3,  
            "connect_using_midpoint" :False, 
            "base_height" :.65, 
            "base_dist_multiplier" :2.5, 
            "connect_ambiguous_points" :True, 
            "fix_overlapping_segments" :False, 
            "layer_size" :.16, 
            "min_Z" :float(np.min(tile.cloud[:,2])),
            "combine_nearby_bases" :True ,
        }

        # New tuned parameters. 
        tuned_arguments = {
            "max_dist": 0.3,
            "base_height" : 1, 
            "layer_size" :0.15, 
            "combine_nearby_bases" :False,
        }

        default_arguments.update(tuned_arguments)
        segment_point_cloud(tile,**default_arguments)
        mask = tile.segment_labels >-2#filters out points that could not be connected, ideal will segment better and this will be uneccesary
        print("UNIQUE LABELS", np.unique(tile.segment_labels))
        
        point_cloud = tile.cloud[mask]
        labels = tile.segment_labels[mask]

        return point_cloud, labels



class SegmenterTreeX:
    """
    From the Pointtree package.
    """
    def __init__(self):
        preset = TreeXPresetOriginal()  # or use TreeXPresetULS()

        # FIX: Lower the intensity threshold to match your data's intensity range (0-33)
        # Default is 6000, but your data only goes to 33
        
        preset = replace(
            preset, 
            stem_search_min_cluster_intensity=2,
            # csf_tree_classification_threshold = 0.5,
            # stem_search_min_z = 0.5,
            # stem_search_max_z = 3,
            stem_search_dbscan_2d_min_points = 50,
            stem_search_voxel_size = 0.005,
            stem_search_circle_fitting_max_std_diameter = 5,
            stem_search_circle_fitting_min_completeness_idx = None,
            stem_search_circle_fitting_min_points = 5 # Lowering this alloewed for a tree trunk to be properly segmented. 
            # Relax circle fitting to detect more trees
            # stem_search_circle_fitting_layer_start=2,
            # stem_search_circle_fitting_max_std_diameter=0.1,  # 0.04 is too strict
            # stem_search_circle_fitting_min_fitting_score=50.0,  # 100.0 is too strict
            # stem_search_circle_fitting_min_points=10,  # Allow slightly fewer points
            # stem_search_circle_fitting_min_completeness_idx=0.2,  # More lenient on completeness
            # For separating closely-spaced trees, consider these parameters:
            # tree_seg_max_search_radius=0.35,  # Reduce from 0.5 to prevent nearby crowns merging
            # tree_seg_seed_diameter_factor=0.85,  # Reduce from 1.05 for tighter initial seeds
            # tree_seg_seed_layer_height=0.4,  # Reduce from 0.6 for finer seed resolution
        )

        self.algorithm = TreeXAlgorithm(**preset)



    def segment(self, point_cloud):
        """
        Perform Tree instance segmentation

        Args:
            point_cloud (numpy.ndarray): Input point cloud with shape (n_points, 4) where columns are [x, y, z, intensity].

        Returns:
            point_cloud (numpy.ndarray): Input point cloud with shape (n_points, 4) where columns are [x, y, z, intensity].
            instance_ids: output instance Ids. 
        """
        instance_ids, trunk_positions, trunk_diameters = self.algorithm(point_cloud[:, :3], intensities=point_cloud[:, 3])

        return point_cloud, instance_ids


