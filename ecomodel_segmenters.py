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
    Standalone implementation of John's segmentation algorithm. 

    point_cloud --> labels. 
    """
    def __init__(self):
        pass

    def segment(self, point_cloud):
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

class SegmenterTreeLearn:
    """
    Wraps the TreeLearn pipeline to provide the same segment() interface
    as SegmenterTreeX: accepts an (N,3) or (N,4) numpy array and returns
    (point_cloud, instance_ids).

    TreeLearn labels:
        0  → non-tree  (remapped to -1 to match ecomodel convention)
        1+ → individual tree instances

    Usage:
        segmenter = SegmenterTreeLearn(config_path="path/to/config.yaml")
        point_cloud, instance_ids = segmenter.segment(point_cloud_array)
    """

    def __init__(self, config_path: str,logger):
        """
        Args:
            config_path: Path to the TreeLearn YAML config file.
                         The config's forest_path will be overridden at
                         runtime with a temporary file for each call to
                         segment(), so it only needs to be a valid path
                         format — the actual file does not have to exist.
        """
        from tree_learn.util import get_config
        self.config_path = config_path
        self.base_config = get_config(config_path)
        self.logger=logger

    def segment(self, point_cloud: np.ndarray, temp_dir: str = None):
        """
        Run TreeLearn instance segmentation on an in-memory point cloud.

        Args:
            point_cloud (np.ndarray): (N, 3) or (N, 4) array [x, y, z, (intensity)].
            temp_dir (str, optional): Directory for intermediate TreeLearn files.
                                      A TemporaryDirectory is used if not supplied.

        Returns:
            point_cloud (np.ndarray): The voxelized/filtered point cloud that
                                      TreeLearn actually operated on (may be
                                      fewer points than the input due to
                                      voxelization).
            instance_ids (np.ndarray): Per-point integer labels.
                                       -1  = non-tree / unassigned
                                        1+ = individual tree instances
        """
        import copy
        import tempfile
        from tree_learn.util import get_config
        from tools.pipeline.pipeline import run_treelearn_pipeline

        xyz = point_cloud[:, :3].astype(np.float64)
        work_dir = temp_dir

        try:
            # --- write input cloud as NPZ where TreeLearn expects it ----------
            forest_dir = os.path.join(work_dir, "forest")
            os.makedirs(forest_dir, exist_ok=True)
            forest_path = os.path.join(forest_dir, os.path.basename(work_dir)+".npz")
            np.savez_compressed(forest_path, points=xyz)

            # --- build a per-call config with the temp paths -----------------
            config = copy.deepcopy(self.base_config)
            config.forest_path = forest_path
            config.tile_generation=True
            # Disable treewise saving to keep things fast
            config.save_cfg.save_treewise = False
            # Keep pointwise results so we can read instance_preds back
            config.save_cfg.save_pointwise = False
            config.save_cfg.save_formats=['npz','laz']


            # --- read back results --------------------------------------------


            # Results land under work_dir/results/input/pointwise_results/
            full_segmented_forest = os.path.join(work_dir, 'results','full_forest',os.path.basename(work_dir)+".npz")

            run_treelearn_pipeline(config)

            if not os.path.exists(full_segmented_forest):
                self.logger.warning("TreeLearn full segmented tree results not found at %s", full_segmented_forest)
                return None, None


            data = np.load(full_segmented_forest, allow_pickle=True)
            coords = data['points'] # full XYZ with intensity
            instance_preds = data['labels'].copy()

            # Remap: TreeLearn 0 (non-tree) → ecomodel -1
            instance_preds[instance_preds == 0] = -1

            return coords, instance_preds

        except Exception as e:
            self.logger.error("TreeLearn segmentation failed: %s", e)
            import traceback
            traceback.print_exc()
            return None, None

