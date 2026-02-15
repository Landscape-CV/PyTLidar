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


class Tile:
    """
    Simpler tile for storing point cloud.
    """
    def __init__(self):
        """"""
        

class PlotterBase:
    def __init__(self):
        self.simple_plotter = SimplePlotter()

    def view_point_cloud(self, point_cloud):
        """"""
        self.simple_plotter.add_point_cloud(point_cloud)
        self.simple_plotter.show()

    def save_point_cloud(self, path, point_cloud):
        """"""
        np.savetxt(path, point_cloud)

    



class PotentialBranch:
    """
    Responsibility of class is to hold information about a specific branch. 
    
    Basically we want the 'graph' to consist of only nearby things. 
    """
    def __init__(self, segment_id):
        """"""
        self.segment_id = segment_id
        self.top_centroid = None
        self.top_centroids = []

    def add_centroid(self, centroid):
        self.top_centroids.append(centroid)

    # def add_branch_segment(self, centroid):
    #     self.top_layer_cendroid = np.mean(self.main_point_cloud[mask], axis=1)
        


class Segmenter(PlotterBase):
    """
    Responsiblity of this class is to return a point cloud that has individial branch chains segmented, 
    Especially from mangrovy type bushes. Maybe can work for trees? I am not sure.     
    """
    def __init__(self):
        super().__init__()

    def process(self, point_cloud, layer_height = 0.1, radius = 0.2):
        """
        Builds layers


        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz = np.concatenate((raw_points_layer, labels), axis=1)
        """

        min_value = min(point_cloud[:, 2])
        layer_min = min_value
        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_classification = np.concatenate((point_cloud_xyz, np.full((point_cloud_xyz.shape[0], 1), -1)), axis=1)
        point_cloud_segment_classification = np.concatenate((point_cloud_xyz, np.full((point_cloud_xyz.shape[0], 1), -1)), axis=1)

        potential_trees = []
        segment_id = 0
        layer_num = 0
        scanned = 1
        num_empty_layers = 0
        max_empty_layers = 20
        while True:
            # radius += 0.01
            layer_pc_mask = (layer_min < point_cloud[:, 2]) & (point_cloud[:, 2] < layer_min + layer_height)

            layer = point_cloud[layer_pc_mask]
            raw_points_layer = layer[:, 0:3]
            labels = self.classify_via_dbscan(raw_points_layer)

            # if layer_num > 10:
            #     break

            if labels.size == 0:
                num_empty_layers += 1

                if num_empty_layers > max_empty_layers:
                    break
                else:
                    layer_min += layer_height
                    layer_num += 1
                    continue
            num_empty_layers = 0

            # print(np.unique(labels))
            point_cloud_segment_classification[layer_pc_mask, 3] = labels
            # labels = labels[:,np.newaxis]
            # np.savetxt("debug_pc.txt", point_cloud_segment_classification)
            # np.savetxt("debug_labels.txt", labels)
            # self.view_point_cloud(point_cloud_classification)

            # print(labels)
            # print("Layer", layer_num)
            # print("Total unique branch segments", np.unique(labels))
            mapping = []
            for label in np.unique(labels):

                if label == -1:
                    continue

                if layer_num ==3 :
                    # branch_segment_mask = point_cloud_classification[:, 3] == label
                    # layer_segment_mask = layer_pc_mask#  & branch_segment_mask   
                    # print("Layer segment mask")    
                    # point_cloud_classification[layer_segment_mask,  3] = segment_id
                    pass
                    # self.view_point_cloud(point_cloud_segment_classification[layer_pc_mask])
                branch_segment_mask = point_cloud_segment_classification[:, 3] == label
                layer_segment_mask = layer_pc_mask & branch_segment_mask       
                
                if layer_num == 0:
                    point_cloud_classification[layer_segment_mask,  3] = segment_id
                    new_branch = PotentialBranch(segment_id)
                    new_branch.add_centroid(np.mean(point_cloud_segment_classification[layer_segment_mask], axis=0))
                    potential_trees.append(new_branch)
                    segment_id += 1
                    # print("Base branch added")
                    continue
                    
                else:
                    brand_new_branch = True
                    for potential_tree in potential_trees:
                        # print("Label", label)
                        # print("seg id", potential_tree.segment_id)
                        # print("existing Centroid", potential_tree.top_centroid[0:2])
                        for centroid in potential_tree.top_centroids:
                            segment_centroid = np.mean(point_cloud_segment_classification[layer_segment_mask], axis=0)
                            # print("new Centroid", segment_centroid[0:2])
                            distance = np.linalg.norm(potential_tree.top_centroid[0:2] - segment_centroid[0:2])

                            # print(distance)

                            if distance < radius:
                                ""
                                brand_new_branch = False
                                tree = potential_tree
                                break
                                
                    if brand_new_branch:
                        new_branch = PotentialBranch(segment_id)
                        new_branch.top_centroid = np.mean(point_cloud_segment_classification[layer_segment_mask], axis=0)    
                        potential_trees.append(new_branch)
                        segment_id += 1
                        # print("new branch added")

                    else:
                        layer_segment_mask = layer_pc_mask & branch_segment_mask     
                        tree.top_centroid = np.mean(point_cloud_segment_classification[layer_segment_mask], axis=0)
                        mapping.append((label, tree.segment_id))
                        point_cloud_classification[layer_segment_mask, 3] = tree.segment_id
                        # print("existing branch")
            
            # Centroids
            # print("Centroids", [tree.top_centroid for tree in potential_trees])
                # print(f"Euclidean Distance: {distance}")

                # branch_segment_mask = point_cloud_classification[layer_pc_mask][] == label

                # potential_trees.append(PotentialBranch())

                # self.view_point_cloud(point_cloud_xyz)

            layer_min += layer_height
            layer_num += 1

            # Classify the entire cloud now with each of the potential branches.
             
            # for tree in potential_trees:
            #     tree: PotentialBranch
            #     point_cloud_classification[tree.]


        print(np.unique(point_cloud_classification[:,3]))

        # self.view_point_cloud(point_cloud_xyz)
        # break
        self.save_point_cloud("output.xyz", point_cloud_classification)
        self.view_point_cloud(point_cloud_classification)
        


    def classify_via_dbscan(self, point_cloud, epsilon=0.05):
        """
        Uses DBSCAN to return a set of labels for each point for the cluster that was in.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        labels = np.array(
            pcd.cluster_dbscan(eps=epsilon, min_points=10, print_progress=True)
        )

        


        return labels


class Ecomodel2(PlotterBase):
    def __init__(self):
        super().__init__()
        self.results_folder = ""
        self.segmenter = Segmenter()


    def run(self):
        """"""


    def remove_ground(self, point_cloud_tile, remove_under_ground = True, write_cloth=False):
        """"""

        csf = CSF.CSF()
        new_min_z = float('inf')

        # prameter settings
        csf.params.cloth_resolution = 2
        csf.params.class_threshold = 0.5
        csf.params.interations = 500

        csf.setPointCloud(point_cloud_tile)
        ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
        non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
        csf.do_filtering(ground, non_ground, exportCloth=write_cloth)
        ground_mask = np.array(ground)
        non_ground_mask = np.array(non_ground)
        ground_points = point_cloud_tile[ground_mask]
        lowest_ground_height = np.min(ground_points[:,2]) 
        print(lowest_ground_height)

        point_cloud_tile = point_cloud_tile[non_ground_mask]

        if remove_under_ground:
            above_ground_mask = point_cloud_tile[:,2] > lowest_ground_height
            point_cloud_tile = point_cloud_tile[above_ground_mask]
        
        return point_cloud_tile


    def classify_wood_leaf_on_array(self,tree_cloud, input_params=None):
            """
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
        Returns a wood and leaf mask. 
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


    def view_point_cloud(self, point_cloud):
        """"""
        self.simple_plotter.add_point_cloud(point_cloud)
        self.simple_plotter.show()

    def save_point_cloud(self, path, point_cloud):
        """"""
        np.savetxt(path, point_cloud)

    
    def perform_instance_segmentation(self, point_cloud):
        """"""
        self.segmenter.process(point_cloud)

if __name__ == "__main__":
    model = Ecomodel2()

    # tile_data, full_data = load_point_cloud(r"G:\Projects\TreeCanopyLidar\Datasets\MVP_tiles\other\tile_573110_2840090.laz", full_data=True)
    # full_data = model.remove_ground(full_data)
    # full_data = model.filter_intensity(full_data, 42000)
    # full_data = model.remove_leaves_rgi(full_data)
    # model.save_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\_experimentation\testing_my_method\leaves_intensity_removed.xyz", full_data)
    # model.view_point_cloud(full_data)



    tile_data, full_data = load_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\_experimentation\testing_my_method\leaves_intensity_removed.xyz", full_data=True)
    # tile_data, full_data = load_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\_experimentation\testing_my_method\two_trees.las", full_data=True)
    model.perform_instance_segmentation(full_data)