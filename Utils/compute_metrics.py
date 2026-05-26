"""
Compares the QSMs cylinders with the ones from the Ecomodel ones. 

But the ecomodel ones do not have individual files per tree, and instead, individual files per tile. 

So we would just need to probably rerun the tiles again, and this time keep the tree 


COLUMN 1: Radii of the cylinders
COLUMN 2: Lengths of the cylinders
COLUMN 3: Starting points of the cylinders, x-coordinate
COLUMN 4: Starting points of the cylinders, y-coord
COLUMN 5: Starting points of the cylinders, z-coord
COLUMN 6: Axes of the cylinders, x-component
COLUMN 7: Axes of the cylinders, y-comp
COLUMN 8: Axes of the cylinders, z-comp
COLUMN 9: Parent cylinder of each cylinder
COLUMN 10: Extension cylinder of each cylinder
COLUMN 11: Branch of the cylinders
COLUMN 12: Branch order of the cylinders
COLUMN 13: Running index of the cylinders in the branch (1 = base)
COLUMN 14: Logical vector indicating cylinders that are added to fill gaps
The coordinate system is oriented to true north and relative to the centre of the plot (0,0,0)
"""


# load the cylinders from the file and view them.
import math
from Utils.plot_tools import SimplePlotter
import numpy as np
import pyvista as pv
plotter = SimplePlotter()
import os
from ecomodel_lite import EcomodelTreeX, SegmenterTreeX, RGIWoodLeafClassifier, GBSeperationWoodLeafClassifier, EcomodelFunctions
from Utils.Utils import load_point_cloud
import laspy

# cylinder = np.loadtxt("G:\Projects\TreeCanopyLidar\PyTLidar\_experimentation\downloaded_qsms_rush\pgv1_rush07_2_2_cylmodel.txt")
# total_volume = 0
# for row in cylinder:
#     radius = row[0]    
#     length = row[1]
#     start = row[2:5]
#     axis = row[5:8]
#     plotter.add_cylinder(start, axis, radius, length)

#     volume += np.pi * radius**2 * length

# plotter.show()
# print(f"Total volume: {total_volume}")

# # Compute total volume: 



# mean_location_of_tree = np.mean(cylinder[2:5], axis=1)

# # Our tile. 
# cylinder = np.loadtxt(r"G:\Projects\TreeCanopyLidar\PyTLidar\results_lite_rush\rush_07_3_5\rush_07_3_5_cylinders.txt")
# for row in cylinder:
#     start = row[0:3]
#     radius = row[3:4]
#     axis = row[4:7]
#     length = row[7:8]
#     plotter.add_cylinder(start, axis, radius, length)

# # plotter.show()

class QuantitativeMetricsPipeline:
    """
    Need to compare two different datasets results. 
    Main goal:  comparing qsms from our method, with the qsms created by the rush dataset. 
    Its only the volume of each tree. 
    Outline:
    1.Get the mean location of the tree from each dataset. 
    2. Compare the location and match it with one from the rush dataset. 
    3. Compute the volumes of each tree. 


    This will be responbible for QSM metrics for trees, leaf removal metrics, and hopefully 
    segmentation metrics.
    """
    
    def __init__(self):
        pass


    def load_data(self, file_path): # Third Loader
        """
        Generator function which returns a point cloud from the las in a specific folder. 
        """
        for las_file in os.listdir(file_path):
            if not las_file.endswith(".laz") and not las_file.endswith(".las"):
                continue
            xyz, full_data = load_point_cloud(os.path.join(file_path, las_file), full_data=True)
            yield full_data

    def load_data_las(self, file_path): # Second Loader
        """
        Generator which returns a laspy file object from the las in a specific folder. 
        """
        for las_file in os.listdir(file_path):
            if not las_file.endswith(".laz") and not las_file.endswith(".las"):
                continue
            las = laspy.read(os.path.join(file_path, las_file))
            arr_struct = las.points.array
            print("BRUH", arr_struct.dtype.names)
            # BRUH ('X', 'Y', 'Z', 'intensity', 'bit_fields', 'raw_classification', 'scan_angle_rank', 'user_data', 'point_source_id', 'red', 'green', 'blue', 'treeID', 'treeSP')
            full_data = np.vstack((las.x, las.y, las.z,las.intensity, las.raw_classification, las.treeID)).T.astype('float64')
            # full_data = full_data.astype(dtype=np.float64)
            # 12: tree id
            # 13: classification
            yield full_data, las_file

    # def load_data_las(self, file_path):
    #     for las_file in os.listdir(file_path):
    #         if not las_file.endswith((".laz", ".las")):
    #             continue
    #         las = laspy.read(os.path.join(file_path, las_file))
    #         # Only extract x, y, z, intensity
    #         full_data = np.column_stack((las.x, las.y, las.z, las.intensity))
    #         full_data = full_data.astype(dtype=np.float64)
    #         yield full_data


    def compute_qsm_metrics(self):
        """
        This method compares the volumes from each tree. 
        
        Double for loop, looping of each tree in the first one, seeing if it matches the approximate location of the second one, and then comparing volumes. 
        """
        # Compute ours: 
        base_dir = r"G:\Projects\TreeCanopyLidar\PyTLidar\results_lite_rush_tree_saved_treeX"
        our_centroids = []
        our_volumes = []
        our_file_paths = []

        ECOMODEL_TREE_ID = 8
        ECOMODEL_RADIUS = 3
        ECOMODEL_LENGTH = 7
        for folder in os.listdir(f"{base_dir}"):
            for data in os.listdir(f"{base_dir}/{folder}"):
                if "cylinders" in data:
                    raw_data = np.loadtxt(f"{base_dir}/{folder}/{data}")
                    if raw_data.size:
                        # start, axis, radius, length, tree instance. 
                        for instance in range(int(np.max(raw_data[:, ECOMODEL_TREE_ID]))):
                            tree_mask = raw_data[:, ECOMODEL_TREE_ID] == instance
                            tree = raw_data[tree_mask]

                            centroid = np.mean(tree[:, 0:3], axis=0)
                            
                            # This is the volume of all cylinders.
                            volume = np.sum(np.pi * tree[:, ECOMODEL_RADIUS]**2 * tree[:, ECOMODEL_LENGTH])


                            our_centroids.append(centroid)
                            our_volumes.append(volume)
                            our_file_paths.append(os.path.join(base_dir, folder, data))
                            print(f"{folder} {instance} vol: ", volume)

        # Compute theirs. They have individual trees, os no need to loop over anything. 
        base_dir = r"G:\Projects\TreeCanopyLidar\PyTLidar\downloaded_cylmodels"
        their_centroids = []
        their_volumes = []
        their_file_paths = []
        for data in os.listdir(f"{base_dir}"):
            print(data)
            raw_data = np.loadtxt(f"{base_dir}/{data}")
            if raw_data.size:
                # start, axis, radius, length
                centroid = np.mean(raw_data[:, 2:5], axis=0)
                volume = np.sum(np.pi * raw_data[:, 0]**2 * raw_data[:, 1])

                their_centroids.append(centroid)
                their_volumes.append(volume)
                their_file_paths.append(os.path.join(base_dir, data))
                print("Their vol: ", volume)

        print("Report:")
        print(f"Our total counted trees: {len(our_volumes)}")
        print(f"Their total counted trees: {len(their_volumes)}")

        # exit()

        # for our_centroid, our_volume in zip(our_centroids, our_volumes):
        #     for their_centroid, their_volume in zip(their_centroids, their_volumes):
        #         our_xy = our_centroid[:, :2]  # Just x, y
        #         their_xy = np.array([c[:2] for c in their_centroids])

        #         if np.linalg.norm(our_xy - their_xy) < 2:  # horizontal distance only
        #             print("POTENTIAL MATCH")
        #             print(our_centroid)
        #             print(their_centroid)
        #         # exit()
        matched_pairs = []
        for i, our_centroid in enumerate(our_centroids):
            min_dist = float('inf')
            best_match_idx = -1
            
            for j, their_centroid in enumerate(their_centroids):
                dist = np.linalg.norm(our_centroid[:2] - their_centroid[:2])  # horizontal only
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = j
            
            if min_dist < 5:  # more lenient threshold
                matched_pairs.append({
                    'our_idx': i,
                    'their_idx': best_match_idx,
                    'distance': min_dist,
                    'our_volume': our_volumes[i],
                    'their_volume': their_volumes[best_match_idx],
                    'our_file': our_file_paths[i],
                    'their_file': their_file_paths[best_match_idx],
                    'our_centroid': our_centroids[i],
                    'their_centroid': their_centroids[best_match_idx]
                })
                print(f"MATCH: Our tree {i} ↔ Their tree {best_match_idx}, distance={min_dist:.2f}m, vol_diff={abs(our_volumes[i]-their_volumes[best_match_idx]):.2f}")
        
        self.plot_matched_trees(matched_pairs)


    def plot_matched_trees(self, matched_pairs):
        """
        Plot matched trees side by side using pyvista for visual comparison.
        
        Args:
            matched_pairs (list): List of dicts containing match information
        """
        from Utils.plot_tools import SimplePlotter
        
        for match_idx, match in enumerate(matched_pairs):
            print(f"\n========== VISUALIZING MATCH {match_idx + 1} ==========")
            print(f"Our Tree {match['our_idx']} ↔ Their Tree {match['their_idx']}")
            print(f"Horizontal Distance: {match['distance']:.2f}m")
            print(f"Our Volume: {match['our_volume']:.4f} m³")
            print(f"Their Volume: {match['their_volume']:.4f} m³")
            print(f"Volume Difference: {abs(match['our_volume'] - match['their_volume']):.4f} m³")
            
            # Load our tree cylinders
            our_data = np.loadtxt(match['our_file'])
            our_tree_mask = our_data[:, 8] == match['our_idx']  # column 8 is tree instance
            our_cylinders = our_data[our_tree_mask]
            
            # Load their tree cylinders
            their_data = np.loadtxt(match['their_file'])
            
            # Create plotter
            plotter = SimplePlotter(terrain_style=True, parallel_projection=False)
            
            # Add our cylinders (red)
            for cylinder_row in our_cylinders:
                start = cylinder_row[0:3]
                radius = cylinder_row[3]
                axis = cylinder_row[4:7]
                length = cylinder_row[7]
                mesh = pv.Cylinder(
                    center=start + (length/2) * axis,
                    direction=axis,
                    radius=radius,
                    height=length
                )
                plotter.plotter.add_mesh(mesh, color='red', opacity=0.6, label='Our Tree')
            
            # Add their cylinders (blue)
            for cylinder_row in their_data:
                radius = cylinder_row[0]
                length = cylinder_row[1]
                start = cylinder_row[2:5]
                axis = cylinder_row[5:8]
                mesh = pv.Cylinder(
                    center=start + (length/2) * axis,
                    direction=axis,
                    radius=radius,
                    height=length
                )
                plotter.plotter.add_mesh(mesh, color='blue', opacity=0.6, label='Their Tree')
            
            # Add text annotations
            plotter.plotter.add_text(
                f"Match {match_idx + 1}: Our Vol={match['our_volume']:.4f}m³ vs Their Vol={match['their_volume']:.4f}m³",
                position=(10, 10),
                font_size=14,
                color='white'
            )
            
            # Display
            plotter.show()


    def compute_leaf_removal_metrics(self, algorithm = "GB"):
        """
        1. Instantiate leaf removal
        2. load point cloud.
        3. perform my own leaf removal. 
        4. compare to the ground truth. 
        5. Thing is the parameters need tuning for each batch of trees...
        """
        # base_folder = r"G:\Projects\TreeCanopyLidar\Datasets\FORinstance_dataset\raw_tiles"
        base_folder = r"G:\Projects\TreeCanopyLidar\Datasets\FORinstance_dataset\to_test"
        # base_folder = r"G:\Projects\TreeCanopyLidar\Datasets\FORinstance_dataset\single_tree_test"
        CLASSIFICATION_INDEX = 4
        TREE_ID_INDEX = 5

        if algorithm == "GB":
            leaf_remover = GBSeperationWoodLeafClassifier()
        else:
            leaf_remover = RGIWoodLeafClassifier(noise_percentile = 5,
                                                angle_deg=15.0, 
                                                curv_thresh=0.07, 
                                                resid_thresh=0.05, 
                                                k=15,
                                                minClusterSize = 3,
                                                maxClusterSize = 100000,
                                                smoothMode = True,
                                                useResidualTest = True,
                                                useCurvatureTest = True)
        ### These are the best ones for the Leaf Removal.
        # noise_percentile = 5,
        # angle_deg=15.0, 
        # curv_thresh=0.07, 
        # resid_thresh=0.05, 
        # k=15,
        # minClusterSize = 3,
        # maxClusterSize = 100000,
        # smoothMode = True,
        # useResidualTest = True,
        # useCurvatureTest = True


        for full_data, las_file in self.load_data_las(base_folder):

            print(np.unique(full_data[:, TREE_ID_INDEX]))
            for tree_index in np.unique(full_data[:, TREE_ID_INDEX]):
                if tree_index == 0:
                    continue
                tree_mask = full_data[:, TREE_ID_INDEX] == tree_index
                tree_data = full_data[tree_mask]

                ground_truth = tree_data[:, CLASSIFICATION_INDEX]
                ground_truth_og = np.copy(ground_truth)
                # np.savetxt("z_test_before.txt", ground_truth)
                # ground_truth[ground_truth == 4] = 1
                ground_truth[np.isin(ground_truth_og, [4, 6])] = True # Wood 
                ground_truth[np.isin(ground_truth_og, [1, 2, 3, 5])] = False # Leaves

                new_tree = np.concatenate((tree_data[:, :3], ground_truth[:, np.newaxis]), axis=1)

                print(new_tree[:, :3].shape)

                # new_tree = EcomodelFunctions.filter_intensity(new_tree, 20000)
                wood_mask, leaf_mask = leaf_remover.classify(new_tree[:, :4])

                percent_wood_real = np.count_nonzero(ground_truth) / ground_truth.shape[0]
                percent_wood_estimated = np.count_nonzero(wood_mask) / ground_truth.shape[0]

                percent = np.count_nonzero(ground_truth == wood_mask) / ground_truth.shape[0]
                print(f"{las_file}, {tree_index:5}, {percent*100}\n")
                with open("final_data.txt", "a") as f:
                    f.write(f"{las_file}, {tree_index:5}, {percent*100}"+ "\n")
                    # f.write("\n")
                np.savetxt(f"classified_rgi_{las_file}_{tree_index}.xyz", new_tree[wood_mask])         
                # exit()       



    def compute_segmentation_metrics(self):
        """
        Steps to compute accuracy: 
        1. Load each tile
        2. Do segmentation. 
        3. Then compare with the segmented trees in that tile. 
        """
        base_folder = r"G:\Projects\TreeCanopyLidar\Datasets\FORinstance_dataset\raw_tiles"
        for full_data, las_file in self.load_data_las(base_folder):
            pass










if __name__ == "__main__":
    metrics = QuantitativeMetricsPipeline()
    # metrics.compute_qsm_metrics()
    metrics.compute_leaf_removal_metrics(algorithm="RGI")
