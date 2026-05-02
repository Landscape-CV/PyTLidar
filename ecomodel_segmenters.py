"""
This module includes a class which streamlines the current ecomodel.
"""


import numpy as np
from Utils.Utils import load_point_cloud
from pathlib import Path
from Utils.plot_tools import SimplePlotter
import open3d as o3d
import networkx as nx
import cc3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import pyvista as pv
from circle_fit import taubinSVD
from pathlib import Path

# John
import time
from TreeQSMSteps.cover_sets import cover_sets
from Utils.TreeSegmentation import segment_point_cloud
from ecomodel import Tile


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

    def process(self, point_cloud, layer_height=0.1, radius=0.2):
        """
        Builds layers


        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz = np.concatenate((raw_points_layer, labels), axis=1)
        """

        min_value = min(point_cloud[:, 2])
        layer_min = min_value
        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_classification = np.concatenate((point_cloud_xyz, np.full((point_cloud_xyz.shape[0], 1), -1)),
                                                    axis=1)
        point_cloud_segment_classification = np.concatenate(
            (point_cloud_xyz, np.full((point_cloud_xyz.shape[0], 1), -1)), axis=1)

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

                if layer_num == 3:
                    # branch_segment_mask = point_cloud_classification[:, 3] == label
                    # layer_segment_mask = layer_pc_mask#  & branch_segment_mask
                    # print("Layer segment mask")
                    # point_cloud_classification[layer_segment_mask,  3] = segment_id
                    pass
                    # self.view_point_cloud(point_cloud_segment_classification[layer_pc_mask])
                branch_segment_mask = point_cloud_segment_classification[:, 3] == label
                layer_segment_mask = layer_pc_mask & branch_segment_mask

                if layer_num == 0:
                    point_cloud_classification[layer_segment_mask, 3] = segment_id
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
                        new_branch.top_centroid = np.mean(point_cloud_segment_classification[layer_segment_mask],
                                                          axis=0)
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

        print(np.unique(point_cloud_classification[:, 3]))

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


class SegmenterScanline:
    """
    Returns labels for each thing.

    point_cloud --> labels.
    """

    def __init__(self):
        pass

    def process_working(self, point_cloud, intensity_threshold=0):
        """
        Segments the point cloud into groups from a single tree
        Parameters:
                min_points (int): Minimum number of points in a cluster.
        Returns:
                numpy.ndarray: Clustered point cloud, shape (n_points, 3).
        """

        # inputs = {'PatchDiam1': 0.08, 'BallRad1':.08, 'nmin1': 15}
        inputs = {'PatchDiam1': 0.15, 'BallRad1': .15, 'nmin1': 25}
        # inputs = {'PatchDiam1': 0.1, 'BallRad1':.125, 'nmin1': 5}

        cover_set_adjust = 0
        for x in range(1):
            tile = Tile(point_cloud[:, :3], point_cloud)
            # logger.info("Segmenting tile: %s",i)

            # tile.to_xyz(f"{self.results_folder}/pre_segmentation.xyz")
            intensity_mask = tile.point_data[:, 3] > intensity_threshold
            tile.point_data = tile.point_data[intensity_mask]
            tile.cloud = tile.cloud[intensity_mask]
            # tile.cover_sets = tile.cover_sets[intensity_mask]
            # tile.segment_labels = tile.segment_labels[intensity_mask]
            # tile.cluster_labels = tile.cluster_labels[intensity_mask]

            # logger.info("Create Cover Sets")
            start = time.time()
            cover = cover_sets(tile.get_cloud_as_array(), inputs, qsm=False, device='cpu',
                               full_point_data=tile.point_data)
            if len(cover['sets']) == 0:
                # logger.info("No cover sets found")
                continue

            labels = cover['sets']

            noise_mask = labels > -1
            tile.cloud = tile.cloud[noise_mask]
            tile.point_data = tile.point_data[noise_mask]
            labels = labels[noise_mask]
            tile.cover_sets = labels
            I = np.argsort(labels)
            cover_set_adjust = len(tile.cover_sets)

            if len(labels) == 0:
                # logger.info("No cover sets found")
                continue
            # logger.info("Time to create cover sets: %s",time.time()-start)

            # logger.info("Segment Cloud")
            start = time.time()
            # settings for Missouri data
            # segment_point_cloud(tile,base_height=.75, connect_ambiguous_points=True, fix_overlapping_segments=False,base_dist_multiplier=1.2,max_dist=.17,combine_nearby_bases=False,initial_size_limit=100000,min_height =.1)

            # Leaf Removal Step
            # intensity_threshold = 42000
            # intensity_mask = tile.point_data[:, 3] > intensity_threshold
            # tile.cloud = tile.cloud[intensity_mask]
            # tile.point_data = tile.point_data[intensity_mask]
            # tile.cover_sets = tile.cover_sets[intensity_mask]
            # wood_mask, leaf_mask = self.remove_leaves_rgi(tile.point_data)
            # tile.cloud = tile.cloud[wood_mask]
            # tile.point_data = tile.point_data[wood_mask]
            # tile.cover_sets = tile.cover_sets[wood_mask]
            # self.save_point_cloud("leaves_removed", tile.point_data)

            # tile.to_xyz(f"{self.results_folder}/pre_segmentation.xyz")

            # Default parameters for the tree instance segmentation.
            default_arguments = {
                "max_dist": 0.16,
                "min_height": .3,
                "connect_using_midpoint": False,
                "base_height": .65,
                "base_dist_multiplier": 2.5,
                "connect_ambiguous_points": True,
                "fix_overlapping_segments": False,
                "layer_size": .16,
                "min_Z": float(np.min(tile.cloud[:, 2])),
                "combine_nearby_bases": True,
            }

            # New tuned parameters.
            tuned_arguments = {
                "max_dist": 0.3,
                "base_height": 1,
                "layer_size": 0.15,
                "combine_nearby_bases": False,
            }

            default_arguments.update(tuned_arguments)
            segment_point_cloud(tile, **default_arguments)
            # tile.to_xyz(f"{self.results_folder}/post_segmentation.xyz", True)
            mask = tile.segment_labels > -2  # filters out points that could not be connected, ideal will segment better and this will be uneccesary
            print("UNIQUE LABELS", np.unique(tile.segment_labels))
            # for label in np.unique(tile.segment_labels):
            #     # if label < 0:
            #     #     continue
            #     save_mask = tile.segment_labels == label
            #     self.save_point_cloud(f"segment_label_{label}", tile.cloud[save_mask])

            tile.cloud = tile.cloud[mask]
            tile.point_data = tile.point_data[mask]
            tile.segment_labels = tile.segment_labels[mask]
            tile.cover_sets = tile.cover_sets[mask]
            try:
                tile.original_data = tile.original_data[intensity_mask.cpu().numpy()][noise_mask][I][mask]
            except:
                pass
            # logger.info("Time to segment cloud: %s",time.time()-start)

            # tile.cluster_labels = labels
            # args_formatted = str(tuned_arguments).replace(" ", "_").replace(",", "_").replace(":","_").replace("'", "")
            # out_path = os.path.join(self.results_folder, f"data_{args_formatted}.xyz")
            # tile.to_xyz(out_path, True)
            # logger.info(f"Clustered tile written to {out_path}")

        # self.save_point_cloud(f"data_{arguments}", )
        # logger.info("Tree segmentation finished.")

        point_cloud = tile.cloud
        labels = tile.segment_labels

        return point_cloud, labels

    def process(self, point_cloud, intensity_threshold=0):
        """
        Segments the point cloud into groups from a single tree
        Parameters:
                min_points (int): Minimum number of points in a cluster.
        Returns:
                numpy.ndarray: Clustered point cloud, shape (n_points, 3).
        """

        inputs = {'PatchDiam1': 0.15, 'BallRad1': .15, 'nmin1': 25}
        tile = Tile(point_cloud[:, :3], point_cloud)

        cover = cover_sets(tile.get_cloud_as_array(), inputs, qsm=False, device='cpu', full_point_data=tile.point_data)
        if len(cover['sets']) == 0:
            return None, None

        labels = cover['sets']

        noise_mask = labels > -1
        tile.cloud = tile.cloud[noise_mask]
        tile.point_data = tile.point_data[noise_mask]
        labels = labels[noise_mask]
        tile.cover_sets = labels

        if len(labels) == 0:
            return None, None

        # Default parameters for the tree instance segmentation.
        default_arguments = {
            "max_dist": 0.16,
            "min_height": .3,
            "connect_using_midpoint": False,
            "base_height": .65,
            "base_dist_multiplier": 2.5,
            "connect_ambiguous_points": True,
            "fix_overlapping_segments": False,
            "layer_size": .16,
            "min_Z": float(np.min(tile.cloud[:, 2])),
            "combine_nearby_bases": True,
        }

        # New tuned parameters.
        tuned_arguments = {
            "max_dist": 0.3,
            "base_height": 1,
            "layer_size": 0.15,
            "combine_nearby_bases": False,
        }

        default_arguments.update(tuned_arguments)
        segment_point_cloud(tile, **default_arguments)
        mask = tile.segment_labels > -2  # filters out points that could not be connected, ideal will segment better and this will be uneccesary
        print("UNIQUE LABELS", np.unique(tile.segment_labels))

        point_cloud = tile.cloud[mask]
        labels = tile.segment_labels[mask]

        return point_cloud, labels

    def old(self, point_cloud, intensity_threshold=0):
        tile = Tile(point_cloud[:, :3], point_cloud)
        inputs = {'PatchDiam1': 0.15, 'BallRad1': .15, 'nmin1': 25}
        cover = cover_sets(point_cloud[:, :3], inputs, qsm=False, full_point_data=point_cloud)
        if len(cover['sets']) == 0:
            return None, None
        cover_set_labels = cover['sets']

        # Remove noise.
        noise_mask = cover_set_labels > -1
        point_cloud = point_cloud[noise_mask]
        cover_set_labels = cover_set_labels[noise_mask]

        if len(cover_set_labels) == 0:
            return None, None

        tile.point_data = point_cloud
        tile.cloud = point_cloud[:, :3]
        tile.cover_sets = cover_set_labels

        default_arguments = {
            "max_dist": 0.16,
            "min_height": .3,
            "connect_using_midpoint": False,
            "base_height": .65,
            "base_dist_multiplier": 2.5,
            "connect_ambiguous_points": True,
            "fix_overlapping_segments": False,
            "layer_size": .16,
            "min_Z": float(np.min(point_cloud[:, 2])),
            "combine_nearby_bases": True,
        }

        # New tuned parameters.
        tuned_arguments = {
            "max_dist": 0.3,
            "base_height": 1,
            "layer_size": 0.15,
            "combine_nearby_bases": False,
        }

        default_arguments.update(tuned_arguments)
        segment_point_cloud(tile, **default_arguments)
        mask = tile.segment_labels > -2  # filters out points that could not be connected, ideal will segment better and this will be uneccesary
        print("UNIQUE LABELS", np.unique(tile.segment_labels))

        point_cloud = point_cloud[mask]
        labels = tile.segment_labels[mask]

        return point_cloud, labels

        # print(point_cloud[point_mask].shape)


class Segmenter2():
    def __init__(self):
        super().__init__()
        self.occupied_voxels = []
        self.cylinder_data_list = []
        self.occupied_nodes = []

    def create_skeleton_and_cylinders(self, point_cloud):
        """
        Creates a Skeleton Graph in preperation for retreiving cylinder information from the segment.
        """
        point_cloud = point_cloud[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # Further clean the point cloud
        # pcd, ind = pcd.remove_statistical_outlier(
        #     nb_neighbors=20,
        #     std_ratio=0.5
        # )
        # o3d.visualization.draw_geometries([pcd])

        if point_cloud.shape[0] > 50000:
            init_contraction = 80
            max_contraction = 2048
        elif point_cloud.shape[0] < 10:
            return None
        else:
            init_contraction = 10
            max_contraction = 2048

        lbc = LBC(point_cloud=pcd,
                  down_sample=0.008, init_contraction=init_contraction, max_contraction=max_contraction)

        try:
            lbc.extract_skeleton()
        except:
            return None
        # points_np = np.asarray(lbc.contracted_point_cloud)   # Shape: (N, 3)
        branch_graph, branch_skeleton_graph = self.get_graph(point_cloud, lbc.contracted_point_cloud)

        if branch_graph is None:
            return None

        return self.get_cylinder_from_branch_segment(point_cloud, branch_skeleton_graph, branch_graph)

    def clean_graph(self, graph):
        """
        Cleans the graph by doing some operations on it.
        """
        # Combine nodes that are in a straight line.
        # print("Before", graph.number_of_nodes())

        # Remove nodes with a single edge.
        # for n in list(graph.nodes()):
        #     if graph.degree(n) == 1:
        #         graph.remove_node(n)

        # Remove nodes with 2 children (branch bifurcation)
        for n in list(graph.nodes()):
            if graph.degree(n) > 2:
                graph.remove_node(n)

        # Remove nodes with 0 children (leftover points)
        for n in list(graph.nodes()):
            if graph.degree(n) == 0:
                graph.remove_node(n)
        # print("After", graph.number_of_nodes())

        return graph

    def get_graph(self, other_pc, skeleton_pc):

        skeleton_pc = skeleton_pc.voxel_down_sample(0.15)

        numpy_points = np.asarray(skeleton_pc.points)
        points = numpy_points

        if points.shape[0] < 10:
            return None, None

        # self.view_point_cloud(numpy_points)

        k = 6  # typical for skeletons

        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, indices = nbrs.kneighbors(points)

        G = nx.Graph()

        # add nodes
        for i in range(len(points)):
            G.add_node(i)

        # add weighted edges
        for i in range(len(points)):
            for j, d in zip(indices[i], distances[i]):
                if i != j:
                    G.add_edge(i, j, weight=d)

        T = nx.minimum_spanning_tree(G, weight='weight')

        T = self.clean_graph(T)

        return T, skeleton_pc

        # Visualize

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(slice_pos)

        # # convert edges to LineSet
        # edges = np.array([[u, v] for u, v in T.edges()])
        # lines = o3d.geometry.LineSet(
        #     points=o3d.utility.Vector3dVector(points),
        #     lines=o3d.utility.Vector2iVector(edges)
        # )

        # o3d.visualization.draw_geometries([pcd, lines, pcd2])

        # self.simple_plotter.plotter.add_points(points, color='blue', point_size=5)

        # # add edges
        # for u, v in G.edges():
        #     line = np.vstack([points[u], points[v]])
        #     self.simple_plotter.plotter.add_lines(line, color='black', width=2)

        # self.simple_plotter.plotter.show()

    def get_points_within_line(self, pt1, pt2, point_cloud, T):

        # Side 1
        normal_vector = (pt1 - pt2) / np.linalg.norm(pt1 - pt2)

        d = np.dot(point_cloud - pt2, normal_vector)

        # slice: keep points on positive side
        slice_pos = point_cloud[d > 0]

        # # Side 2:
        normal_vector = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)

        d = np.dot(slice_pos - pt1, normal_vector)

        # slice: keep points on positive side
        line_points = slice_pos[d > 0]

        # Keep points within a certian radius of the two other points.

        # Get cylinder information while we are at it.
        # Project points onto plane
        # fitter = RobustCylinderFitter()
        # on that plane, get the max distance between two points.
        normal_vector = (pt1 - pt2) / np.linalg.norm(pt1 - pt2)

        pc1, pc2, pc3 = self.orthonormal_basis_from_vector(normal_vector)

        projection1 = np.dot(line_points, pc2)
        projection2 = np.dot(line_points, pc3)
        circle_projection = np.column_stack([projection1, projection2])
        # self.circle_projection = np.transpose(circle_projection)

        # plt.scatter(circle_projection[:, 0], circle_projection[:, 1])
        # plt.axis('equal')
        # plt.show()

        db = DBSCAN(eps=0.1, min_samples=5).fit(circle_projection)
        labels = db.labels_

        # Pick the label that is closes to 0,0
        min_distance = 1000
        for label in labels:
            label_mask = labels == label
            test_point = circle_projection[label_mask][0]
            distance = np.linalg.norm(test_point - np.array((0, 0)))
            if distance < min_distance:
                candidate_label = label

            continue

        label_mask = labels == candidate_label
        cluster_circle = circle_projection[label_mask]

        ########### Attempt 1 Robust Cylinder Fitter
        # fitter = RobustCylinderFitter()
        # self.circle_projection = np.transpose(circle_projection)
        # circle_params = fitter._WRLTS(self.circle_projection) # For Me WRLTS had better results on tree branches.
        # x, y, r, s = circle_params
        # l = np.linalg.norm(pt1 - pt2)

        # center = fitter.get_cylinder_center(pc2, pc3, x, y)
        # axis = fitter.get_cylinder_axis(pc1)
        # start = fitter.get_cylinder_start(pc2, pc3, center, axis, l)

        # print("max distance",max_distance)

        ########### Attempt 2 Max distance
        # max_distance = np.max(pdist(cluster_circle))
        # start = pt2
        # axis = normal_vector
        # r = max_distance/2
        # l = np.linalg.norm(pt1 - pt2)

        # self.simple_plotter.add_cylinder(start, axis, r, l)

        ########### Attempt 3 better circle fitter.

        # plt.scatter(cluster_circle[:, 0], cluster_circle[:, 1])
        # plt.axis('equal')
        # plt.show()

        # Define your points

        # Fit the circle
        try:
            xc, yc, r, sigma = taubinSVD(cluster_circle)
        except:
            return

        print(f"Center: ({xc}, {yc}), Radius: {r}, Residual Error: {sigma}")

        start = pt2
        axis = normal_vector
        # r = max_distance/2
        l = np.linalg.norm(pt1 - pt2)

        self.simple_plotter.add_cylinder(start, axis, r, l)
        # [cylinder start x, start y, start z, radius, axis x, axis y, axis z, length]
        self.cylinder_data_list.append([start[0], start[1], start[2], r, axis[0], axis[1], axis[2], l])

    def orthonormal_basis_from_vector(self, v):
        v = v / np.linalg.norm(v)

        # pick a helper vector not parallel to v
        if abs(v[0]) < 0.9:
            tmp = np.array([1.0, 0.0, 0.0])
        else:
            tmp = np.array([0.0, 1.0, 0.0])

        u = np.cross(v, tmp)
        u /= np.linalg.norm(u)

        w = np.cross(v, u)
        w /= np.linalg.norm(w)

        return v, u, w

    def get_cylinder_from_branch_segment(self, branch_points, branch_skeleton_points, branch_graph):
        """
        Returns the cylinder information given the branch segment.
        """
        # points = T.nodes(data=True)
        skeleton_points_numpy = np.asarray(branch_skeleton_points.points)
        edges = np.array([[u, v] for u, v in branch_graph.edges()])
        lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(skeleton_points_numpy),
            lines=o3d.utility.Vector2iVector(edges)
        )

        points = np.asarray(lines.points)  # shape (N, 3)
        lines = np.asarray(lines.lines)  # shape (M, 2)

        # For each line process the points to get the cylinders
        for u, v in lines:
            p1 = points[u]
            p2 = points[v]
            print("edge:", u, v, "coords:", p1, p2)

            # Plot lines

            self.get_points_within_line(p1, p2, branch_points, branch_graph)
            line = pv.Line(p1, p2)
            self.simple_plotter.plotter.add_mesh(line, color="blue", line_width=2)

            # self.get_cylinder_from_branch_segment(line_points)

            # break
            # # We slice the space to grab the points
            # from scipy.spatial import cKDTree

            # tree = cKDTree(points)

        # Prune the cylinders to remove the outliers.
        cylinder_data_array = np.array(self.cylinder_data_list)

        arr = cylinder_data_array[:, 3]
        z = np.abs((arr - arr.mean()) / arr.std())
        clean_mask = arr < 0.15  # keep values less than 15 cm

        cleaned_cylinders = cylinder_data_array[clean_mask]

        # self.simple_plotter.add_point_cloud(branch_points)

        return cleaned_cylinders

        # Then we convert the points

    def process(self, point_cloud):
        """"""
        data_xyz = point_cloud[:, :3]
        self.save_point_cloud("Debug.txt", data_xyz)

        # data_xyz = data_xyz - np.mean(data_xyz, axis=1)[:,np.newaxis]
        # self.view_point_cloud(point_cloud)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(data_xyz)
        # octree = o3d.geometry.Octree(max_depth=6)  # tweak depth as needed
        # octree.convert_from_point_cloud(pcd, max_depth=6)
        # # o3d.visualization.draw_geometries([pcd])        # raw point cloud
        # # o3d.visualization.draw_geometries([octree])     # octree cells
        # octree.traverse(self.traverse_callback)
        # self.octree = octree

        # voxels = voxel_grid.get_voxels()  # returns list of voxels
        # indices = np.stack(list(vx.grid_index for vx in voxels))
        # colors = np.stack(list(vx.color for vx in voxels))

        voxel_grid, vg = self.create_voxel_grid(0.1)

        voxel_labels = self.classify_connected_components(voxel_grid)
        # classified_point_cloud = self.restore_point_cloud(data_xyz, voxel_labels)
        classified_point_cloud = self.restore_point_cloud(point_cloud, voxel_labels, vg, 0.1)
        # self.simple_plotter.add_point_cloud(classified_point_cloud)
        self.view_point_cloud(classified_point_cloud)
        # Then we turn those voxels back into the worldspace.

    def traverse_callback(self, node, node_info):
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            self.occupied_nodes.append(node)
            self.occupied_voxels.append(node_info)

    def create_voxel_grid(self, voxel_size):
        """
        Creates a uniform voxel grid (binary occupancy) from the point cloud.
        """
        vg = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, voxel_size)

        # Build a dense 3D numpy grid
        min_corner = vg.origin
        max_corner = vg.origin + vg.get_max_bound() - vg.get_min_bound()

        grid_shape = np.ceil((max_corner - min_corner) / voxel_size).astype(int)
        voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

        for voxel in vg.get_voxels():
            idx = voxel.grid_index  # (i, j, k)
            voxel_grid[idx[0], idx[1], idx[2]] = 1

        return voxel_grid, vg

    def restore_point_cloud(self, point_cloud, voxel_grid, vg, voxel_size):
        """
        Assigns connected-component labels back to the original points.
        """
        classified = np.concatenate(
            (point_cloud, np.zeros((point_cloud.shape[0], 1))), axis=1
        )

        min_corner = vg.origin

        # For each point, compute its voxel index
        idxs = ((point_cloud - min_corner) / voxel_size).astype(int)

        # Clamp indices to grid bounds
        idxs = np.clip(idxs, [0, 0, 0], np.array(voxel_grid.shape) - 1)

        # Assign labels
        classified[:, 3] = voxel_grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]]

        return classified

    # def create_voxel_grid(self, point_cloud):
    #     """
    #     Converts a list of voxels (from octree) into a 3d matrix of 1s and 0s.
    #     """

    #     # points = np.zeros((len(self.occupied_voxels), 3))
    #     # for index, voxel in enumerate(self.occupied_voxels):
    #     #     points[index, :] = voxel.origin

    #     # print("Total num voxels", len(self.occupied_voxels))

    #     # voxel_size = self.occupied_voxels[0].size
    #     # min_corner = self.octree.origin
    #     # max_corner = self.octree.origin + self.octree.size

    #     # grid_shape = np.ceil((max_corner - min_corner) / voxel_size).astype(int)
    #     # voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(point_cloud)

    #     voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 0.10)
    #     voxels = voxel_grid.get_voxels()  # returns list of voxels
    #     indices = np.stack(list(vx.grid_index for vx in voxels))
    #     print(indices)
    #     exit()

    #     numpy_array = np.zeros(())

    #     for idx in indices:
    #         voxel_grid[idx[0], idx[1], idx[2]] = 1

    #     return voxel_grid

    def classify_connected_components(self, voxel_grid):
        labels_out = cc3d.connected_components(voxel_grid, connectivity=26)  # 26-connected
        print(np.unique(labels_out))
        return labels_out

    # def restore_point_cloud(self, point_cloud, voxel_grid):
    #     """
    #     Returns a classified point cloud.
    #     """
    #     # self.view_point_cloud(point_cloud)
    #     classified = np.concatenate((point_cloud, np.zeros((point_cloud.shape[0], 1))), axis=1)

    #     min_corner = self.octree.origin

    #     for node, info in zip(self.occupied_nodes, self.occupied_voxels):
    #         idx = ((info.origin - min_corner) / info.size).astype(int)
    #         # print(idx)
    #         classification = voxel_grid[idx[0], idx[1], idx[2]]

    #         min_point = info.origin
    #         max_point = info.origin + info.size

    #         mask = np.all((classified[:,:3] >= min_point) & (classified[:,:3] <= max_point), axis=1)
    #         # np.savetxt("mask.txt", mask)

    #         # print(min_point, max_point, classification, mask.shape)
    #         # print(mask.shape)
    #         # print(classification)
    #         classified[mask, 3] = classification
    #         # print("origin", info.origin)
    #         # print("size", voxel_size)
    #         # print(node.indices)

    #     np.savetxt("classified.txt",classified )

    #     print(classified.shape)

    #     return classified

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # # voxel_grid is a 3D numpy array of 0/1
    # ax.voxels(voxel_grid.astype(bool), facecolors='blue', edgecolor='k')

    # plt.show()

    # voxels = []
    # for info in self.occupied_voxels:
    #     center = info.origin + info.size * 0.5
    #     voxels.append(o3d.geometry.Voxel(center, voxel_size))

    # vg = o3d.geometry.VoxelGrid()
    # vg.voxels = o3d.utility.Vector3iVector(voxels)
    # vg.voxel_size = voxel_size

    # o3d.visualization.draw_geometries([vg])

    # early_stop = False

    # if isinstance(node, o3d.geometry.OctreeInternalNode):
    #     if isinstance(node, o3d.geometry.OctreeInternalPointNode):
    #         n = 0
    #         for child in node.children:
    #             if child is not None:
    #                 n += 1
    #         print(
    #             "{}{}: Internal node at depth {} has {} children and {} points ({})"
    #             .format('    ' * node_info.depth,
    #                     node_info.child_index, node_info.depth, n,
    #                     len(node.indices), node_info.origin))

    #         # we only want to process nodes / spatial regions with enough points
    #         early_stop = len(node.indices) < 250
    # elif isinstance(node, o3d.geometry.OctreeLeafNode):
    #     if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
    #         print("{}{}: Leaf node at depth {} has {} points with origin {}".
    #             format('    ' * node_info.depth, node_info.child_index,
    #                     node_info.depth, len(node.indices), node_info.origin))
    # else:
    #     raise NotImplementedError('Node type not recognized!')

    # # early stopping: if True, traversal of children of the current node will be skipped
    # return early_stop

    # octree.traverse(collect)


if __name__ == "__main__":
    classifier = DistanceBasedNoiseRemoval()

    path = Path(r"G:\Projects\TreeCanopyLidar\Datasets\2025_1cm\tile_00005_2_0_leaves_removed.xyz")
    folder_path = path.parent
    basename = path.stem
    tile_data, full_data = load_point_cloud(str(path), full_data=True)

    classifier.remove_distant_small_clusters(full_data)
