"""
Python adaptation and extension of TREEQSM:

a sample test to show the code flow and progress.

Version: 0.0.1
Date: 18 Jan 2025
Authors: Fan Yang, John Hagood, Amir Hossein Alikhah Mishamandani
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

from tools.define_input import define_input
from plotting.visualize_point_cloud import visualize_point_cloud
from plotting.point_cloud_plotting import point_cloud_plotting
from tools.load_point_cloud import load_point_cloud
import numpy as np

if __name__ == "__main__":
    # Specify the path to your LAS file
    # file_path = r'E:\5-Study\OMSCS\CS8903_Research\data\segmented_trees\tree_1.las'
    file_path = r'/Users/johnhagood/Documents/LiDAR/segmented_trees/tree_1.las'
    # Step 1: Read the point cloud from the LAS file
    points = load_point_cloud(file_path)
    if points is not None:
        print(f"Loaded point cloud with {points.shape[0]} points.")

    # Step 2: Visualize the point cloud
    points = points - np.mean(points,axis = 0)
    visualize_point_cloud(points)
    point_cloud_plotting(points, marker_size=5)

    # Step 3: Define inputs for TreeQSM
    inputs = define_input(points, 1, 1, 1)
    print("TreeQSM Inputs:")
    print(inputs)
