"""
Python adaptation and extension of TREEQSM:

Load a point cloud from LAS or LAZ files.


% -----------------------------------------------------------
% This file is part of TREEQSM.
%
% TREEQSM is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% TREEQSM is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with TREEQSM.  If not, see <http://www.gnu.org/licenses/>.
% -----------------------------------------------------------


Version: 0.0.1
Date: 18 Jan 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import open3d as o3d


def visualize_point_cloud(points):
    """
    Visualizes the point cloud using Open3D.

    Parameters:
    points : ndarray
        Nx3 NumPy array of point coordinates (x, y, z).
    """
    # Create Open3D PointCloud object
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pc], window_name="Point Cloud of Tree")
