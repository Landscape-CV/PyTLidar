"""
Python adaptation and extension of TREEQSM:

Downsamples the given point cloud by averaging points from each cube of side length `cube_size`.


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
Authors: Fan Yang, John Hagood, Amir Hossein Alikhah Mishamandani
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import numpy as np

def cubical_averaging(points, cube_size):
    """
    Downsamples the given point cloud by averaging points from each cube of side length `cube_size`.

    Parameters:
        points (numpy.ndarray): An (n x 3) array representing the point cloud.
        cube_size (float): The edge length of each cube.

    Returns:
        numpy.ndarray: The downsampled point cloud.
    """
    # Convert points to numpy array if not already
    points = np.array(points, dtype=float)

    # Find the min and max coordinates
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # Calculate the number of cubes in each dimension
    n_cubes = np.ceil((max_coords - min_coords) / cube_size).astype(int) + 1

    # Assign each point to a cube
    cube_coords = np.floor((points - min_coords) / cube_size).astype(int)

    # Compute lexicographical order for cubes
    lex_order = (
        cube_coords[:, 0] +
        cube_coords[:, 1] * n_cubes[0] +
        cube_coords[:, 2] * n_cubes[0] * n_cubes[1]
    )

    # Sort points by lexicographical order
    sort_order = np.argsort(lex_order)
    sorted_points = points[sort_order]
    sorted_lex_order = lex_order[sort_order]

    # Initialize the downsampled point cloud
    downsampled_points = []

    # Group points by cube and calculate their average
    i = 0
    n_points = len(points)
    while i < n_points:
        # Find the range of points in the current cube
        j = i + 1
        while j < n_points and sorted_lex_order[j] == sorted_lex_order[i]:
            j += 1

        # Compute the average of the points in this cube
        cube_average = np.mean(sorted_points[i:j], axis=0)
        downsampled_points.append(cube_average)

        # Move to the next group of points
        i = j

    # Convert the list to a numpy array
    downsampled_points = np.array(downsampled_points)

    return downsampled_points