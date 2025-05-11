"""
Python adaptation and extension of TREEQSM:

Downsamples the given point cloud by selecting one point from each
cube of side length "cube_size".

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

def cubical_downsampling(points, cube_size):
    """
    Downsamples the given point cloud by selecting one point from each
    cube of side length "cube_size".

    Parameters:
    points (numpy.ndarray): n x 3 array of 3D points.
    cube_size (float): The size of each cube for downsampling.

    Returns:
    numpy.ndarray: A boolean array indicating which points are retained.
    """
    # Ensure the input is a numpy array
    points = np.asarray(points, dtype=float)

    # Calculate the bounding box of the points
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # Number of cubes along each dimension
    n_cubes = np.ceil((max_coords - min_coords) / cube_size).astype(int) + 1

    # Number of points and block size
    num_points = points.shape[0]
    block_size = int(1e7)
    if num_points < block_size:
        block_size = num_points

    # Number of blocks
    num_blocks = int(np.ceil(num_points / block_size))

    # Initialize storage for selected points
    retained_indices = []

    # Process points in blocks
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_points)

        # Points in the current block
        block_points = points[start_idx:end_idx]

        # Compute the cube coordinates for the points
        cube_coords = np.floor((block_points - min_coords) / cube_size).astype(int) + 1

        # Compute unique cube indices in lexicographical order
        lex_order = (cube_coords[:, 0]
                     + (cube_coords[:, 1] - 1) * n_cubes[0]
                     + (cube_coords[:, 2] - 1) * n_cubes[0] * n_cubes[1])

        # Get unique cubes and their corresponding indices
        unique_lex, unique_indices = np.unique(lex_order, return_index=True)

        # Map the indices back to the original point indices
        retained_indices.append(start_idx + unique_indices)

    # Flatten retained indices
    retained_indices = np.concatenate(retained_indices)

    # Create a boolean mask to mark retained points
    mask = np.zeros(num_points, dtype=bool)
    mask[retained_indices] = True

    return mask