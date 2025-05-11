"""
Python adaptation and extension of TREEQSM:

Partition the point cloud into cubic cells.


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

Version: 0.0.1
Date: 9 Feb 2025
Authors: Fan Yang, John Hagood, Amir Hossein Alikhah Mishamandani
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import numpy as np

def cubical_partition(P, EL, NE=3, return_cubes = True):
    """
    Partition the point cloud into cubic cells.

    Parameters:
    P (numpy.ndarray): Point cloud, shape (n_points, 3).
    EL (float): Length of the cube edges.
    NE (int): Number of empty edge layers (default=3).

    Returns:
    tuple: Partition (list of lists of point indices), CubeCoord (n_points x 3 matrix of cube coordinates),
           Info (list containing [Min, N, EL, NE]), and optionally Cubes (3D numpy array).
    """
    # Convert P to a numpy array if not already
    P = np.array(P, dtype=float)

    # The vertices of the bounding box containing P
    Min = np.min(P, axis=0)
    Max = np.max(P, axis=0)

    # Calculate the number of cubes in each direction
    N = np.ceil((Max - Min) / EL).astype(int) + 2 * NE + 1

    # Adjust edge length and re-calculate N if too large
    t = 0
    while t < 10 and 8 * np.prod(N) > 4e9:
        t += 1
        EL *= 1.1
        N = np.ceil((Max - Min) / EL).astype(int) + 2 * NE + 1

    if 8 * np.prod(N) > 4e9:
        NE = 3
        N = np.ceil((Max - Min) / EL).astype(int) + 2 * NE + 1

    #Info = [Min, N, EL, NE]
    # Info: [Min, N, EL, NE] as a 1D array (Min and N are concatenated)
    Info = np.concatenate((Min, N.astype(float), np.array([EL, NE], dtype=float)))

    # Calculate cube coordinates of each point
    CubeCoord = np.floor((P - Min) / EL).astype(int) + NE + 1

    # Lexicographical order for sorting
    LexOrd = (CubeCoord[:, 0]
              + (CubeCoord[:, 1] - 1) * N[0]
              + (CubeCoord[:, 2] - 1) * (N[0] * N[1]))
    SortOrd = np.lexsort((CubeCoord[:, 2], CubeCoord[:, 1], CubeCoord[:, 0]))
    # Sort points by LexOrd
    # SortOrd = np.argsort(LexOrd)
    LexOrd = LexOrd[SortOrd]
    #print(LexOrd)
    #print(SortOrd)
    if return_cubes:
        # Initialize outputs
        Partition = []
        np_points = P.shape[0]

        # Group points into cubes
        p = 0
        while p < np_points:
            t = 1
            while (p + t < np_points) and (LexOrd[p] == LexOrd[p + t]):
                t += 1

            # Collect indices for the current cube
            Partition.append(SortOrd[p:p + t].tolist())
            p += t

        # Optionally create a Cubes array
        Cubes = np.zeros(N, dtype=int)
        for c_idx, points in enumerate(Partition):
            cube_coords = CubeCoord[points[0]]  # Representative point's cube coordinate
            Cubes[cube_coords[0], cube_coords[1], cube_coords[2]] = c_idx + 1  # Non-zero index

        return Partition, CubeCoord, Info, Cubes
    else:
        Partition = np.empty((N[0], N[1], N[2]), dtype=object)

        np_points = P.shape[0]  # number of points
        p = 0  

        while p < np_points:
            t = 1
            while (p + t < np_points) and (LexOrd[p] == LexOrd[p + t]):
                t += 1
            q = SortOrd[p]
            #print(SortOrd[p:p + t])
            # Assign the indices of points in the current cube to the corresponding cell in Partition
            Partition[CubeCoord[q, 0] - 1, CubeCoord[q, 1] - 1, CubeCoord[q, 2] - 1] = SortOrd[p:p + t]
            p += t
        return Partition,CubeCoord,Info
