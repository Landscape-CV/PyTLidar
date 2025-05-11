"""
Python adaptation and extension of TREEQSM:

Calculates the distances of points to a line in 3D space.


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
from numba import jit

@jit()
def distances_to_line(Q, LineDirec, LinePoint):
    """
    Calculates the distances of points to a line in 3D space.

    Parameters:
        Q (ndarray): An (n x 3) array of points in 3D space.
        LineDirec (ndarray): A 1x3 unit vector representing the line's direction.
        LinePoint (ndarray): A 1x3 vector representing a point on the line.

    Returns:
        d (ndarray): A (n x 1) array of distances of points to the line.
        V (ndarray): An (n x 3) array of perpendicular vectors from the line to the points.
        h (ndarray): An (n x 1) array of projections of the vectors onto the line.
        B (ndarray): An (n x 3) array of the projections along the line direction.
    """
    # Calculate vectors from LinePoint to points in Q
    A = Q - LinePoint
    LineDirec = LineDirec.astype(np.float64)
    # Project A onto the line direction
    h = np.dot(A, LineDirec)

    # Calculate projections along the line
    B = np.outer(h, LineDirec)

    # Calculate perpendicular vectors
    V = A - B

    # Calculate distances
    d = np.sqrt(np.sum(V**2,axis=1))
    # d = np.linalg.norm(V, axis=1)

    return d, V, h, B