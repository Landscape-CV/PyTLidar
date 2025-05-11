"""
Python adaptation and extension of TREEQSM:

Calculates the distances between a ray and multiple lines.



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
from tools.mat_vec_subtraction import mat_vec_subtraction


def distances_between_lines(PointRay, DirRay, PointLines, DirLines):
    """
    Calculates the distances between a ray and multiple lines.

    Parameters:
    -----------
    PointRay : array-like, shape (3,)
        A point on the ray.
    DirRay : array-like, shape (3,)
        A unit direction vector of the ray.
    PointLines : array-like, shape (n, 3)
        One point on every line (each row corresponds to a line).
    DirLines : array-like, shape (n, 3)
        Unit direction vectors for the lines (each row corresponds to a line).

    Returns:
    --------
    DistLines : numpy.ndarray, shape (n,)
        The shortest distance between the ray and each line.
    DistOnRay : numpy.ndarray, shape (n,)
        Distance along the ray (from PointRay) to the closest approach to each line.
    DistOnLines : numpy.ndarray, shape (n,)
        Distance along each line (from PointLines) to the closest approach to the ray.
    """
    # Ensure inputs are numpy arrays of type float
    PointRay = np.array(PointRay, dtype=float)
    DirRay = np.array(DirRay, dtype=float)
    PointLines = np.array(PointLines, dtype=float)
    DirLines = np.array(DirLines, dtype=float)

    # Calculate unit vectors N that are orthogonal to both the ray and each line via cross product.
    # For each line, N = DirRay x DirLines[i]
    # When DirLines is (n,3) and DirRay is (3,), we use broadcasting.
    N = np.column_stack((
        DirRay[1] * DirLines[:, 2] - DirRay[2] * DirLines[:, 1],
        DirRay[2] * DirLines[:, 0] - DirRay[0] * DirLines[:, 2],
        DirRay[0] * DirLines[:, 1] - DirRay[1] * DirLines[:, 0]
    ))

    # Normalize N so that each row is a unit vector.
    l = np.linalg.norm(N, axis=1)

    # To avoid division by zero (i.e. when the ray and a line are parallel),
    # you might want to handle that separately. For now, we assume non-parallel.
    N_unit = (N.T / l).T  # Transpose division for broadcasting row-wise

    # Compute A = -(PointRay - PointLines) = PointLines - PointRay
    A = -mat_vec_subtraction(PointLines, PointRay)  # This subtracts PointRay from each row of PointLines

    # Calculate the perpendicular distance (projection of A on N_unit)
    # Use the dot product for each row and take the absolute value
    DistLines = np.sqrt(np.abs(np.sum(A * N_unit, axis=1)))

    # Now, calculate the distances along the ray and lines.
    # Let:
    #   d = A dot DirRay
    #   e = A dot DirLines (each row, so row-wise dot product)
    #   b = DirLines dot DirRay  (each row dot the ray direction)
    b = np.sum(DirLines * DirRay, axis=1)
    d = np.sum(A * DirRay, axis=1)
    e = np.sum(A * DirLines, axis=1)

    # Solve for the scalar parameters along the ray (s) and the line (t)
    # as derived from the perpendicularity conditions:
    #   s = (b*e - d) / (1 - b^2)
    #   t = (e - b*d) / (1 - b^2)
    # Again, we assume 1-b^2 is not zero.
    denom = 1 - b ** 2
    DistOnRay = (b * e - d) / denom
    DistOnLines = (e - b * d) / denom

    return DistLines, DistOnRay, DistOnLines