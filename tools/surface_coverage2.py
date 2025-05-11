"""
Python adaptation and extension of TREEQSM:

Compute surface coverage (fraction of covered cells on a cylindrical surface).


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
from tools.orthonormal_vectors import orthonormal_vectors


def surface_coverage2(axis, length, vec, height, nl, ns):
    """
    Compute surface coverage (fraction of covered cells on a cylindrical surface).

    Parameters:
        axis (array-like): Axis vector of the cylinder.
        length (float): Length of the cylinder.
        vec (np.ndarray): Vectors connecting points to the axis (n x 3).
        height (np.ndarray): Heights of points from the base of the cylinder (n x 1).
        nl (int): Number of layers along the cylinder height.
        ns (int): Number of angular segments around the cylinder.

    Returns:
        float: Surface coverage (value between 0 and 1).
    """
    # Compute orthonormal basis
    u, w = orthonormal_vectors(axis)

    # Project vectors into the cylinder's local 2D plane
    vec = vec @ np.array([u, w]).T

    # Compute angular coordinates
    ang = np.arctan2(vec[:, 1], vec[:, 0]) + np.pi

    # Map points to layer indices
    i = np.ceil(height / length * nl).astype(int)
    i = np.clip(i, 1, nl)

    # Map points to angular segment indices
    j = np.ceil(ang / (2 * np.pi) * ns).astype(int)
    j = np.clip(j, 1, ns)

    # Compute unique cell indices
    k = (i - 1) + (j - 1) * nl
    unique_k = np.unique(k)

    # Compute surface coverage
    surf_cov = len(unique_k) / (nl * ns)
    return surf_cov