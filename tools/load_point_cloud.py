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
Authors: Fan Yang, John Hagood, Amir Hossein Alikhah Mishamandani
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import numpy as np
import laspy

def load_point_cloud(file_path, intensity_threshold = 0):
    """
    Load a point cloud from LAS or LAZ files.

    Parameters:
    file_path : str
        Path to the LAS or LAZ file.

    Returns:
    point_cloud : ndarray
        Nx3 matrix of point coordinates (x, y, z).
    """
    with laspy.open(file_path) as las:
        point_data = las.read()
        point_data = point_data[point_data.intensity > intensity_threshold]
        point_cloud = np.vstack((point_data.x, point_data.y, point_data.z)).T.astype('float64')
    return point_cloud