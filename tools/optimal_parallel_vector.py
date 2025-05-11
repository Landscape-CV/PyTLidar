"""
Python adaptation and extension of TREEQSM:

Compute the optimal parallel vector using Singular Value Decomposition (SVD).


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


def optimal_parallel_vector(V ):
    """
    Compute the optimal parallel vector using Singular Value Decomposition (SVD).

    Parameters:
    V : ndarray
        Nx3 matrix of vectors.

    Returns:
    axis_dir : ndarray
        Optimal parallel vector (principal axis).
    """
    _, _, vh = np.linalg.svd(V, full_matrices=False)


    return vh[0]