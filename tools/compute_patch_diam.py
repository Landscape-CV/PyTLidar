"""
Python adaptation and extension of TREEQSM:

Compute a range of PatchDiam values based on a given center value and count.


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


def compute_patch_diam(pd, n):
    """
    Compute a range of PatchDiam values based on a given center value and count.

    Parameters:
    pd : float
        Center value of PatchDiam.
    n : int
        Number of PatchDiam values to compute.

    Returns:
    patch_diam : ndarray
        Array of PatchDiam values.
    """
    if n == 1:
        return np.array([pd])
    return np.linspace((0.90 - (n - 2) * 0.1) * pd, (1.10 + (n - 2) * 0.1) * pd, n)