"""
Python adaptation and extension of TREEQSM:

Adjust the precision of each element in the vector v based on its magnitude.


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


def change_precision(v):
    """
    Adjust the precision of each element in the vector v based on its magnitude.

    Parameters:
        v (list or np.ndarray): Input vector.

    Returns:
        np.ndarray: Vector with adjusted precision.
    """
    v = np.array(v, dtype=float)  # Ensure input is a NumPy array
    n = len(v)

    for i in range(n):
        if abs(v[i]) >= 1e3:
            v[i] = round(v[i])
        elif abs(v[i]) >= 1e2:
            v[i] = round(10 * v[i]) / 10
        elif abs(v[i]) >= 1e1:
            v[i] = round(100 * v[i]) / 100
        elif abs(v[i]) >= 1e0:
            v[i] = round(1000 * v[i]) / 1000
        elif abs(v[i]) >= 1e-1:
            v[i] = round(10000 * v[i]) / 10000
        else:
            v[i] = round(100000 * v[i]) / 100000

    return v