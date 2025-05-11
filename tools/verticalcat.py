"""
Python adaptation and extension of TREEQSM:

Vertical concatenation of a list of arrays into a single vector.


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


def verticalcat(cell_array):
    """
    Vertical concatenation of a list of arrays into a single vector.

    Parameters:
    cell_array (list of np.ndarray): A list where each element is a numpy array.

    Returns:
    tuple: A tuple (vector, ind_elements) where:
        - vector is a 1D numpy array containing the concatenated values.
        - ind_elements is a 2D numpy array where each row specifies the start
          and end indices of the corresponding cell's elements in the vector.
    """
    # Determine the size of each array in the cell array
    cell_size = np.array([len(cell) for cell in cell_array])

    # Compute cumulative sum to determine index ranges
    ind_elements = np.zeros((len(cell_array), 2), dtype=int)
    ind_elements[:, 1] = np.cumsum(cell_size) - 1  # End indices
    ind_elements[1:, 0] = 1 + ind_elements[:-1, 1]  # Start indices (shifted ends)

    # Create the output vector and fill it
    total_size = sum(cell_size)
    vector = np.zeros(total_size, dtype=int)
    for i, cell in enumerate(cell_array):
        vector[ind_elements[i, 0]:ind_elements[i, 1] + 1] = cell

    return vector, ind_elements
