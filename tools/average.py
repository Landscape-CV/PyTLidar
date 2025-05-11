"""
Python adaptation and extension of TREEQSM:

Computes the average of columns of the matrix X.


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


def average(X):
    """
    Computes the average of columns of the matrix X.

    Parameters:
        X (np.ndarray): A 2D NumPy array.

    Returns:
        np.ndarray: A row vector containing the averages of each column if X has more than one row;
                    otherwise, returns X unchanged.
    """
    # Ensure X is a numpy array
    X = np.array(X)

    # Determine the number of rows
    m = X.shape[0]

    if m > 1:
        # Calculate the column-wise average
        A = np.sum(X, axis=0) / m
    else:
        # If there's only one row, return X directly
        A = X

    return A