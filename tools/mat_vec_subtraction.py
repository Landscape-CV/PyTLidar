"""
Python adaptation and extension of TREEQSM:

Subtracts the vector v from each row of the matrix A.
    If A is an (n x m) matrix, then v needs to be an m-dimensional vector.



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


def mat_vec_subtraction(A, v):
    """
    Subtracts the vector v from each row of the matrix A.
    If A is an (n x m) matrix, then v needs to be an m-dimensional vector.

    Parameters:
    A (numpy.ndarray): An (n x m) matrix.
    v (numpy.ndarray): An m-dimensional vector.

    Returns:
    numpy.ndarray: The modified matrix A after subtraction.
    """
    if A.shape[1] != len(v):
        raise ValueError("The length of the vector v must match the number of columns in matrix A.")

    A = A - v  # Broadcasting automatically handles row-wise subtraction
    return A