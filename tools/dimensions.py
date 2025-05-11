"""
Python adaptation and extension of TREEQSM:

Calculates the box dimensions and dimension estimates of the point set "points".
Also returns the corresponding direction vectors.

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


def dimensions(points, *args):
    """
    Calculates the box dimensions and dimension estimates of the point set "points".
    Also returns the corresponding direction vectors.

    Parameters:
        points (np.ndarray): A numpy array of shape (n_points, d) where d = 2 or 3.
        *args:
            If one extra argument is provided:
                P (np.ndarray): An indexable array used to re-map "points".
                points = P[points, :]
            If two extra arguments are provided:
                P (np.ndarray): An indexable array.
                Bal (list or dict): A collection where indexing by "points" returns index arrays.
                I = np.concatenate([Bal[idx] for idx in points])
                points = P[I, :]

    Returns:
        D (np.ndarray): A vector containing extents and variance ratios.
        dir (np.ndarray): A matrix whose rows are the principal direction vectors.
                          For a d-dimensional space, the output shape will be (d, d).
    """

    # Handle optional arguments to remap 'points'
    if len(args) == 1:
        P = args[0]
        points = P[points, :]
    elif len(args) == 2:
        P = args[0]
        Bal = args[1]
        # Assuming Bal is a list-like structure where each element Bal[idx] is an index array.
        I = np.concatenate([np.asarray(Bal[idx]) for idx in points])
        points = P[I, :]

    # Calculate the covariance matrix.
    # np.cov expects data variables as rows by default, so we need to set rowvar=False.
    X = np.cov(points, rowvar=False)

    # Compute the Singular Value Decomposition.
    U, S_vals, _ = np.linalg.svd(X)

    # Create diagonal matrix S so that S(i,i) = S_vals[i]
    S = np.diag(S_vals)

    d = points.shape[1]

    if d == 3:
        # Project the points on the principal axes.
        dp1 = points @ U[:, 0]
        dp2 = points @ U[:, 1]
        dp3 = points @ U[:, 2]

        # Calculate extents along each principal direction.
        extent1 = np.max(dp1) - np.min(dp1)
        extent2 = np.max(dp2) - np.min(dp2)
        extent3 = np.max(dp3) - np.min(dp3)

        # Calculate variance ratios.
        ratio1 = (S[0, 0] - S[1, 1]) / S[0, 0] if S[0, 0] != 0 else 0
        ratio2 = (S[1, 1] - S[2, 2]) / S[0, 0] if S[0, 0] != 0 else 0
        ratio3 = S[2, 2] / S[0, 0] if S[0, 0] != 0 else 0

        # Dimensions vector: extents and ratios.
        D = np.array([extent1, extent2, extent3, ratio1, ratio2, ratio3])

        # Direction vectors as rows.
        dir_vectors = np.vstack((U[:, 0].T, U[:, 1].T, U[:, 2].T))
    elif d == 2:
        dp1 = points @ U[:, 0]
        dp2 = points @ U[:, 1]

        extent1 = np.max(dp1) - np.min(dp1)
        extent2 = np.max(dp2) - np.min(dp2)

        ratio1 = (S[0, 0] - S[1, 1]) / S[0, 0] if S[0, 0] != 0 else 0
        ratio2 = S[1, 1] / S[0, 0] if S[0, 0] != 0 else 0

        D = np.array([extent1, extent2, ratio1, ratio2])
        dir_vectors = np.vstack((U[:, 0].T, U[:, 1].T))
    else:
        raise ValueError("The dimension of points must be either 2 or 3.")

    return D, dir_vectors