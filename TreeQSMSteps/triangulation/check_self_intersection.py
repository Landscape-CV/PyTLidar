"""
Python adaptation and extension of TREEQSM:

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

Version: 0.0.1
Date: 9 Feb 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""
import numpy as np
# import sys
# sys.path.append('../')
from Utils.Utils import distances_between_lines
def check_self_intersection(Curve):
    # Check if the curve intersects itself
    if Curve.size > 0:
        dim = Curve.shape[1]  # 2 or 3 dimensional curve
        n = Curve.shape[0]  # number of points in the curve
        # line elements forming the curve (with wrap-around for the last one)
        V = Curve[np.concatenate([np.arange(1, n), [0]]), :] - Curve
        L = np.sqrt(np.sum(V**2, axis=1))  # the lengths of the line elements
        Ind = np.arange(n)  # 0-based indexes of the line elements

        # Cell-array analogue: column 0 holds intersecting line indices,
        # column 1 holds distances along the line to the intersection points.
        IntersectLines = np.empty((n, 2), dtype=object)
        for r in range(n):
            IntersectLines[r, 0] = np.array([], dtype=int)
            IntersectLines[r, 1] = np.array([], dtype=float)
        Intersect = False

        if dim == 2:  # 2d curves
            # directions (unit vectors) of the line elements
            DirLines = np.column_stack([V[:, 0] / L, V[:, 1] / L])
            for i in range(n - 1):
                # Select the line elements that can intersect element i
                if i > 0:
                    I = np.logical_or(Ind > i + 1, Ind < i - 1)
                else:
                    I = np.logical_and(Ind > i + 1, Ind < n - 1)
                ind = Ind[I]
                for j in ind:
                    # Solve for the crossing point of the two line elements
                    A = np.column_stack([DirLines[j, :], -DirLines[i, :]])
                    b = Curve[i, :] - Curve[j, :]
                    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
                    Ainv = (1.0 / det) * np.array([[A[1, 1], -A[0, 1]],
                                                   [-A[1, 0], A[0, 0]]])
                    x = Ainv @ b  # signed lengths along the line elements to the crossing
                    if x[0] >= 0 and x[0] <= L[j] and x[1] >= 0 and x[1] <= L[i]:
                        Intersect = True
                        IntersectLines[i, 0] = np.append(IntersectLines[i, 0], j)
                        IntersectLines[j, 0] = np.append(IntersectLines[j, 0], i)
                        IntersectLines[i, 1] = np.append(IntersectLines[i, 1], x[0])
                        IntersectLines[j, 1] = np.append(IntersectLines[j, 1], x[1])
            # remove possible multiple values
            for r in range(n):
                IntersectLines[r, 0] = np.unique(IntersectLines[r, 0])
                if IntersectLines[r, 1].size > 0:
                    IntersectLines[r, 1] = np.array([np.min(IntersectLines[r, 1])])
            return Intersect, IntersectLines

        elif dim == 3:  # 3d curves
            # directions (unit vectors) of the line elements
            DirLines = np.column_stack([V[:, 0] / L, V[:, 1] / L, V[:, 2] / L])
            for i in range(n - 1):
                # Select the line elements that can intersect element i
                if i > 0:
                    I = np.logical_or(Ind > i + 1, Ind < i - 1)
                else:
                    I = np.logical_and(Ind > i + 1, Ind < n - 1)
                # Solve for possible intersection points
                D, DistOnRay, DistOnLines = distances_between_lines(
                    Curve[i, :], DirLines[i, :], Curve[I, :], DirLines[I, :]
                )
                mask = ((DistOnRay >= 0) & (DistOnRay <= L[i]) &
                        (DistOnLines > 0) & (DistOnLines <= L[I]))
                if np.any(mask):
                    Intersect = True
                    ind = Ind[I]
                    ind = ind[mask]
                    DistOnLines = DistOnLines[mask]
                    IntersectLines[i, 0] = ind
                    IntersectLines[i, 1] = DistOnRay[mask]
                    for j in range(len(ind)):
                        IntersectLines[ind[j], 0] = np.append(IntersectLines[ind[j], 0], i)
                        IntersectLines[ind[j], 1] = np.append(IntersectLines[ind[j], 1], DistOnLines[j])
            # remove possible multiple values
            for r in range(n):
                IntersectLines[r, 0] = np.unique(IntersectLines[r, 0])
                if IntersectLines[r, 1].size > 0:
                    IntersectLines[r, 1] = np.array([np.min(IntersectLines[r, 1])])
            return Intersect, IntersectLines
    else:
        # Empty curve
        return False, np.empty((1, 1), dtype=object)