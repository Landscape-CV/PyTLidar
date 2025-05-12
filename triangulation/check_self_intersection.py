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
Authors: Fan Yang, John Hagood, Amir Hossein Alikhah Mishamandani
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
        V = Curve[1:, :] - Curve[:-1, :]  # line elements forming the curve
        V = np.vstack([V, Curve[0, :] - Curve[-1, :]])  # Wrap around for last line
        L = np.sqrt(np.sum(V**2, axis=1))  # the lengths of the line elements
        Ind = np.arange(n)  # indexes of the line elements
        
        if dim == 2:  # 2d curves
            # directions (unit vectors) of the line elements
            DirLines = np.hstack([V[:, 0][:, np.newaxis] / L[:, np.newaxis], 
                                  V[:, 1][:, np.newaxis] / L[:, np.newaxis]])
            Intersect = False
            IntersectLines = None
            
            while not Intersect:
                for i in range(n-1):
                    if i > 0:
                        I = np.logical_or(Ind > i+1, Ind < i-1)
                    else:
                        I = np.logical_and(Ind > i+1, Ind < n)
                    ind = Ind[I]
                    
                    for j in ind:
                        A = np.array([DirLines[j, :], -DirLines[i, :]])
                        b = Curve[i, :] - Curve[j, :]
                        Ainv = np.linalg.inv(A.T @ A) @ A.T
                        x = Ainv @ b  # signed length along the line elements to the crossing
                        
                        if 0 <= x[0] <= L[j] and 0 <= x[1] <= L[i]:
                            Intersect = True
                            break
                if Intersect:
                    break
            return Intersect, IntersectLines
        elif dim == 3:  # 3d curves
            # directions (unit vectors) of the line elements
            DirLines = np.hstack([V[:, 0][:, np.newaxis] / L[:, np.newaxis], 
                                  V[:, 1][:, np.newaxis] / L[:, np.newaxis],
                                  V[:, 2][:, np.newaxis] / L[:, np.newaxis]])
            Intersect = False
            IntersectLines = None
            
            while not Intersect:
                for i in range(n-1):
                    if i > 0:
                        I = np.logical_or(Ind > i+1, Ind < i-1)
                    else:
                        I = np.logical_and(Ind > i+1, Ind < n)
                    ind = Ind[I]
                    
                    # Solve for possible intersection points (this assumes the function exists)
                    D, DistOnRay, DistOnLines = distances_between_lines(
                        Curve[i, :], DirLines[i, :], Curve[ind, :], DirLines[ind, :]
                    )
                    
                    if np.any((DistOnRay >= 0) & (DistOnRay <= L[i]) & 
                               (DistOnLines > 0) & (DistOnLines <= L[I])):
                        Intersect = True
                        break
                if Intersect:
                    break
            return Intersect, IntersectLines
    else:
        # Empty curve
        return False, []