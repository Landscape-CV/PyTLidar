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


def boundary_curve2(P, Curve0, rball, dmax):
    """
    Determines the boundary curve based on the previously defined boundary curve.
    
    Inputs:
    P         Point cloud of the cross section
    Curve0    Seed points from previous cross section curve
    rball     Radius of the balls centered at seed points
    dmax      Maximum distance between consecutive curve points; if larger,
              create a new one between the points

    Returns:
    Curve     The boundary curve as an ndarray of shape (nc, 3)
    """

    # Partition the point cloud into cubes
    Min = np.min(np.vstack([P[:, :2], Curve0[:, :2]]), axis=0)
    Max = np.max(np.vstack([P[:, :2], Curve0[:, :2]]), axis=0)
    N = np.ceil((Max - Min) / rball) + 5
    N = N.astype(int)
    
    CC = np.floor(np.hstack([P[:, :2] - Min, np.ones((P.shape[0], 1))]) / rball) + 3
    CC = CC[:, :2].astype(int)  # Cube coordinates of the section points
    
    # Sort the points according to lexicographical order
    S = np.dot(CC[:, 0:2], np.array([1, N[0]]))
    S, I = zip(*sorted(zip(S, range(len(S)))))
    
    # Define the "partition"
    np_points = len(P)
    partition = {key: [] for key in range(N[0] * N[1])}
    p = 0
    while p < np_points:
        t = 1
        while p + t <= np_points and S[p] == S[p + t]:
            t += 1
        q = I[p]
        partition[tuple(CC[q, :])] = I[p:p + t]
        p += t

    # Define segments using the previous points
    CC = np.floor(np.hstack([Curve0[:, :2] - Min, np.ones((len(Curve0), 1))]) / rball) + 3
    CC[CC < 3] = 3
    nc = len(Curve0)
    Dist = np.ones(np_points) * 1e8  # distance of point to the closest center
    SoP = np.zeros(np_points)  # the segment the points belong to
    Radius = rball ** 2

    for i in range(nc):
        points = partition.get(tuple(CC[i, :]), [])
        V = np.hstack([P[points, 0:2] - Curve0[i, 0:2]])
        dist = np.sum(V**2, axis=1)
        PointsInBall = dist < Radius
        points = np.array(points)[PointsInBall]
        dist = dist[PointsInBall]
        D = Dist[points]
        L = dist < D
        I = points[L]
        Dist[I] = dist[L]
        SoP[I] = i

    # Finalize the segments
    Num = np.zeros(nc)
    IndPoints = np.zeros(np_points)
    
    for i in range(np_points):
        if SoP[i] > 0:
            Num[int(SoP[i])] += 1
            IndPoints[i] = Num[int(SoP[i])]

    if np.count_nonzero(Num) > 0.05 * nc:
        # Initialize the "Seg"
        Seg = {i: [] for i in range(nc)}
        
        # Define the "Seg"
        for i in range(np_points):
            if SoP[i] > 0:
                Seg[int(SoP[i])].append(i)

        # Define the new curve points as the average of the segments
        Curve = np.zeros((nc, 3))
        for i in range(nc):
            S = Seg[i]
            if len(S) > 0:
                Curve[i, :] = np.mean(P[S, :], axis=0)
                if np.linalg.norm(Curve[i, :] - Curve0[i, :]) > 1.25 * dmax:
                    Curve[i, :] = Curve0[i, :]
            else:
                Curve[i, :] = Curve0[i, :]

        # Add new points if the distances are too large
        V = Curve[np.arange(1, nc), :] - Curve[0:nc, :]
        d = np.sum(V**2, axis=1)
        Large = d > dmax**2
        m = np.count_nonzero(Large)

        if m > 0:
            Curve0 = np.zeros((nc + m, 3))
            t = 0
            for i in range(nc):
                if Large[i]:
                    t += 1
                    Curve0[t, :] = Curve[i, :]
                    t += 1
                    Curve0[t, :] = Curve[i, :] + 0.5 * V[i, :]
                else:
                    t += 1
                    Curve0[t, :] = Curve[i, :]
            Curve = Curve0

        # Remove new points if distances are too small
        nc = len(Curve)
        V = Curve[np.arange(1, nc), :] - Curve[0:nc, :]
        d = np.sum(V**2, axis=1)
        Small = d < (0.333 * dmax)**2
        m = np.count_nonzero(Small)

        if m > 0:
            for i in range(nc - 1):
                if Small[i] and Small[i + 1]:
                    Small[i + 1] = False
            if not Small[nc - 1] and Small[0]:
                Small[0] = False
                Small[nc - 1] = True
            Curve = Curve[~Small, :]

    else:
        # If not enough new points, return the old curve
        Curve = Curve0

    return Curve