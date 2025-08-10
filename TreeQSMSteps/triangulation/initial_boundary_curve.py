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
from TreeQSMSteps.triangulation import check_self_intersection
from Utils.Utils import distances_to_line
from TreeQSMSteps.triangulation import boundary_curve2
def initial_boundary_curve(P, TriaWidth):
    """
    Determines the boundary curve adaptively based on the given point cloud and triangular width.

    Inputs:
    P          - The point cloud (n x 3 array, each row is a point in 3D space)
    TriaWidth  - The width of the triangle for adaptive curve adjustment

    Returns:
    Curve      - The boundary curve as an ndarray (m x 3), where m is the number of curve points.
    """

    # Define suitable center
    Top = np.max(P[:, 2])
    P[:, 2] = Top * np.ones(len(P))

    # Define the "center" of points as the mean
    Center = np.mean(P, axis=0)
    Center0 = Center.copy()

    # Initialize variables
    i = 0
    A0 = 61
    ShortestDist = 0
    while ShortestDist < 0.075 and i < 100:
        # Randomly move the center
        Center = Center0 + np.array([3 * ShortestDist * np.random.randn(), 3 * ShortestDist * np.random.randn(), 0])

        # Compute angles of points as seen from the center
        V = P[:, :2] - Center[:2]
        angle = np.degrees(np.arctan2(V[:, 1], V[:, 0])) + 180

        A = np.zeros(70, dtype=bool)
        a = np.ceil(angle / 5).astype(int)
        I = a > 0
        A[a[I]] = True
        
        if i == 0:
            ShortestDist = 0.025
        elif np.sum(A) < A0:
            ShortestDist = 0.05
        else:
            PointDist = np.linalg.norm(V, axis=1)#PointDist = sqrt(sum(V.*V,2));
            ShortestDist = np.min(PointDist)
            FirstPoint = np.argmin(PointDist)

        i += 1
        if i == 100 and ShortestDist < 0.075:
            i = 0
            A0 -= 2

    # Define first boundary curve based on the center
    Curve = np.zeros(18, dtype=int)
    Curve[0] = FirstPoint  # Start the curve from the closest point to the center
    a0 = angle[FirstPoint]
    I = angle < a0
    angle[I] += (360 - a0)
    angle[~I] -= a0

    np_points = len(P)

    Ind = np.arange(np_points)
    t = 0
    for i in range(1, 19):
        J = (angle > 12.5 + 20 * (i - 2)) & (angle < 27.5 + 20 * (i - 2))
        if not np.any(J):  # If no points, try 18-degree sector
            J = (angle > 11 + 20 * (i - 2)) & (angle < 29 + 20 * (i - 2))
        if np.any(J):
            D = PointDist[J]
            ind = Ind[J]
            closest = np.argmin(D)
            t += 1
            Curve[t] = ind[closest]

    Curve = Curve[:t+1]
    if len(Curve) == 0:
        return Curve

    I = np.ones(np_points, dtype=bool)
    I[Curve] = False
    Ind = Ind[I]

    # Adapt the initial curve to the data
    V = P[Curve[1:t], :] - P[Curve[:t], :]
    D = np.linalg.norm(V[:, :2], axis=1)
    n = t
    n0 = 1

    while np.any(D > 1.25 * TriaWidth) and n > n0:
        N = np.hstack([V[:, 1].reshape(-1, 1), -V[:, 0].reshape(-1, 1), V[:, 2].reshape(-1, 1)])
        M = P[Curve, :] + 0.5 * V

        Curve1 = Curve.copy()
        t = 0
        for i in range(n):
            if D[i] > 1.25 * TriaWidth:
                d, _, hc = distances_to_line(P[Curve, :], N[i, :], M[i, :])
                I = (hc > 0.01) & (d < D[i] / 2)
                if np.any(I):
                    H = np.min(hc[I])
                else:
                    H = 1
                d, _, h = distances_to_line(P[Ind, :], N[i, :], M[i, :])
                I = (d < D[i] / 3) & (h > -TriaWidth / 2) & (h < H)
                if np.any(I):
                    ind = Ind[I]
                    h = h[I]
                    closest = np.argmin(h)
                    I = ind[closest]
                    t += 1
                    Curve1 = np.insert(Curve1, t, I)
                    Ind = Ind[Ind != I]
                    t += 1
                else:
                    t += 1
            else:
                t += 1

        Curve = Curve1[:t]
        n0 = n
        n = len(Curve)
        V = P[Curve[1:n], :] - P[Curve[:n], :]
        D = np.linalg.norm(V[:, :2], axis=1)

    # Refine the curve for longer edges if far away points
    n0 = n - 1
    while n > n0:
        N = np.hstack([V[:, 1].reshape(-1, 1), -V[:, 0].reshape(-1, 1), V[:, 2].reshape(-1, 1)])
        M = P[Curve, :] + 0.5 * V

        Curve1 = Curve.copy()
        t = 0
        for i in range(n):
            if D[i] > 0.5 * TriaWidth:
                d, _, hc = distances_to_line(P[Curve, :], N[i, :], M[i, :])
                I = (hc > 0.01) & (d < D[i] / 2)
                if np.any(I):
                    H = np.min(hc[I])
                else:
                    H = 1
                d, _, h = distances_to_line(P[Ind, :], N[i, :], M[i, :])
                I = (d < D[i] / 3) & (h > -TriaWidth / 3) & (h < H)
                ind = Ind[I]
                h = h[I]
                closest = np.argmin(h)

                if h > TriaWidth / 10:
                    I = ind[closest]
                    t += 1
                    Curve1 = np.insert(Curve1, t, I)
                    Ind = Ind[Ind != I]
                    t += 1
                else:
                    t += 1
            else:
                t += 1

        Curve = Curve1[:t]
        n0 = n
        n = len(Curve)
        V = P[Curve[1:n], :] - P[Curve[:n], :]
        D = np.linalg.norm(V[:, :2], axis=1)

    # Smooth the curve by defining the points by means of neighbors
    Curve = P[Curve, :]
    Curve = boundary_curve2(P, Curve, 0.04, TriaWidth)
    if len(Curve) == 0:
        return Curve

    # Add points for too long edges
    n = len(Curve)
    V = Curve[1:n, :] - Curve[:n - 1, :]
    D = np.linalg.norm(V[:, :2], axis=1)
    Curve1 = Curve.copy()
    t = 0
    for i in range(n):
        if D[i] > TriaWidth:
            m = int(np.floor(D[i] / TriaWidth))
            t += 1
            W = np.zeros((m, 3))
            for j in range(m):
                W[j, :] = Curve[i, :] + (j + 1) / (m + 1) * V[i, :]
            Curve1 = np.insert(Curve1, t, W, axis=0)
            t += m
        else:
            t += 1
    Curve = Curve1

    # Define the curve again by equalizing the point distances along the curve
    n = len(Curve)
    V = Curve[1:n, :] - Curve[:n - 1, :]
    D = np.linalg.norm(V[:, :2], axis=1)
    L = np.cumsum(D)
    m = int(np.ceil(L[-1] / TriaWidth))
    TriaWidth = L[-1] / m
    Curve1 = np.zeros((m, 3))
    Curve1[0, :] = Curve[0, :]
    b = 1
    for i in range(1, m):
        while L[b] < (i - 1) * TriaWidth:
            b += 1
        if b > 1:
            a = ((i - 1) * TriaWidth - L[b - 1]) / D[b]
            Curve1[i, :] = Curve[b, :] + a * V[b, :]
        else:
            a = (L[b] - (i - 1) * TriaWidth) / D[b]
            Curve1[i, :] = Curve[b, :] + a * V[b, :]

    Curve = Curve1

    # Check if the curve intersects itself
    Intersect = check_self_intersection(Curve[:, :2])
    if Intersect:
        Curve = np.zeros((0, 3))

    return Curve