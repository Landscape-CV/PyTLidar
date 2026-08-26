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
from . import check_self_intersection
def boundary_curve(P,Curve0,rball,dmax):



    # %% Partition the point cloud into cubes
    Min = np.min(np.vstack([P[:, :2], Curve0[:, :2]]), axis=0)
    Max = np.max(np.vstack([P[:, :2], Curve0[:, :2]]), axis=0)
    N = np.ceil((Max - Min) / rball).astype(int) + 5
    # % cube coordinates of the section points
    CC = np.floor((P[:, :2] - Min) / rball).astype(int) + 3
    # % Sorts the points according a lexicographical order
    # MATLAB: S = [CC(:,1) CC(:,2)-1]*[1 N(1)]';
    S = CC[:, 0] + (CC[:, 1] - 1) * N[0]
    I = np.argsort(S, kind='stable')
    S = S[I]

    # % Define "partition" (dict keyed by cube coordinate tuple)
    nump = np.size(P, 0)
    partition = {}
    p = 0              # % The index of the point under comparison
    while p < nump:
        t = 1
        while (p + t < nump) and (S[p] == S[p + t]):
            t = t + 1
        q = I[p]
        partition[(int(CC[q, 0]), int(CC[q, 1]))] = I[p:p + t]
        p = p + t

    # %% Define segments using the previous points
    # % cube coordinates of the seed points:
    CC = np.floor((Curve0[:, :2] - Min) / rball).astype(int) + 3
    CC[CC < 3] = 3
    nc = len(Curve0)#  % number of sets
    Dist = 1e8*np.ones(nump)#  % distance of point to the closest center
    SoP = np.zeros(nump)#  % the segment the points belong to
    Radius = rball**2
    for i in range(nc):
        cx = int(CC[i, 0])
        cy = int(CC[i, 1])
        # Gather points from the 5x5 cube neighborhood (MATLAB CC-2:CC+2)
        points = []
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                points.extend(partition.get((cx + dx, cy + dy), []))
        points = np.array(points, dtype=int)
        if points.size == 0:
            continue
        V = P[points, :2] - Curve0[i, :2]#[P(points,1)-Curve0(i,1) P(points,2)-Curve0(i,2)];
        dist = np.sum(V ** 2, axis=1)
        PointsInBall = dist < Radius
        points = points[PointsInBall]
        dist = dist[PointsInBall]
        D = Dist[points]
        L = dist < D
        I = points[L]
        Dist[I] = dist[L]
        SoP[I] = i + 1  # store 1-based segment id so 0 stays the 'unassigned' sentinel


    # %% Finalise the segments
    # % Number of points in each segment and index of each point in its segment
    Num = np.zeros(nc)
    IndPoints = np.zeros(nump,dtype=int)
    for i in range(nump):
        if SoP[i] > 0:
            Num[int(SoP[i]) - 1] = Num[int(SoP[i]) - 1] + 1
            IndPoints[i] = Num[int(SoP[i]) - 1]

    # % Continue if enough non-emtpy segments
    if np.count_nonzero(Num) > 0.05 * nc:
        # % Initialization of the "Seg"
        Seg = {i: np.zeros(int(Num[i])) for i in range(nc)}
        
        for i in range(nump):
            if SoP[i] > 0:
                Seg[int(SoP[i] - 1)][IndPoints[i] - 1] = i

        # %% Define the new curve points as the average of the segments
        Curve = np.zeros((nc, 3))
        Empty = np.zeros(nc, dtype=bool)
        for i in range(nc):
            S = Seg[i]
            if len(S) > 0:
                Curve[i, :] = np.mean(P[S.astype(int), :], axis=0)
                if np.linalg.norm(Curve[i, :] - Curve0[i, :]) > 1.25 * dmax:
                    Curve[i, :] = Curve0[i, :]
            else:
                Empty[i] = True

        # %% Interpolate for empty segments
        # % For empty segments create points by interpolation from neighboring 
        # % non-empty segments
        if np.any(Empty):
            for i in range(nc):
                if Empty[i]:
                    if 0 < i < nc - 1:
                        k = 0
                        while i + k < nc and Empty[i + k]:
                            k += 1
                        if i + k < nc:
                            LineEle = Curve[i + k, :] - Curve[i - 1, :]
                        else:
                            LineEle = Curve[0, :] - Curve[i - 1, :]
                        if k < 5:
                            for j in range(k):
                                Curve[i + j, :] = Curve[i - 1, :] + (j + 1) / (k + 1) * LineEle
                        else:
                            Curve[i:i + k, :2] = Curve0[i:i + k, :2]
                            Curve[i:i + k, 2] = Curve0[i:i + k, 2]
                    elif i == 0:
                        # Wrap-around gap: a trailing empties + b leading empties (rows 0..b-1).
                        # Here b is the 0-based index of the first non-empty row, so it equals
                        # MATLAB's b-1; the loop bounds below are written for that convention.
                        a = 0
                        while Empty[-(a + 1)]:
                            a += 1
                        b = 1
                        while Empty[b]:
                            b += 1
                        LineEle = Curve[b, :] - Curve[nc - a - 1, :]
                        n = a + b            # MATLAB: n = a+b-1 with 1-based b
                        if n < 5:
                            for j in range(a - 1):
                                Curve[nc - a + 1 + j, :] = Curve[nc - a - 1, :] + (j + 1) / n * LineEle
                            for j in range(b):   # MATLAB: for j = 1:b-1 (1-based b)
                                Curve[j, :] = Curve[nc - a - 1, :] + (j + a) / n * LineEle
                        else:
                            Curve[nc - a + 1:nc, :2] = Curve0[nc - a + 1:nc, :2]
                            Curve[nc - a + 1:nc, 2] = Curve0[nc - a + 1:nc, 2]
                            Curve[0:b, :2] = Curve0[0:b, :2]   # MATLAB: Curve(1:b-1,:) with 1-based b
                            Curve[0:b, 2] = Curve0[0:b, 2]
                    elif i == nc - 1:
                        LineEle = Curve[0, :] - Curve[nc - 2, :]
                        Curve[i, :] = Curve[nc - 2, :] + 0.5 * LineEle


        Curve[:, 2] = np.min(Curve[:, 2])

        # Check self-intersection
        Intersect, IntersectLines = check_self_intersection(Curve[:, :2])

        # If self-intersection, try to modify the curve
        j = 0
        while Intersect and j < 5:
            n = Curve.shape[0]
            InterLines = np.arange(n)  # 0-based line-element indices
            NumberOfIntersections = [len(line) for line in IntersectLines[:, 0]]
            I = np.array(NumberOfIntersections) > 0
            InterLines = InterLines[I]
            CrossLen = np.concatenate(list(IntersectLines[I, 1])) if np.any(I) else np.array([])
            if len(CrossLen) == len(InterLines):
                LineEle = np.vstack([Curve[1:, :] - Curve[:-1, :], Curve[0, :] - Curve[-1, :]])
                d = np.linalg.norm(LineEle, axis=1)
                m = len(InterLines)
                for i in range(0, m, 2):
                    li = InterLines[i]
                    if li != n - 1:
                        Curve[li + 1, :] = Curve[li, :] + 0.9 * CrossLen[i] / d[li] * LineEle[li, :]
                    else:
                        Curve[0, :] = Curve[li, :] + 0.9 * CrossLen[i] / d[li] * LineEle[li, :]
                Intersect, IntersectLines = check_self_intersection(Curve[:, :2])
                j += 1
            else:
                j = 6

        # %% Add new points if too large distances
        LineEle = np.vstack([Curve[1:, :] - Curve[:-1, :], Curve[0, :] - Curve[-1, :]])
        d = np.sum(LineEle ** 2, axis=1)
        Large = d > dmax ** 2
        m = np.count_nonzero(Large)

        if m > 0:
            Curve0_new = np.zeros((nc + m, 3))
            Ind = np.zeros((nc + m, 2), dtype=int)
            t = 0
            for i in range(nc):
                if Large[i]:
                    Curve0_new[t, :] = Curve[i, :]
                    # 1-based ring indices (wrap -> 1) to match MATLAB
                    Ind[t, :] = [i + 1, i + 2] if i < nc - 1 else [i + 1, 1]
                    t += 1
                    Curve0_new[t, :] = Curve[i, :] + 0.5 * LineEle[i, :]
                    # midpoint: second connection is the sentinel 0
                    Ind[t, :] = [i + 2, 0] if i < nc - 1 else [1, 0]
                    t += 1
                else:
                    Curve0_new[t, :] = Curve[i, :]
                    Ind[t, :] = [i + 1, i + 2] if i < nc - 1 else [i + 1, 1]
                    t += 1
            Curve = Curve0_new
        else:
            # MATLAB: Ind = [(1:nc)' [(2:nc)'; 1]]  (1-based ring, wrap -> 1)
            Ind = np.column_stack([np.arange(1, nc + 1),
                                   np.concatenate([np.arange(2, nc + 1), [1]])])


        # %% Remove new points if too small distances
        nc = len(Curve)
        LineEle = np.vstack([Curve[1:, :] - Curve[:-1, :], Curve[0, :] - Curve[-1, :]])
        d = np.sum(LineEle ** 2, axis=1)
        Small = d < (0.333 * dmax) ** 2
        m = np.count_nonzero(Small)
        if m > 0:
            for i in range(nc - 1):
                if not Small[i] and Small[i + 1]:
                    Ind[i, 1] = -1
                elif Small[i] and Small[i + 1]:
                    Small[i + 1] = False
            if not Small[nc - 1] and Small[0]:
                Ind[nc - 1, 1] = -1
                Ind[0, 1] = -1
                Small[0] = False
                Small[nc - 1] = True
                I = Ind[:, 1] > 0
                Ind[1:, 0] = Ind[1:, 0] + 1
                Ind[I, 1] = Ind[I, 1] + 1

            Ind = Ind[~Small, :]
            Curve = Curve[~Small, :]

    else:
        # MATLAB: Ind = [(1:nc)' [(2:nc)'; 1]]  (1-based ring, wrap -> 1)
        Ind = np.column_stack([np.arange(1, nc + 1),
                               np.concatenate([np.arange(2, nc + 1), [1]])])
        Curve = Curve0

    return Curve, Ind