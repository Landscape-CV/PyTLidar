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
Date: 15 Mar 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import numpy as np

try:
    from ..Utils import Utils
except ImportError:
    import Utils.Utils as Utils

def point_model_distance(P, cylinder):
    '''
    Computes the distances of the points to the cylinder model
    Args:
        P:          numpy array of point cloud
        cylinder:   cylinder model

    Returns:
        pmdistance: distance data structure that includes
            CylDist:  Total distance between the cylinder and the point cloud
            median:   Median distance
            mean:     Mean distance
            max:      Max distance
            std:      Standard deviation of distance

    '''
    # Extract cylinder data
    Rad = cylinder['radius']
    Len = cylinder['length']
    Sta = cylinder['start']
    Axe = cylinder['axis']
    BOrd = cylinder['BranchOrder']

    # Select randomly 25 % or max one million points for the distance compute.
    np.random.seed(15)  ####### FY note: for test, can remove later
    np0 = P.shape[0]
    if np0 == 0:
        return {}
    a = min(int(0.25 * np0), 1000000)
    if a < np0:
        scale = 0.5 / (1 - a / np0)
        I = np.round(scale * np.random.rand(np0)).astype(bool)
        P_sampled = P[I, :]
    else:
        P_sampled = P.copy()
    #print(I)
    # Cubical partitioning parameters
    L = 2 * np.median(Len)
    NE_input = max(3, min(10, int(np.ceil(np.max(Len) / L)))) + 3
    partition, _, Info = Utils.cubical_partition(P_sampled, L, NE_input, False)
    Min = Info[0:3]
    EL = Info[6]
    NE = int(Info[7])
    dims = Info[3:6].astype(int)
    #print(Info)
    #print(partition)
    #print(cubes)

    # Calculates the cube-coordinates of the starting points
    CC = np.floor((Sta - Min) / EL).astype(int) + NE

    # Compute the N number of cubes needed for each starting point
    N = np.ceil(Len / L).astype(int)
    n_cylinders = Rad.shape[0]

    # Correct N based on CC and dims, so that cube indexes are not too small or large
    for i in range(n_cylinders):
        for dim in range(3):
            if CC[i, dim] < N[i] + 1:
                N[i] = CC[i, dim] - 1
            if CC[i, dim] + N[i] + 1 > dims[dim]:
                N[i] = dims[dim] - CC[i, dim] - 1
        N[i] = max(N[i], 0)
    #print(N)
    # Calculate the distances to the cylinders
    np_sampled = P_sampled.shape[0]
    Dist = np.full((np_sampled, 2), -1, dtype=float)  # Distance and the closest cylinder of each points
    Dist[:, 0] = 2  # Large distance initially
    #Points = np.zeros((int(np.ceil(np_sampled / 10)), 1), dtype=np.int32)
    Data = [None] * n_cylinders

    # Calculate distances for each cylinder
    for i in range(n_cylinders):
        cc = CC[i]
        n_i = N[i]
        points = []
        # Collect points in nearby cubes
        for x in range(cc[0] - n_i, cc[0] + n_i + 1):
            for y in range(cc[1] - n_i, cc[1] + n_i + 1):
                for z in range(cc[2] - n_i, cc[2] + n_i + 1):
                    key = (x, y, z)
                    if partition[key] is not None:
                        points.extend(partition[key])
        if not points:
            Data[i] = None
            continue
        points = np.array(points, dtype=int)
        #print(points)
        d, _, h, _ = Utils.distances_to_line(P_sampled[points], Axe[i], Sta[i])
        d = np.abs(d - Rad[i])
        #print(d)
        # Filter points within cylinder length and close
        valid = (h >= 0) & (h <= Len[i]) & (d < 0.5)
        improved = d < Dist[points, 0]
        mask = valid & improved
        #print(mask)
        improved_points = points[mask]
        if improved_points.size > 0:
            Dist[improved_points, 0] = d[mask]
            Dist[improved_points, 1] = i
        Data[i] = (d, h, points)
    #print(Dist)
    # Calculate the distances to the cylinders for points not yet calculated
    # because they are not "on side of cylinder
    # Filter points slightly outside cylinder ends
    for i in range(n_cylinders):
        if Data[i] is None:
            continue
        d, h, points = Data[i]
        valid = ((h >= -0.1) & (h < 0)) | ((h >= Len[i]) & (h <= Len[i] + 0.1)) & (d < 0.5)
        improved = d < Dist[points, 0]
        mask = valid & improved
        improved_points = points[mask]
        #print(mask)
        if improved_points.size > 0:
            Dist[improved_points, 0] = d[mask]
            Dist[improved_points, 1] = i
    #print(Dist)
    # Select only the shortest 95% of distances for each cylinder
    '''
    Cyl = [[] for _ in range(n_cylinders)]
    for idx in range(np_sampled):
        cyl_idx = int(Dist[idx, 1])
        if cyl_idx != -1:
            Cyl[cyl_idx].append(Dist[idx, 0])
    '''
    N = np.zeros(n_cylinders)
    O = np.zeros(np_sampled)
    for i in range(np_sampled):
        if Dist[i, 1] >= 0:
            N[int(Dist[i, 1])] = N[int(Dist[i, 1])] + 1
            O[i] = N[int(Dist[i, 1])]
    Cyl = [[] for _ in range(n_cylinders)]
    #print(O, Cyl)
    for i in range(n_cylinders):
        Cyl[i] = np.zeros(int(N[i]))
    for i in range(np_sampled):
        if Dist[i, 1] >= 0:
            Cyl[int(Dist[i, 1])][int(O[i]) - 1] = i
    #print(Cyl)
    DistCyl = np.zeros(n_cylinders, dtype=float)  # Average point distance to each cylinder
    for i in range(n_cylinders):
        I = np.array(Cyl[i], dtype=int)
        m = np.size(I)
        #print(I, m)
        if m <= 0:
            DistCyl[i] = 0.0
            continue
        if m > 19:  # select the smallest 95% of distances
            cutoff = int(0.95 * m)
            sorted_d = np.sort(Dist[I, 0])
            DistCyl[i] = np.mean(sorted_d[:cutoff])
        else:
            DistCyl[i] = np.mean(Dist[I, 0])
    #print(DistCyl)
    # Prepare output
    pmdistance = {
        'CylDist': DistCyl.astype(np.float32),
        'median': float(np.median(DistCyl)),
        'mean': float(np.mean(DistCyl)),
        'max': float(np.max(DistCyl)),
        'std': float(np.std(DistCyl, ddof=1))  # Use sample standard deviation
    }

    # Trunk and branch statistics
    T = (BOrd == 0).flatten()
    B = (BOrd > 0).flatten()
    B1 = (BOrd == 1).flatten()
    B2 = (BOrd == 2).flatten()

    trunk_dists = DistCyl[T]
    branch_dists = DistCyl[B]
    branch1_dists = DistCyl[B1]
    branch2_dists = DistCyl[B2]

    pmdistance['TrunkMedian'] = float(np.median(trunk_dists)) if trunk_dists.size > 0 else 0.0
    pmdistance['TrunkMean'] = float(np.mean(trunk_dists)) if trunk_dists.size > 0 else 0.0
    pmdistance['TrunkMax'] = float(np.max(trunk_dists)) if trunk_dists.size > 0 else 0.0
    pmdistance['TrunkStd'] = float(np.std(trunk_dists)) if trunk_dists.size > 0 else 0.0

    pmdistance['BranchMedian'] = float(np.median(branch_dists)) if branch_dists.size > 0 else 0.0
    pmdistance['BranchMean'] = float(np.mean(branch_dists)) if branch_dists.size > 0 else 0.0
    pmdistance['BranchMax'] = float(np.max(branch_dists)) if branch_dists.size > 0 else 0.0
    pmdistance['BranchStd'] = float(np.std(branch_dists)) if branch_dists.size > 0 else 0.0

    pmdistance['Branch1Median'] = float(np.median(branch1_dists)) if branch1_dists.size > 0 else 0.0
    pmdistance['Branch1Mean'] = float(np.mean(branch1_dists)) if branch1_dists.size > 0 else 0.0
    pmdistance['Branch1Max'] = float(np.max(branch1_dists)) if branch1_dists.size > 0 else 0.0
    pmdistance['Branch1Std'] = float(np.std(branch1_dists)) if branch1_dists.size > 0 else 0.0

    pmdistance['Branch2Median'] = float(np.median(branch2_dists)) if branch2_dists.size > 0 else 0.0
    pmdistance['Branch2Mean'] = float(np.mean(branch2_dists)) if branch2_dists.size > 0 else 0.0
    pmdistance['Branch2Max'] = float(np.max(branch2_dists)) if branch2_dists.size > 0 else 0.0
    pmdistance['Branch2Std'] = float(np.std(branch2_dists)) if branch2_dists.size > 0 else 0.0

    return pmdistance