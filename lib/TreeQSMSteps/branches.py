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
Date: 19 Mar 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import numpy as np
import math

def branches(cylinder):
    """
    Determines the branching structure and computes branch attributes
    
    Determines the branches (cylinders in a segment define a branch), their order
    and topological parent-child-relation. Branch number one is the trunk and
    its order is zero. Notice that branch number does not tell its age in the
    sense that branch number two would be the oldest branch and the number
    three the second oldest.
    
    Args:
    cylinder(Dictionary):  Dictionary containing cylinder data generated during the TreeQSM process.
    
    Returns:
    dictionary :   Branch structure array, contains fields:
                Branch order, parent, volume, length, angle, height, azimuth
                and diameter
    """

    Rad = cylinder['radius']  # radii (nc,)
    Len = cylinder['length']  # lengths (nc,)
    Axe = cylinder['axis']  # axes (nc, 3)
    Branch = cylinder['branch']  # branch number for each cylinder (nc,)
    BranchOrder = cylinder['BranchOrder']
    added = cylinder['added']
    parent = cylinder['parent']
    start = cylinder['start']
    extension = cylinder['extension']
    # Branches
    nc = Rad.shape[0]  # number of cylinders
    ns = int(np.max(Branch)) + 1  # number of segments (branches)
    # BData will store branch attributes: order, diameter, volume, area, length, angle, height, azimuth, zenith
    BData = np.zeros((ns, 12), dtype=np.float32)
    ind = np.arange(0, nc)
    # CiB will hold the list of cylinder indices
    CiB = [None] * ns

    # Loop over branches
    for i in range(0, ns):
        # Find all cylinder indices that belong to branch i.
        C = ind[Branch == i]
        CiB[i] = C
        if C.size > 0:
            first_idx = int(C[0])

            BData[i, 0] = BranchOrder[first_idx]  # Branch order
            BData[i, 1] = 2 * Rad[first_idx]  # Branch diameter
            # Branch volume
            BData[i, 2] = 1000 * math.pi * np.sum(Len[(C.astype(int))] * (Rad[(C.astype(int))] ** 2))
            # Branch area
            BData[i, 3] = 2 * math.pi * np.sum(Len[(C.astype(int))] * Rad[(C.astype(int))])
            # Branch length in meters: sum of lengths
            BData[i, 4] = np.sum(Len[(C.astype(int))])

            # Determine the cylinder used to compute the branch angle.
            # if the first cylinder is added to fill a gap, then use the second cylinder to compute the angle:
            if added[first_idx] and C.size > 1:
                FC = int(C[1])  # First cylinder in the branch
                PC = int(parent[first_idx])  # Parent cylinder of the branch
            else:
                FC = first_idx
                PC = int(parent[FC])
            if parent[FC] >= 0:
                dot_val = np.dot(Axe[FC, :], Axe[PC, :])
                # Clamp the value to avoid domain errors in acos
                dot_val = np.clip(dot_val, -1.0, 1.0)
                BData[i, 5] = 180 / math.pi * math.acos(dot_val)  # branch angle

            # Branch height
            BData[i, 6] = start[first_idx, 2] - start[0, 2]
            # Branch azimuth
            BData[i, 7] = 180 / np.pi * np.arctan2(Axe[C[0], 1], Axe[C[0], 0])  # branch azimuth
            BData[i, 8] = 180 / np.pi * np.arccos(Axe[C[0], 2]) 
            BData[i, 9] = start[first_idx,0]
            BData[i, 10] = start[first_idx,1]
            BData[i,11] = start[first_idx,2]
            # BData[i, 7] = 180 / math.pi * math.atan2(Axe[FC, 1], Axe[FC, 0])
            # # Branch zenith
            # BData[i, 8] = 180 / math.pi * math.acos(np.clip(Axe[FC, 2], -1.0, 1.0))

    BData = BData.astype(np.float32)

    # Branching structure (topology, parent-child relation)
    branch_dict = {}
    branch_dict['order'] = BData[:, 0].astype(np.uint8)
    # Parent branch numbers
    BPar = np.ones(ns, dtype=np.int64) * -1
    #print(BPar)
    # Children cylinder indices
    Chi = [None] * nc

    for i in range(0, nc):
        c = ind[parent == i]
        c = c[c != extension[i]]
        Chi[i] = c
    # Assign parent branch for each branch based on child cylinder relationships.
    for i in range(0, ns):
        C = CiB[i]  # cylinders belonging to branch i
        if C.size > 0:
            # Collect children from each cylinder in branch i
            child_cyls = np.array([], dtype=int)
            for cyl in C:
                if cyl >= 0:  # cylinder have children
                    child_cyls = np.concatenate((child_cyls, Chi[int(cyl)]))
            child_cyls = np.unique(child_cyls)
            if child_cyls.size > 0:
                CB = np.unique(Branch[(child_cyls.astype(int))])
                # Set the parent branch for each child branch to be i
                for cb in CB:
                    BPar[int(cb)] = i

    ### FY note: convert to unit16 will change -1 to 65535
    #if ns <= 2 ** 16:
    #    branch_parent = BPar.astype(np.uint16)
    #else:
    #   branch_parent = BPar.astype(np.uint32)
    branch_dict['parent'] = BPar
    #branch_dict['parent'] = branch_parent

    # Assign the remaining branch attributes from BData.
    branch_dict['diameter'] = BData[:, 1]  # in meters
    branch_dict['volume'] = BData[:, 2]  # in liters
    branch_dict['area'] = BData[:, 3]  # in square meters
    branch_dict['length'] = BData[:, 4]  # in meters
    branch_dict['angle'] = BData[:, 5]  # in degrees
    branch_dict['height'] = BData[:, 6]  # in meters
    branch_dict['azimuth'] = BData[:, 7]  # in degrees
    branch_dict['zenith'] = BData[:, 8]  # in degrees
    branch_dict['x'] = BData[:,9]
    branch_dict['y'] = BData[:,10]
    branch_dict['z'] = BData[:,11]

    return branch_dict
