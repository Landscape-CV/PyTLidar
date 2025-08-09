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
Date: 21 Feb 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import numpy as np


def relative_size(P, cover, segment):
    """
    Determines relative cover set size for points in new covers
    Uses existing segmentation and its branching structure to determine
    relative size of the cover sets distributed over new covers. The idea is
    to decrease the relative size as the branch size decreases. This is
    realised so that the relative size at the base of a branch is
    proportional to the size of the stem's base, measured as number of
    cover sets in the first few layers. Also when we approach the
    tip of the branch, the relative size decreases to the minimum.
    Maximum relative size is 256 at the bottom of the
    stem and the minimum is 1 at the tip of every branch.

    Args:
        P (numpy.ndarray): Point cloud
        cover           : Structure array containing the following fields:
            ball            : Cover sets, (n_sets x 1)-cell
            center          : Center points of the cover sets, (n_sets x 1)-vector
            neighbor        : Neighboring cover sets of each cover set, (n_sets x 1)-cell
        segment         : Dictionary containing the following fields:
            segments        : Segments found, list of lists, each list contains the cover sets
            ParentSegment   : Parent segment of each segment, list of integers,
                              equals to zero if no parent segment
            ChildSegment    : Children segments of each segment, list of lists

    Returns:
        np.Array: Relative size (1-256), uint8-vector, (n_points x 1)
    """
    Bal = cover['ball']
    Cen = cover['center']
    Nei = cover['neighbor']
    Segs = segment['segments']
    SChi = segment['ChildSegment']
    np_points = P.shape[0]  # number of points
    ns = len(Segs)  # number of segments

    # Use branching order and height as apriori info
    # Determine branch orders
    Ord = np.zeros(ns, dtype=int)
    if SChi:
        C = SChi[0]  # children of the first segment (0-based)
    else:
        C = []
    order = 0
    while len(C)>0:
        ##############
        # FY: The next line I don't think make sense as order will remain 0 for all branches
        ##############
        order += order
        # order += 1  # FY: This might make more sense
        Ord[C] = order
        next_C = []
        for c in C:
            if c < len(SChi):
                next_C.extend(SChi[c])
        C = next_C

    maxO = order + 1  # maximum branching order (plus one)
    #print(Ord)

    # Determine tree height
    if Cen.size == 0:
        H = 0.0
    else:
        Top = np.max(P[Cen, 2])
        Bot = np.min(P[Cen, 2])
        H = Top - Bot

    # Determine "base size" compared to the stem base
    """
    BaseSize is the relative size of the branch base compared to the stem
    base, measured as number of cover sets in the first layers of the cover
    sets. If it is larger than apriori upper limit based on branching order
    and branch height, then correct to the apriori limit
    """
    BaseSize = np.zeros(ns)
    # Determine first the base size at the stem
    if ns > 0:
        S = Segs[0]
        n = len(S)
        if n >= 2:
            m = min(6, n)
            BaseSize[0] = np.mean([len(S[j]) for j in range(1, m)])
        elif n == 1:
            BaseSize[0] = len(S[0])

    # Then define base size for other segments
    for i in range(1, ns):
        S = Segs[i]
        n = len(S)
        if n >= 2:
            m = min(6, n)
            BaseSize[i] = int(np.ceil(np.mean([len(S[j]) for j in range(1, m)]) / BaseSize[0] * 256))
        elif n == 1:
            BaseSize[i] = len(S[0]) / BaseSize[0] * 256
        bot = np.min(P[Cen[S[0]], 2])
        h = bot - Bot  # height of the segment's base
        #print(f"h {h}")
        bs_term = 256 * ((maxO - Ord[i]) / maxO) * ((H - h) / H)
        BS = int(np.ceil(bs_term))  # maximum apriori base size
        if BaseSize[i] > BS:
            BaseSize[i] = BS
    BaseSize[0] = 256


    # Determine relative size for points
    TS = 1
    RS = np.zeros(np_points, dtype=np.uint8)
    for i in range(ns):
        S = Segs[i]
        s = len(S)
        if s == 0:
            continue
        for j in range(s):
            layer = S[j]
            for q in layer:
                points = Bal[q]
                if points.size == 0:
                    continue
                #print(f"basesize[i] {BaseSize[i]}")
                rs_val = BaseSize[i] - (BaseSize[i] - TS) * np.sqrt(j / s)
                rounded_rs_val = round(rs_val)
                if rs_val-rounded_rs_val == .5:
                    rounded_rs_val +=1          
                #print(rs_val)
                rs_val = max(0, min(255, int(rounded_rs_val)))  # unit8 only allows 0-255
                RS[points] = rs_val



    # Adjust the relative size at the base of child segments
    RS0 = RS.copy()
    for i in range(ns):
        C = SChi[i]
        if len(C)==0:
            continue
        for child_seg in C:
            if child_seg >= len(Segs):
                continue
            S_child = Segs[child_seg]
            if len(S_child) == 0:
                continue
            B = S_child[0] if len(S_child) > 0 else []
            if len( B)==0:
                continue
            # Get neighbors of B
            N = []
            for b in B:
                if b < len(Nei):
                    N.extend(Nei[b])
            N = list(set(N))
            # Remove points from S_child[1] if present
            if len(S_child) > 1:
                layer1 = S_child[1]
                layer1_set = set(layer1)
                N = [n for n in N if n not in layer1_set]
            # Combine B and N
            N_B = list(set(N) | set(B))
            # Get all points in these cover sets
            points = []
            for nb in N_B:
                if nb < len(Bal):
                    points.extend(Bal[nb])
            points = np.unique(points)
            #print(points)
            if points.size > 0:
                #added rounding to be the same as Matlab -- only important for testing
                adj_RS = RS0[points]/2
                rd_ars = np.floor(adj_RS)
                diff = np.abs(adj_RS-rd_ars)
                plus = (diff>=.5).astype(int)
                
                RS[points] = (rd_ars+plus).astype(np.uint8)


    return RS