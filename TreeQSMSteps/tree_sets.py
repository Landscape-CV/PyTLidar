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
from scipy.spatial.distance import cdist
try:
    from ..Utils import Utils
except ImportError:
    import Utils.Utils as Utils
def tree_sets(P:np.ndarray,cover:dict,inputs:dict,segment=None):
    """
    Defines the location of the base of the trunk on the first pass, and the main branches on the second

    Args:
        P(np.Array): Point cloud
        cover(dictionary): Cover sets, their centers and neighbors
        PatchDiam(float)     Minimum diameter of the cover sets
        OnlyTree(boolean)      Logical value indicating if the point cloud contains only
                        points from the tree to be modelled -- Feature is not implemented in this version
        segment(dictionary)       Previous segments
    
     Returns:
     ( Dictionary, np.Array, np.Array):     Cover sets with updated neigbors
           Base of the trunk (the cover sets forming the base)
           Cover sets not part of the tree
    """
    # Define auxiliary object
    aux = {}
    aux['nb'] = len(cover['center'])  # number of cover sets
    aux['Fal'] = [False] * aux['nb']  # Initialize with False values
    aux['Ind'] = np.array(list(range( aux['nb'])))  # Index list from 1 to aux.nb
    aux['Ce'] = P[cover['center'], :3]  # Coordinates of the center points 
    aux['Hmin'] = aux['Ce'][:, 2].min()  # Minimum height (Z-coordinate)
    aux['Height'] = aux['Ce'][:, 2].max() - aux['Hmin']  # Height range

    # Define the base of the trunk and the forbidden sets
    if segment is None:  # Assuming args is a list containing arguments passed to the function
        Base, Forb, cover = define_base_forb(P, cover, aux, inputs)
    else:
        inputs['OnlyTree'] = True
        Base, Forb, cover = define_base_forb(P, cover, aux, inputs, segment)

    # Define the trunk (and the main branches)
    if segment is None:
        Trunk, cover = define_trunk(cover, aux, Base, Forb, inputs)
    else:
        Trunk, cover = define_main_branches(cover, segment, aux, inputs)

    # Update neighbor-relation to make the whole tree connected
    cover, Forb = make_tree_connected(cover, aux, Forb, Base, Trunk, inputs)
    return cover, Base, Forb
def define_base_forb(P, cover, aux, inputs, segment=None):
    """
    Defines the base of base of the trunk and any points that should be ignored (Forb)
    Used by tree_sets function only, not a standalone function. 
    """
    Ce = aux['Ce']
    
    if inputs['OnlyTree'] and segment is None:
        # No ground in the point cloud, the base is the lowest part
        BaseHeight = min(1.5, 0.02 * aux['Height'])
        I = Ce[:, 2] < aux['Hmin'] + BaseHeight
        Base = aux['Ind'][I]
        Forb = aux['Fal']

        # Ensure the base is not in multiple parts
        Wb = np.max(np.max(Ce[Base, :2]) - np.min(Ce[Base, :2]))
        Wt = np.max(np.max(Ce[:, :2]) - np.min(Ce[:, :2]))
        k = 1
        while k <= 5 and Wb > 0.3 * Wt:
            BaseHeight -= 0.05
            BaseHeight = max(BaseHeight, 0.05)
            if BaseHeight > 0:
                I = Ce[:, 2] < aux['Hmin'] + BaseHeight
            else:
                I = np.argmin(Ce[:, 2])  # Find the index of the minimum height
            Base = aux['Ind'][I]
            Wb = np.max(np.max(Ce[Base, :2]) - np.min(Ce[Base, :2]))
            k += 1

    elif inputs['OnlyTree']:
        # Select the stem sets from the previous segmentation and define the base
        BaseHeight = min(1.5, 0.02 * aux['Height'])
        SoP = segment['SegmentOfPoint'][cover['center']]
        stem = aux['Ind'][SoP == 0]
        I = Ce[stem, 2] < aux['Hmin'] + BaseHeight
        Base = stem[I]
        Forb = aux['Fal']
    
    else:
        
        raise NotImplementedError("This Feature has not been implemented") 

    return np.array(Base).astype(int), Forb, cover

def define_trunk(cover, aux, Base, Forb, inputs):
    """
    Determines the trunk of the tree by expanding the base upwards through connnected cover sets.
    Used by tree_sets function only, not a standalone function.
    """
    Nei = np.array(cover['neighbor'],dtype = 'object')
    Ce = aux['Ce']
    
    # Initialize the Trunk with the base
    Trunk = np.array(aux['Fal'])
    Trunk[Base] = True
    Forb = np.array(Forb)
    # Expand Trunk from the base above with neighbors as long as possible
    Exp = Base  # the current "top" of Trunk
    Exp = Utils.unique_elements_array(np.concatenate([Nei[i] for i in Exp]),np.array(aux['Fal'])).astype(int)
    I = Trunk[Exp]
    J = Forb[Exp]
    Exp = Exp[~(I | J)]  # Only non-forbidden sets that are not already in Trunk
    Trunk[Exp] = True  # Add the expansion Exp to Trunk
    
    L = 0.25  # maximum height difference in Exp from its top to bottom
    H = np.max(Ce[Trunk, 2]) - L  # the minimum bottom height for the current Exp
    
    FirstMod = True
    while len(Exp) > 0:
        # Expand Trunk similarly as above as long as possible
        H0 = H
        Exp0 = Exp
        Exp = np.union1d(Exp, np.concatenate([Nei[i] for i in Exp]))
        I = Trunk[Exp]
        Exp = Exp[~I]
        I = Ce[Exp, 2] >= H
        Exp = Exp[I]
        Trunk[Exp] = True
        if len(Exp) > 0:
            H = np.max(Ce[Exp, 2]) - L
        
        # If the expansion Exp is empty and the top of the tree is still over 5 meters higher
        if len(Exp) == 0 or H < H0 + inputs['PatchDiam1'] / 2:

            if H < aux['Height'] - 5:


                if FirstMod:
                    FirstMod = False
                    
                    # The vertices of the rectangle containing C
                    Min = np.min(Ce[:, :2], axis=0)
                    Max = np.max(Ce[:, :2], axis=0)
                    nb = Ce.shape[0]

                    # Number of rectangles with edge length "E" in the plane
                    EdgeLenth = 0.2
                    NRect = np.ceil((Max - Min) / EdgeLenth).astype(int) + 1

                    # Calculates the rectangular-coordinates of the points
                    px = np.floor((Ce[:, 0] - Min[0]) / EdgeLenth).astype(int) + 1
                    py = np.floor((Ce[:, 1] - Min[1]) / EdgeLenth).astype(int) + 1

                    # Sorts the points according to lexicographical order
                    LexOrd = np.dot(np.column_stack([px, py - 1]), [1, NRect[0]])
                    
                    SortOrd = np.argsort(LexOrd)
                    LexOrd.sort()
                    Partition = np.empty((NRect[0], NRect[1]), dtype=object)

                    p = 0  
                    while p < nb:
                        t = 1
                        while (p + t < nb) and (LexOrd[p] == LexOrd[p + t]):
                            t += 1
                        q = SortOrd[p]
                        J = SortOrd[p:p + t]
                        Partition[px[q] -1, py[q]-1 ] = J  
                        p += t
                        

                if len(Exp) > 0:
                    Region = Exp
                else:
                    Region = Exp0
                
                # Select the minimum and maximum rectangular coordinates of the region
                X1 = np.min(px[Region])
                if X1 <= 2:
                    X1 = 3
                X2 = np.max(px[Region])
                if X2 >= NRect[0] - 1:
                    X2 = NRect[0] - 2
                Y1 = np.min(py[Region])
                if Y1 <= 2:
                    Y1 = 3
                Y2 = np.max(py[Region])
                if Y2 >= NRect[1] - 1:
                    Y2 = NRect[1] - 2

                # Select the sets in the 2-meter layer above the region
                sets = Partition[int(X1) - 3:int(X2) + 2, int(Y1) - 3:int(Y2) + 2].flatten()
                sets = np.concatenate([s for s in sets if s is not None])
                K = np.array(aux['Fal'])
                K[sets] = True  # the potential sets
                I = Ce[:, 2] > H
                J = Ce[:, 2] < H + 2
                I = I & J & K
                I[Trunk] = False  # Must be non-Trunk sets
                SetsAbove = aux['Ind'][I]

                # Search the closest connection between Region and SetsAbove
                if len(SetsAbove) > 0:
                    # Compute the distances and cosines of the connections
                    n = len(Region)
                    m = len(SetsAbove)
                    Dist = np.zeros((n, m))
                    Cos = np.zeros((n, m))
                    for i in range(n):
                        V = Ce[SetsAbove, :] - Ce[Region[i], :]
                        Len = np.sum(V**2, axis=1)
                        v = V / np.sqrt(Len[:, np.newaxis])
                        Dist[i, :] = Len
                        Cos[i, :] = v[:, 2]

                    I = Cos > 0.7  # Select those connections with large enough cosines
                    # If not any, search with smaller cosines
                    t = 0
                    while not np.any(I):
                        t += 1
                        I = Cos > 0.7 - t * 0.05

                    # Search the minimum distance
                    Dist[~I] = 3
                    if n > 1 and m > 1:
                        d, I = np.min(Dist, axis=0), np.argmin(Dist, axis=0)
                        _, J = np.min(d), np.argmin(d)
                        I = I[J]
                    elif n == 1 and m > 1:
                        _, J = np.min(Dist, axis=1), np.argmin(Dist, axis=1)[0]
                        I = 0
                    elif m == 1 and n > 1:
                        _, I = np.min(Dist, axis=0), np.argmin(Dist, axis=0)[0]
                        J = 0
                    else:
                        I = 0  # The set in component to be connected
                        J = 0  # The set in "trunk" to be connected

                    # Join to "SetsAbove"
                    I = Region[I]
                    J = SetsAbove[J]
                    # Make the connection
                    
                    Nei[I] = np.append(Nei[I], J) 
                    Nei[J] = np.append(Nei[J], I)

                    # Expand "Trunk" again
                    Exp = np.union1d(Region, np.concatenate([Nei[r] for r in Region]))
                    I = Trunk[Exp]
                    Exp = Exp[~I]
                    I = Ce[Exp, 2] >= H
                    Exp = Exp[I]
                    Trunk[Exp] = True
                    H = np.max(Ce[Exp, 2]) - L

    cover['neighbor'] = Nei
    return Trunk, cover


def define_main_branches(cover, segment, aux, inputs):
    """Determines location of the primary branches of tree.
    Used by tree_sets function only, not a standalone function.
    """
    Bal = cover['ball']

    Nei = np.array([np.array(n, dtype=int) for n in cover['neighbor']], dtype=object)
    Ce = aux['Ce']
    Fal = np.array(aux['Fal'], dtype=bool)
    # Initialize the main branches
    nb = len(Bal)
    MainBranches = np.zeros(nb, dtype=int)-1
    SegmentOfPoint = segment['SegmentOfPoint']
    
    # Determine which branch indexes define the main branches
    max_segment = np.max(SegmentOfPoint) if SegmentOfPoint.size > 0 else 0
    MainBranchIndexes = np.zeros(max_segment + 1, dtype=bool)
    MainBranchIndexes[0] = True  # 0-based stem index
    
    # Unified branch index handling
    branch_attrs = ['branch1indexes', 'branch2indexes', 'branch3indexes']
    for attr in branch_attrs:
        idxs = segment.get(attr, np.array([], dtype=int))
        if idxs.size > 0:
            MainBranchIndexes[idxs] = True


    for i in range(nb):
        branch_ind = SegmentOfPoint[cover['ball'][i]]
        branch_ind = branch_ind[branch_ind != -1]  # Filter zeros
        if branch_ind.size > 0:
            ind = np.min(branch_ind)
            if MainBranchIndexes[ind]:
                MainBranches[i] = ind

    Trunk = Fal.copy()
    Trunk[MainBranches > -1] = True

    Par, CC, _ = Utils.cubical_partition(Ce, 3*inputs['PatchDiam2Max'], 10, return_cubes=False)

    BI = np.max(MainBranches) if MainBranches.size > 0 else 0
    N = Par.shape
    ind = 0
    for i in range(BI+1):
        Sets = np.zeros(aux['nb'], dtype=np.int32)
        if MainBranchIndexes[i]:
            Branch = (MainBranches == i)
            Comps, cs = Utils.connected_components_array(Nei, Branch, 1, Fal)
            n_comps = len(Comps)
            
            while n_comps > 1:
                
                for j in range(n_comps):
                    comp = Comps[j].copy()
                    NC = len(comp)
                    c = np.unique(CC[comp.astype(int)], axis=0)
                    m = c.shape[0]
                    t = 0
                    NearSets = np.array([], dtype=int)
                    
                    # Expand search area until nearby sets are found
                    while NearSets.size == 0:
                        NearSets = np.zeros(len(aux['Fal']), dtype=bool)
                        t += 1


                        

                        for k in range(m):
                            ind+=1
                            x1 = max(1,c[k,0]-t)-1
                            x2 = min(c[k,0]+t,N[0])
                            y1 = max(1,c[k,1]-t)-1
                            y2 = min(c[k,1]+t,N[1])
                            z1 = max(1,c[k,2]-t)-1
                            z2 = min(c[k,2]+t,N[2])
                            
                            # Collect all sets in the expanded partition
                            part = Par[x1:x2, y1:y2, z1:z2]
                            balls0 = np.array([p for p in part.flatten() if p is not None ],dtype = 'object')

                            if t==1:
                                
                                balls = np.concatenate([b for b in balls0]).astype(int)
                            else:
                                
                                # Filter non-empty cells and compute lengths
                                S = np.array([c.size for c in balls0], dtype=int)
                                I = S > 0
                                S = S[I]
                                balls0 = balls0[I]
                                balls = np.concatenate([balls,np.concatenate([b for b in balls0]).astype(int)])

                                
                            if balls.size > 0:
                                I = Branch[balls]
                                balls = balls[I]
                                NearSets[balls] = True
                        
                        # Exclude current component and get indices
                        NearSets[comp.astype(int)] = False
                        NearSets = aux["Ind"][NearSets]
                    
                    # Find closest connection between component and nearby sets
                    if NearSets.size == 0:
                        continue
                    
                    d = cdist(Ce[comp.astype(int)], Ce[NearSets])
                    if NC == 1 and len(NearSets) == 1:
                        IU = 0  # the set in component to be connected
                        JU = 0  # the set in "trunk" to be connected
                    elif NC==1:
                        d, JU = np.min(d), np.argmin(d)
                        IU = 0
                    elif len(NearSets)==1:
                        d, IU = np.min(d), np.argmin(d)
                        JU = 0
                    else:
                        d, IU = np.min(d,axis = 0), np.argmin(d, axis = 0)
                        dt, JU = np.min(d), np.argmin(d)
                        IU = IU[JU]
                        
                    
                    # Connect the closest pair
                    I = comp[IU].astype(int)
                    J = NearSets[JU]
                    Nei[I]= np.append(Nei[I],J)
                    Nei[J]= np.append(Nei[J],I)
                
                # Recompute components after connections
                Comps, cs = Utils.connected_components_array(Nei, Branch, 1, Fal)
                n_comps = len(Comps)
    
    
    Stem = MainBranches == 0
    Stem = aux["Ind"][Stem]
    # Create MainBranchIndexes array
    max_segment = np.max(SegmentOfPoint)
    MainBranchIndexes = np.zeros(max_segment + 1, dtype=bool)  # 1-based indexing
    MainBranchIndexes[segment["branch1indexes"]] = True

    # Determine BI
    if segment["branch1indexes"].size == 0:
        BI = 0
    else:
        BI = np.max(segment["branch1indexes"])

    # Process main branches
    for i in range(1, BI+1):
        if MainBranchIndexes[i]:
            Branch = MainBranches == i
            Branch = aux["Ind"][Branch]
            if Branch.size>0:
                Neighbors = MainBranches[np.concatenate(Nei[Branch])] == 0
                if not np.any(Neighbors):
                    d = cdist(Ce[Branch, :], Ce[Stem, :])
                    if len(Branch) > 1 and len(Stem) > 1:
                        d, I = np.min(d,axis = 0), np.argmin(d, axis = 0)
                        dt, J = np.min(d), np.argmin(d)
                        I = I[J]
                    elif len(Branch) == 1 and len(Stem) > 1:
                        d, J = np.min(d), np.argmin(d)
                        I = 0
                    elif len(Stem) == 1 and len(Branch) > 1:
                        d, I = np.min(d), np.argmin(d)
                        J = 0
                    elif len(Branch) == 1 and len(Stem) == 1:
                        I = 0  # the set in component to be connected
                        J = 0  # the set in "trunk" to be connected

                    # Join the Branch to Stem
                    I = Branch[I]
                    J = Stem[J]
                    Nei[I] = np.append(Nei[I], J)
                    Nei[J] = np.append(Nei[J], I)
    
    
#     % Check if the trunk is still in mutliple components and select the bottom
#     % component to define "Trunk":
    comps, cs = Utils.connected_components_array(Nei, Trunk,1, Fal)
    comps = np.array(comps,dtype='object')
    if len(cs) > 1:
        I = np.argsort(-cs)
        sorted_comps = comps[I]
        Stem = (MainBranches == 0)  # 0-based stem
        Trunk[:] = False
        
        for comp in sorted_comps:
            if np.any(Stem[comp]):
                Trunk[comp] = True
                break

    cover['neighbor'] = Nei
    return Trunk, cover


def make_tree_connected(cover, aux, Forb, Base, Trunk, inputs):
    """Connects unconnected parts of the point cloud to the nearest sets.
    Used by tree_sets function only, not a standalone function.
    """
    Nei = np.array(cover["neighbor"],dtype='object')
    Ce = aux["Ce"]
    Forb = np.array(Forb)
    # Expand trunk as much as possible
    Trunk[Forb] = False
    Exp = Trunk.copy()

    while np.any(Exp):
        Exp[np.concatenate(Nei[Exp])] = True
        Exp[Trunk] = False
        Exp[Forb] = False
        Exp[Base] = False
        Trunk[Exp] = True
    Fal = np.array(aux["Fal"])
    # Define "Other", sets not yet connected to trunk or Forb
    Other = np.invert( Fal)
    Other[Forb] = False
    Other[Trunk] = False
    Other[Base] = False

    # Determine parameters on the extent of the "Nearby Space" and acceptable component size
    k0 = min(10, np.ceil(0.2 / inputs["PatchDiam1"]))
    k = k0
    if inputs["OnlyTree"]:
        Cmin = 0
    else:
        Cmin = np.ceil(0.1 / inputs["PatchDiam1"])  # minimum accepted component size

    # Determine the components of "Other"
    if np.any(Other):
        Comps, _comp_size = Utils.connected_components_array(Nei, Other, 1, Fal)
        nc = len(Comps)
        NonClassified = np.ones(nc, dtype=bool)
    else:
        NonClassified = np.zeros(0, dtype=bool)

    bottom = np.min(Ce[Base, 2])
    # repeat search and connecting as long as "Other" sets exists
    nc_ind = 0
    while np.any(NonClassified):
        nc_ind+=1
        npre = np.count_nonzero(NonClassified)  # number of "Other" sets before new connections
        again = True  # check connections again with same "distance" if true

        # Partition the centers of the cover sets into cubes with size k*dmin
        Par,CC,_Info  = Utils.cubical_partition(Ce, k * inputs["PatchDiam1"],return_cubes=False)
        Par = np.array(Par,dtype = 'object')
        Neighbors = [None] * nc
        Sizes = np.zeros((nc, 2))
        Pass = np.ones(nc, dtype=bool)
        first_round = True
        again_ind=0
        while again:
            again_ind+=1
            # Check each component: part of "Tree" or "Forb"
            for i in range(nc):
                # print(nc_ind, again_ind,i)
                if NonClassified[i] and Pass[i]:
                    comp = Comps[i].astype(int)#np.concatenate([arr for arr in Comps[i]])   # candidate component for joining to the tree

                    # If the component is neighbor of forbidden sets, remove it
                    J = Forb[np.concatenate([Nei[i] for i in comp])]
                    if np.any(J):
                        NonClassified[i] = False
                        Forb[comp] = True
                        Other[comp] = False
                    else:
                        # Otherwise check nearest sets for a connection
                        NC = len(comp)
                        if first_round:
                            # Select the cover sets the nearest to the component
                            c = np.unique(CC[comp, :], axis=0)
                            m = len(c)
                            B = [None] * m
                            for j in range(m):
                                balls = Par[c[j, 0]-2 : c[j, 0]+1, 
                                     c[j, 1]-2 : c[j, 1]+1,
                                     c[j, 2]-2 : c[j, 2]+1]
                                _balls = np.array([b for b in balls.flatten() if b is not None ],dtype = 'object')
                                # _balls = []
                                # for x in balls:
                                #     for y in x:
                                #         for z in y:
                                #             if z is not None:
                                #                 _balls.append(z)
                                B[j] = np.concatenate(_balls)#np.concatenate([np.concatenate([np.array(r) for r in ball ]) for ball in balls if ball is not None])
                            NearSets = np.concatenate([np.array(row) for row in B]).astype(int)
                            # Only the non-component cover sets
                            Fal[comp] = True
                            I = Fal[NearSets].copy()
                            NearSets = NearSets[~I].copy()
                            Fal[comp] = False
                            NearSets = np.unique(NearSets)
                            Neighbors[i] = NearSets.copy()
                            if len(NearSets) == 0:
                                Pass[i] = False
                            # No "Other" sets
                            I = Other[NearSets]
                            NearSets = NearSets[~I].copy()
                        else:
                            NearSets = Neighbors[i].copy()
                            # No "Other" sets
                            I = Other[NearSets]
                            NearSets = NearSets[~I].copy()

                        # Select different class from NearSets
                        I = Trunk[NearSets]
                        J = Forb[NearSets]
                        trunk = NearSets[I].copy()  # "Trunk" sets
                        forb = NearSets[J].copy()  # "Forb" sets
                        if len(trunk) != Sizes[i, 0] or len(forb) != Sizes[i, 1]:
                            Sizes[i, :] = [len(trunk), len(forb)]

                            # If large component is tall and close to ground, then
                            # search the connection near the component's bottom
                            if NC > 100:
                                hmin = np.min(Ce[comp, 2])
                                H = np.max(Ce[comp, 2]) - hmin
                                if H > 5 and hmin < bottom + 5:
                                    I = Ce[NearSets, 2] < hmin + 0.5
                                    NearSets = NearSets[I].copy()
                                    I = Trunk[NearSets]
                                    J = Forb[NearSets]
                                    trunk = NearSets[I]  # "Trunk" sets
                                    forb = NearSets[J]  # "Forb" sets

                            # Determine the closest sets for "trunk"
                            if len(trunk) > 0:
                                d = cdist(Ce[comp, :], Ce[trunk, :])
                                if NC == 1 and len(trunk) == 1:
                                    dt = d
                                    IC = 0  # the set in component to be connected
                                    IT = 0  # the set in "trunk" to be connected
                                elif NC == 1:
                                    dt, IT = np.min(d), np.argmin(d)
                                    IC = 0
                                elif len(trunk) == 1:
                                    dt, IC = np.min(d), np.argmin(d)
                                    IT = 0
                                else:
                                    d, IC = np.min(d,axis = 0), np.argmin(d, axis = 0)
                                    dt, IT = np.min(d), np.argmin(d)
                                    IC = IC[IT]
                            else:
                                dt = 700

                            # Determine the closest sets for "forb"
                            if len(forb) > 0:
                                d = cdist(Ce[comp, :], Ce[forb, :])
                                df = np.min(d)
                                try:
                                    if len(d) > 1 and len(df)>1:
                                        df = np.min(df)
                                except TypeError:
                                    pass
                            else:
                                df = 1000

                            # Determine what to do with the component
                            if (dt > 12 and dt < 100) or (NC < Cmin and dt > 0.5 and dt < 10):
                                # Remove small isolated component
                                Forb[comp] = True
                                Other[comp] = False
                                NonClassified[i] = False
                            elif 3 * df < dt or (df < dt and df > 0.25):
                                # Join the component to "Forb"
                                Forb[comp] = True
                                Other[comp] = False
                                NonClassified[i] = False
                            elif (df == 1000 and dt == 700) or dt > k * inputs["PatchDiam1"]:
                                # Isolated component, do nothing
                                pass
                            else:
                                # Join to "Trunk"
                                I = comp[IC]
                                J = trunk[IT]
                                Other[comp] = False
                                Trunk[comp] = True
                                NonClassified[i] = False
                                # make the connection
                                Nei[I] = np.concatenate([Nei[I], [J]])
                                Nei[J] = np.concatenate([Nei[J], [I]])

            first_round = False
            # If "Other" has decreased, do another check with same "distance"
            if np.count_nonzero(NonClassified) < npre:
                again = True
                npre = np.count_nonzero(NonClassified)
            else:
                again = False

        k += k0  # increase the cell size of the nearby search space
        Cmin *= 3  # increase the acceptable component size

    Forb[Base] = False
    cover["neighbor"] = Nei

    return cover, Forb