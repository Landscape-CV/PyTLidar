"""
Python adaptation and extension of TREEQSM:

Creates cover sets (surface patches) and their neighbor-relation for a point cloud

Version: 0.0.1
Date: Feb 19 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

from numba import jit
import numpy as np
try:
    from ..Utils import Utils
except ImportError:
    import Utils.Utils as Utils
# import csv
import time
import torch 

def cover_sets(P, inputs, RelSize=None, qsm = True, device = 'cpu', full_point_data = None):
    """
    Creates cover sets (surface patches) and their neighbor-relation for a point cloud

    Args:
        P (numpy.ndarray): Point cloud
        inputs: Input structure, the following fields are needed:
            PatchDiam1   Minimum distance between centers of cover sets; i.e. the
                         minimum diameter of cover set in uniform covers. Does
                         not need nor use the third optional input "RelSize".
            PatchDiam2Min   Minimum diameter of cover sets for variable-size
                            covers. Needed if "RelSize" is given as input.
            PatchDiam2Max   Maximum diameter of cover sets for variable-size
                            covers. Needed if "RelSize" is given as input.
            BallRad1    Radius of the balls used to generate the uniform cover.
                        These balls are also used to determine the neighbors
            BallRad2    Maximum radius of the balls used to generate the
                        variable-size cover.
            nmin1, nmin2    Minimum number of points in a BallRad1- and
                            BallRad2-balls
        RelSize: Relative cover set size for each point

    Returns:
        dictionary: Dictionary containing the following fields:
            ball        Cover sets, (n_sets x 1)-cell
            center      Center points of the cover sets, (n_sets x 1)-vector
            neighbor    Neighboring cover sets of each cover set, (n_sets x 1)-cell
    """
    if device == 'cpu' and P.dtype != np.float64:
        P = P.astype(np.float64)
        np_points = P.shape[0]  # number of points
    else:
        np_points = len(P)

    if RelSize is None:
        return uniform_cover(P, inputs, np_points, qsm, device, full_point_data)
    else:
        return variable_cover(P, inputs, RelSize, np_points)


def uniform_cover(P, inputs, np_points, qsm =True, device = 'cpu', full_point_data = None):
    """
    Creates uniform cover sets and neighbor-relation of a point cloud using fixed-radius balls

    Args:
        P (numpy.ndarray): Point cloud
        inputs: Input structure, the following fields are needed:
            PatchDiam1   Minimum distance between centers of cover sets; i.e. the
                         minimum diameter of cover set in uniform covers. Does
                         not need nor use the third optional input "RelSize".
            BallRad1    Radius of the balls used to generate the uniform cover.
                        These balls are also used to determine the neighbors
            nmin1   Minimum number of points in a BallRad1 ball
        np_points (int): The total number of points in the point cloud

    Returns:
        Dictionary: Dictionary containing the following fields:
            ball        Cover sets, (n_sets x 1)-cell
            center      Center points of the cover sets, (n_sets x 1)-vector
            neighbor    Neighboring cover sets of each cover set, (n_sets x 1)-cell
    """
    BallRad = float(inputs['BallRad1'])
    PatchDiamMax = float(inputs['PatchDiam1'])
    nmin = int(inputs['nmin1'])

    # Partition, CC, Info, Cubes = Utils.cubical_partition(P, BallRad,return_cubes=True)  # Partition the point cloud into cubes for quick neighbor search
    Partition, CC, Info = Utils.cubical_partition(P, BallRad,return_cubes=False)
    Partition = np.array(Partition, dtype = 'object')
    
    NotExa = np.ones(np_points, dtype=bool)  # the points not yet examined
    Dist = np.full(np_points, 1e8)  # distance of point to the closest center
    BoP = np.zeros(np_points, dtype=np.int64)  # the balls/cover sets the points belong
    Ball = []  # Large balls for generation of the cover sets and their neighbors
    Cen = []  # the center points of the balls/cover sets
    nb = 0  # number of sets generated

    # random permutation of points, produces different covers for the same inputs:
    # np.random.seed(0)
    rg = np.random.Generator(np.random.Philox(0))
    RandPerm = rg.permutation(np_points)
    # Generate the balls
    Radius_sq = BallRad ** 2
    MaxDist_sq = (PatchDiamMax) ** 2

    for _i,i in enumerate(RandPerm):
        if NotExa[i]:  # point not yet examined
            Q = i
            cube = CC[Q]  # get cube coordinates

            points = Partition[CC[Q,0]-2:CC[Q,0]+1,
                               CC[Q,1]-2:CC[Q,1]+1,
                               CC[Q,2]-2:CC[Q,2]+1]
            
            points = np.concatenate([p for p in points.flatten() if p is not None ])
            # #print(points)
            if len(points) == 0:
                continue
            # Compute distances of the points to the seed
            V = P[points] - P[Q]
            dist = np.sum(V ** 2, axis=1)
            # Select the points inside the ball
            inside = dist < Radius_sq
            ball_points = points[inside]  # the points forming the ball
            #print(ball_points)
            if len(ball_points) >= nmin:
                d = dist[inside]  # the distances of the ball's points
                core = d < MaxDist_sq  # the core points of the cover set
                NotExa[ball_points[core]] = False  # mark points as examined
                # Define the new ball
                nb += 1
                Ball.append(ball_points)
                Cen.append(Q)
                # Update the cover sets the points belong to and their distances to the closest seed
                closer = d < Dist[ball_points]
                closer_idx = np.where(closer)[0]
                ball_closer = ball_points[closer_idx]
                BoP[ball_closer] = nb
                Dist[ball_closer] = d[closer_idx]
    # Create cover sets
    cover = create_cover(Ball, Cen, BoP, nb, np_points)
    return cover
    

def variable_cover(P, inputs, RelSize, np_points):
    """
    Creates variable cover sets and neighbor-relation of a point cloud using variable-radius balls

    Args:
        P (numpy.ndarray): Point cloud
        inputs: Input structure, the following fields are needed:
            PatchDiam2Min   Minimum diameter of cover sets for variable-size
                            covers. Needed if "RelSize" is given as input.
            PatchDiam2Max   Maximum diameter of cover sets for variable-size
                            covers. Needed if "RelSize" is given as input.
            BallRad2    Maximum radius of the balls used to generate the
                        variable-size cover.
            nmin2   Minimum number of points in a BallRad2 ball
        np_points (int): The total number of points in the point cloud

    Returns:
        cover: Structure array containing the following fields:
            ball        Cover sets, (n_sets x 1)-cell
            center      Center points of the cover sets, (n_sets x 1)-vector
            neighbor    Neighboring cover sets of each cover set, (n_sets x 1)-cell
    """
    BallRad = float(inputs['BallRad2'])
    PatchDiamMin = float(inputs['PatchDiam2Min'])
    PatchDiamMax = float(inputs['PatchDiam2Max'])
    nmin = int(inputs['nmin2'])
    MRS = PatchDiamMin / PatchDiamMax
    # Calculate minimum radius
    r = 1.5 * (np.min(RelSize) / 256 * (1 - MRS) + MRS) * BallRad + 1e-5
    NE = 1 + int(np.ceil(BallRad / r))  # Number of empty edge layers
    if NE > 4:
        r = PatchDiamMax / 4
        NE = 1 + int(np.ceil(BallRad / r))

    # Partition, CC, Info, Cubes = Utils.cubical_partition(P, r, NE, return_cubes = True)
    Partition, CC, Info = Utils.cubical_partition(P, BallRad,return_cubes=False)
    Partition = np.array(Partition, dtype = 'object')
    NotExa = np.ones(np_points, dtype=bool)
    NotExa[RelSize == 0] = False
    Dist = np.full(np_points, 1e8)  # distance of point to the closest center
    BoP = np.zeros(np_points, dtype=np.int64)  # the balls/cover sets the points belong
    Ball = []  # Large balls for generation of the cover sets and their neighbors
    Cen = []  # the center points of the balls/cover sets
    nb = 0  # number of sets generated

     # Simplified permutation for small sets first
    

    RandPerm = np.argsort(RelSize) 
    e = BallRad - PatchDiamMax
    ind = 0
    for i in RandPerm:
        ind+=1
        if NotExa[i]:
            Q = i  # the index of the center/seed point of the current cover set
            # Compute the set size and the cubical neighborhood of the seed point
            rs = (RelSize[Q] / 256) * (1 - MRS) + MRS  # relative radius
            MaxDist = PatchDiamMax * rs  # diameter of the cover set
            Radius = MaxDist + np.sqrt(rs) * e  # radius of the ball including the cover set
            N = int(np.ceil(Radius / r))  # = number of cube cells needed to be included in the ball
            points = Partition[CC[Q,0]-N-1:CC[Q,0]+N,
                               CC[Q,1]-N-1:CC[Q,1]+N,
                               CC[Q,2]-N-1:CC[Q,2]+N]
            
            if len(points.flatten()) == 0:
                continue
            points = np.concatenate([p for p in points.flatten() if p is not None ])
            if len(points) == 0:
                continue
            # Compute the distance of the "points" to the seed:
            V = P[points] - P[Q]
            dist = np.sum(V ** 2, axis=1)
            Radius_sq = Radius ** 2
            # Select the points inside the ball:
            inside = dist < Radius_sq
            ball_points = points[inside]
            if len(ball_points) >= nmin:
                d = dist[inside]  # the distances of the ball's points
                core = d < (MaxDist ** 2)  # the core points of the cover set
                NotExa[ball_points[core]] = False  # mark points as examined
                # define new ball:
                nb += 1
                Ball.append(ball_points)
                Cen.append(Q)
                # Select which points belong to this ball, i.e. are closer to this
                # seed than previously tested seeds:
                closer = d < Dist[ball_points]  # which points are closer to this seed
                closer_idx = np.where(closer)[0]
                ball_closer = ball_points[closer_idx]
                BoP[ball_closer] = nb
                Dist[ball_closer] = d[closer_idx]

    cover = create_cover(Ball, Cen, BoP, nb, np_points)
    return cover


@jit(nopython=True,cache=True)
def create_neighbors(Ball,BoP,nb):
    """Helper Function
        Creates neighbor relation for cover sets
        Separated out for numba compilation
    """
    # Nei = [np.array([],dtype=np.int64) for _ in range(nb)]
    Nei=[]
    
    for i in range(nb):
        B = Ball[i]  # the points in the big ball of cover set "i"
        bops = BoP[B]
        mask = (bops != (i + 1))
        N = bops[mask]  # the points of B not in the cover set "i"
        N = np.unique(N)#unique_elements_array(N,Fal)#
        N = N[N != 0]
        Nei.append(N - 1)
    #print(Nei)
    # Make the relation symmetric by adding, if needed, A as B's neighbor in the case B is A's neighbor
    for i in range(nb):
        for j in Nei[i]:
            if i not in Nei[j]:
                Nei[j]=np.append(Nei[j],i)

    
    return Nei
@jit(nopython=True,cache=True)
def create_PointsInSets(nb,np_points,BoP):
    """
    Generates array of points in each cover set
    Separated out for numba compilation 
    """
    Num = np.zeros(nb, dtype=np.int64)  # number of points in each ball
    Ind = np.zeros(np_points, dtype=np.int64)  # index of each point in its ball
    for i in range(np_points):
        bop = BoP[i]
        if bop > 0:
            Num[bop - 1] += 1
            Ind[i] = Num[bop - 1]
    # Initialization of the "PointsInSets"
    PointsInSets = []
    for i in range(nb):
        PointsInSets.append(np.zeros(Num[i], dtype=np.int64))
    # Define the "PointsInSets"
    for i in range(np_points):
        bop = BoP[i]
        if bop > 0:
            idx = bop - 1
            pos = Ind[i] - 1
            PointsInSets[idx][pos] = i

    return PointsInSets

def create_cover(Ball, Cen, BoP, nb, np_points):
    """

    Args:
        Ball (list): Large balls for generation of the cover sets and their neighbors
        Cen (list): the center points of the balls/cover sets
        BoP (numpy.ndarray): the balls/cover sets the points belong
        nb (int): number of sets generated
        np_points (int): The total number of points in the point cloud

    Returns:
        dictionary: Dictionary containing the following fields:
            ball        Cover sets, (n_sets x 1)-cell
            center      Center points of the cover sets, (n_sets x 1)-vector
            neighbor    Neighboring cover sets of each cover set, (n_sets x 1)-cell

    """
    if len(Ball) ==0:
        cover = {'ball': Ball,
        'center': np.array(Cen, dtype=np.int64),
        'sets':BoP.copy()-1}
        
        return cover
    PointsInSets=create_PointsInSets(nb,np_points,BoP)
    Nei=create_neighbors(Ball,BoP,nb)
    Nei = np.array([np.array(neighbors).astype(np.int64) for neighbors in Nei],dtype=object)
    cover = {
        'ball': PointsInSets,
        'center': np.array(Cen, dtype=np.int64),
        'neighbor': Nei,
        'sets':BoP.copy()-1
    }




   

    return cover
