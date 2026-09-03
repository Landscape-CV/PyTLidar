"""
Python adaptation and extension of TREEQSM:

Creates cover sets (surface patches) and their neighbor-relation for a point cloud

Version: 0.0.1
Date: Feb 19 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

from numba import jit, types
from numba.typed import List
import numpy as np
try:
    from ..Utils import Utils
except ImportError:
    import Utils.Utils as Utils
# import csv
import time


def _cube_index(P, EL, NE):
    """
    Cube coordinates of the points, the same way Utils.cubical_partition computes
    them, plus the points sorted by cube and the sorted cube keys, so a cube's
    points can be found by binary search instead of through a table of cubes.
    Returns (CubeCoord, N, order, keys).
    """
    P = np.array(P, dtype=float)
    Min = np.min(P, axis=0)
    Max = np.max(P, axis=0)
    N = np.ceil((Max - Min) / EL).astype(int) + 2 * NE + 1
    t = 0
    while t < 10 and 8 * np.prod(N) > 4e9:
        t += 1
        EL *= 1.1
        N = np.ceil((Max - Min) / EL).astype(int) + 2 * NE + 1
    if 8 * np.prod(N) > 4e9:
        NE = 3
        N = np.ceil((Max - Min) / EL).astype(int) + 2 * NE + 1
    CubeCoord = np.floor((P - Min) / EL).astype(int) + NE + 1
    order = np.lexsort((CubeCoord[:, 2], CubeCoord[:, 1], CubeCoord[:, 0]))
    keys = (CubeCoord[:, 0] * N[1] + CubeCoord[:, 1]) * N[2] + CubeCoord[:, 2]
    return CubeCoord, N, np.ascontiguousarray(order), np.ascontiguousarray(keys[order])


@jit(nopython=True, cache=True)
def _grow_balls(P, CC, N, order, keys, RandPerm, NotExa, nmin, uniform, Radius_sq_u,
                MaxDist_sq_u, RelSize, MRS, PatchDiamMax, e, r):
    """
    The seed loop shared by the uniform and the variable cover. For each seed
    the candidate points are the ones in the cube window around it, in cube
    order, and the ball is the candidates within the radius. Returns
    (Ball, Cen, BoP, nb).
    """
    n = P.shape[0]
    Dist = np.full(n, 1e8)
    BoP = np.zeros(n, dtype=np.int64)
    Ball = List.empty_list(types.int64[::1])
    Cen = List.empty_list(types.int64)
    nb = 0
    for Q in RandPerm:
        if not NotExa[Q]:
            continue
        if uniform:
            W = 1
            Radius_sq = Radius_sq_u
            MaxDist_sq = MaxDist_sq_u
        else:
            rs = RelSize[Q] / 256 * (1 - MRS) + MRS
            MaxDist = PatchDiamMax * rs
            Radius = MaxDist + np.sqrt(rs) * e
            W = int(np.ceil(Radius / r))
            Radius_sq = Radius * Radius
            MaxDist_sq = MaxDist * MaxDist
        xa = max(CC[Q, 0] - W, 1)
        xb = min(CC[Q, 0] + W, N[0])
        ya = max(CC[Q, 1] - W, 1)
        yb = min(CC[Q, 1] + W, N[1])
        za = max(CC[Q, 2] - W, 1)
        zb = min(CC[Q, 2] + W, N[2])
        cnt = 0
        for x in range(xa, xb + 1):
            for y in range(ya, yb + 1):
                for z in range(za, zb + 1):
                    key = (x * N[1] + y) * N[2] + z
                    cnt += np.searchsorted(keys, key, side='right') - np.searchsorted(keys, key, side='left')
        if cnt == 0:
            continue
        pts = np.empty(cnt, dtype=np.int64)
        d = np.empty(cnt)
        k = 0
        qx = P[Q, 0]
        qy = P[Q, 1]
        qz = P[Q, 2]
        for x in range(xa, xb + 1):
            for y in range(ya, yb + 1):
                for z in range(za, zb + 1):
                    key = (x * N[1] + y) * N[2] + z
                    lo = np.searchsorted(keys, key, side='left')
                    hi = np.searchsorted(keys, key, side='right')
                    for s in range(lo, hi):
                        p = order[s]
                        dx = P[p, 0] - qx
                        dy = P[p, 1] - qy
                        dz = P[p, 2] - qz
                        pts[k] = p
                        d[k] = dx * dx + dy * dy + dz * dz
                        k += 1
        m = 0
        for k in range(cnt):
            if d[k] < Radius_sq:
                m += 1
        if m < nmin:
            continue
        ball_points = np.empty(m, dtype=np.int64)
        dist = np.empty(m)
        m = 0
        for k in range(cnt):
            if d[k] < Radius_sq:
                ball_points[m] = pts[k]
                dist[m] = d[k]
                m += 1
        for k in range(m):
            if dist[k] < MaxDist_sq:
                NotExa[ball_points[k]] = False
        nb += 1
        Ball.append(ball_points)
        Cen.append(Q)
        for k in range(m):
            p = ball_points[k]
            if dist[k] < Dist[p]:
                BoP[p] = nb
                Dist[p] = dist[k]
    return Ball, Cen, BoP, nb

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

    # Empty input would crash cubical_partition's np.min
    if np_points == 0:
        return {'ball': [], 'center': np.array([], dtype=np.int64), 'sets': np.array([], dtype=np.int64)}

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

    P = np.ascontiguousarray(P, dtype=np.float64)
    CC, N, order, keys = _cube_index(P, BallRad, 3)

    NotExa = np.ones(np_points, dtype=bool)  # the points not yet examined

    # random permutation of points, produces different covers for the same inputs:
    RandPerm = np.random.permutation(np_points).astype(np.int64)
    # Generate the balls
    Radius_sq = BallRad ** 2
    MaxDist_sq = (PatchDiamMax) ** 2

    Ball, Cen, BoP, nb = _grow_balls(P, CC, N, order, keys, RandPerm, NotExa, nmin, True,
                                     Radius_sq, MaxDist_sq, np.zeros(1, dtype=np.uint8),
                                     0.0, 0.0, 0.0, 1.0)
    # Create cover sets
    cover = create_cover(list(Ball), list(Cen), BoP, nb, np_points)
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

    P = np.ascontiguousarray(P, dtype=np.float64)
    CC, N, order, keys = _cube_index(P, r, NE)
    NotExa = np.ones(np_points, dtype=bool)
    NotExa[RelSize == 0] = False

    # Define random permutation of points (results in different covers for
    # same input) so that first small sets are generated. The order is
    # randomized within three RelSize buckets (<=32, 33-128, >128).
    RelSizeArr = np.asarray(RelSize)
    ind = np.arange(np_points)
    I1 = ind[RelSizeArr <= 32]
    I2 = ind[(RelSizeArr <= 128) & (RelSizeArr > 32)]
    I3 = ind[RelSizeArr > 128]
    RandPerm = np.concatenate([
        np.random.permutation(I1),
        np.random.permutation(I2),
        np.random.permutation(I3),
    ]).astype(np.int64)
    e = BallRad - PatchDiamMax
    Ball, Cen, BoP, nb = _grow_balls(P, CC, N, order, keys, RandPerm, NotExa, nmin, False,
                                     0.0, 0.0, np.ascontiguousarray(RelSizeArr), MRS,
                                     PatchDiamMax, e, r)

    cover = create_cover(list(Ball), list(Cen), BoP, nb, np_points)
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
