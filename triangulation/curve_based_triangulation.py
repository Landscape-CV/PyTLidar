
"""% This file is part of TREEQSM.
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

This derivative work is released under the GNU General Public License (GPL)."""

import numpy as np
from scipy.spatial import Delaunay
from triangulation import initial_boundary_curve
from triangulation import boundary_curve
from triangulation import check_self_intersection
from Utils.Utils import cubical_partition
def curve_based_triangulation(P, TriaHeight, TriaWidth):
    """
    Reconstructs a triangulation for the stem-buttress surface based on boundary curves

    Inputs:
        P             Point cloud of the stem to be triangulated
        TriaHeight    Height of the triangles
        TriaWidth     Width of the triangles

    Output:
        triangulation  Dictionary containing triangulation data with keys:
                       'vert', 'facet', 'fvd', 'volume', 'bottom', 'top', 'triah', 'triaw'
    """

    # Initialize variables
    np_points = len(P)
    I = np.argsort(P[:, 2])[::-1]
    P = P[I, :]
    
    Hbot = np.mean(P[-100:, 2])
    Htop = P[0, 2]
    N = int(np.ceil((Htop - Hbot) / TriaHeight))

    Vert = np.zeros((int(1e5), 3))
    Tria = np.zeros((int(1e5), 3), dtype=int)
    TriaLay = np.zeros((int(1e5),), dtype=int)
    VertLay = np.zeros((int(1e5),), dtype=np.uint16)
    
    Curve = np.zeros((0, 3))
    i = 0  # the layer whose cross-section is under reconstruction
    ps = 0

    while P[ps, 2] > Htop - i * TriaHeight:
        ps += 1
    pe = ps

    while i < N / 4 and Curve.size == 0:
        i += 1
        ps = pe + 1
        k = 1
        while P[ps + k, 2] > Htop - i * TriaHeight:
            k += 1
        pe = ps + k - 1
        PSection = P[ps:pe, :]

        # Create initial boundary curve
        iter = 0
        while iter <= 15 and Curve.size == 0:
            iter += 1
            Curve = initial_boundary_curve(PSection, TriaWidth)

    if Curve.size == 0:
        
        triangulation = np.zeros((0, 1))
        #might want to raise exception instead
        print('No triangulation: Problem with the first curve')
        return triangulation

    # Make the height of the curve even
    Curve[:, 2] = np.max(Curve[:, 2])

    # Save vertices
    nv = len(Curve)  # number of vertices in the curve
    Vert[:nv, :] = Curve
    VertLay[:nv] = i
    t = 0
    m00 = len(Curve)

    # Further logic for the triangulation layers
    i0 = i
    i = i0 + 1
    nv0 = 0
    LayerBottom = Htop - i * TriaHeight

    while i <= N and pe < np_points:
        ps = pe + 1
        k = 1
        while ps + k <= np_points and P[ps + k, 2] > LayerBottom:
            k += 1
        pe = ps + k - 1
        PSection = P[ps:pe, :]

        if i > i0+1:
            nv0 = nv1
        # Define seed points
        Curve[:, 2] = Curve[:, 2] - TriaHeight
        Curve0 = Curve

        # Create new boundary curve
        Curve, Ind = boundary_curve(PSection, Curve, 2 * TriaWidth, 1.5 * TriaWidth)

        if Curve.size == 0:
            triangulation  = np.zeros((0, 1))
            print('No triangulation: Empty curve')
            return triangulation

        Curve[:, 2] = np.max(Curve[:, 2])

        # Check for self-intersection
        Intersect, IntersectLines = check_self_intersection(Curve[:, :2])

        # If self-intersects, try modifying the curve
        j = 1
        while Intersect and j <= 10:
            n = len(Curve)
            CrossLines = np.arange(1, n + 1)
            NumberOfIntersections = np.array([len(x) for x in IntersectLines[:, 0]])
            I = NumberOfIntersections > 0
            CrossLines = CrossLines[I]
            CrossLen = np.concatenate([IntersectLines[I, 1]])

            if len(CrossLen) == len(CrossLines):
                LineEle = np.roll(Curve, -1, axis=0) - Curve
                d = np.linalg.norm(LineEle, axis=1)
                m = len(CrossLines)
                for k in range(0, m, 2):
                    if CrossLines[k] != n:
                        Curve[CrossLines[k] + 1, :] = Curve[CrossLines[k], :] + 0.9 * CrossLen[k] / d[CrossLines[k]] * LineEle[CrossLines[k], :]
                    else:
                        Curve[0, :] = Curve[CrossLines[k], :] + 0.9 * CrossLen[k] / d[CrossLines[k]] * LineEle[CrossLines[k], :]
                Intersect, IntersectLines = check_self_intersection(Curve[:, :2])
                j += 1
            else:
                j = 11

        m = len(Curve)
        if Intersect:
            # Handle self-intersection cases
            H = Curve0[0, 2] - Hbot
            if H > 0.75:
                triangulation = np.zeros((0, 1))
                print(f'No triangulation: Self-intersection at {H} m from the bottom')
                return triangulation

            Curve = Curve0
            Curve[:, 2] = Curve[:, 2] - TriaHeight
            Nadd = int(np.floor(H / TriaHeight) + 1)
            m = len(Curve)
            Ind = np.column_stack((np.arange(m), np.roll(np.arange(m), -1)))

            T = H / Nadd
            for k in range(1, Nadd + 1):
                if k > 1:
                    Curve[:, 2] = Curve[:, 2] - T

                Vert[nv:nv + m, :] = Curve
                VertLay[nv:nv + m] = i

                nv1 = nv
                nv += m
                t0 = t + 1
                pass_flag = False
                for j in range(m):
                    if Ind[j,1] > 0 and j < m:
                        t = t+1
                        Tria[t,:] = [nv1+j ,nv0+Ind[j,:]]
                        t = t+1
                        Tria[t,:] = [nv1+j ,nv0+Ind[j,1], nv1+j+1]
                    elif Ind[j,1] > 0 and not pass_flag:
                        t = t+1
                        Tria[t,:] = [nv1+j, nv0+Ind[j,:]]
                        t = t+1
                        Tria[t,:] = [nv1+j, nv0+Ind[j,1], nv1+1]
                    elif Ind[j,1] == 0 and j < m:
                        t = t+1
                        Tria[t,:] = [nv1+j, nv0+Ind[j,0], nv1+j+1]
                    elif Ind[j,1] == 0 and not pass_flag:
                        t = t+1
                        Tria[t,:] = [nv1+j, nv0+Ind[j,0], nv1+1]
                    elif j == 1 and Ind[j,1] == -1:
                        t = t+1
                        Tria[t,:] = [nv, nv1, nv0+1]
                        t = t+1
                        Tria[t,:] = [nv, nv0+1 ,nv1+1]
                        t = t+1
                        Tria[t,:] = [nv0+1 ,nv0+2 ,nv1+1]
                        t = t+1
                        Tria[t,:] = [nv1+1, nv0+2, nv0+3]
                        t = t+1
                        Tria[t,:] = [nv1+1, nv0+3 ,nv1+2]
                        pass_flag = True
                    elif Ind[j,1] == -1 and j < m:
                        t = t+1
                        Tria[t,:] = [nv1+j ,nv0+Ind[j,0], nv0+Ind[j,0]+1]
                        t = t+1
                        Tria[t,:] = [nv1+j, nv0+Ind[j,0]+1 ,nv1+j+1]
                        t = t+1
                        Tria[t,:] = [nv0+Ind[j,1]+1, nv0+Ind[j,1]+2 ,nv1+j+1]
                    elif Ind[j,1] == -1 and not pass_flag:
                        t = t+1
                        Tria[t,:] = [nv1+j ,nv0+Ind[j,0] ,nv0+Ind[j,0]+1]
                        t = t+1
                        Tria[t,:] = [nv1+j ,nv0+Ind[j,0]+1 ,nv1+1]
                        t = t+1
                        Tria[t,:] = [nv0+Ind[j,0]+1, nv0+1 ,nv1+1]
                TriaLay[t0:t] = i
                i += 1
                nv0 = nv1

            i = N+1

        else:
            # Handle no self-intersection cases
            C = np.intersect1d(Curve0, Curve, axis=0)
            if C.shape[0] > 0.7 * Curve.shape[0]:
                N = i

            # If the boundary curve has grown much longer than originally, decrease the triangle height
            if m > 3 * m00:
                TriaHeight = TriaHeight / 2  # use half the height
                N = N + np.ceil((N - i) / 2).astype(int)  # update the number of layers
                m00 = m

            # Define the triangulation between two boundary curves
            nv1 = nv
            nv = nv + m
            t0 = t + 1
            pass_ = False
            for j in range(m):
                if Ind[j, 1] > 0 and j < m - 1:
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, :]]
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 1], nv1 + j + 1]
                elif Ind[j, 1] > 0 and not pass_:
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, :]]
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 1], nv1 + 1]
                elif Ind[j, 1] == 0 and j < m - 1:
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0], nv1 + j + 1]
                elif Ind[j, 1] == 0 and not pass_:
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0], nv1 + 1]
                elif j == 0 and Ind[j, 1] == -1:
                    t = t + 1
                    Tria[t, :] = [nv, nv1, nv0 + 1]
                    t = t + 1
                    Tria[t, :] = [nv, nv0 + 1, nv1 + 1]
                    t = t + 1
                    Tria[t, :] = [nv0 + 1, nv0 + 2, nv1 + 1]
                    t = t + 1
                    Tria[t, :] = [nv1 + 1, nv0 + 2, nv0 + 3]
                    t = t + 1
                    Tria[t, :] = [nv1 + 1, nv0 + 3, nv1 + 2]
                    pass_ = True
                elif Ind[j, 1] == -1 and j < m - 1:
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0], nv0 + Ind[j, 0] + 1]
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0] + 1, nv1 + j + 1]
                    t = t + 1
                    Tria[t, :] = [nv0 + Ind[j, 0] + 1, nv0 + Ind[j, 0] + 2, nv1 + j + 1]
                elif Ind[j, 1] == -1 and not pass_:
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0], nv0 + Ind[j, 0] + 1]
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0] + 1, nv1 + 1]
                    t = t + 1
                    Tria[t, :] = [nv0 + Ind[j, 0] + 1, nv0 + 1, nv1 + 1]

            # Update TriaLay array
            TriaLay[t0:t] = i

            # Increment the iteration index and update LayerBottom
            i = i + 1
            LayerBottom = LayerBottom - TriaHeight

    # Clean up and format the output
    Vert = Vert[:nv, :]
    VertLay = VertLay[:nv]
    Tria = Tria[:t, :]
    TriaLay = TriaLay[:t]

    a = round(t / 10)  # select the top triangles
    U = Vert[Tria[:a, 1], :] - Vert[Tria[:a, 0], :]
    V = Vert[Tria[:a, 2], :] - Vert[Tria[:a, 0], :]
    Center = np.mean(Vert[:nv - 1, :], axis=0)  # the center of the stem
    C = Vert[Tria[:a, 0], :] + 0.25 * V + 0.25 * U
    W = C[:, :2] - Center[:2]  # vectors from the triangles to the stem's center
    Normals = np.cross(U, V)
    if np.count_nonzero(np.sum(Normals[:, :2] * W, axis=1) < 0) > 0.5 * len(C):
        Tria[:t, :2] = Tria[:t, [1, 0]]

    # Remove possible double triangles
    nt = len(Tria)
    Keep = np.ones(nt, dtype=bool)
    Scoord = Vert[Tria[:, 0], :] + Vert[Tria[:, 1], :] + Vert[Tria[:, 2], :]
    S = np.sum(Scoord, axis=1)

    part, CC = cubical_partition(Scoord, 2 * TriaWidth)

    for j in range(nt - 1):
        if Keep[j]:
            points = part[CC[j, 0] - 1:CC[j, 0] + 1, CC[j, 1] - 1:CC[j, 1] + 1, CC[j, 2] - 1:CC[j, 2] + 1]
            points = np.vstack(points)
            I = S[j] == S[points]
            J = points != j
            I = I & J & Keep[points]
            if np.any(I):
                p = points[I]
                I = np.intersect1d(Tria[j, :], Tria[p, :])
                if len(I) == 3:
                    Keep[p] = False

    Tria = Tria[Keep, :]
    TriaLay = TriaLay[Keep]

    # Triangles of the ground layer
    N = float(np.max(VertLay))
    I = VertLay == N
    Vert[I, 2] = Hbot
    ind = np.arange(1, nv + 1)
    ind = ind[I]
    Curve = Vert[I, :]  # Boundary curve of the bottom
    n = len(Curve)
    if n < 10:
        triangulation = np.zeros((0, 1))
        print('No triangulation: Ground layer boundary curve too small')
        return
    
    C = np.zeros((n, 2), dtype=int)
    C[:, 0] = np.arange(1, n + 1)
    C[:-1, 1] = np.arange(2, n + 1)
    C[-1, 1] = 1
    dt = Delaunay(Curve[:, :2])  # Delaunay triangulation for the bottom
    In = dt.is_interior
    GroundTria = dt.simplices[In]#may need to check on this
    Points = dt.points
    if Points.shape[0] > Curve.shape[0]:
        print('No triangulation: Problem with Delaunay in the bottom layer')
        triangulation = np.zeros((0, 1))
        return
    
    GroundTria0 = GroundTria
    GroundTria[:, 0] = ind[GroundTria[:, 0]]
    GroundTria[:, 1] = ind[GroundTria[:, 1]]
    GroundTria[:, 2] = ind[GroundTria[:, 2]]

    # Compute the normals and areas
    U = Curve[GroundTria0[:, 1], :] - Curve[GroundTria0[:, 0], :]
    V = Curve[GroundTria0[:, 2], :] - Curve[GroundTria0[:, 0], :]
    Cg = Curve[GroundTria0[:, 0], :] + 0.25 * V + 0.25 * U
    Ng = np.cross(U, V)
    I = Ng[:, 2] > 0  # Check orientation
    Ng[I, :] = -Ng[I, :]
    Ag = 0.5 * np.sqrt(np.sum(Ng * Ng, axis=1))
    Ng = 0.5 * np.column_stack([Ng[:, 0] / Ag, Ng[:, 1] / Ag, Ng[:, 2] / Ag])

    # Remove possible negative area triangles
    I = Ag > 0
    Ag = Ag[I]
    Cg = Cg[I, :]
    Ng = Ng[I, :]
    GroundTria = GroundTria[I, :]

    # Update the triangles
    Tria = np.vstack([Tria, GroundTria])
    TriaLay = np.concatenate([TriaLay, (N + 1) * np.ones(GroundTria.shape[0], dtype=int)])

    # Check triangulation validity
    if np.abs(np.sum(Ag) - np.polyarea(Curve[:, 0], Curve[:, 1])) > 0.001 * np.sum(Ag):
        print('No triangulation: Problem with Delaunay in the bottom layer')
        triangulation = np.zeros((0, 1))
        return

    # Triangles of the top layer
    N = float(np.min(VertLay))
    I = VertLay == N
    ind = np.arange(1, nv + 1)
    ind = ind[I]
    Curve = Vert[I, :]
    CenterTop = np.mean(Curve, axis=0)

    n = len(Curve)
    C = np.zeros((n, 2), dtype=int)
    C[:, 0] = np.arange(1, n + 1)
    C[:-1, 1] = np.arange(2, n + 1)
    C[-1, 1] = 1
    dt = Delaunay(Curve[:, :2])
    Points = dt.points
    if dt.vertices.shape[0] == 0 or Points.shape[0] > Curve.shape[0]:
        print('No triangulation: Problem with Delaunay in the top layer')
        triangulation = np.zeros((0, 1))
        return
    In = dt.is_interior
    TopTria = dt.simplices[In]
    TopTria0 = TopTria
    TopTria[:, 0] = ind[TopTria[:, 0]]
    TopTria[:, 1] = ind[TopTria[:, 1]]
    TopTria[:, 2] = ind[TopTria[:, 2]]

    # Compute the normals and areas
    U = Curve[TopTria0[:, 1], :] - Curve[TopTria0[:, 0], :]
    V = Curve[TopTria0[:, 2], :] - Curve[TopTria0[:, 0], :]
    Ct = Curve[TopTria0[:, 0], :] + 0.25 * V + 0.25 * U
    Nt = np.cross(U, V)
    I = Nt[:, 2] < 0
    Nt[I, :] = -Nt[I, :]
    At = 0.5 * np.sqrt(np.sum(Nt * Nt, axis=1))
    Nt = 0.5 * np.column_stack([Nt[:, 0] / At, Nt[:, 1] / At, Nt[:, 2] / At])

    # Remove possible negative area triangles
    I = At > 0
    At = At[I]
    Ct = Ct[I, :]
    Nt = Nt[I, :]
    TopTria = TopTria[I, :]

    # Update the triangles
    Tria = np.vstack([Tria, TopTria])
    TriaLay = np.concatenate([TriaLay, N * np.ones(TopTria.shape[0], dtype=int)])

    # Triangles of the side
    B = (TriaLay <= np.max(VertLay)) & (TriaLay > 1)
    U = Vert[Tria[B, 1], :] - Vert[Tria[B, 0], :]
    V = Vert[Tria[B, 2], :] - Vert[Tria[B, 0], :]
    Cs = Vert[Tria[B, 0], :] + 0.25 * V + 0.25 * U
    Ns = np.cross(U, V)
    As = 0.5 * np.sqrt(np.sum(Ns * Ns, axis=1))
    Ns = 0.5 * np.column_stack([Ns[:, 0] / As, Ns[:, 1] / As, Ns[:, 2] / As])
    I = As > 0
    Ns = Ns[I, :]
    As = As[I]
    Cs = Cs[I, :]

    # Volumes in liters
    VTotal = np.sum(At * np.sum(Ct * Nt, axis=1)) + np.sum(As * np.sum(Cs * Ns, axis=1)) + np.sum(Ag * np.sum(Cg * Ng, axis=1))
    VTotal = round(10000 * VTotal / 3) / 10

    if VTotal < 0:
        print('No triangulation: Problem with volume')
        triangulation = np.zeros((0, 1))

    # Final triangulation output
    V = Vert[Tria[:, 0], :2] - CenterTop[:2]
    fvd = np.sqrt(np.sum(V * V, axis=1))
    triangulation = {
        'vert': Vert.astype(np.float32),
        'facet': Tria.astype(np.uint16),
        'fvd': fvd.astype(np.float32),
        'volume': VTotal,
        'SideArea': np.sum(As),
        'BottomArea': np.sum(Ag),
        'TopArea': np.sum(At),
        'bottom': np.min(Vert[:, 2]),
        'top': np.max(Vert[:, 2]),
        'triah': TriaHeight,
        'triaw': TriaWidth
    }

    return triangulation