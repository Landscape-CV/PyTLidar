
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
from TreeQSMSteps.triangulation import initial_boundary_curve
from TreeQSMSteps.triangulation import boundary_curve
from TreeQSMSteps.triangulation import check_self_intersection
from Utils.Utils import cubical_partition


def _polyarea(x, y):
    """Shoelace area of the polygon defined by the ordered vertices (x, y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _triangulate_polygon(pts):
    """Ear-clipping triangulation of a simple polygon given by ordered 2D
    vertices ``pts`` (n x 2). Returns an (k x 3) array of 0-based indices into
    ``pts`` (the original vertex order) and a boolean ``ok`` that is True when
    the polygon was fully triangulated into n-2 triangles (i.e. it is a simple,
    non self-intersecting polygon). This replaces MATLAB's constrained
    delaunayTriangulation + isInterior for the horizontal cap layers; the
    covered region and hence the enclosed volume are identical even though the
    individual facets differ from MATLAB's Delaunay result."""
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n < 3:
        return np.zeros((0, 3), dtype=int), False

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def point_in_tri(p, a, b, c):
        d1 = cross(a, b, p)
        d2 = cross(b, c, p)
        d3 = cross(c, a, p)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    idx = list(range(n))
    # Ensure counter-clockwise orientation so that convex (ear) vertices have
    # a positive cross product.
    sa = np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))
    if sa < 0:
        idx = idx[::-1]

    tris = []
    guard = 0
    while len(idx) > 3 and guard < 100000:
        guard += 1
        ear = False
        L = len(idx)
        for k in range(L):
            i0 = idx[(k - 1) % L]
            i1 = idx[k]
            i2 = idx[(k + 1) % L]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 0:  # reflex or degenerate
                continue
            good = True
            for j in idx:
                if j == i0 or j == i1 or j == i2:
                    continue
                if point_in_tri(pts[j], a, b, c):
                    good = False
                    break
            if good:
                tris.append([i0, i1, i2])
                del idx[k]
                ear = True
                break
        if not ear:
            break
    if len(idx) == 3:
        tris.append([idx[0], idx[1], idx[2]])
    Tri = np.array(tris, dtype=int) if tris else np.zeros((0, 3), dtype=int)
    ok = (Tri.shape[0] == n - 2)
    return Tri, ok


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
    
    Hbot = np.mean(P[-101:, 2])
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
        PSection = P[ps:pe + 1, :]

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
        while ps + k < np_points and P[ps + k, 2] > LayerBottom:
            k += 1
        pe = ps + k - 1
        PSection = P[ps:pe + 1, :]

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
            CrossLines = np.arange(n)
            NumberOfIntersections = np.array([len(x) for x in IntersectLines[:, 0]])
            I = NumberOfIntersections > 0
            CrossLines = CrossLines[I]
            CrossLen = np.concatenate(list(IntersectLines[I, 1])) if np.any(I) else np.array([])

            if len(CrossLen) == len(CrossLines):
                LineEle = np.roll(Curve, -1, axis=0) - Curve
                d = np.linalg.norm(LineEle, axis=1)
                m = len(CrossLines)
                for k in range(0, m, 2):
                    if CrossLines[k] != n - 1:
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
            # 1-based curve-point indices so that the sentinel 0 (no second
            # connection) and -1 (removed point) do not collide with a valid
            # wrap-around target.
            Ind = np.column_stack((np.arange(1, m + 1),
                                   np.concatenate([np.arange(2, m + 1), [1]])))

            T = H / Nadd
            for k in range(1, Nadd + 1):
                if k > 1:
                    Curve[:, 2] = Curve[:, 2] - T

                Vert[nv:nv + m, :] = Curve
                VertLay[nv:nv + m] = i

                nv1 = nv
                nv += m
                t0 = t
                pass_flag = False
                for j in range(m):
                    if Ind[j,1] > 0 and j < m-1:
                        Tria[t,:] = [nv1+j, nv0+Ind[j,0]-1, nv0+Ind[j,1]-1]
                        t = t+1
                        Tria[t,:] = [nv1+j, nv0+Ind[j,1]-1, nv1+j+1]
                        t = t+1
                    elif Ind[j,1] > 0 and not pass_flag:
                        Tria[t,:] = [nv1+j, nv0+Ind[j,0]-1, nv0+Ind[j,1]-1]
                        t = t+1
                        Tria[t,:] = [nv1+j, nv0+Ind[j,1]-1, nv1]
                        t = t+1
                    elif Ind[j,1] == 0 and j < m-1:
                        Tria[t,:] = [nv1+j, nv0+Ind[j,0]-1, nv1+j+1]
                        t = t+1
                    elif Ind[j,1] == 0 and not pass_flag:
                        Tria[t,:] = [nv1+j, nv0+Ind[j,0]-1, nv1]
                        t = t+1
                    elif j == 0 and Ind[j,1] == -1:
                        Tria[t,:] = [nv-1, nv1-1, nv0]
                        t = t+1
                        Tria[t,:] = [nv-1, nv0, nv1]
                        t = t+1
                        Tria[t,:] = [nv0, nv0+1, nv1]
                        t = t+1
                        Tria[t,:] = [nv1, nv0+1, nv0+2]
                        t = t+1
                        Tria[t,:] = [nv1, nv0+2, nv1+1]
                        t = t+1
                        pass_flag = True
                    elif Ind[j,1] == -1 and j < m-1:
                        Tria[t,:] = [nv1+j, nv0+Ind[j,0]-1, nv0+Ind[j,0]]
                        t = t+1
                        Tria[t,:] = [nv1+j, nv0+Ind[j,0], nv1+j+1]
                        t = t+1
                        Tria[t,:] = [nv0+Ind[j,0], nv0+Ind[j,0]+1, nv1+j+1]
                        t = t+1
                    elif Ind[j,1] == -1 and not pass_flag:
                        Tria[t,:] = [nv1+j, nv0+Ind[j,0]-1, nv0+Ind[j,0]]
                        t = t+1
                        Tria[t,:] = [nv1+j, nv0+Ind[j,0], nv1]
                        t = t+1
                        Tria[t,:] = [nv0+Ind[j,0], nv0, nv1]
                        t = t+1
                TriaLay[t0:t] = i
                i += 1
                nv0 = nv1

            i = N+1

        else:
            # Handle no self-intersection cases
            # Save the new curve's vertices
            Vert[nv:nv + m, :] = Curve
            VertLay[nv:nv + m] = i

            # If little change between Curve and Curve0, stop the reconstruction
            set0 = set(map(tuple, Curve0))
            nC = sum(1 for r in set(map(tuple, Curve)) if r in set0)
            if nC > 0.7 * Curve.shape[0]:
                N = i

            # If the boundary curve has grown much longer than originally, decrease the triangle height
            if m > 3 * m00:
                TriaHeight = TriaHeight / 2  # use half the height
                N = N + np.ceil((N - i) / 2).astype(int)  # update the number of layers
                m00 = m

            # Define the triangulation between two boundary curves
            nv1 = nv
            nv = nv + m
            t0 = t
            pass_ = False
            for j in range(m):
                if Ind[j, 1] > 0 and j < m - 1:
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0] - 1, nv0 + Ind[j, 1] - 1]
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 1] - 1, nv1 + j + 1]
                    t = t + 1
                elif Ind[j, 1] > 0 and not pass_:
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0] - 1, nv0 + Ind[j, 1] - 1]
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 1] - 1, nv1]
                    t = t + 1
                elif Ind[j, 1] == 0 and j < m - 1:
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0] - 1, nv1 + j + 1]
                    t = t + 1
                elif Ind[j, 1] == 0 and not pass_:
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0] - 1, nv1]
                    t = t + 1
                elif j == 0 and Ind[j, 1] == -1:
                    Tria[t, :] = [nv - 1, nv1 - 1, nv0]
                    t = t + 1
                    Tria[t, :] = [nv - 1, nv0, nv1]
                    t = t + 1
                    Tria[t, :] = [nv0, nv0 + 1, nv1]
                    t = t + 1
                    Tria[t, :] = [nv1, nv0 + 1, nv0 + 2]
                    t = t + 1
                    Tria[t, :] = [nv1, nv0 + 2, nv1 + 1]
                    t = t + 1
                    pass_ = True
                elif Ind[j, 1] == -1 and j < m - 1:
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0] - 1, nv0 + Ind[j, 0]]
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0], nv1 + j + 1]
                    t = t + 1
                    Tria[t, :] = [nv0 + Ind[j, 0], nv0 + Ind[j, 0] + 1, nv1 + j + 1]
                    t = t + 1
                elif Ind[j, 1] == -1 and not pass_:
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0] - 1, nv0 + Ind[j, 0]]
                    t = t + 1
                    Tria[t, :] = [nv1 + j, nv0 + Ind[j, 0], nv1]
                    t = t + 1
                    Tria[t, :] = [nv0 + Ind[j, 0], nv0, nv1]
                    t = t + 1

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

    part, CC, _ = cubical_partition(Scoord, 2 * TriaWidth, return_cubes=False)

    for j in range(nt - 1):
        if Keep[j]:
            nbr = part[CC[j, 0] - 1:CC[j, 0] + 2, CC[j, 1] - 1:CC[j, 1] + 2, CC[j, 2] - 1:CC[j, 2] + 2]
            cells = [c for c in nbr.ravel() if c is not None and len(c) > 0]
            if not cells:
                continue
            points = np.concatenate(cells).astype(int)
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
    ind = np.arange(nv)
    ind = ind[I]
    Curve = Vert[I, :]  # Boundary curve of the bottom
    n = len(Curve)
    if n < 10:
        triangulation = np.zeros((0, 1))
        print('No triangulation: Ground layer boundary curve too small')
        return triangulation

    # Triangulate the interior of the (simple, closed) bottom boundary polygon
    GroundTria, ok = _triangulate_polygon(Curve[:, :2])
    if not ok:
        print('No triangulation: Problem with Delaunay in the bottom layer')
        triangulation = np.zeros((0, 1))
        return triangulation

    GroundTria0 = GroundTria.copy()
    GroundTria[:, 0] = ind[GroundTria0[:, 0]]
    GroundTria[:, 1] = ind[GroundTria0[:, 1]]
    GroundTria[:, 2] = ind[GroundTria0[:, 2]]

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
    if np.abs(np.sum(Ag) - _polyarea(Curve[:, 0], Curve[:, 1])) > 0.001 * np.sum(Ag):
        print('No triangulation: Problem with Delaunay in the bottom layer')
        triangulation = np.zeros((0, 1))
        return triangulation

    # Triangles of the top layer
    N = float(np.min(VertLay))
    I = VertLay == N
    ind = np.arange(nv)
    ind = ind[I]
    Curve = Vert[I, :]
    CenterTop = np.mean(Curve, axis=0)

    n = len(Curve)
    TopTria, ok = _triangulate_polygon(Curve[:, :2])
    if TopTria.shape[0] == 0 or not ok:
        print('No triangulation: Problem with Delaunay in the top layer')
        triangulation = np.zeros((0, 1))
        return triangulation
    TopTria0 = TopTria.copy()
    TopTria[:, 0] = ind[TopTria0[:, 0]]
    TopTria[:, 1] = ind[TopTria0[:, 1]]
    TopTria[:, 2] = ind[TopTria0[:, 2]]

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

    if np.abs(np.sum(At) - _polyarea(Curve[:, 0], Curve[:, 1])) > 0.001 * np.sum(At):
        print('No triangulation: Problem with Delaunay in the top layer')
        triangulation = np.zeros((0, 1))
        return triangulation

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
        return triangulation

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