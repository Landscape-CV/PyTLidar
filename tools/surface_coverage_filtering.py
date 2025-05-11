"""
Python adaptation and extension of TREEQSM:

Filters a point cloud based on the assumption that it samples a cylinder

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
Authors: Fan Yang, John Hagood, Amir Hossein Alikhah Mishamandani
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import numpy as np
from Utils.Utils import distances_to_line
from tools.orthonormal_vectors import orthonormal_vectors
from numba import jit

@jit(nopython=True)
def surface_coverage_filtering(P, axis,start,length, lh, ns):
    """
    Filter a 3d-point cloud based on given cylinder (axis and radius) by
    dividing the point cloud into "ns" equal-angle sectors and "lh"-height
    layers along the axis. For each sector-layer intersection (a region in
    the cylinder surface) keep only the points closest to the axis.

    Inputs:
    P             Point cloud, (n_points x 3)-matrix
    c             Cylinder, stucture array with fields "axis", "start",
                    "length"
    lh            Height of the layers
    ns            Number of sectors

    Returns:
    Pass          Logical vector indicating which points pass the filtering
    c             Cylinder, stucture array with additional fields "radius",
                    "SurfCov", "mad", "conv", "rel", estimated from the
                    filtering
    """
    # Compute the distances, heights and angles of the points
    
    d, V, h, B = distances_to_line(P, axis, start)
    h = h - np.min(h)
    U, W = orthonormal_vectors(axis)
    #print(U)
    #V_proj = np.dot(V, np.column_stack((U, W)))
    V_proj = V @ np.column_stack((U, W))
    ang = np.arctan2(V_proj[:, 1], V_proj[:, 0]) + np.pi
    #print(ang)

    # Initial layer and sector computation
    nl_initial = max(int(np.ceil(length / lh)), 1)
    Layer = np.ceil(h / length * nl_initial).astype(np.int64)
    Layer = np.clip(Layer, 1, nl_initial)
    Sector = np.ceil(ang / (2 * np.pi) * ns).astype(np.int64)
    Sector = np.clip(Sector, 1, ns)
    #print(Sector)

    # Sort based on lexicographic order of (sector,layer)
    LexOrd = Layer + (Sector - 1) * nl_initial  # Equivalent to MATLAB's [Layer Sector-1]*[1 nl]'
    SortOrd = np.argsort(LexOrd)
    LexOrd = LexOrd[SortOrd]
    ds = d[SortOrd]
    #print(SortOrd)
    #print(LexOrd)
    #print(ds)

    # Estimate the distances for each sector-layer intersection
    Dis = np.zeros((nl_initial, ns))
    #print(nl_initial, ns)
    #print(LexOrd // nl_initial)
    np_points = P.shape[0]
    p = 0
    max_ns=np.int64(36)
    min_ns=np.int64(8)
    while p < np_points:
        t = 1
        while (p + t < np_points) and (LexOrd[p] == LexOrd[p + t]):
            t =t+ 1
        D = np.min(ds[p:p + t])
        current_Layer = LexOrd[p] % nl_initial+1
        # if current_Layer == 0:
        #     current_Layer = 1
        current_Sector = (LexOrd[p] - current_Layer) // nl_initial + 1
        #print(current_Layer, current_Sector)
        Dis[current_Layer - 1, current_Sector - 1] = min(1.05 * D, D + 0.02)
        p =p+ t
    #print(Dis)
    # Compute the number of sectors (new ns and nl) based on estimated radius
    Dis=Dis.flatten()
    non_zero_Dis = Dis[Dis > 0]
    if len(non_zero_Dis) == 0:
        R = 0.0
    else:
        R = np.median(non_zero_Dis)
    a = max(0.02, 0.2 * R)
    ns_new = int(np.ceil(2 * np.pi * R / a))
    ns_new = np.maximum(min_ns,np.minimum(ns_new,max_ns))#np.clip(ns_new, 8, 36)
    nl_new = int(np.ceil(length / a))
    nl_new = np.maximum(nl_new, 3)
    #print(ns_new, nl_new)

    # Recompute layers and sectors with new ns and nl
    Layer_new = np.ceil(h / length * nl_new).astype(np.int64)
    Layer_new = np.minimum(Layer_new,nl_new)#np.clip(Layer_new, 1, nl_new)
    Sector_new = np.ceil(ang / (2 * np.pi) * ns_new).astype(np.int64)
    Sector_new = np.maximum(Sector_new,ns_new)#np.clip(Sector_new, 1, ns_new)

    # Sort based on lexicographic order of (Sector_new,Layer_new)
    LexOrd_new = Layer_new + (Sector_new - 1) * nl_new
    SortOrd_new = np.argsort(LexOrd_new)
    LexOrd_new = LexOrd_new[SortOrd_new]
    sorted_d_new = d[SortOrd_new]
    #print(LexOrd_new)
    #print(SortOrd_new)
    #print(sorted_d_new)

    # Filtering for each sector-layer intersection
    Dis_new = np.zeros((nl_new, ns_new))
    try:
        Pass = np.zeros(len(P), dtype=np.bool)
    except:
        Pass = np.zeros(len(P), dtype=np.bool_)
    p = 0  # index of point under processing
    k = 0  # number of nonempty cells
    r = max(0.01, 0.05 * R)  # cell diameter from the closest point
    #print(np_points)
    while p < np_points:
        t = 1
        while (p + t < np_points) and (LexOrd_new[p] == LexOrd_new[p + t]):
            t =t+ 1
        ind = np.arange(p, p + t)
        D = sorted_d_new[ind]
        #print(D)
        #print(ind)
        Dmin = np.min(D)
        I = D <= Dmin + r
        Pass[ind[I]] = True
        current_Layer = LexOrd_new[p] % nl_new
        #print(current_Layer)
        if current_Layer == 0:
            current_Layer = nl_new
        current_Sector = (LexOrd_new[p] - current_Layer) // nl_new + 1
        Dis_new[current_Layer - 1, current_Sector - 1] = min(1.05 * Dmin, Dmin + 0.02)
        p =p+ t
        k =k+ 1
    #print(Dis_new)
    # Sort the "Pass"-vector back to original point cloud order
    inv_sorted_indices = np.argsort(SortOrd_new)
    Pass_ordered = Pass[inv_sorted_indices]

    # Compute radius, SurfCov and mad
    d_filtered = d[Pass_ordered]
    Dis_new=Dis_new.flatten()
    non_zero_Dis_new = Dis_new[Dis_new > 0]
    if len(non_zero_Dis_new) == 0:
        R_final = 0.0
    else:
        R_final = np.median(non_zero_Dis_new)
    if len(d_filtered) == 0:
        mad = 0.0
    else:
        mad = np.sum(np.abs(d_filtered - R_final)) / len(d_filtered)
    k = np.sum(Dis_new > 0)
    SurfCov = k / (nl_new * ns_new)
    #print(k, nl_new, ns_new)
    # Update cylinder dictionary
    

    return Pass_ordered, R_final,SurfCov,mad

"""
def distances_to_line(P, axis, start):
    V = P - start
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        h = np.zeros(len(P))
        d = np.linalg.norm(V, axis=1)
        return d, V, h
    unit_axis = axis / axis_norm
    h = np.dot(V, unit_axis)
    proj = h[:, np.newaxis] * unit_axis
    vec = V - proj
    d = np.linalg.norm(vec, axis=1)
    return d, V, h

def orthonormal_vectors(axis):
    axis = axis / np.linalg.norm(axis)
    if np.abs(axis[0]) < 0.6:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([0.0, 1.0, 0.0])
    u -= np.dot(u, axis) * axis
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    return u, w
"""