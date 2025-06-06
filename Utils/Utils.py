"""
Python adaptation and extension of TREEQSM.

Version: 0.0.4
Date: 4 March 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""
import time
import math
import numpy as np
from scipy.io import loadmat
import copy
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
# from scipy.spatial import ConvexHull
# import alphashape
# from shapely.geometry import Polygon
from numba import jit
from numba.experimental import jitclass
import laspy
from plotting.qsm_plotting import qsm_plotting
import open3d as o3d
import sys
# class Utils:

# class Utils:


    
    

def load_point_cloud(file_path, intensity_threshold = 0, full_data = False):
    """
    Load a point cloud from LAS or LAZ files.

    Parameters:
    file_path : str
        Path to the LAS or LAZ file.

    Returns:
    point_cloud : ndarray
        Nx3 matrix of point coordinates (x, y, z).
    """
    if ".xyz" in file_path:
        # Load point cloud from an XYZ file
        point_data = np.loadtxt(file_path, dtype=np.float64)
        if point_data.shape[1] == 3:
            point_cloud = point_data
        elif point_data.shape[1] == 4:
            I = point_data[:, 3] > intensity_threshold
            point_cloud = point_data[I, :3]
        else:
            raise ValueError("Unsupported format in XYZ file.")
        return point_cloud if not full_data else (point_cloud, point_data)
    with laspy.open(file_path) as las:
        point_data = las.read()
        point_data = np.vstack((point_data.x, point_data.y, point_data.z,point_data.intensity)).T.astype('float64')
        I = point_data[:,3]>intensity_threshold
        point_data = point_data[I]
        point_cloud = point_data[:,0:3]
    return point_cloud if not full_data else (point_cloud,point_data)



                



@jit()
def average(X):

    """
    Computes the average of the columns of the matrix X.

    Parameters:
        X (array-like): Input matrix.

    Returns:
        numpy.ndarray: Column-wise average of X if more than one row exists,
                        otherwise returns X unchanged.
    """
    # Convert input to numpy array in case it isn't already one.
    # X = np.array(X,dtype=np.float64)

    # Determine the number of rows.
    n = X.shape[0]

    # compute column-wise average.
    return np.sum(X, axis=0) / n



def change_precision(v):
    """
    Decrease the number of nonzero decimals in the vector v according to the
    exponent of the number for displaying and writing.

    Parameters:
        v (array-like): Input vector.

    Returns:
        numpy.ndarray: Vector with reduced precision.
    """
    # Convert the input to a numpy array.
    v = np.array(v)

    # Create a copy to preserve the original values.
    new_v = v.copy()

    # Process each element in the vector.

    for i in range(len(new_v)):
        try:
            len(new_v[i])
            iterable = True
        except:
            iterable = False
        if iterable:
            new_v[i] = change_precision(new_v[i])
        else:
            abs_val = abs(new_v[i])
            if abs_val >= 1e3:
                new_v[i] = np.round(new_v[i])
            elif abs_val >= 1e2:
                new_v[i] = np.round(10 * new_v[i]) / 10
            elif abs_val >= 1e1:
                new_v[i] = np.round(100 * new_v[i]) / 100
            elif abs_val >= 1e0:
                new_v[i] = np.round(1000 * new_v[i]) / 1000
            elif abs_val >= 1e-1:
                new_v[i] = np.round(10000 * new_v[i]) / 10000
            else:
                new_v[i] = np.round(100000 * new_v[i]) / 100000
    return new_v


def cross_product(A, B):
    """
    Calculates the cross product C of the 3-vectors A and B.

    Parameters:
        A (array-like): A 3-element vector.
        B (array-like): A 3-element vector.

    Returns:
        numpy.ndarray: The cross product vector.
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array([
        A[1]*B[2] - A[2]*B[1],
        A[2]*B[0] - A[0]*B[2],
        A[0]*B[1] - A[1]*B[0]
    ])
    return C

def compute_patch_diam(pd, n):
    """
    Compute a range of PatchDiam values based on a given center value and count.

    Parameters:
    pd : float
        Center value of PatchDiam.
    n : int
        Number of PatchDiam values to compute.

    Returns:
    patch_diam : ndarray
        Array of PatchDiam values.
    """
    if n == 1:
        return np.array([pd])
    return np.linspace((0.90 - (n - 2) * 0.1) * pd, (1.10 + (n - 2) * 0.1) * pd, n)

def dot_product(A, B):
    """
    Computes the dot product of the corresponding rows of the matrices A and B.

    Parameters:
        A (array-like): Input matrix.
        B (array-like): Input matrix with the same shape as A.

    Returns:
        numpy.ndarray: A 1D array containing the row-wise dot products.
    """
    A = np.array(A)
    B = np.array(B)
    return np.sum(A * B, axis=1)



@jit()
def distances_to_line(Q, LineDirec, LinePoint):
    """
    Calculates the distances of points to a line in 3D space.

    Parameters:
        Q (ndarray): An (n x 3) array of points in 3D space.
        LineDirec (ndarray): A 1x3 unit vector representing the line's direction.
        LinePoint (ndarray): A 1x3 vector representing a point on the line.

    Returns:
        d (ndarray): A (n x 1) array of distances of points to the line.
        V (ndarray): An (n x 3) array of perpendicular vectors from the line to the points.
        h (ndarray): An (n x 1) array of projections of the vectors onto the line.
        B (ndarray): An (n x 3) array of the projections along the line direction.
    """
    # Calculate vectors from LinePoint to points in Q
    A = Q - LinePoint
    LineDirec = LineDirec.astype(np.float64)
    # Project A onto the line direction
    h = np.dot(A, LineDirec)

    # Calculate projections along the line
    B = np.outer(h, LineDirec)

    # Calculate perpendicular vectors
    V = A - B

    # Calculate distances
    d = np.sqrt(np.sum(V**2,axis=1))
    # d = np.linalg.norm(V, axis=1)

    return d, V, h, B


def distances_between_lines(PointRay, DirRay, PointLines, DirLines):
    """
    Calculates the distances between a ray and multiple lines.

    Parameters:
    -----------
    PointRay : array-like, shape (3,)
        A point on the ray.
    DirRay : array-like, shape (3,)
        A unit direction vector of the ray.
    PointLines : array-like, shape (n, 3)
        One point on every line (each row corresponds to a line).
    DirLines : array-like, shape (n, 3)
        Unit direction vectors for the lines (each row corresponds to a line).

    Returns:
    --------
    DistLines : numpy.ndarray, shape (n,)
        The shortest distance between the ray and each line.
    DistOnRay : numpy.ndarray, shape (n,)
        Distance along the ray (from PointRay) to the closest approach to each line.
    DistOnLines : numpy.ndarray, shape (n,)
        Distance along each line (from PointLines) to the closest approach to the ray.
    """
    # Ensure inputs are numpy arrays of type float
    PointRay = np.array(PointRay, dtype=float)
    DirRay = np.array(DirRay, dtype=float)
    PointLines = np.array(PointLines, dtype=float)
    DirLines = np.array(DirLines, dtype=float)

    # Calculate unit vectors N that are orthogonal to both the ray and each line via cross product.
    # For each line, N = DirRay x DirLines[i]
    # When DirLines is (n,3) and DirRay is (3,), we use broadcasting.
    N = np.column_stack((
        DirRay[1] * DirLines[:, 2] - DirRay[2] * DirLines[:, 1],
        DirRay[2] * DirLines[:, 0] - DirRay[0] * DirLines[:, 2],
        DirRay[0] * DirLines[:, 1] - DirRay[1] * DirLines[:, 0]
    ))

    # Normalize N so that each row is a unit vector.
    l = np.linalg.norm(N, axis=1)

    # To avoid division by zero (i.e. when the ray and a line are parallel),
    # you might want to handle that separately. For now, we assume non-parallel.
    N_unit = (N.T / l).T  # Transpose division for broadcasting row-wise

    # Compute A = -(PointRay - PointLines) = PointLines - PointRay
    A = -mat_vec_subtraction(PointLines, PointRay)  # This subtracts PointRay from each row of PointLines

    # Calculate the perpendicular distance (projection of A on N_unit)
    # Use the dot product for each row and take the absolute value
    DistLines = np.sqrt(np.abs(np.sum(A * N_unit, axis=1)))

    # Now, calculate the distances along the ray and lines.
    # Let:
    #   d = A dot DirRay
    #   e = A dot DirLines (each row, so row-wise dot product)
    #   b = DirLines dot DirRay  (each row dot the ray direction)
    b = np.sum(DirLines * DirRay, axis=1)
    d = np.sum(A * DirRay, axis=1)
    e = np.sum(A * DirLines, axis=1)

    # Solve for the scalar parameters along the ray (s) and the line (t)
    # as derived from the perpendicularity conditions:
    #   s = (b*e - d) / (1 - b^2)
    #   t = (e - b*d) / (1 - b^2)
    # Again, we assume 1-b^2 is not zero.
    denom = 1 - b ** 2
    DistOnRay = (b * e - d) / denom
    DistOnLines = (e - b * d) / denom

    return DistLines, DistOnRay, DistOnLines

def sec2min(T):
    """
    Converts a time in seconds T into minutes and remaining seconds.

    Parameters:
        T (float): Time in seconds.

    Returns:
        (int, float): A tuple containing minutes (as an integer) and the remaining seconds.
    """
    minutes = int(T // 60)
    seconds = T - minutes * 60
    return minutes, seconds


def display_time(T1, T2, string, display):
    """
    Display the two times given. T1 is the time named with the "string" and
    T2 is named "Total".
    """
    if display:
        tmin, tsec = sec2min(T1)
        Tmin, Tsec = sec2min(T2)

        if tmin < 60 and Tmin < 60:
            if tmin < 1 and Tmin < 1:
                result = f"{string} {tsec} sec.   Total: {Tsec} sec"
            elif tmin < 1:
                result = f"{string} {tsec} sec.   Total: {Tmin} min {Tsec} sec"
            else:
                result = f"{string} {tmin} min {tsec} sec.   Total: {Tmin} min {Tsec} sec"
        elif tmin < 60:
            Thour = Tmin // 60
            Tmin %= 60
            result = f"{string} {tmin} min {tsec} sec.   Total: {Thour} hours {Tmin} min"
        else:
            thour = tmin // 60
            tmin %= 60
            Thour = Tmin // 60
            Tmin %= 60
            result = f"{string} {thour} hours {tmin} min.   Total: {Thour} hours {Tmin} min"

        sys.stdout.write(result+'\n')


def median2(X):
    """
    Computes the median of the given vector.

    If the vector has more than one element, it sorts the vector and computes the median.
    For an even number of elements, the median is the average of the two middle elements.
    For an odd number of elements, the median is the middle element.
    If the vector has only one element, it returns that element.

    Parameters:
        X (array-like): Input vector.

    Returns:
        float: The median of the vector.
    """
    X = np.array(X).flatten()  # Ensure X is a 1D array.
    n = X.shape[0]
    if n > 1:
        X_sorted = np.sort(X)
        m = n // 2  # Floor division.
        if 2 * m == n:  # Even number of elements.
            return (X_sorted[m - 1] + X_sorted[m]) / 2.0
        else:  # Odd number of elements.
            return X_sorted[m]
    else:
        return X[0]


def normalize(A):
    """
    Normalize the rows of the matrix A.

    Parameters:
        A (array-like): Input matrix.

    Returns:
        A_normalized (numpy.ndarray): The matrix with each row normalized.
        L (numpy.ndarray): 1D array containing the norm (Euclidean length) of each row.
    """
    A = np.array(A, dtype=float)
    L = np.sqrt(np.sum(A**2, axis=1))
    A_normalized = A / L[:, None]  # Broadcast division over rows.
    return A_normalized, L


def mat_vec_subtraction(A, v):
    """
    Subtracts from each row of the matrix A the vector v.
    If A is an (n x m)-matrix, then v needs to be an m-element vector.

    Parameters:
        A (array-like): Input matrix of shape (n, m).
        v (array-like): 1D array of length m.

    Returns:
        numpy.ndarray: The matrix after subtracting v from each row.
    """
    A = np.array(A, dtype=float)
    v = np.array(v, dtype=float)
    return A - v


def verticalcat(cell_array):
    """
    Vertical concatenation of a list of arrays into a single vector.

    Parameters:
    cell_array (list of np.ndarray): A list where each element is a numpy array.

    Returns:
    tuple: A tuple (vector, ind_elements) where:
        - vector is a 1D numpy array containing the concatenated values.
        - ind_elements is a 2D numpy array where each row specifies the start
          and end indices of the corresponding cell's elements in the vector.
    """
    # Determine the size of each array in the cell array
    cell_size = np.array([len(cell) for cell in cell_array])

    # Compute cumulative sum to determine index ranges
    ind_elements = np.zeros((len(cell_array), 2), dtype=int)
    ind_elements[:, 1] = np.cumsum(cell_size) - 1  # End indices
    ind_elements[1:, 0] = 1 + ind_elements[:-1, 1]  # Start indices (shifted ends)

    # Create the output vector and fill it
    total_size = sum(cell_size)
    vector = np.zeros(total_size, dtype=int)
    for i, cell in enumerate(cell_array):
        vector[ind_elements[i, 0]:ind_elements[i, 1] + 1] = cell

    return vector, ind_elements

@jit()
def rotation_matrix(A, angle):
    """
    Returns the rotation matrix for the given axis A and angle (in radians).

    Parameters:
        A (array-like): The axis of rotation (a 3-element vector).
        angle (float): The angle of rotation in radians.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    # A = np.array(A)
    A = A / np.linalg.norm(A)
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.zeros((3, 3))
    R[0, :] = [A[0]**2 + (1 - A[0]**2)*c,      A[0]*A[1]*(1-c) - A[2]*s,       A[0]*A[2]*(1-c) + A[1]*s]
    R[1, :] = [A[0]*A[1]*(1-c) + A[2]*s,        A[1]**2 + (1 - A[1]**2)*c,       A[1]*A[2]*(1-c) - A[0]*s]
    R[2, :] = [A[0]*A[2]*(1-c) - A[1]*s,        A[1]*A[2]*(1-c) + A[0]*s,        A[2]**2 + (1 - A[2]**2)*c]
    return R

@jit
def orthonormal_vectors(U):
    """
    Generate two unit vectors (V and W) that are orthogonal to each other
    and to the input vector U.
    """
    # Generate a random vector V
    V = np.random.rand(3)
    # keeping vector same as vector generated by matlab for now: 
    # V = np.array([0.223505555240651,0.942321673912143,0.504261406484429])
    

    # Compute cross product with U to get an orthogonal vector
    V = np.cross(V, U)

    # Ensure V is a valid non-zero vector
    while np.linalg.norm(V) == 0:
        V = np.random.rand(3)
        V = np.cross(V, U)

    # Compute the second orthogonal vector W
    W = np.cross(V, U)

    # Normalize both vectors
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)

    return V, W


def optimal_parallel_vector(V):
    """
    For a given set of unit vectors (the rows of the matrix V), returns a unit vector v that is the most parallel to them all
    in the sense that the sum of squared dot products of v with the vectors of V is maximized.

    Parameters:
        V (array-like): A 2D array where each row is a unit vector.

    Returns:
        v (numpy.ndarray): A 1D unit vector that maximizes the sum of squared dot products with V.
        mean_res (float): The mean of the absolute dot products between each row of V and v.
        sigmah (float): The standard deviation of these absolute dot products.
        residual (numpy.ndarray): 1D array containing the absolute dot products for each row.
    """
    _, _, vh = np.linalg.svd(V, full_matrices=False)


    return vh[0]


def expand(Nei, C, n, Forb=None):
    """
    Expands the given subset "C" of cover sets n times with their neighbors,
    and optionally prevents the expansion into "Forb" sets. C is a vector (list or array)
    and Forb can be a number vector (list/array) or a logical (boolean) vector (numpy array).

    Parameters:
        Nei (list of list of int): A list where each element is a list (or array) of neighboring indices.
        C (list or array-like): Initial subset of indices.
        n (int): Number of expansion iterations.
        Forb (None or array-like): Optional. Either a boolean numpy array or a list/array of forbidden indices.

    Returns:
        numpy.ndarray: A 1D array containing the expanded set of indices (sorted in ascending order).
    """
    # Work with C as a list of integers.
    C = list(C)
    if Forb is None:
        for _ in range(n):
            # Concatenate neighbors from all indices in C.
            new_neighbors = []
            for idx in C:
                new_neighbors.extend(Nei[idx])
            # Union: combine current indices with new neighbors.
            C = sorted(set(C).union(new_neighbors))
        return np.array(C)
    else:
        # Branch depending on the type of Forb.
        if isinstance(Forb, np.ndarray) and Forb.dtype == bool:
            # Forb is a boolean vector.
            for _ in range(n):
                new_neighbors = []
                for idx in C:
                    new_neighbors.extend(Nei[idx])
                C = sorted(set(C).union(new_neighbors))
                # Remove indices where Forb is True.
                C = [c for c in C if not Forb[c]]
        else:
            # Forb is assumed to be a number vector.
            Forb_set = set(Forb)
            for _ in range(n):
                new_neighbors = []
                for idx in C:
                    new_neighbors.extend(Nei[idx])
                C = sorted(set(C).union(new_neighbors))
                # Remove forbidden indices.
                C = sorted(set(C) - Forb_set)
        return np.array(C)


def unique2(Set):
    """
    Returns the unique elements of the given vector Set.
    The input is first sorted, and then consecutive duplicates are removed.

    Parameters:
        Set (array-like): Input vector.

    Returns:
        numpy.ndarray: A 1D array containing the unique elements.
    """
    Set = np.array(Set)
    n = Set.size
    if n > 0:
        sorted_set = np.sort(Set)
        # For a single element, just return the sorted array.
        if n == 1:
            return sorted_set
        # Compute differences between consecutive elements.
        d = sorted_set[1:] - sorted_set[:-1]
        # A contains the elements from the second element onward.
        A = sorted_set[1:]
        # Logical mask for differences greater than zero.
        I = d > 0
        # Concatenate the first element with those elements where a change occurs.
        SetUni = np.concatenate(([sorted_set[0]], A[I]))
        return SetUni
    else:
        return Set



def unique_elements_array(arr,False_mask=None):


    return np.unique(arr)


    if False_mask is None:
        False_mask = np.zeros((len(arr),)).astype(bool)
    False_mask = False_mask.copy()
    n = len(arr)
    
    if n > 2:
        I = [True] * n
        for j in range(n):
            if not False_mask[arr[j].astype(int)]:
                False_mask[arr[j].astype(int)] = True
            else:
                I[j] = False
        arr = [arr[i] for i in range(n) if I[i]]
        
    elif n == 2:
        if arr[0] == arr[1]:
            arr = [arr[0]]
    
    return np.array(arr)


def dimensions(points, *args):
    """
    Calculates the box dimensions and dimension estimates of the point set "points".
    Also returns the corresponding direction vectors.

    Parameters:
        points (np.ndarray): A numpy array of shape (n_points, d) where d = 2 or 3.
        *args:
            If one extra argument is provided:
                P (np.ndarray): An indexable array used to re-map "points".
                points = P[points, :]
            If two extra arguments are provided:
                P (np.ndarray): An indexable array.
                Bal (list or dict): A collection where indexing by "points" returns index arrays.
                I = np.concatenate([Bal[idx] for idx in points])
                points = P[I, :]

    Returns:
        D (np.ndarray): A vector containing extents and variance ratios.
        dir (np.ndarray): A matrix whose rows are the principal direction vectors.
                          For a d-dimensional space, the output shape will be (d, d).
    """

    # Handle optional arguments to remap 'points'
    if len(args) == 1:
        P = args[0]
        points = P[points, :]
    elif len(args) == 2:
        P = args[0]
        Bal = args[1]
        # Assuming Bal is a list-like structure where each element Bal[idx] is an index array.
        I = np.concatenate([np.asarray(Bal[idx]) for idx in points])
        points = P[I, :]

    # Calculate the covariance matrix.
    # np.cov expects data variables as rows by default, so we need to set rowvar=False.
    X = np.cov(points, rowvar=False)

    # Compute the Singular Value Decomposition.
    U, S_vals, _ = np.linalg.svd(X)

    # Create diagonal matrix S so that S(i,i) = S_vals[i]
    S = np.diag(S_vals)

    d = points.shape[1]

    if d == 3:
        # Project the points on the principal axes.
        dp1 = points @ U[:, 0]
        dp2 = points @ U[:, 1]
        dp3 = points @ U[:, 2]

        # Calculate extents along each principal direction.
        extent1 = np.max(dp1) - np.min(dp1)
        extent2 = np.max(dp2) - np.min(dp2)
        extent3 = np.max(dp3) - np.min(dp3)

        # Calculate variance ratios.
        ratio1 = (S[0, 0] - S[1, 1]) / S[0, 0] if S[0, 0] != 0 else 0
        ratio2 = (S[1, 1] - S[2, 2]) / S[0, 0] if S[0, 0] != 0 else 0
        ratio3 = S[2, 2] / S[0, 0] if S[0, 0] != 0 else 0

        # Dimensions vector: extents and ratios.
        D = np.array([extent1, extent2, extent3, ratio1, ratio2, ratio3])

        # Direction vectors as rows.
        dir_vectors = np.vstack((U[:, 0].T, U[:, 1].T, U[:, 2].T))
    elif d == 2:
        dp1 = points @ U[:, 0]
        dp2 = points @ U[:, 1]

        extent1 = np.max(dp1) - np.min(dp1)
        extent2 = np.max(dp2) - np.min(dp2)

        ratio1 = (S[0, 0] - S[1, 1]) / S[0, 0] if S[0, 0] != 0 else 0
        ratio2 = S[1, 1] / S[0, 0] if S[0, 0] != 0 else 0

        D = np.array([extent1, extent2, ratio1, ratio2])
        dir_vectors = np.vstack((U[:, 0].T, U[:, 1].T))
    else:
        raise ValueError("The dimension of points must be either 2 or 3.")

    return D, dir_vectors


def intersect_elements(Set1, Set2, tracker1, tracker2):
    """
    Determines the intersection of Set1 and Set2 using the provided boolean trackers.
    The function first computes the union of Set1 and Set2 (using unique2),
    then marks the elements present in Set1 in tracker1 and those in Set2 in tracker2.
    Finally, it selects those elements from the union that are marked True in both trackers.

    Parameters:
        Set1 (array-like): A list or array of integer indices.
        Set2 (array-like): A list or array of integer indices.
        tracker1 (numpy.ndarray of bool): A boolean tracker array (sized to cover possible indices).
        tracker2 (numpy.ndarray of bool): A second boolean tracker array.

    Returns:
        numpy.ndarray: An array containing the elements that are present in both Set1 and Set2.
    """
    # Compute the union of Set1 and Set2 using unique2.
    union_set = unique2(np.concatenate((Set1, Set2)))
    # Ensure union_set is of integer type for indexing.
    union_set = union_set.astype(int)
    # Mark the occurrences of elements in Set1 and Set2.
    for elem in Set1:
        tracker1[elem] = True
    for elem in Set2:
        tracker2[elem] = True
    # Select those elements from the union for which both trackers are True.
    mask = np.logical_and(tracker1[union_set], tracker2[union_set])
    return union_set[mask]


def connected_components_array(Nei, Sub, MinSize, Fal=None):
    """
    Version of connected components using Numpy and accepting Fal as an input

    Inputs:
    Nei       : List of neighboring cover sets for each cover set (list of lists or list of arrays)
    Sub       : Subset whose components are determined. 
                If length(Sub) <= 3 and not a logical array, it is treated as a small subset.
                If Sub is a single 0, it means all cover sets.
                Otherwise, Sub is a logical array or a list of indices.
    MinSize   : Minimum number of cover sets in an acceptable component.
    Fal       : Logical false vector for the cover sets (optional).

    Outputs:
    Components: List of connected components (list of arrays).
    CompSize  : Number of sets in the components (list of integers).
    """
    Sub = Sub.copy()
    if len(Sub) <= 3 and Sub[0] > 0:
        # Very small subset, i.e., at most 3 cover sets
        n = len(Sub)
        if n == 1:
            Components = [np.array(Sub, dtype=np.uint32)]
            CompSize = [1]
        elif n == 2:
            if Sub[1] in Nei[Sub[0]]:
                Components = [np.array(Sub, dtype=np.uint32)]
                CompSize = [1]
            else:
                Components = [np.array([Sub[0]], dtype=np.uint32), np.array([Sub[1]], dtype=np.uint32)]
                CompSize = [1, 1]
        elif n == 3:
            I = Sub[1] in Nei[Sub[0]]
            J = Sub[2] in Nei[Sub[0]]
            K = Sub[2] in Nei[Sub[1]]
            if I + J + K >= 2:
                Components = [np.array(Sub, dtype=np.uint32)]
                CompSize = [1]
            elif I:
                Components = [np.array([Sub[0], Sub[1]], dtype=np.uint32), np.array([Sub[2]], dtype=np.uint32)]
                CompSize = [2, 1]
            elif J:
                Components = [np.array([Sub[0], Sub[2]], dtype=np.uint32), np.array([Sub[1]], dtype=np.uint32)]
                CompSize = [2, 1]
            elif K:
                Components = [np.array([Sub[1], Sub[2]], dtype=np.uint32), np.array([Sub[0]], dtype=np.uint32)]
                CompSize = [2, 1]
            else:
                Components = [np.array([Sub[0]], dtype=np.uint32), np.array([Sub[1]], dtype=np.uint32), np.array([Sub[2]], dtype=np.uint32)]
                CompSize = [1, 1, 1]
        return Components, CompSize

    elif any(Sub) or (len(Sub) == 1 and Sub[0] == 0):
        nb = len(Nei)
        if Fal is None:
            Fal = np.zeros(nb, dtype=bool)
        Fal = Fal.copy()
        if len(Sub) == 1 and Sub[0] == 0:
            # All the cover sets
            ns = nb
            Sub = ~Fal
        elif not isinstance(Sub, (np.ndarray, list)):
            # Subset of cover sets
            ns = len(Sub)
            sub = np.zeros(nb, dtype=bool)
            sub[Sub] = True
            Sub = sub
        else:
            # Subset of cover sets
            ns = np.sum(Sub)

        Components = []
        CompSize = []
        nc = 0  # number of components found
        m = 0
        while m < nb and not Sub[m]:
            m += 1
        i = 0
        Comp = np.zeros(ns, dtype=np.uint32)
        while i < ns:
            Add = Nei[m]
            I = Sub[Add]
            Add = Add[I]
            a = len(Add)
            Comp = Comp.copy()
            Comp[0] = m
            Sub[m] = False
            t = 1
            while a > 0:
                if t+a > len(Comp):
                    Comp = np.concatenate([Comp,np.zeros((t+a-len(Comp)))])
                Comp[t:t + a] = Add
                Sub[Add] = False
                t += a
                Add = np.concatenate([Nei[a] for a in Add])
                I = Sub[Add]
                Add = Add[I]
                Add = np.unique(Add)
                a = len(Add)
            i += t
            if t >= MinSize:
                nc += 1
                Components.append(Comp[:t])
                CompSize.append(t)
            if i < ns:
                while m < nb and not Sub[m]:
                    m += 1
        return Components, np.array(CompSize)
    else:
        return [], 0


def cubical_averaging(P, CubeSize):
    """
    Downsamples the given point cloud P by averaging points from each cube of side length CubeSize.

    Parameters:
        P (numpy.ndarray): An (n, 3) array representing the point cloud.
        CubeSize (float): The edge length of each cube.

    Returns:
        DSP (numpy.ndarray): Downsampled point cloud; one averaged point per occupied cube.
    """
    start_time = time.time()

    # Compute the minimum and maximum coordinates of P.
    Min = np.min(P, axis=0).astype(float)
    Max = np.max(P, axis=0).astype(float)

    # Number of cubes along each dimension: N = ceil((Max - Min) / CubeSize) + 1
    N = np.ceil((Max - Min) / CubeSize).astype(int) + 1

    # Determine the cube coordinates for each point in P.
    CubeCoord = np.floor((P - Min) / CubeSize).astype(int) + 1

    # Compute lexicographical order for each point.
    LexOrd = (CubeCoord[:, 0] +
                (CubeCoord[:, 1] - 1) * N[0] +
                (CubeCoord[:, 2] - 1) * N[0] * N[1])

    # Sort points by lexicographical order.
    SortOrd = np.argsort(LexOrd)
    LexOrd_sorted = LexOrd[SortOrd]

    # Determine the number of unique cubes occupied.
    nc = len(np.unique(LexOrd))
    np_total = P.shape[0]

    DSP_list = []
    p_index = 0

    while p_index < np_total:
        t = 1
        while (p_index + t < np_total) and (LexOrd_sorted[p_index] == LexOrd_sorted[p_index + t]):
            t += 1
        indices = SortOrd[p_index : p_index + t]
        avg_point = average(P[indices, :])
        # Ensure that the averaged point is a flat (1D) array.
        avg_point = np.asarray(avg_point).flatten()
        DSP_list.append(avg_point)
        p_index += t

    DSP = np.array(DSP_list)
    elapsed = time.time() - start_time

    print(f"Time {elapsed:.3f} sec.   Total: {elapsed:.3f} sec")
    print(f"    Points before:  {np_total}")
    print(f"  Filtered points:  {np_total - nc}")
    print(f"      Points left:  {nc}")

    return DSP


def create_input():
    """
    Creates the input parameter dictionary needed to run the 'treeqsm' and 'filtering' functions.
    This replicates the MATLAB 'create input' script.
    """
    inputs = {}
    # QSM reconstruction parameters
    # The three input parameters to be optimized.
    inputs['PatchDiam1'] = np.array([0.08, 0.12])
    inputs['PatchDiam2Min'] = np.array([0.02, 0.03])
    inputs['PatchDiam2Max'] = np.array([0.07, 0.1])

    # Additional patch generation parameters.
    # Ball radius in the first uniform-size cover generation.
    inputs['BallRad1'] = inputs['PatchDiam1'] + 0.015
    # Maximum ball radius in the second cover generation.
    inputs['BallRad2'] = inputs['PatchDiam2Max'] + 0.01

    # Fixed parameters.
    inputs['nmin1'] = 3       # Minimum number of points in BallRad1-balls.
    inputs['nmin2'] = 1       # Minimum number of points in BallRad2-balls.
    inputs['OnlyTree'] = 1    # Does the point cloud contain points only from the tree.
    inputs['Tria'] = 0        # Produce a triangulation of the stem's bottom part.
    inputs['Dist'] = 1        # Compute the point-model distances.

    # Radius correction options.
    inputs['MinCylRad'] = 0.0025
    inputs['ParentCor'] = 1
    inputs['TaperCor'] = 1
    inputs['GrowthVolCor'] = 0
    inputs['GrowthVolFac'] = 1.5

    # Filtering parameters.
    inputs['filter'] = {}
    inputs['filter']['k'] = 10             # k-nearest neighbors.
    inputs['filter']['radius'] = 0.00        # Ball neighborhood radius.
    inputs['filter']['nsigma'] = 1.5         # Multiplier of standard deviation.
    inputs['filter']['PatchDiam1'] = 0.05    # Filtering patch diameter.
    inputs['filter']['BallRad1'] = 0.075     # Filtering ball radius.
    inputs['filter']['ncomp'] = 2            # Minimum number of patches in a component.
    inputs['filter']['EdgeLength'] = 0.004   # Cube edge length for downsampling.
    inputs['filter']['plot'] = 1             # Automatically plot filtering results.

    # Other inputs.
    inputs['name'] = 'tree'   # Name string for saving output files.
    inputs['tree'] = 1        # Tree index.
    inputs['model'] = 1       # Model index.
    inputs['savemat'] = 1     # Save output as a MATLAB file.
    inputs['savetxt'] = 1     # Save models in text files.
    inputs['plot'] = 2        # What to plot during reconstruction.
    inputs['disp'] = 2        # Verbosity of displayed information.

    return inputs


def define_input(Clouds, nPD1, nPD2Min, nPD2Max):
    """
    Defines the required inputs (PatchDiam and BallRad parameters) for TreeQSM based on
    estimated tree stem radius and tree height.

    Inputs:
        Clouds  : Either a point cloud (n x 3 numpy array) for a single tree OR a string
                    specifying the base name of a .mat file containing multiple point clouds.
        nPD1    : Number of parameter values for PatchDiam1.
        nPD2Min : Number of parameter values for PatchDiam2Min.
        nPD2Max : Number of parameter values for PatchDiam2Max.

    Output:
        inputs  : A list of input dictionaries (one per tree) with the estimated parameter values.
    """
    # Create default input structure using create_input (assumed to be implemented).
    default_input = create_input()
    inputs_list = []

    # Check if Clouds is a string (i.e. multiple trees from a MAT file)
    if isinstance(Clouds, str):
        mat_data = loadmat(Clouds + ".mat")
        # Exclude MATLAB hidden fields and the 'Properties' field.
        names = [key for key in mat_data.keys() if not key.startswith('__') and key != 'Properties']
        names.sort()  # Sort names alphabetically.
        nt = len(names)  # Number of trees/point clouds.
    else:
        # Single tree case: Clouds is a point cloud.
        P = np.array(Clouds, dtype=float)
        nt = 1

    # Pre-allocate the inputs list.
    if nt > 1:
        for _ in range(nt):
            inputs_list.append(copy.deepcopy(default_input))
    else:
        inputs_list.append(copy.deepcopy(default_input))

    # Process each tree.
    if nt > 1:
        # Multiple trees: load each point cloud from the MAT file.
        for i in range(nt):
            tree_input = inputs_list[i]
            tree_input['name'] = names[i]
            tree_input['tree'] = i + 1
            tree_input['plot'] = 0
            tree_input['savetxt'] = 0
            tree_input['savemat'] = 0
            tree_input['disp'] = 0

            # Extract point cloud P for this tree.
            P = np.array(mat_data[names[i]], dtype=float)

            # Estimate stem parameters.
            Hb = np.min(P[:, 2])
            Ht = np.max(P[:, 2])
            TreeHeight = Ht - Hb
            Hei = P[:, 2] - Hb

            hSecTop = min(4, 0.1 * TreeHeight)
            hSecBot = 0.02 * TreeHeight
            hSec = hSecTop - hSecBot
            Sec = (Hei > hSecBot) & (Hei < hSecTop)
            StemBot = P[Sec, :3]

            # Estimate stem axis.
            AxisPoint = np.mean(StemBot, axis=0)
            V = StemBot - AxisPoint
            V_normalized, _ = normalize(V)
            AxisDir, _, _, _ = optimal_parallel_vector(V_normalized)
            d, _, _, _ = distances_to_line(StemBot, AxisDir, AxisPoint)
            Rstem = float(np.median(d))
            Res = np.sqrt((2 * np.pi * Rstem * hSec) / StemBot.shape[0])

            # Define PatchDiam1.
            pd1 = Rstem / 3.0
            if nPD1 == 1:
                tree_input['PatchDiam1'] = pd1
            else:
                n = nPD1
                tree_input['PatchDiam1'] = np.linspace((0.90 - (n - 2) * 0.1) * pd1,
                                                        (1.10 + (n - 2) * 0.1) * pd1, n)

            # Define PatchDiam2Min.
            pd2 = Rstem / 6.0 * min(1, 20 / TreeHeight)
            if nPD2Min == 1:
                tree_input['PatchDiam2Min'] = pd2
            else:
                n = nPD2Min
                tree_input['PatchDiam2Min'] = np.linspace((0.90 - (n - 2) * 0.1) * pd2,
                                                            (1.10 + (n - 2) * 0.1) * pd2, n)

            # Define PatchDiam2Max.
            pd3 = Rstem / 2.5
            if nPD2Max == 1:
                tree_input['PatchDiam2Max'] = pd3
            else:
                n = nPD2Max
                tree_input['PatchDiam2Max'] = np.linspace((0.90 - (n - 2) * 0.1) * pd3,
                                                            (1.10 + (n - 2) * 0.1) * pd3, n)

            # Define the BallRad parameters.
            # For PatchDiam1, if it is an array, use its first element.
            pd1_val = tree_input['PatchDiam1'][0] if not np.isscalar(tree_input['PatchDiam1']) else tree_input['PatchDiam1']
            pd3_val = tree_input['PatchDiam2Max'][0] if not np.isscalar(tree_input['PatchDiam2Max']) else tree_input['PatchDiam2Max']
            tree_input['BallRad1'] = max(pd1_val + 1.5 * Res, min(1.25 * pd1_val, pd1_val + 0.025))
            tree_input['BallRad2'] = max(pd3_val + 1.25 * Res, min(1.2 * pd3_val, pd3_val + 0.025))
    else:
        # Single tree case.
        tree_input = inputs_list[0]
        P = np.array(Clouds, dtype=float)
        Hb = np.min(P[:, 2])
        Ht = np.max(P[:, 2])
        TreeHeight = Ht - Hb
        Hei = P[:, 2] - Hb

        hSecTop = min(4, 0.1 * TreeHeight)
        hSecBot = 0.02 * TreeHeight
        hSec = hSecTop - hSecBot
        Sec = (Hei > hSecBot) & (Hei < hSecTop)
        StemBot = P[Sec, :3]

        AxisPoint = np.mean(StemBot, axis=0)
        V = StemBot - AxisPoint
        V_normalized, _ = normalize(V)
        AxisDir, _, _, _ = optimal_parallel_vector(V_normalized)
        d, _, _, _ = distances_to_line(StemBot, AxisDir, AxisPoint)
        Rstem = float(np.median(d))
        Res = np.sqrt((2 * np.pi * Rstem * hSec) / StemBot.shape[0])

        pd1 = Rstem / 3.0
        if nPD1 == 1:
            tree_input['PatchDiam1'] = pd1
        else:
            n = nPD1
            tree_input['PatchDiam1'] = np.linspace((0.90 - (n - 2) * 0.1) * pd1,
                                                    (1.10 + (n - 2) * 0.1) * pd1, n)

        pd2 = Rstem / 6.0 * min(1, 20 / TreeHeight)
        if nPD2Min == 1:
            tree_input['PatchDiam2Min'] = pd2
        else:
            n = nPD2Min
            tree_input['PatchDiam2Min'] = np.linspace((0.90 - (n - 2) * 0.1) * pd2,
                                                        (1.10 + (n - 2) * 0.1) * pd2, n)

        pd3 = Rstem / 2.5
        if nPD2Max == 1:
            tree_input['PatchDiam2Max'] = pd3
        else:
            n = nPD2Max
            tree_input['PatchDiam2Max'] = np.linspace((0.90 - (n - 2) * 0.1) * pd3,
                                                        (1.10 + (n - 2) * 0.1) * pd3, n)

        pd1_val = tree_input['PatchDiam1'][0] if not np.isscalar(tree_input['PatchDiam1']) else tree_input['PatchDiam1']
        pd3_val = tree_input['PatchDiam2Max'][0] if not np.isscalar(tree_input['PatchDiam2Max']) else tree_input['PatchDiam2Max']
        tree_input['BallRad1'] = max(pd1_val + 1.5 * Res, min(1.25 * pd1_val, pd1_val + 0.025))
        tree_input['BallRad2'] = max(pd3_val + 1.25 * Res, min(1.2 * pd3_val, pd3_val + 0.025))

    return inputs_list


def set_difference(Set1,Set2,Fal):

# % Performs the set difference so that the common elements of Set1 and Set2
# % are removed from Set1, which is the output. Uses logical vector whose
# % length must be up to the maximum element of the sets.
   
    Fal[Set2] = True
    I = Fal[Set1]
    Set1 = Set1[~I]
    return Set1


def save_model_text(QSM, savename):
    """
    Saves QSM (cylinder, branch, treedata) into text files in the "results" folder.

    The function creates three files:
        - results/cylinder_{savename}.txt
        - results/branch_{savename}.txt
        - results/treedata_{savename}.txt

    Parameters:
        QSM (dict): Dictionary with keys "cylinder", "branch", and "treedata".
        savename (str): String used to define the file names.
    """
    # Ensure results directory exists.
    os.makedirs("results", exist_ok=True)

    # --------------------
    # Process cylinder data.
    cylinder = QSM["cylinder"]
    # Round with 4 decimals.
    Rad   = np.round(10000 * cylinder["radius"]) / 10000
    Len   = np.round(10000 * cylinder["length"]) / 10000
    Sta   = np.round(10000 * cylinder["start"]) / 10000
    Axe   = np.round(10000 * cylinder["axis"]) / 10000
    CPar  = np.array(cylinder["parent"], dtype=np.float32)
    CExt  = np.array(cylinder["extension"], dtype=np.float32)
    Added = np.array(cylinder["added"], dtype=np.float32)
    Rad0  = np.round(10000 * cylinder["UnmodRadius"]) / 10000
    B     = np.array(cylinder["branch"], dtype=np.float32)
    BO    = np.array(cylinder["BranchOrder"], dtype=np.float32)
    PIB   = np.array(cylinder["PositionInBranch"], dtype=np.float32)
    Mad   = np.array(np.round(10000 * cylinder["mad"]) / 10000, dtype=np.float32)
    SC    = np.array(np.round(10000 * cylinder["SurfCov"]) / 10000, dtype=np.float32)

    # Stack the cylinder data as columns.
    CylData = np.column_stack((Rad, Len, Sta, Axe, CPar, CExt, B, BO, PIB, Mad, SC, Added, Rad0))
    NamesC = ['radius (m)', 'length (m)', 'start_point', 'axis_direction',
                'parent', 'extension', 'branch', 'branch_order', 'position_in_branch',
                'mad', 'SurfCov', 'added', 'UnmodRadius (m)']

    # --------------------
    # Process branch data.
    branch = QSM["branch"]
    BOrd = np.array(branch["order"], dtype=np.float32)
    BPar = np.array(branch["parent"], dtype=np.float32)
    BDia = np.round(10000 * branch["diameter"]) / 10000
    BVol = np.round(10000 * branch["volume"]) / 10000
    BAre = np.round(10000 * branch["area"]) / 10000
    BLen = np.round(1000 * branch["length"]) / 1000
    BAng = np.round(10 * branch["angle"]) / 10
    BHei = np.round(1000 * branch["height"]) / 1000
    BAzi = np.round(10 * branch["azimuth"]) / 10
    BZen = np.round(10 * branch["zenith"]) / 10

    BranchData = np.column_stack((BOrd, BPar, BDia, BVol, BAre, BLen, BHei, BAng, BAzi, BZen))
    NamesB = ["order", "parent", "diameter (m)", "volume (L)", "area (m^2)",
                "length (m)", "height (m)", "angle (deg)", "azimuth (deg)", "zenith (deg)"]

    # --------------------
    # Process treedata.
    treedata = QSM["treedata"]
    # Extract the field names up to (but not including) 'location'
    treedata_keys = list(treedata.keys())
    n = 0
    for key in treedata_keys:
        if key == "location":
            break
        n += 1
    selected_keys = treedata_keys[:n]
    # Build the TreeData vector.
    TreeData = np.array([treedata[k] for k in selected_keys], dtype=object)
    # Use less decimals (assuming change_precision is available)
    TreeData = change_precision(TreeData)
    NamesD = [str(k) for k in selected_keys]

    # --------------------
    # Save cylinder data.
    cyl_filename = os.path.join("results", f"cylinder_{savename}.txt")
    with open(cyl_filename, "wt") as fid:
        # Write header
        fid.write("\t".join(NamesC) + "\n")
        # Write each row in CylData.
        for row in CylData:
            row_str = "\t".join(str(x) for x in row)
            fid.write(row_str + "\n")

    # Save branch data.
    branch_filename = os.path.join("results", f"branch_{savename}.txt")
    with open(branch_filename, "wt") as fid:
        fid.write("\t".join(NamesB) + "\n")
        for row in BranchData:
            row_str = "\t".join(str(x) for x in row)
            fid.write(row_str + "\n")

    # Save treedata.
    treedata_filename = os.path.join("results", f"treedata_{savename}.txt")
    with open(treedata_filename, "wt") as fid:
        # Each line contains a field name and its corresponding value.
        for name, val in zip(NamesD, TreeData):
            fid.write(f"{name}\t {val}\n")


def cubical_partition(P, EL, NE=3, return_cubes = True):
    """
    Partition the point cloud into cubic cells.

    Parameters:
    P (numpy.ndarray): Point cloud, shape (n_points, 3).
    EL (float): Length of the cube edges.
    NE (int): Number of empty edge layers (default=3).

    Returns:
    tuple: Partition (list of lists of point indices), CubeCoord (n_points x 3 matrix of cube coordinates),
           Info (list containing [Min, N, EL, NE]), and optionally Cubes (3D numpy array).
    """
    # Convert P to a numpy array if not already
    P = np.array(P, dtype=float)

    # The vertices of the bounding box containing P
    Min = np.min(P, axis=0)
    Max = np.max(P, axis=0)

    # Calculate the number of cubes in each direction
    N = np.ceil((Max - Min) / EL).astype(int) + 2 * NE + 1

    # Adjust edge length and re-calculate N if too large
    t = 0
    while t < 10 and 8 * np.prod(N) > 4e9:
        t += 1
        EL *= 1.1
        N = np.ceil((Max - Min) / EL).astype(int) + 2 * NE + 1

    if 8 * np.prod(N) > 4e9:
        NE = 3
        N = np.ceil((Max - Min) / EL).astype(int) + 2 * NE + 1

    #Info = [Min, N, EL, NE]
    # Info: [Min, N, EL, NE] as a 1D array (Min and N are concatenated)
    Info = np.concatenate((Min, N.astype(float), np.array([EL, NE], dtype=float)))

    # Calculate cube coordinates of each point
    CubeCoord = np.floor((P - Min) / EL).astype(int) + NE + 1

    # Lexicographical order for sorting
    LexOrd = (CubeCoord[:, 0]
              + (CubeCoord[:, 1] - 1) * N[0]
              + (CubeCoord[:, 2] - 1) * (N[0] * N[1]))
    SortOrd = np.lexsort((CubeCoord[:, 2], CubeCoord[:, 1], CubeCoord[:, 0]))
    # Sort points by LexOrd
    # SortOrd = np.argsort(LexOrd)
    LexOrd = LexOrd[SortOrd]
    #print(LexOrd)
    #print(SortOrd)
    if return_cubes:
        # Initialize outputs
        Partition = []
        np_points = P.shape[0]

        # Group points into cubes
        p = 0
        while p < np_points:
            t = 1
            while (p + t < np_points) and (LexOrd[p] == LexOrd[p + t]):
                t += 1

            # Collect indices for the current cube
            Partition.append(SortOrd[p:p + t].tolist())
            p += t

        # Optionally create a Cubes array
        Cubes = np.zeros(N, dtype=int)
        for c_idx, points in enumerate(Partition):
            cube_coords = CubeCoord[points[0]]  # Representative point's cube coordinate
            Cubes[cube_coords[0], cube_coords[1], cube_coords[2]] = c_idx + 1  # Non-zero index

        return Partition, CubeCoord, Info, Cubes
    else:
        Partition = np.empty((N[0], N[1], N[2]), dtype=object)

        np_points = P.shape[0]  # number of points
        p = 0  

        while p < np_points:
            t = 1
            while (p + t < np_points) and (LexOrd[p] == LexOrd[p + t]):
                t += 1
            q = SortOrd[p]
            #print(SortOrd[p:p + t])
            # Assign the indices of points in the current cube to the corresponding cell in Partition
            Partition[CubeCoord[q, 0] - 1, CubeCoord[q, 1] - 1, CubeCoord[q, 2] - 1] = SortOrd[p:p + t]
            p += t
        return Partition,CubeCoord,Info


def cubical_downsampling(P, CubeSize):
    """
    Downsamples the given point cloud by selecting one point from each cube of side length CubeSize.

    Parameters:
        P (numpy.ndarray): (n_points x 3) array representing the point cloud.
        CubeSize (float): Length of the cube edges.

    Returns:
        Pass (numpy.ndarray): Boolean array of length n_points, where True indicates that the corresponding
                                point is selected as the representative for its cube.
    """
    P = np.array(P, dtype=float)
    np_points = P.shape[0]

    # Determine the bounding box of P.
    Min = np.min(P, axis=0).astype(float)
    Max = np.max(P, axis=0).astype(float)

    # Number of cubes along each dimension: N = ceil((Max - Min) / CubeSize) + 1 (elementwise)
    N = np.ceil((Max - Min) / CubeSize).astype(int) + 1

    # Process the data in blocks of m points (to reduce memory consumption)
    m = int(1e7)
    if np_points < m:
        m = np_points
    nblocks = int(np.ceil(np_points / m))

    R_list = []  # List to store [S, index] pairs for each block.
    p = 0
    for i in range(nblocks):
        if i < nblocks - 1:
            block = P[p : p + m, :]
            block_end = p + m
        else:
            block = P[p:, :]
            block_end = np_points

        # Compute cube coordinates for the current block:
        # Each coordinate: floor((P - Min) / CubeSize) + 1 (to mimic MATLAB 1-indexing)
        C = np.floor((block - Min) / CubeSize).astype(int) + 1

        # Compute lexicographical order S for each point in the block:
        # S = C[:,0] + (C[:,1]-1)*N[0] + (C[:,2]-1)*N[0]*N[1]
        S = C[:, 0] + (C[:, 1] - 1) * N[0] + (C[:, 2] - 1) * N[0] * N[1]

        # Get the unique cube values (and the indices of their first occurrence) within the block.
        unique_S, unique_idx = np.unique(S, return_index=True)
        # Create an array of the original indices corresponding to the current block.
        J = np.arange(p, block_end)
        # For the unique cubes in this block, save [S, index] pairs.
        R_block = np.column_stack((S[unique_idx], J[unique_idx]))
        R_list.append(R_block)
        p = block_end

    # Concatenate all blocks.
    R_all = np.vstack(R_list)
    # Across all blocks, select unique cubes (first occurrence).
    unique_all, idx_all = np.unique(R_all[:, 0], return_index=True)
    selected_indices = R_all[idx_all, 1].astype(int)

    # Build the output boolean mask.
    Pass = np.zeros(np_points, dtype=bool)
    Pass[selected_indices] = True

    return Pass


def growth_volume_correction(cylinder, inputs):
    """
    Uses a growth volume allometry approach to modify the radii of cylinders.
    The allometry model is: Predicted Radius = a * (GrowthVolume)^b + c.

    Parameters:
        cylinder (dict): Dictionary containing at least:
            - "radius": measured radii (array-like)
            - "length": lengths (array-like)
            - "parent": parent indices (array-like, 1-indexed; 0 indicates no parent)
            - "extension": array indicating cylinder extension (0 for tips)
        inputs (dict): Dictionary containing at least:
            - "GrowthVolFac": the factor controlling allowed deviation.

    Returns:
        cylinder (dict): The updated cylinder dictionary with corrected "radius" field.
    """
    print('----------')
    print('Growth volume based correction of cylinder radii:')

    # Convert fields to arrays.
    Rad = np.array(cylinder["radius"], dtype=float)
    Rad0 = Rad.copy()
    Len = np.array(cylinder["length"], dtype=float)
    CPar = np.array(cylinder["parent"], dtype=int)  # 1-indexed; 0 indicates no parent.
    CExt = np.array(cylinder["extension"], dtype=int)

    # Compute initial volume in liters.
    initial_volume = int(round(1000 * np.pi * np.sum(Rad**2 * Len)))
    print(' Initial_volume (L):', initial_volume)

    n = len(Rad)
    # Build child lists for each cylinder.
    CChi = [[] for _ in range(n)]
    for j in range(n):
        parent = CPar[j]
        if parent > 0:
            CChi[parent - 1].append(j)

    # Compute growth volume for each cylinder.
    GrowthVol = np.zeros(n, dtype=float)
    S = np.array([len(children) for children in CChi])
    tip_mask = (S == 0)
    GrowthVol[tip_mask] = np.pi * (Rad[tip_mask]**2) * Len[tip_mask]

    parents = np.unique(CPar[tip_mask])
    parents = parents[parents != 0]
    while parents.size > 0:
        V = np.pi * (Rad[parents - 1]**2) * Len[parents - 1]
        for i, parent in enumerate(parents):
            children = CChi[parent - 1]
            GrowthVol[parent - 1] = V[i] + (np.sum(GrowthVol[children]) if children else V[i])
        new_parents = np.unique(CPar[parents - 1])
        new_parents = new_parents[new_parents != 0]
        parents = new_parents

    # Define the allometry function with proper signature.
    def allometry(gv, a, b, c):
        return a * gv**b + c

    initial_guess = [0.5, 0.5, 0.0]
    popt, _ = curve_fit(allometry, GrowthVol, Rad, p0=initial_guess, maxfev=10000)
    print(' Allometry model parameters R = a*GV^b+c:')
    print('   Multiplier a:', popt[0])
    print('   Exponent b:', popt[1])
    print('   Intersect c:', popt[2])

    # Compute predicted radii.
    PredRad = allometry(GrowthVol, *popt)

    # Determine which cylinders need correction.
    fac = inputs["GrowthVolFac"]
    modify_idx = np.where((Rad < PredRad/fac) | (Rad > fac*PredRad))[0]
    # For tip cylinders (extension==0) where Rad is too low, do not increase the radius.
    modify_idx = np.array([i for i in modify_idx if not ((Rad[i] < PredRad[i]/fac) and (CExt[i] == 0))], dtype=int)
    CorRad = PredRad[modify_idx]

    if modify_idx.size > 0:
        R_diff = np.abs(Rad[modify_idx] - CorRad)
        D_max = np.max(R_diff)
        idx_max = modify_idx[np.argmax(R_diff)]
        D = CorRad[np.argmax(R_diff)] - Rad[idx_max]
    else:
        D = 0.0

    Rad[modify_idx] = CorRad
    cylinder["radius"] = Rad

    print(' Modified', len(modify_idx), 'of the', n, 'cylinders')
    print(' Largest radius change (cm):', round(1000 * D) / 10)
    corrected_volume = int(round(1000 * np.pi * np.sum(Rad**2 * Len)))
    print(' Corrected volume (L):', corrected_volume)
    print(' Change in volume (L):', corrected_volume - initial_volume)
    print('----------')

    # Plotting the allometry and corrections.
    gvm = np.max(GrowthVol)
    gv = np.linspace(0, gvm, int(gvm/0.001) + 1)
    PRad = allometry(gv, *popt)
    plt.figure(1)
    plt.plot(GrowthVol, Rad, '.b', markersize=2, label='radius')
    plt.plot(gv, PRad, '-r', linewidth=2, label='predicted radius')
    plt.plot(gv, PRad/fac, '-g', linewidth=2, label='minimum radius')
    plt.plot(gv, fac*PRad, '-g', linewidth=2, label='maximum radius')
    plt.grid(True)
    plt.xlabel('Growth volume (m^3)')
    plt.ylabel('Radius (m)')
    plt.legend(loc='upper left')

    plt.figure(2)
    if modify_idx.size > 0:
        plt.hist(CorRad - Rad[modify_idx], bins=20)
        plt.xlabel('Change in radius')
        plt.title('Number of cylinders per change in radius class')
    else:
        plt.title('No cylinders modified')
    plt.show()

    return cylinder


def select_cylinders(cylinder, Ind):
    """
    For each field in the dictionary 'cylinder', selects the rows specified by Ind.
    Assumes that each value in 'cylinder' is a 2D NumPy array.

    Parameters:
        cylinder (dict): Dictionary whose keys correspond to fields and whose values
                            are NumPy arrays.
        Ind (array-like): Indices of the rows to select.

    Returns:
        dict: The updated cylinder dictionary with each field indexed by Ind.
    """
    for key in cylinder.keys():
        # Select the rows indicated by Ind from each field.
        cylinder[key] = np.array(cylinder[key])[Ind, :]
    return cylinder


@jit(nopython=True)
def surface_coverage_prep(P, Axis, Point, nl, ns, Dmin=None, Dmax=None):
    """
    First half of surface coverage moved out for compilation
    """
    # Compute distances, projections and heights from points to the cylinder axis.
    d, V, h, _ = distances_to_line(P, Axis, Point)
    h = h - np.min(h)
    Len = np.max(h)
    #print(d)

    # Optional filtering: keep only points with d > Dmin (and, if provided, d < Dmax)
    if Dmin is not None:
        Keep = d > Dmin
        if Dmax is not None:
            Keep = Keep & (d < Dmax)
        #V = V[Keep, :]
        V = V[Keep, : ]
        h = h[Keep]
        d = d[Keep]

    # Compute SurfCov over four rotated baselines.
    V0 = V.copy()
    U, W = orthonormal_vectors(Axis)  # U and W: 1D arrays of length 3.
    R = rotation_matrix(Axis, 2 * np.pi / ns / 4)
    surf_cov_array = np.zeros(4)
    lexord_final = None  # to store lex order from the final rotation.

    for i in range(4):
        if i > 0:
            U = R @ U
            W = R @ W
        # Form transformation matrix from the two planar axes.
        T = np.column_stack((U, W))  # shape (3,2)
        V_proj = V0 @ T             # shape (n_points,2)
        # Compute angles in [0,2pi)
        ang = np.arctan2(V_proj[:, 1], V_proj[:, 0]) + np.pi
        # Determine layer: 1-indexed: layer = ceil(h/Len*nl)
        Layer = np.ceil(h / Len * nl).astype(np.int64)
        Layer[Layer < 1] = 1
        Layer[Layer > nl] = nl
        # Determine sector: 1-indexed: sector = ceil(ang/(2*pi)*ns)
        Sector = np.ceil(ang / (2 * np.pi) * ns).astype(np.int64)
        Sector[Sector < 1] = 1
        Sector[Sector > ns] = ns
        # Compute lexicographic order: for 1-indexing, we use:
        # lex = (Layer - 1) + (Sector - 1)*nl, which gives indices 0...nl*ns-1.
        lexord = (Layer - 1) + (Sector - 1) * nl
        # Build coverage matrix Cov of shape (nl, ns)
        Cov = np.zeros((nl, ns))
        unique_lex = np.unique(lexord)
        for val in unique_lex:
            Cov.flat[int(val)] = 1
        surf_cov_array[i] = np.count_nonzero(Cov) / (nl * ns)
        # Save lex order from final rotation for further processing.
        if i == 3:
            lexord_final = lexord.copy()
            d_final = d.copy()
            sort_idx = np.argsort(lexord_final)
            lexord_sorted = lexord_final[sort_idx]
            d_sorted = d_final[sort_idx]
    SurfCov = np.max(surf_cov_array)

    # Compute Dis: mean distance for each (layer, sector) cell using lexord from final rotation.
    Dis = np.zeros((nl, ns))
    np_total = lexord_sorted.size
    p = 0
    while p < np_total:
        t = 1
        while (p + t < np_total) and (lexord_sorted[p] == lexord_sorted[p + t]):
            t =t+ 1
        avg_val = average(d_sorted[p:p+t])
        idx = lexord_sorted[p]  # 0-indexed index into a flattened (nl x ns) array.
        row = np.int64(idx % nl)
        col = np.int64(idx // nl)
        Dis[row, col] = avg_val
        p =p+ t

    
    return Dis,SurfCov,Len


    

def surface_coverage(P, Axis, Point, nl, ns, Dmin=None, Dmax=None):
    """
    Computes point surface coverage measure of a cylinder.

    Parameters:
        P     : (n_points x 3) NumPy array representing the point cloud.
        Axis  : (1 x 3) axis direction vector.
        Point : (1 x 3) starting point of the cylinder.
        nl    : Number of layers (in the axis direction) used to partition the cylinder surface.
        ns    : Number of angular sectors used to partition each layer.
        Dmin  : Optional minimum point distance from the axis (only points with d > Dmin are included).
        Dmax  : Optional maximum point distance from the axis (only points with d < Dmax are included).

    Returns:
        SurfCov : A number between 0 and 1 describing the fraction of the cylinder surface covered by points.
        Dis     : (nl x ns) matrix of mean distances for each layer-sector cell.
        CylVol  : Cylinder volume estimate (in liters) computed from the mean distances.
        dis     : (nl x ns) matrix of distances where missing values are interpolated.
    """
    
    Dis,SurfCov,Len = surface_coverage_prep(P, Axis, Point, nl, ns, Dmin=None, Dmax=None)
    # If volume estimation is requested, compute interpolation for missing distances.
    # (In MATLAB: if nargout > 2)
    # Create an extended matrix D_ext by replicating D (here D = Dis)
    CylVol = None
    dis_out = None
    D = Dis.copy()
    dis_out = Dis.copy()
    D_inv = D[::-1, :]  # reverse rows
    D_ext = np.block([
        [D_inv, D_inv, D_inv],
        [D,     D,     D],
        [D_inv, D_inv, D_inv]
    ])
    
    Zero = (D == 0)
    if np.count_nonzero(D) > 0:
        D=D.flatten()
        RadMean = average(D[D > 0])
    else:
        RadMean = 0
    for i in range(nl):
        for j in range(ns):
            if Zero[i, j]:
                # First try a 3x3 window.
                window = D_ext[i+nl-1:i+nl+2, j+ns-1:j+ns+2].flatten()
                if np.count_nonzero(window) > 1:
                    dis_out[i, j] = average(window[window > 0])
                else:
                    # Try a 5x5 window.
                    window = D_ext[i+nl-2:i+nl+3, j+ns-2:j+ns+3]
                    if np.count_nonzero(window) > 1:
                        dis_out[i, j] = average(window[window > 0])
                    else:
                        # Try a 7x7 window.
                        window = D_ext[i+nl-3:i+nl+4, j+ns-3:j+ns+4]
                        if np.count_nonzero(window) > 1:
                            dis_out[i, j] = average(window[window > 0])
                        else:
                            dis_out[i, j] = RadMean
    # Compute volume estimate.
    r = dis_out.flatten()
    CylVol = 1000 * np.pi * np.sum(r**2) / ns * (Len / nl)
    

    return SurfCov, Dis, CylVol, dis_out


def surface_coverage2(axis, length, vec, height, nl, ns):
    """
    Compute surface coverage (fraction of covered cells on a cylindrical surface).

    Parameters:
        axis (array-like): Axis vector of the cylinder.
        length (float): Length of the cylinder.
        vec (np.ndarray): Vectors connecting points to the axis (n x 3).
        height (np.ndarray): Heights of points from the base of the cylinder (n x 1).
        nl (int): Number of layers along the cylinder height.
        ns (int): Number of angular segments around the cylinder.

    Returns:
        float: Surface coverage (value between 0 and 1).
    """
    # Compute orthonormal basis
    u, w = orthonormal_vectors(axis)

    # Project vectors into the cylinder's local 2D plane
    vec = vec @ np.array([u, w]).T

    # Compute angular coordinates
    ang = np.arctan2(vec[:, 1], vec[:, 0]) + np.pi

    # Map points to layer indices
    i = np.ceil(height / length * nl).astype(int)
    i = np.clip(i, 1, nl)

    # Map points to angular segment indices
    j = np.ceil(ang / (2 * np.pi) * ns).astype(int)
    j = np.clip(j, 1, ns)

    # Compute unique cell indices
    k = (i - 1) + (j - 1) * nl
    unique_k = np.unique(k)

    # Compute surface coverage
    surf_cov = len(unique_k) / (nl * ns)
    return surf_cov


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


def update_tree_data(QSM, cylinder, branch, inputs):
    """
    Updates the treedata structure after QSM simplification.

    Inputs:
        QSM      : Dictionary with at least key "treedata" (and optionally "triangulation")
        cylinder : Dictionary with cylinder fields including "radius", "length", "start", "axis", "branch"
        branch   : Dictionary with branch fields (order, volume, area, length, height, angle, azimuth, zenith, diameter)
        inputs   : Dictionary with processing options (e.g., "Tria", "disp", "plot")

    Output:
        treedata : Updated dictionary with tree attributes.
    """
    # Copy existing treedata from QSM.
    treedata = copy.deepcopy(QSM["treedata"])

    # Extract cylinder data.
    Rad = np.array(cylinder["radius"], dtype=float)
    Len = np.array(cylinder["length"], dtype=float)
    Sta = np.array(cylinder["start"], dtype=float)  # (n x 3)
    Axe = np.array(cylinder["axis"], dtype=float)   # (n x 3)
    nc = len(Rad)
    ind = np.arange(1, nc+1)  # 1-indexed indices

    # Identify trunk cylinders (cylinder.branch == 1).
    Trunk = (np.array(cylinder["branch"]) == 1)

    # Compute basic tree attributes.
    treedata["TotalVolume"] = 1000 * np.pi * np.sum(Rad**2 * Len)
    treedata["TrunkVolume"] = 1000 * np.pi * np.sum(Rad[Trunk]**2 * Len[Trunk])
    treedata["BranchVolume"] = 1000 * np.pi * np.sum(Rad[~Trunk]**2 * Len[~Trunk])
    bottom = np.min(Sta[:,2])
    top_val = np.max(Sta[:,2])
    top_idx = np.argmax(Sta[:,2])
    if Axe[top_idx,2] > 0:
        top_val = top_val + Len[top_idx] * Axe[top_idx,2]
    treedata["TreeHeight"] = top_val - bottom
    treedata["TrunkLength"] = np.sum(Len[Trunk])
    treedata["BranchLength"] = np.sum(Len[~Trunk])
    treedata["TotalLength"] = treedata["TrunkLength"] + treedata["BranchLength"]
    NB = len(branch["order"]) - 1
    treedata["NumberBranches"] = NB
    BO = np.max(branch["order"])
    treedata["MaxBranchOrder"] = BO
    treedata["TrunkArea"] = 2 * np.pi * np.sum(Rad[Trunk] * Len[Trunk])
    treedata["BranchArea"] = 2 * np.pi * np.sum(Rad[~Trunk] * Len[~Trunk])
    treedata["TotalArea"] = 2 * np.pi * np.sum(Rad * Len)

    # Compute crown measures, vertical profile, and spreads.
    treedata, spreads = crown_measures(treedata, cylinder, branch)

    # Set crown attributes.
    branch_orders = np.array(branch["order"])
    # Simple rule: if any branch has order > 1, crown base height is 1.0 m; else, crown base equals tree height.
    if np.any(branch_orders > 1):
        CrownBaseHeight = 1.0
    else:
        CrownBaseHeight = treedata["TreeHeight"]
    treedata["CrownBaseHeight"] = CrownBaseHeight
    treedata["CrownLength"] = treedata["TreeHeight"] - CrownBaseHeight
    treedata["CrownRatio"] = (treedata["CrownLength"] / treedata["TreeHeight"]
                                if treedata["TreeHeight"] != 0 else 0.0)

    # Update triangulation if requested.
    if inputs["Tria"]:
        treedata = update_triangulation(QSM, treedata, cylinder)
    else:
        # Ensure mix keys exist.
        treedata["MixTrunkVolume"] = 0.0
        treedata["MixTotalVolume"] = 0.0
        treedata["MixTrunkArea"] = 0.0
        treedata["MixTotalArea"] = 0.0

    # Tree location.
    treedata["location"] = Sta[0, :]

    # Stem taper.
    R_trunk = Rad[Trunk]
    n_trunk = len(R_trunk)
    Taper = np.zeros((n_trunk+1, 2))
    if n_trunk > 0:
        Taper[0, 1] = 2 * R_trunk[0]
    if n_trunk > 1:
        trunk_lengths = Len[Trunk]
        Taper[1:, 0] = np.cumsum(trunk_lengths)
        Taper[1:, 1] = np.concatenate((2 * R_trunk[1:], [2 * R_trunk[-1]]))
    treedata["StemTaper"] = Taper.T

    # Vertical profile and spreads.
    treedata["VerticalProfile"] = np.mean(spreads, axis=1)
    treedata["spreads"] = spreads

    # Cylinder distributions.
    treedata = cylinder_distribution(treedata, Rad, Len, Axe, "Dia")
    treedata = cylinder_height_distribution(treedata, Rad, Len, Sta, Axe, ind)
    treedata = cylinder_distribution(treedata, Rad, Len, Axe, "Zen")
    treedata = cylinder_distribution(treedata, Rad, Len, Axe, "Azi")

    # Branch distributions.
    treedata = branch_order_distribution(treedata, branch)
    treedata = branch_distribution(treedata, branch, "Dia")
    treedata = branch_distribution(treedata, branch, "Hei")
    treedata = branch_distribution(treedata, branch, "Ang")
    treedata = branch_distribution(treedata, branch, "Azi")
    treedata = branch_distribution(treedata, branch, "Zen")

    # Convert all fields to single precision.
    for key in list(treedata.keys()):
        treedata[key] = np.array(treedata[key], dtype=np.float32)

    # Optionally display treedata.
    if inputs["disp"] == 2:
        print("------------")
        print("  Tree attributes:")
        for key in treedata.keys():
            v = change_precision(np.atleast_1d(treedata[key]))
            print(f"  {key} = {v}")
        print("  -----")

    # Optionally plot distributions (placeholder).
    if inputs["plot"] > 1:
        print("Plotting distributions (placeholder).")

    return treedata


def crown_measures(treedata, cylinder, branch):
    """
    Simplified crown measures implementation.
    Generates a crown point cloud from cylinder start and tip positions,
    then computes average and maximum crown diameters.
    Also returns dummy spreads.
    """
    Axe = cylinder["axis"]
    Len = cylinder["length"]
    Sta = cylinder["start"]
    Tip = Sta + (np.array(Len).reshape(-1,1) * np.array(Axe))
    nc = len(Len)
    P = np.vstack((Sta, Tip))
    P = np.unique(P, axis=0)
    treedata["CrownDiamAve"] = np.mean(np.linalg.norm(P[:, :2], axis=1)) * 2
    treedata["CrownDiamMax"] = np.max(np.linalg.norm(P[:, :2], axis=1)) * 2
    treedata["CrownAreaConv"] = 1.0
    treedata["CrownAreaAlpha"] = 1.0
    m = 5  # number of layers for crown measures
    spreads = np.zeros((m, 18))
    return treedata, spreads


def update_triangulation(QSM, treedata, cylinder):
    """
    Simplified update of triangulation-related fields.
    If QSM.triangulation exists, update mix volumes and areas.
    """
    if "triangulation" in QSM and QSM["triangulation"]:
        triang = QSM["triangulation"]
        treedata["MixTrunkVolume"] = treedata["TrunkVolume"] * 0.9 + triang.get("volume", 0.0)
        treedata["MixTotalVolume"] = treedata["MixTrunkVolume"] + treedata["BranchVolume"]
        treedata["MixTrunkArea"] = treedata["TrunkArea"] * 0.9 + triang.get("SideArea", 0.0)
        treedata["MixTotalArea"] = treedata["MixTrunkArea"] + treedata["BranchArea"]
    return treedata


def cylinder_distribution(treedata, Rad, Len, Axe, dist):
    if dist == "Dia":
        Par = Rad
        n = int(np.ceil(np.max(100 * np.array(Par))))
        a = 0.005
    elif dist == "Zen":
        Par = np.degrees(np.arccos(np.array(Axe)[:,2]))
        n = 18
        a = 10
    elif dist == "Azi":
        Par = np.degrees(np.arctan2(np.array(Axe)[:,1], np.array(Axe)[:,0])) + 180
        n = 36
        a = 10
    else:
        raise ValueError("Unknown distribution type")
    CylDist = np.zeros((3, n))
    for i in range(n):
        I = (np.array(Par) >= i * a) & (np.array(Par) < (i+1) * a)
        CylDist[0, i] = 1000 * np.pi * np.sum(np.array(Rad)[I]**2 * np.array(Len)[I])
        CylDist[1, i] = 2 * np.pi * np.sum(np.array(Rad)[I] * np.array(Len)[I])
        CylDist[2, i] = np.sum(np.array(Len)[I])
    treedata["VolCyl" + dist] = CylDist[0, :].tolist()
    treedata["AreCyl" + dist] = CylDist[1, :].tolist()
    treedata["LenCyl" + dist] = CylDist[2, :].tolist()
    return treedata


def cylinder_height_distribution(treedata, Rad, Len, Sta, Axe, ind):
    MaxHei = int(np.ceil(treedata["TreeHeight"]))
    treedata["VolCylHei"] = np.zeros(MaxHei)
    treedata["AreCylHei"] = np.zeros(MaxHei)
    treedata["LenCylHei"] = np.zeros(MaxHei)
    End = Sta + (np.array(Len).reshape(-1, 1) * np.array(Axe))
    bot = np.min(Sta[:,2])
    B = Sta[:,2] - bot
    T = End[:,2] - bot
    for j in range(1, MaxHei+1):
        idx = np.where((B >= (j-1)) & (B < j))[0]
        v1 = 1000 * np.pi * np.sum(np.array(Rad)[idx]**2 * np.array(Len)[idx])
        a1 = 2 * np.pi * np.sum(np.array(Rad)[idx] * np.array(Len)[idx])
        l1 = np.sum(np.array(Len)[idx])
        treedata["VolCylHei"][j-1] = v1
        treedata["AreCylHei"][j-1] = a1
        treedata["LenCylHei"][j-1] = l1
    return treedata


def branch_distribution(treedata, branch, dist):
    BOrd = branch["order"][1:]
    if dist == "Dia":
        Par = branch["diameter"][1:]
        n = int(np.ceil(np.max(100 * np.array(Par))))
        a = 0.005
    elif dist == "Hei":
        Par = branch["height"][1:]
        n = int(np.ceil(treedata["TreeHeight"]))
        a = 1
    elif dist == "Ang":
        Par = branch["angle"][1:]
        n = 18
        a = 10
    elif dist == "Zen":
        Par = branch["zenith"][1:]
        n = 18
        a = 10
    elif dist == "Azi":
        Par = np.array(branch["azimuth"][1:]) + 180
        n = 36
        a = 10
    else:
        raise ValueError("Unknown branch distribution type")
    BranchDist = np.zeros((8, n))
    Par = np.array(Par)
    BOrd = np.array(BOrd)
    for i in range(n):
        I = (Par >= i * a) & (Par < (i+1) * a)
        BranchDist[0, i] = np.sum(np.array(branch["volume"][1:])[I])
        BranchDist[1, i] = np.sum(np.array(branch["volume"][1:])[I & (BOrd == 1)])
        BranchDist[2, i] = np.sum(np.array(branch["area"][1:])[I])
        BranchDist[3, i] = np.sum(np.array(branch["area"][1:])[I & (BOrd == 1)])
        BranchDist[4, i] = np.sum(np.array(branch["length"][1:])[I])
        BranchDist[5, i] = np.sum(np.array(branch["length"][1:])[I & (BOrd == 1)])
        BranchDist[6, i] = np.count_nonzero(I)
        BranchDist[7, i] = np.count_nonzero(I & (BOrd == 1))
    treedata["VolBranch" + dist] = BranchDist[0, :].tolist()
    treedata["VolBranch1" + dist] = BranchDist[1, :].tolist()
    treedata["AreBranch" + dist] = BranchDist[2, :].tolist()
    treedata["AreBranch1" + dist] = BranchDist[3, :].tolist()
    treedata["LenBranch" + dist] = BranchDist[4, :].tolist()
    treedata["LenBranch1" + dist] = BranchDist[5, :].tolist()
    treedata["NumBranch" + dist] = BranchDist[6, :].tolist()
    treedata["NumBranch1" + dist] = BranchDist[7, :].tolist()
    return treedata


def branch_order_distribution(treedata, branch):
    BO = np.max(branch["order"])
    BranchOrdDist = np.zeros((BO, 4))
    for i in range(1, BO+1):
        I = (np.array(branch["order"]) == i)
        vol = np.sum(np.array(branch["volume"])[I])
        area = np.sum(np.array(branch["area"])[I])
        leng = np.sum(np.array(branch["length"])[I])
        count = np.count_nonzero(I)
        BranchOrdDist[i-1, :] = [vol, area, leng, count]
    treedata["VolBranchOrd"] = BranchOrdDist[:, 0].tolist()
    treedata["AreBranchOrd"] = BranchOrdDist[:, 1].tolist()
    treedata["LenBranchOrd"] = BranchOrdDist[:, 2].tolist()
    treedata["NumBranchOrd"] = BranchOrdDist[:, 3].tolist()
    return treedata

def package_outputs(models,cyl_htmls):
    
    tree_data_figures =[]
    segment_plots=[]
    cyl_plots=[]
    for i in range(len(models)):
        run_name = models[i]['rundata']['inputs']['name']+"_"+str(i)
        figs =[]
        for j,fig in enumerate(models[i]['treedata']['figures']):
            save_name = f"results/tree_data_{run_name}_{models[i]['rundata']['inputs']['tree']}_{models[i]['rundata']['inputs']['model']}_{j}.pdf"
            fig.dpi=1000
            fig.savefig(save_name,format ='pdf')
            figs.append(save_name)
        figs = tuple(figs)
        tree_data_figures.append(figs)

        # segment_plots.append(qsm_plotting(models[i]['points'], models[i]['cover'], models[i]['segment'],models[i]))
        #keeping segments out this for now, need to make more efficient

        

    return {"tree_data":tuple(tree_data_figures),"cylinders":tuple(cyl_htmls)}

@jit(nopython=True)
def assign_segments(cloud,segments,cover_sets):
    point_segments = np.zeros((cloud.shape[0]),dtype = np.int64)-1
    for i,segment in enumerate(segments):
        I = np.where(np.isin(cover_sets, segment))[0]
        point_segments[I] = i
    return point_segments

def select_metric(Metric):
    """Convert metric string to corresponding numeric code.
    
    Args:
        Metric (str): Metric description string
        
    Returns:
        tuple: (met, Metric) where met is the numeric code and Metric is the validated string
    """
    # Mean distance metrics:
    if Metric == 'all_mean_dis':
        met = 1
    elif Metric == 'trunk_mean_dis':
        met = 2
    elif Metric == 'branch_mean_dis':
        met = 3
    elif Metric == '1branch_mean_dis':
        met = 4
    elif Metric == '2branch_mean_dis':
        met = 5
    elif Metric == 'trunk+branch_mean_dis':
        met = 6
    elif Metric == 'trunk+1branch_mean_dis':
        met = 7
    elif Metric == 'trunk+1branch+2branch_mean_dis':
        met = 8
    elif Metric == '1branch+2branch_mean_dis':
        met = 9

    # Maximum distance metrics:
    elif Metric == 'all_max_dis':
        met = 10
    elif Metric == 'trunk_max_dis':
        met = 11
    elif Metric == 'branch_max_dis':
        met = 12
    elif Metric == '1branch_max_dis':
        met = 13
    elif Metric == '2branch_max_dis':
        met = 14
    elif Metric == 'trunk+branch_max_dis':
        met = 15
    elif Metric == 'trunk+1branch_max_dis':
        met = 16
    elif Metric == 'trunk+1branch+2branch_max_dis':
        met = 17
    elif Metric == '1branch+2branch_max_dis':
        met = 18

    # Mean plus Maximum distance metrics:
    elif Metric == 'all_mean+max_dis':
        met = 19
    elif Metric == 'trunk_mean+max_dis':
        met = 20
    elif Metric == 'branch_mean+max_dis':
        met = 21
    elif Metric == '1branch_mean+max_dis':
        met = 22
    elif Metric == '2branch_mean+max_dis':
        met = 23
    elif Metric == 'trunk+branch_mean+max_dis':
        met = 24
    elif Metric == 'trunk+1branch_mean+max_dis':
        met = 25
    elif Metric == 'trunk+1branch+2branch_mean+max_dis':
        met = 26
    elif Metric == '1branch+2branch_mean+max_dis':
        met = 27

    # Standard deviation metrics:
    elif Metric == 'tot_vol_std':
        met = 28
    elif Metric == 'trunk_vol_std':
        met = 29
    elif Metric == 'branch_vol_std':
        met = 30
    elif Metric == 'trunk+branch_vol_std':
        met = 31
    elif Metric == 'tot_are_std':
        met = 32
    elif Metric == 'trunk_are_std':
        met = 33
    elif Metric == 'branch_are_std':
        met = 34
    elif Metric == 'trunk+branch_are_std':
        met = 35
    elif Metric == 'trunk_len_std':
        met = 36
    elif Metric == 'trunk+branch_len_std':
        met = 37
    elif Metric == 'branch_len_std':
        met = 38
    elif Metric == 'branch_num_std':
        met = 39

    # Branch order distribution metrics:
    elif Metric == 'branch_vol_ord3_mean':
        met = 40
    elif Metric == 'branch_are_ord3_mean':
        met = 41
    elif Metric == 'branch_len_ord3_mean':
        met = 42
    elif Metric == 'branch_num_ord3_mean':
        met = 43
    elif Metric == 'branch_vol_ord3_max':
        met = 44
    elif Metric == 'branch_are_ord3_max':
        met = 45
    elif Metric == 'branch_len_ord3_max':
        met = 46
    elif Metric == 'branch_num_ord3_max':
        met = 47
    elif Metric == 'branch_vol_ord6_mean':
        met = 48
    elif Metric == 'branch_are_ord6_mean':
        met = 49
    elif Metric == 'branch_len_ord6_mean':
        met = 50
    elif Metric == 'branch_num_ord6_mean':
        met = 51
    elif Metric == 'branch_vol_ord6_max':
        met = 52
    elif Metric == 'branch_are_ord6_max':
        met = 53
    elif Metric == 'branch_len_ord6_max':
        met = 54
    elif Metric == 'branch_num_ord6_max':
        met = 55

    # Cylinder distribution metrics:
    elif Metric == 'cyl_vol_dia10_mean':
        met = 56
    elif Metric == 'cyl_are_dia10_mean':
        met = 57
    elif Metric == 'cyl_len_dia10_mean':
        met = 58
    elif Metric == 'cyl_vol_dia10_max':
        met = 59
    elif Metric == 'cyl_are_dia10_max':
        met = 60
    elif Metric == 'cyl_len_dia10_max':
        met = 61
    elif Metric == 'cyl_vol_dia20_mean':
        met = 62
    elif Metric == 'cyl_are_dia20_mean':
        met = 63
    elif Metric == 'cyl_len_dia20_mean':
        met = 64
    elif Metric == 'cyl_vol_dia20_max':
        met = 65
    elif Metric == 'cyl_are_dia20_max':
        met = 66
    elif Metric == 'cyl_len_dia20_max':
        met = 67
    elif Metric == 'cyl_vol_zen_mean':
        met = 68
    elif Metric == 'cyl_are_zen_mean':
        met = 69
    elif Metric == 'cyl_len_zen_mean':
        met = 70
    elif Metric == 'cyl_vol_zen_max':
        met = 71
    elif Metric == 'cyl_are_zen_max':
        met = 72
    elif Metric == 'cyl_len_zen_max':
        met = 73

    # Mean surface coverage metrics:
    elif Metric == 'all_mean_surf':
        met = 74
    elif Metric == 'trunk_mean_surf':
        met = 75
    elif Metric == 'branch_mean_surf':
        met = 76
    elif Metric == '1branch_mean_surf':
        met = 77
    elif Metric == '2branch_mean_surf':
        met = 78
    elif Metric == 'trunk+branch_mean_surf':
        met = 79
    elif Metric == 'trunk+1branch_mean_surf':
        met = 80
    elif Metric == 'trunk+1branch+2branch_mean_surf':
        met = 81
    elif Metric == '1branch+2branch_mean_surf':
        met = 82

    # Minimum surface coverage metrics:
    elif Metric == 'all_min_surf':
        met = 83
    elif Metric == 'trunk_min_surf':
        met = 84
    elif Metric == 'branch_min_surf':
        met = 85
    elif Metric == '1branch_min_surf':
        met = 86
    elif Metric == '2branch_min_surf':
        met = 87
    elif Metric == 'trunk+branch_min_surf':
        met = 88
    elif Metric == 'trunk+1branch_min_surf':
        met = 89
    elif Metric == 'trunk+1branch+2branch_min_surf':
        met = 90
    elif Metric == '1branch+2branch_min_surf':
        met = 91

    # Not given in right form, take the default option
    else:
        met = 1
        Metric = 'all_mean_dis'

    return met

def get_all_metrics():
    """
    Returns a list of all available metrics.
    """
    return [
    "all_mean_dis",
    "trunk_mean_dis",
    "branch_mean_dis",
    "1branch_mean_dis",
    "2branch_mean_dis",
    "trunk+branch_mean_dis",
    "trunk+1branch_mean_dis",
    "trunk+1branch+2branch_mean_dis",
    "1branch+2branch_mean_dis",
    "all_max_dis",
    "trunk_max_dis",
    "branch_max_dis",
    "1branch_max_dis",
    "2branch_max_dis",
    "trunk+branch_max_dis",
    "trunk+1branch_max_dis",
    "trunk+1branch+2branch_max_dis",
    "1branch+2branch_max_dis",
    "all_mean+max_dis",
    "trunk_mean+max_dis",
    "branch_mean+max_dis",
    "1branch_mean+max_dis",
    "2branch_mean+max_dis",
    "trunk+branch_mean+max_dis",
    "trunk+1branch_mean+max_dis",
    "trunk+1branch+2branch_mean+max_dis",
    "1branch+2branch_mean+max_dis",
    "tot_vol_std",
    "trunk_vol_std",
    "branch_vol_std",
    "trunk+branch_vol_std",
    "tot_are_std",
    "trunk_are_std",
    "branch_are_std",
    "trunk+branch_are_std",
    "trunk_len_std",
    "trunk+branch_len_std",
    "branch_len_std",
    "branch_num_std",
    "branch_vol_ord3_mean",
    "branch_are_ord3_mean",
    "branch_len_ord3_mean",
    "branch_num_ord3_mean",
    "branch_vol_ord3_max",
    "branch_are_ord3_max",
    "branch_len_ord3_max",
    "branch_num_ord3_max",
    "branch_vol_ord6_mean",
    "branch_are_ord6_mean",
    "branch_len_ord6_mean",
    "branch_num_ord6_mean",
    "branch_vol_ord6_max",
    "branch_are_ord6_max",
    "branch_len_ord6_max",
    "branch_num_ord6_max",
    "cyl_vol_dia10_mean",
    "cyl_are_dia10_mean",
    "cyl_len_dia10_mean",
    "cyl_vol_dia10_max",
    "cyl_are_dia10_max",
    "cyl_len_dia10_max",
    "cyl_vol_dia20_mean",
    "cyl_are_dia20_mean",
    "cyl_len_dia20_mean",
    "cyl_vol_dia20_max",
    "cyl_are_dia20_max",
    "cyl_len_dia20_max",
    "cyl_vol_zen_mean",
    "cyl_are_zen_mean",
    "cyl_len_zen_mean",
    "cyl_vol_zen_max",
    "cyl_are_zen_max",
    "cyl_len_zen_max",
    "all_mean_surf",
    "trunk_mean_surf",
    "branch_mean_surf",
    "1branch_mean_surf",
    "2branch_mean_surf",
    "trunk+branch_mean_surf",
    "trunk+1branch_mean_surf",
    "trunk+1branch+2branch_mean_surf",
    "1branch+2branch_mean_surf",
    "all_min_surf",
    "trunk_min_surf",
    "branch_min_surf",
    "1branch_min_surf",
    "2branch_min_surf",
    "trunk+branch_min_surf",
    "trunk+1branch_min_surf",
    "trunk+1branch+2branch_min_surf",
    "1branch+2branch_min_surf"]
    

def collect_data(QSMs):
    """
    Collects tree data and attributes from QSM models
    
    Args:
        QSMs: List of QSM model dictionaries
        names: List of attribute names to collect
        
        
    Returns:
        tuple: (treedata, inputs, TreeId, Data)
            treedata: Array of tree attributes (Nattri x Nmod)
            inputs: Array of input parameters (Nmod x 3)
            TreeId: Array of tree and model indexes (Nmod x 2)
            Data: Dictionary containing various distributions
    """
    Nmod = len(QSMs)  # number of models
    names = list(QSMs[0]['treedata'].keys())  # attribute names from the first model
    # Initialize output arrays
    treedata = np.zeros((len(names), Nmod),dtype = object)  # Collect all tree attributes
    inputs = np.zeros((Nmod, 3),dtype = object)  # Input parameters
    CylDist = np.zeros((Nmod, 10),dtype = object)  # Cylinder distances
    CylSurfCov = np.zeros((Nmod, 10),dtype = object)  # Surface coverages
    s = 6  # maximum branch order
    OrdDis = np.zeros((Nmod, 4*s),dtype = object)  # Branch order distributions
    r = 20  # maximum cylinder diameter
    CylDiaDis = np.zeros((Nmod, 3*r),dtype = object)  # Cylinder diameter distributions
    CylZenDis = np.zeros((Nmod, 54),dtype = object)  # Zenith direction distributions
    TreeId = np.zeros((Nmod, 2),dtype = object)  # Tree and model indexes
    Keep = np.ones(Nmod, dtype=bool)  # Non-empty models flag

    for i in range(Nmod):
        if len(QSMs[i].get('cylinder',[]))>0:
            # Collect input-parameter values and tree IDs
            p = QSMs[i]['rundata']['inputs']
            inputs[i,:] = [p['PatchDiam1'], p['PatchDiam2Min'], p['PatchDiam2Max']]
            TreeId[i,:] = [p['tree'], p['model']]

            # Collect cylinder-point distances
            D = QSMs[i]['pmdistance']
            CylDist[i,:] = [
                D['mean'], D['TrunkMean'], D['BranchMean'], D['Branch1Mean'], 
                D['Branch2Mean'], D['max'], D['TrunkMax'], D['BranchMax'], 
                D['Branch1Max'], D['Branch2Max']
            ]

            # Collect surface coverages
            D = QSMs[i]['cylinder']['SurfCov']
            T = QSMs[i]['cylinder']['branch'] == 0
            B1 = QSMs[i]['cylinder']['BranchOrder'] == 1
            B2 = QSMs[i]['cylinder']['BranchOrder'] == 2
            
            if not np.any(B1):
                CylSurfCov[i,:] = [
                    np.mean(D), np.mean(D[T]), 0, 0, 0,
                    np.min(D), np.min(D[T]), 0, 0, 0
                ]
            elif not np.any(B2):
                CylSurfCov[i,:] = [
                    np.mean(D), np.mean(D[T]), np.mean(D[~T]), np.mean(D[B1]), 0,
                    np.min(D), np.min(D[T]), np.min(D[~T]), np.min(D[B1]), 0
                ]
            else:
                CylSurfCov[i,:] = [
                    np.mean(D), np.mean(D[T]), np.mean(D[~T]), np.mean(D[B1]), 
                    np.mean(D[B2]), np.min(D), np.min(D[T]), np.min(D[~T]), 
                    np.min(D[B1]), np.min(D[B2])
                ]

            # Collect branch-order distributions
            d = QSMs[i]['treedata']['VolBranchOrd']
            nd = len(d) if d is not None else 0
            if nd > 0:
                a = min(nd, s)
                OrdDis[i, :a] = d[:a]
                OrdDis[i, s:s+a] = QSMs[i]['treedata']['AreBranchOrd'][:a]
                OrdDis[i, 2*s:2*s+a] = QSMs[i]['treedata']['LenBranchOrd'][:a]
                OrdDis[i, 3*s:3*s+a] = QSMs[i]['treedata']['NumBranchOrd'][:a]

            # Collect cylinder diameter distributions
            d = QSMs[i]['treedata']['VolCylDia']
            nd = len(d) if d is not None else 0
            if nd > 0:
                a = min(nd, r)
                CylDiaDis[i, :a] = d[:a]
                CylDiaDis[i, r:r+a] = QSMs[i]['treedata']['AreCylDia'][:a]
                CylDiaDis[i, 2*r:2*r+a] = QSMs[i]['treedata']['LenCylDia'][:a]

            # Collect cylinder zenith direction distributions
            d = QSMs[i]['treedata']['VolCylZen']
            if d is not None and len(d) > 0:
                CylZenDis[i, :18] = d
                CylZenDis[i, 18:36] = QSMs[i]['treedata']['AreCylZen']
                CylZenDis[i, 36:54] = QSMs[i]['treedata']['LenCylZen']

            # Collect the treedata values from each model
            for j in range(len(names)):
                treedata[j,i] = QSMs[i]['treedata'][names[j]]

        else:
            Keep[i] = False

    # Filter out empty models
    treedata = treedata[:, Keep]
    inputs = inputs[Keep, :]
    TreeId = TreeId[Keep, :]
    
    Data = {
        'CylDist': CylDist[Keep, :],
        'CylSurfCov': CylSurfCov[Keep, :],
        'BranchOrdDis': OrdDis[Keep, :],
        'CylDiaDis': CylDiaDis[Keep, :],
        'CylZenDis': CylZenDis[Keep, :]
    }

    return treedata, inputs, TreeId, Data


def compute_metric_value(met, T, treedata, Data):
    """
    Computes metric values based on the specified metric code and input data
    
    Args:
        met: Metric code (1-91)
        T: Index array for selecting data
        treedata: Array of tree attributes
        Data: Dictionary containing various distributions
        
    Returns:
        D: Computed metric value
    """
    
    if met <= 27:  # cylinder distance metrics
        
        D = np.mean(Data['CylDist'][T,:], axis=0) if type(T) is np.ndarray else Data['CylDist'][T,:]
        D[5:10] = 0.5 * D[5:10]  # Half the maximum values 
    
    if met < 10:  # mean cylinder distance metrics
        if met == 1:   # all_mean_dis
            D = D[0]
        elif met == 2:  # trunk_mean_dis
            D = D[1]
        elif met == 3:  # branch_mean_dis
            D = D[2]
        elif met == 4:  # 1branch_mean_dis
            D = D[3]
        elif met == 5:  # 2branch_mean_dis
            D = D[4]
        elif met == 6:  # trunk+branch_mean_dis
            D = D[1] + D[2]
        elif met == 7:  # trunk+1branch_mean_dis
            D = D[1] + D[3]
        elif met == 8:  # trunk+1branch+2branch_mean_dis
            D = D[1] + D[3] + D[4]
        elif met == 9:  # 1branch+2branch_mean_dis
            D = D[3] + D[4]
    
    elif met < 19:  # maximum cylinder distance metrics
        if met == 10:  # all_max_dis
            D = D[5]
        elif met == 11:  # trunk_max_dis
            D = D[6]
        elif met == 12:  # branch_max_dis
            D = D[7]
        elif met == 13:  # 1branch_max_dis
            D = D[8]
        elif met == 14:  # 2branch_max_dis
            D = D[9]
        elif met == 15:  # trunk+branch_max_dis
            D = D[6] + D[7]
        elif met == 16:  # trunk+1branch_max_dis
            D = D[6] + D[8]
        elif met == 17:  # trunk+1branch+2branch_max_dis
            D = D[6] + D[8] + D[9]
        elif met == 18:  # 1branch+2branch_max_dis
            D = D[8] + D[9]
    
    elif met < 28:  # Mean plus maximum cylinder distance metrics
        if met == 19:  # all_mean+max_dis
            D = D[0] + D[5]
        elif met == 20:  # trunk_mean+max_dis
            D = D[1] + D[6]
        elif met == 21:  # branch_mean+max_dis
            D = D[2] + D[7]
        elif met == 22:  # 1branch_mean+max_dis
            D = D[3] + D[8]
        elif met == 23:  # 2branch_mean+max_dis
            D = D[4] + D[9]
        elif met == 24:  # trunk+branch_mean+max_dis
            D = D[1] + D[2] + D[6] + D[7]
        elif met == 25:  # trunk+1branch_mean+max_dis
            D = D[1] + D[3] + D[6] + D[8]
        elif met == 26:  # trunk+1branch+2branch_mean+max_dis
            D = D[1] + D[3] + D[4] + D[6] + D[8] + D[9]
        elif met == 27:  # 1branch+2branch_mean+max_dis
            D = D[3] + D[4] + D[8] + D[9]
    
    elif met < 39:  # Standard deviation metrics
        if met == 28:  # tot_vol_std
            D = np.std(treedata[0,T])
        elif met == 29:  # trunk_vol_std
            D = np.std(treedata[1,T])
        elif met == 30:  # branch_vol_std
            D = np.std(treedata[2,T])
        elif met == 31:  # trunk+branch_vol_std
            D = np.std(treedata[1,T]) + np.std(treedata[2,T])
        elif met == 32:  # tot_are_std
            D = np.std(treedata[11,T])  # Note: Python uses 0-based indexing
        elif met == 33:  # trunk_are_std
            D = np.std(treedata[9,T])
        elif met == 34:  # branch_are_std
            D = np.std(treedata[10,T])
        elif met == 35:  # trunk+branch_are_std
            D = np.std(treedata[9,T]) + np.std(treedata[10,T])
        elif met == 36:  # trunk_len_std
            D = np.std(treedata[4,T])
        elif met == 37:  # branch_len_std
            D = np.std(treedata[5,T])
        elif met == 38:  # trunk+branch_len_std
            D = np.std(treedata[4,T]) + np.std(treedata[5,T])
        elif met == 39:  # branch_num_std
            D = np.std(treedata[7,T])
    
    elif met < 56:  # Branch order metrics

        if type(T) is np.ndarray:
            dis = np.max(Data['BranchOrdDis'][T,:], axis=0) - np.min(Data['BranchOrdDis'][T,:], axis=0)
            M = np.mean(Data['BranchOrdDis'][T,:], axis=0)
            I = M > 0
            dis[I] = dis[I] / M[I]
        else:
            dis = Data['BranchOrdDis'][T,:]
            M = np.mean(dis, axis=0)
            I = M > 0
            dis[I] = dis[I] / M[I]
        
        if met == 40:  # branch_vol_ord3_mean
            D = np.mean(dis[0:3])
        elif met == 41:  # branch_are_ord3_mean
            D = np.mean(dis[6:9])
        elif met == 42:  # branch_len_ord3_mean
            D = np.mean(dis[12:15])
        elif met == 43:  # branch_num_ord3_mean
            D = np.mean(dis[18:21])
        elif met == 44:  # branch_vol_ord3_max
            D = np.max(dis[0:3])
        elif met == 45:  # branch_are_ord3_max
            D = np.max(dis[6:9])
        elif met == 46:  # branch_len_ord3_max
            D = np.max(dis[12:15])
        elif met == 47:  # branch_vol_ord3_max
            D = np.max(dis[18:21])
        elif met == 48:  # branch_vol_ord6_mean
            D = np.mean(dis[0:6])
        elif met == 49:  # branch_are_ord6_mean
            D = np.mean(dis[6:12])
        elif met == 50:  # branch_len_ord6_mean
            D = np.mean(dis[12:18])
        elif met == 51:  # branch_num_ord6_mean
            D = np.mean(dis[18:24])
        elif met == 52:  # branch_vol_ord6_max
            D = np.max(dis[0:6])
        elif met == 53:  # branch_are_ord6_max
            D = np.max(dis[6:12])
        elif met == 54:  # branch_len_ord6_max
            D = np.max(dis[12:18])
        elif met == 55:  # branch_vol_ord6_max
            D = np.max(dis[18:24])
    
    elif met < 68:  # Cylinder diameter distribution metrics
        if type(T) is np.ndarray:
            dis = np.max(Data['CylDiaDis'][T,:], axis=0) - np.min(Data['CylDiaDis'][T,:], axis=0)
            M = np.mean(Data['CylDiaDis'][T,:], axis=0)
            I = M > 0
            dis[I] = dis[I] / M[I]
        else:
            dis = Data['CylDiaDis'][T,:]
            M = np.mean(dis, axis=0)
            I = M > 0
            dis[I] = dis[I] / M[I]
        
        if met == 56:  # cyl_vol_dia10_mean
            D = np.mean(dis[0:10])
        elif met == 57:  # cyl_are_dia10_mean
            D = np.mean(dis[20:30])
        elif met == 58:  # cyl_len_dia10_mean
            D = np.mean(dis[40:50])
        elif met == 59:  # cyl_vol_dia10_max
            D = np.max(dis[0:10])
        elif met == 60:  # cyl_are_dia10_max
            D = np.max(dis[20:30])
        elif met == 61:  # cyl_len_dia10_max
            D = np.max(dis[40:50])
        elif met == 62:  # cyl_vol_dia20_mean
            D = np.mean(dis[0:20])
        elif met == 63:  # cyl_are_dia20_mean
            D = np.mean(dis[20:40])
        elif met == 64:  # cyl_len_dia20_mean
            D = np.mean(dis[40:60])
        elif met == 65:  # cyl_vol_dia20_max
            D = np.max(dis[0:20])
        elif met == 66:  # cyl_are_dia20_max
            D = np.max(dis[20:40])
        elif met == 67:  # cyl_len_dia20_max
            D = np.max(dis[40:60])
    
    elif met < 74:  # Cylinder zenith distribution metrics
        if type(T) is np.ndarray:
            dis = np.max(Data['CylZenDis'][T,:], axis=0) - np.min(Data['CylZenDis'][T,:], axis=0)
            M = np.mean(Data['CylZenDis'][T,:], axis=0)
            I = M > 0
            dis[I] = dis[I] / M[I]
        else:
            dis = Data['CylZenDis'][T,:]
            M = np.mean(dis, axis=0)
            I = M > 0
            dis[I] = dis[I] / M[I]
        
        if met == 68:  # cyl_vol_zen_mean
            D = np.mean(dis[0:18])
        elif met == 69:  # cyl_are_zen_mean
            D = np.mean(dis[18:36])
        elif met == 70:  # cyl_len_zen_mean
            D = np.mean(dis[36:54])
        elif met == 71:  # cyl_vol_zen_max
            D = np.max(dis[0:18])
        elif met == 72:  # cyl_are_zen_max
            D = np.max(dis[18:36])
        elif met == 73:  # cyl_len_zen_max
            D = np.max(dis[36:54])
    
    elif met < 92:  # Surface coverage metrics
        if type(T) is np.ndarray:
            D = 1 - np.mean(Data['CylSurfCov'][T,:], axis=0)
        else:
            D = 1 - Data['CylSurfCov'][T,:]

        
        if met == 74:  # all_mean_surf
            D = D[0]
        elif met == 75:  # trunk_mean_surf
            D = D[1]
        elif met == 76:  # branch_mean_surf
            D = D[2]
        elif met == 77:  # 1branch_mean_surf
            D = D[3]
        elif met == 78:  # 2branch_mean_surf
            D = D[4]
        elif met == 79:  # trunk+branch_mean_surf
            D = D[1] + D[2]
        elif met == 80:  # trunk+1branch_mean_surf
            D = D[1] + D[3]
        elif met == 81:  # trunk+1branch+2branch_mean_surf
            D = D[1] + D[3] + D[4]
        elif met == 82:  # 1branch+2branch_mean_surf
            D = D[3] + D[4]
        elif met == 83:  # all_min_surf
            D = D[5]
        elif met == 84:  # trunk_min_surf
            D = D[6]
        elif met == 85:  # branch_min_surf
            D = D[7]
        elif met == 86:  # 1branch_min_surf
            D = D[8]
        elif met == 87:  # 2branch_min_surf
            D = D[9]
        elif met == 88:  # trunk+branch_min_surf
            D = D[6] + D[7]
        elif met == 89:  # trunk+1branch_min_surf
            D = D[6] + D[8]
        elif met == 90:  # trunk+1branch+2branch_min_surf
            D = D[6] + D[8] + D[9]
        elif met == 91:  # 1branch+2branch_min_surf
            D = D[8] + D[9]
    
    return D

def cloud_to_image(cloud,resolution=.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size = resolution)
    voxels = voxel_grid.get_voxels()
    indices = np.stack(np.array([np.array(vx.grid_index) for vx in voxels]))
    voxel_array = np.ones((indices.max(axis=0)+1))
    for x, y, z in indices: voxel_array[x, y, z] = 0
    return voxel_array,voxel_grid,indices


def parse_args(argv):
    """
        Define run values based on command line args. Options for params are:
        --intensity: filter point cloud based on intensity
        --custominput: user sets specific patch diameters to test
        --ipd: initial patch diameter
        --minpd: min patch diameter
        --maxpd: maximum patch diameter
        --name: specificies a name of the tree different than the file
        --parallel: runs in parallel
        --numcores: specify number of cores to use in parallel mode
        --optimum: specify an optimum value to select best model to save
        --help: displays the run options
        -verbose: verbose mode
        -h: displays the run options
        -v: verbose mode
    """
    i = 0
    current_arg = "Invalid Arg"
    args = {"Custom":False,"Verbose":False,"Parallel":False,"Normalize":False,"Intensity":0, "PatchDiam1":1,"PatchDiam2Min":1,"PatchDiam2Max":1,"Name":"","Cores":1,"Optimum":[],"Directory":None}
    help= """List of valid arguments. Filename must be first, followed by the below arguments
    --intensity: filter point cloud based on intensity: 
        Must be followed with a valid integer
    --normalize: recenter point cloud locations. Use this if your point cloud location values are very large
    --custominput: user sets specific patch diameters to test
    --ipd: initial patch diameter 
        Must be followed by at least one value. A single integer if --custominput is not indicated, a series of decimals if --custominput is indicated
    --minpd: min patch diameter
        Must be followed by at least one value. A single integer if --custominput is not indicated, a series of decimals if --custominput is indicated
    --maxpd: maximum patch diameter
        Must be followed by at least one value. A single integer if --custominput is not indicated, a series of decimals if --custominput is indicated
    --name: specifies a name of the run. This will be appended to the name generated by TreeQSM
    --outputdirectory: specifies the directory to put the "results" folder
    --numcores: specify number of cores to use to process files in parallel. Only valid in batched mode
        Must be a single integer
    --optimum: specify an optimum metric to select best model to save
        Must be a valid optimum as defined by the documentation. If multiple optimums are listed, the best model will be saved for each optimum metric
    --help: displays the run options
    -verbose: verbose mode, displays outputs from TreeQSM as it runs
    -h: displays the run options
    -v: verbose mode"""
    while i <len(argv):
        match argv[i]:
            case "--threshold":
                current_arg = "Intensity"
            case "--custominput":
                args["Custom"] = True
                current_arg = "Invalid Arg"
            case "--ipd":
                current_arg = "PatchDiam1"
            case "--minpd":
                current_arg = "PatchDiam2Min"
            case "--maxpd":
                current_arg = "PatchDiam2Max"
            case "--name":
                current_arg = "Name"
            case "--parallel":
                args["Parallel"] = True
                current_arg = "Invalid Arg"
            case "--numcores":
                current_arg = "Cores"
            case "--optimum":
                current_arg = "Optimum"
            case "--help":
                sys.stdout.write(help)
                return "Help"
            case "-h":
                sys.stdout.write(help)
                return "Help"
            case "--verbose":
                args["Verbose"]=True
                current_arg = "Invalid Arg"
            case "-v":
                args["Verbose"]=True
                current_arg = "Invalid Arg"
            case "--normalize":
                args["Normalize"]=True
                current_arg = "Invalid Arg"
            case "--outputdirectory":
                current_arg="Directory"
            case _:
                if current_arg == "Invalid Arg":
                    sys.stdout.write(f"Argument {argv[i]} not valid in this position. See --help if you need help with arguments. System will continue with remaining arguments")
                elif current_arg in ["PatchDiam1","PatchDiam2Min","PatchDiam2Max"]:
                    for item in argv[i].split(","):
                        arg = item.strip().strip(",")
                        try:
                            arg = float(arg)
                        except:
                            sys.stdout.write(f"Argument {argv[i]} should be a valid number")
                            continue
                        if arg != "":
                            if args[current_arg]==1:
                                args[current_arg] = [arg]
                            else:
                                args[current_arg].append(arg)
                elif current_arg == 'Optimum':
                    arg = argv[i].strip().strip(",")
                    args[current_arg].append(arg)  
                elif current_arg in ["Name","Directory"]:
                    arg = argv[i].strip().strip(",")
                    args[current_arg] = arg 
                else:
                    try:
                        arg = int(float(argv[i].strip().strip(",")))
                        args[current_arg] = arg  
                    except:
                        sys.stdout.write(f"Argument {argv[i]} should be a valid integer")
                    

                
        i+=1
    if args["Custom"]:
        if type(args["PatchDiam1"]) != list or type(args["PatchDiam2Min"]) != list or type(args["PatchDiam2Max"]) != list:
            print(args)
            sys.stdout.write(f"If --custominput is selected, values for --ipd (PatchDiam1) --minpd (PatchDiam2Min) --maxpd (PatchDiam2Max). See --help if needed")
            return "ERROR"
    else:
        if type(args["PatchDiam1"]) != list:
            args["PatchDiam1"]=args["PatchDiam1"][0]
        if type(args["PatchDiam2Min"]) != list:
            args["PatchDiam2Min"]=args["PatchDiam2Min"][0]
        if type(args["PatchDiam2Max"]) != list:
            args["PatchDiam2Max"]=args["PatchDiam2Max"][0]

    return args


