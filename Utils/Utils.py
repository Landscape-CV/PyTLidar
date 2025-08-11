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
import csv
import sys
# from scipy.spatial import ConvexHull
# import alphashape
# from shapely.geometry import Polygon
from numba import jit
from numba.experimental import jitclass
import laspy
try:
    from plotting.qsm_plotting import qsm_plotting
except ImportError:
    from ..plotting.qsm_plotting import qsm_plotting
import open3d as o3d
from robpy.covariance import DetMCD,FastMCD
from scipy.spatial.transform import Rotation 




    
    

def load_point_cloud(file_path, intensity_threshold = 0, full_data = False):
    """
    Load a point cloud from LAS or LAZ files.

    Args:
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
            I = point_data[:, 3] >= intensity_threshold
            point_cloud = point_data[I, :3]
        else:
            raise ValueError("Unsupported format in XYZ file.")
        return point_cloud if not full_data else (point_cloud, point_data)
    with laspy.open(file_path) as las:
        point_data = las.read()
        point_data = np.vstack((point_data.x, point_data.y, point_data.z,point_data.intensity)).T.astype('float64')
        I = point_data[:,3]>=intensity_threshold
        point_data = point_data[I]
        point_cloud = point_data[:,0:3]
    return point_cloud if not full_data else (point_cloud,point_data)



                



@jit()
def average(X):

    """
    Computes the average of the columns of the matrix X.

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
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








def mat_vec_subtraction(A, v):
    """
    Subtracts from each row of the matrix A the vector v.
    If A is an (n x m)-matrix, then v needs to be an m-element vector.

    Args:
        A (array-like): Input matrix of shape (n, m).
        v (array-like): 1D array of length m.

    Returns:
        numpy.ndarray: The matrix after subtracting v from each row.
    """
    A = np.array(A, dtype=float)
    v = np.array(v, dtype=float)
    return A - v




@jit()
def rotation_matrix(A, angle):
    """
    Returns the rotation matrix for the given axis A and angle (in radians).

    Args:
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

    Args:
        V (array-like): A 2D array where each row is a unit vector.

    Returns:
        v (numpy.ndarray): A 1D unit vector that maximizes the sum of squared dot products with V.
        mean_res (float): The mean of the absolute dot products between each row of V and v.
        sigmah (float): The standard deviation of these absolute dot products.
        residual (numpy.ndarray): 1D array containing the absolute dot products for each row.
    """
    _, _, vh = np.linalg.svd(V, full_matrices=False)


    return vh[0]




def unique_elements_array(arr,False_mask=None):
    """
    Alias for np.unique(arr) to maintain consistency with other functions.
    Args:
        arr (array-like): Input array.
        False_mask (optional): Not used, included for compatibility.
    Returns:
        numpy.ndarray: A 1D array containing the unique elements."""

    return np.unique(arr)



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

def set_difference(Set1,Set2,Fal):
    """
        Performs the set difference so that the common elements of Set1 and Set2
        are removed from Set1, which is the output. Uses logical vector whose
        length must be up to the maximum element of the sets.

        Args:
            Set1 (array-like): A list or array of integer indices.
            Set2 (array-like): A list or array of integer indices.
            Fal (numpy.ndarray of bool): A boolean tracker array (sized to cover possible indices).
        Returns:
            numpy.ndarray: An array containing the elements of Set1 that are not in Set2.
    """

   
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

    Args:
        QSM (dict): Dictionary with keys "cylinder", "branch", and "treedata". Created during the TreeQSM process.
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
    Bx = np.round(1000 * branch["x"]) / 1000
    By = np.round(1000 * branch["y"]) / 1000
    Bz = np.round(1000 * branch["z"]) / 1000

    BranchData = np.column_stack((BOrd, BPar, BDia, BVol, BAre, BLen, BHei, BAng, BAzi, BZen,Bx,By,Bz))
    NamesB = ["order", "parent", "diameter (m)", "volume (L)", "area (m^2)",
                "length (m)", "height (m)", "angle (deg)", "azimuth (deg)", "zenith (deg)","Location X", "Location Y", "Location Z"]

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

    Args:
    P (numpy.ndarray): Point cloud, shape (n_points, 3).
    EL (float): Length of the cube edges.
    NE (int): Number of empty edge layers (default=3).

    Returns:
    (list, np.Array,list): Partition (list of lists of point indices), CubeCoord (n_points x 3 matrix of cube coordinates),
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


def growth_volume_correction(cylinder, inputs):
    """
    Uses a growth volume allometry approach to modify the radii of cylinders.
    The allometry model is: Predicted Radius = a * (GrowthVolume)^b + c.
    

    Args:
        cylinder (dict): Cylinder data used in the TreeQSM process:
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





@jit(nopython=True)
def surface_coverage_prep(P, Axis, Point, nl, ns, Dmin=None, Dmax=None):
    """
    First half of surface coverage moved out for numba compilation
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

    Args:
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

    Args:
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




def package_outputs(models,cyl_htmls):
    
    tree_data_figures =[]
    segment_plots=[]
    cyl_plots=[]
    for i in range(len(models)):
        run_name = models[i]['rundata']['inputs']['name']+"_"+str(i)
        figs =[]
        for j,fig in enumerate(models[i]['treedata']['figures']):
            save_name = os.path.join("results",f"tree_data_{run_name}_charts_{j}_{models[i]['file_id']}.pdf")
            fig.dpi=1000
            fig.savefig(save_name,format ='pdf')
            figs.append(save_name)
        figs = tuple(figs)
        tree_data_figures.append(figs)

        # segment_plots.append(qsm_plotting(models[i]['points'], models[i]['cover'], models[i]['segment'],models[i]))
        #keeping segments out this for now, need to make more efficient

        

    return {"tree_data":tuple(tree_data_figures),"cylinders":tuple(cyl_htmls)}

# @jit(nopython=True,parallel=True,nogil=True)
def assign_segments(cloud,segments,cover_sets,array =False):
    if not array:
        point_segments = np.zeros((cloud.shape[0]),dtype = np.int64)-1
        for i,segment in enumerate(segments):
            I = np.where(np.isin(cover_sets, segment))[0]
            point_segments[I] = i
        
        return point_segments
    else:
        point_segments = np.zeros((cloud.shape[0]),dtype = np.int64)-1
        
        for segment in np.unique(segments):
            I = np.where(cover_sets == segment)[0]
            
            point_segments[I] = segment
        
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

def save_fit(cyl_dist,filename):
    filename = filename+"_qsm_cyldist.csv"
    header = ['mean' ,'TrunkMean', 'BranchMean', 'Branch1Mean', 'Branch2Mean', 'max', 'TrunkMax', 'BranchMax', 'Branch1Max', 'Branch2Max']
    with open(filename,"w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(cyl_dist)



def check_for_bends(segment_cloud,num_test_regions = 5,threshold = .2):
    """
    Check for bends in a point cloud segment by analyzing the angles between segments.
    
    Args:
        segment_cloud: Point cloud data as a numpy array of shape (N, 3).
        num_test_regions: Number of regions to test for bends.
        
    Returns:
        Boolean indicating if bends were detected.
    """
    segsize = len(segment_cloud)//num_test_regions
    if segsize <20:
        return False
    
    
    

    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(segment_cloud)
    try:
        obb = pcd.get_oriented_bounding_box()
    except:
        return False


    center = obb.center


    bend = True
    prev_dims = np.sort(obb.extent)
    for i in range(num_test_regions):
        
        # next_seg = segment_cloud[i*segsize:(i+1)*segsize]
        # pcd.points = o3d.utility.Vector3dVector(next_seg)
        next_seg = segment_cloud[:(i+1)*segsize]
        pcd.points = o3d.utility.Vector3dVector(next_seg)
        try:
            obb = pcd.get_oriented_bounding_box()
        except:
            pass

        # try:
        #     obb = pcd.get_oriented_bounding_box()

        # except:
        #     return False
        max_bound = obb.get_max_bound()
        min_bound = obb.get_min_bound()
        bound_range = max_bound - min_bound
        if np.all(center < max_bound-bound_range*.2) and np.all(center > min_bound+bound_range*.2):
            bend = False

        dims = np.sort(obb.extent)
        # if i>0:
        both_dims = np.array([dims,prev_dims])
        min_dims = np.min(both_dims,axis=0)
        max_dims = np.max(both_dims,axis=0)
        if min_dims[0]*2<max_dims[0] or (min_dims[0]<max_dims[0] and min_dims[1]*2<max_dims[1]):
            return True
            
        # prev_dims = dims
        

    return bend
    

def split_segments(segment_cloud, num_test_regions = 5, angle_threshold = 60):
    """
    Find bends in a point cloud based on the distance between points.
    
    Args:
        cloud: Point cloud data as a numpy array of shape (N, 3).
        num_test_regions: Number of regions to test for bends.
        
    Returns:
        Array indicating if point is in new segment
    """

    
    segs = np.zeros(len(segment_cloud),dtype=int)

    
    if not check_for_bends(segment_cloud,num_test_regions):
        return segs

    segsize = len(segment_cloud)//num_test_regions
    initial_seg = segment_cloud[:segsize]
    last_seg = segment_cloud[1*segsize:2*segsize]
    a = np.mean(initial_seg, axis=0)
    b = np.mean(last_seg, axis=0)
    initial_vec = (b-a)/(np.linalg.vector_norm(b-a))
    last_vec = initial_vec

    # full_pcd = o3d.geometry.PointCloud()
    # full_pcd.points = o3d.utility.Vector3dVector(segment_cloud)


    for i in range(2,num_test_regions):
        
        next_seg = segment_cloud[i*segsize:(i+1)*segsize]

        
        if len(next_seg) < 10: 
            break
        new_vec1 = (next_seg[-1]-next_seg[0])#/(np.linalg.vector_norm(c-b))
        
        
        angle = np.rad2deg(np.arccos(np.dot(last_vec, new_vec1)/(np.linalg.norm(last_vec)*np.linalg.norm(new_vec1))))
        if angle > angle_threshold:
            segs[i*segsize:] = 1
            return segs
            
        last_vec = new_vec1
        last_seg = next_seg

    return segs

    


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
    if len(argv)>0 and argv[0] == "-m":
        argv=argv[1:]  # Remove -m flag for compatibility with existing code
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
    # else:
    #     if type(args["PatchDiam1"]) != list:
    #         args["PatchDiam1"]=[args["PatchDiam1"]]
    #     if type(args["PatchDiam2Min"]) != list:
    #         args["PatchDiam2Min"]=[args["PatchDiam2Min"]]
    #     if type(args["PatchDiam2Max"]) != list:
    #         args["PatchDiam2Max"]=[args["PatchDiam2Max"]]

    return args


def color_o3d_clouds(clouds):
    for cloud in clouds:
        color = np.random.random(3)
        color_array = np.repeat(np.array([color]),len(np.asarray(cloud.points)),axis = 0)
        cloud.colors =  o3d.utility.Vector3dVector(color_array)

def get_axis(point_cloud):

    point_cloud = np.random.permutation(point_cloud)[:15]
    # mcd = FastMCD()
    # try:
    #     covariance = mcd.calculate_covariance(point_cloud)
    # except:
    # try:
    #     mcd = DetMCD()
    #     covariance = mcd.calculate_covariance(point_cloud)
    # except Exception as e:
    #     print("Failed to find covariance")
    #     raise e
            

    
    # mean = mcd.location_
    U, S, Vt = np.linalg.svd(point_cloud, full_matrices=False)
    first_pc = Vt[0, :] 
    return first_pc
def rotate_cloud(point_cloud,axis):
    """
    Rotate a point cloud around the z-axis to align it with the x-axis.
    
    Args:
        point_cloud: Point cloud data as a numpy array of shape (N, 3).
        axis: Axis to align with, should be a 3D vector.
        
    Returns:
        Rotated point cloud as a numpy array of shape (N, 3).
    """
    rotvec = Rotation.from_rotvec(axis)
    rotated_cloud = rotvec.as_matrix() @ point_cloud.T
    rotated_cloud = rotated_cloud.T
    # rotated_cloud = rotvec.apply(Q0)
    return rotated_cloud
    
def get_axis_sort(point_cloud,axis):
    """
    Returns the indexes sorted from "bottom" to "top" of cloud rotated to align with axis.
    Args:
        point_cloud: Point cloud data as a numpy array of shape (N, 3).
        axis: Axis to align with, should be a 3D vector."""
    rotated_cloud =rotate_cloud(point_cloud,axis)
    sorted_indices = np.argsort(rotated_cloud[:, 2])
    return sorted_indices


    





