import numpy as np

def svd_eigen(arr):
    """
    Calculates eigenvalues and eigenvectors of an array using SVD.

    Parameters
    ----------
    arr : array
        nxm numpy.ndarray where n is the number of samples and m is the number
        of dimensions.

    Returns
    -------
    centroid : array
        The centroid coordinate of 'points'.
    evals : array
        1xm numpy.ndarray containing the calculated eigenvalues in decrescent
        order.
    evecs : array
        Eigenvectors corresponding to eigenvalues.
    """

    # Calculating centroid coordinates of points in 'arr'.
    centroid = np.average(arr, axis=0)
    # Decentralization
    norm_points = arr - centroid
    # Calculate scatter matrix, one step of SVD
    scatter_matrix = (np.dot(np.transpose(norm_points), norm_points))
    # Running SVD on scatter matrix.
    _, evals, evecs = np.linalg.svd(scatter_matrix)

    return centroid, evals, evecs

def pca_transform(points, centroid, evecs):
    """
    Coordinate transformation through PCA.

    Parameters
    ----------
    points : array
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).
    centroid : array
        The centroid coordinate of 'points'.
    evecs : array
        Eigenvectors corresponding to eigenvalues.

    Returns
    -------
    points_transformed : array
        The 3-D points (z,x,y) after coordinate transformation.
    """

    # Decentralization
    norm_points = points - centroid

    trans_matrix = evecs[0:, :]
    # Perform coordinates transformation
    points_transformed = np.dot(norm_points, np.transpose(trans_matrix))

    return points_transformed
