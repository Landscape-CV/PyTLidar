import numpy as np
import math

def circleFit(arr):
    """
    Least square 2D-circle fitting

    Parameters
    ----------
    arr : array
        input points with (x,y) coordinate.
    Returns
    -------
    circle_x : float
         x coordinate of circle center point.
    circle_y : float
         y coordinate of circle center point.
    circle_r : float
         radius of fitted circle.
    """

    n = arr.shape[0]
    assert n >= 3, "point_cloud must be an array with at least\
     3 points"
    sum_x1 = 0.0
    sum_y1 = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_x3 = 0.0
    sum_y3 = 0.0
    sum_x1y1 = 0.0
    sum_x1y2 = 0.0
    sum_x2y1 = 0.0
    for i in arr:
        x = i[0]
        y = i[1]
        sum_x1 += x
        sum_x2 += x * x
        sum_x3 += x * x * x
        sum_y1 += y
        sum_y2 += y * y
        sum_y3 += y * y * y
        sum_x1y1 += x * y
        sum_x1y2 += x * y * y
        sum_x2y1 += x * x * y

    c = n * sum_x2 - sum_x1 * sum_x1
    d = n * sum_x1y1 - sum_x1 * sum_y1
    e = n * sum_x3 + n * sum_x1y2 - (sum_x2 + sum_y2) * sum_x1
    g = n * sum_y2 - sum_y1 * sum_y1
    h = n * sum_x2y1 + n * sum_y3 - (sum_x2 + sum_y2) * sum_y1

    a = (h * d - e * g) / (c * g - d * d)
    b = (h * c - e * d) / (d * d - g * c)
    c = -(sum_x2 + sum_y2 + a * sum_x1 + b * sum_y1) / n

    circle_x = -0.5 * a
    circle_y = -0.5 * b
    circle_r = 0.5 * math.sqrt(a * a + b * b - 4 * c)

    return circle_x, circle_y, circle_r

def getRootPt(arr, lower_h=0.00, upper_h=0.05):
    """
    Fit a root point at the trunk base.

    Parameters
    ----------
    arr : array
        input point cloud with (x,y,z) coordinate.
    lower_h : float
        the relative lower height of trunk segment used to fit a circle.
    upper_h : float
        the relative upper height of trunk segment used to fit a circle.
    Returns
    -------
    fitted root point : array
        1x3 2d-array of root point.
    trunk segment points : array
        index of extracted trunk segment points.
    """
    min_z = np.min(arr[:, 2])

    # get the trunk segment points idx.
    arr_sage = []
    for i, (p) in enumerate(arr):
        if((p[2]>min_z+lower_h)and(p[2]<min_z+upper_h)):
            arr_sage.append(i)
    # fit a 2D-circle based on the trunk segment points, only x, y coordinates are used.
    x, y, r = circleFit(arr[arr_sage])

    return np.array([[x, y, min_z+lower_h]]), np.array(arr_sage)

def circleFitError(arr):
    """
    fit a 2D circle and calculate the fit error.

    Parameters
    ----------
    arr : array
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).

    Returns
    -------
    fitError : float
    circle_r : float
    """
    circle_x, circle_y, circle_r = circleFit(arr)
    n = arr.shape[0]
    sum = 0.0
    for point in arr:
        sum += math.pow(math.sqrt(math.pow(point[0]-circle_x, 2) +
                                      math.pow(point[1]-circle_y, 2))-circle_r, 2)
    return math.sqrt(sum/n)/circle_r, circle_r






