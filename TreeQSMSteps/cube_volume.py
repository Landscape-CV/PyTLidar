import numpy as np
try:
    from ..Utils import Utils
except ImportError:
    import Utils.Utils as Utils 



def cube_volume(P, cylinder, EL, NE = 3):
    """
    Calculate cylinder volume in each cube of user defined size.

    Args:
        P (numpy.ndarray): Point cloud, shape (n_points, 3).
        cylinder : TreeQSM fitted cylinders.
        EL (float): Length of the cube edges.
        NE (int): Number of empty edge layers (default=3).

    Returns:
        (3D numpy array): Cylinder volume in each cube.
    """

    P = np.array(P, dtype=float)

    # The vertices of the bounding box containing P
    Min = np.min(P, axis=0)
    Max = np.max(P, axis=0)
    #print(Max - Min)  # tree size

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

    # Calculate cube coordinates of each cylinder
    CubeCoord = np.floor((cylinder['start'] - Min) / EL).astype(int) + NE + 1

    # Lexicographical order for sorting
    LexOrd = (CubeCoord[:, 0]
              + (CubeCoord[:, 1] - 1) * N[0]
              + (CubeCoord[:, 2] - 1) * (N[0] * N[1]))  # start from cylinder 0
    SortOrd = np.lexsort((CubeCoord[:, 2], CubeCoord[:, 1], CubeCoord[:, 0]))  # cylinder index in lexsort sequence
    LexOrd = LexOrd[SortOrd]  # start from min cube

    cube_volume = np.zeros((N[0], N[1], N[2]), dtype=object)

    n_cylinders = cylinder['radius'].shape[0]  # number of cylinders
    c = 0
    total_volume = 0

    while c < n_cylinders:
        t = 1
        curr_volume = 1000 * cylinder['radius'][SortOrd[c]] ** 2 * np.pi * cylinder['length'][SortOrd[c]]  # volume of first cylinder
        while (c + t < n_cylinders) and (LexOrd[c] == LexOrd[c + t]):
            curr_volume += 1000 * cylinder['radius'][SortOrd[c + t]] ** 2 * np.pi * cylinder['length'][SortOrd[c + t]]
            t += 1
        q = SortOrd[c]
        # Calculate the volume of cylinders in current cube
        cube_volume[CubeCoord[q, 0] - 1, CubeCoord[q, 1] - 1, CubeCoord[q, 2] - 1] = curr_volume
        #print(CubeCoord[q, 0] - 1, CubeCoord[q, 1] - 1, CubeCoord[q, 2] - 1, curr_volume)
        total_volume += curr_volume
        c += t
    #print(total_volume)
    return cube_volume