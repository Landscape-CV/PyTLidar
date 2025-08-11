import numpy as np
from scipy.spatial import ConvexHull
from alphashape import alphashape 
try:
    from ..Utils import Utils
except ImportError:
    import Utils.Utils as Utils
try:
    from . import LSF
except ImportError:
    import LSF

try:
    from ..plotting import PlottingUtils
except ImportError:
    import plotting.PlottingUtils as PlottingUtils
import matplotlib.pyplot as plt
import matplotlib
def tree_data(cylinder, branch, trunk, inputs, iter = 0):
    """
    Calculate tree attributes from cylinder QSM data.

    Args :
        cylinder (dict): Dictionary containing cylinder data (radius, length, start, axis, branch).
        branch (dict): Dictionary containing branch data (order, volume, length, diameter, height, angle, zenith, azimuth).
        trunk (numpy.ndarray): Point cloud of the trunk.
        inputs (dict): Input structure defining if results are displayed, plotted, and if triangulation is computed.

    Returns:
        (dict, int/dict): Tree data and triangulation results, triangulation results are 0 if not computed.
    """
    matplotlib.rcParams.update({'font.size': 8})
    # Define some variables from cylinder
    Rad = cylinder['radius']
    Len = cylinder['length']
    nc = len(Rad)
    ind = np.arange(nc)
    Trunk = cylinder['branch'] == 0  # Trunk cylinders

    # Initialize treedata dictionary
    treedata = {}

    # --------------------------------------------------------------
    # Tree attributes from cylinders
    # ---------------------------------------------------------------------
    # Volumes, areas, lengths, branches
    treedata['TotalVolume'] = 1000 * np.pi * np.sum(Rad**2 * Len)
    treedata['TrunkVolume'] = 1000 * np.pi * np.sum(Rad[Trunk]**2 * Len[Trunk])
    treedata['BranchVolume'] = 1000 * np.pi * np.sum(Rad[~Trunk]**2 * Len[~Trunk])
    
    # Tree height
    bottom = np.min(cylinder['start'][:, 2])
    top_idx = np.argmax(cylinder['start'][:, 2])
    top = cylinder['start'][top_idx, 2]
    if cylinder['axis'][top_idx, 2] > 0:
        top += Len[top_idx] * cylinder['axis'][top_idx, 2]
    treedata['TreeHeight'] = top - bottom
    
    # Lengths
    treedata['TrunkLength'] = np.sum(Len[Trunk])
    treedata['BranchLength'] = np.sum(Len[~Trunk])
    treedata['TotalLength'] = treedata['TrunkLength'] + treedata['BranchLength']
    
    # Number of branches and maximum branch order
    NB = len(branch['order']) - 1  # Number of branches
    treedata['NumberBranches'] = NB
    BO = np.max(branch['order'])  # Maximum branch order
    treedata['MaxBranchOrder'] = BO
    
    # Areas
    treedata['TrunkArea'] = 2 * np.pi * np.sum(Rad[Trunk] * Len[Trunk])
    treedata['BranchArea'] = 2 * np.pi * np.sum(Rad[~Trunk] * Len[~Trunk])
    treedata['TotalArea'] = 2 * np.pi * np.sum(Rad * Len)

    # ---------------------------------------------------------------------
    # Diameter at breast height (DBH)
    # ---------------------------------------------------------------------
    treedata = dbh_cylinder(treedata, trunk, Trunk, cylinder, ind)

    # ---------------------------------------------------------------------
    # Crown measures, vertical profile, and spreads
    # ---------------------------------------------------------------------
    treedata, spreads = crown_measures(treedata, cylinder, branch)
    treedata['VerticalProfile'] = np.mean(spreads, axis=1)
    treedata['spreads'] = spreads

    # ---------------------------------------------------------------------
    # Trunk volume and DBH from triangulation
    # ---------------------------------------------------------------------
    if inputs['Tria']:
        raise Exception("Triangulation is not yet tested and is currently disabled.")   
        treedata, triangulation = triangulate_stem(treedata, cylinder, branch, trunk)
    else:
        triangulation = 0

    # ---------------------------------------------------------------------
    # Tree location
    # ---------------------------------------------------------------------
    treedata['location'] = cylinder['start'][0, :]

    # ---------------------------------------------------------------------
    # Stem taper
    # ---------------------------------------------------------------------
    R = Rad[Trunk]
    n = len(R)
    Taper = np.zeros((n + 1, 2))
    Taper[0, 1] = 2 * R[0]
    Taper[1:, 0] = np.cumsum(Len[Trunk])
    Taper[1:, 1] = np.hstack((2 * R[1:], 2 * R[-1]))
    treedata['StemTaper'] = Taper.T

    # ---------------------------------------------------------------------
    # CYLINDER DISTRIBUTIONS
    # ---------------------------------------------------------------------
    # Wood part diameter distributions
    treedata = cylinder_distribution(treedata, cylinder, 'Dia')

    # Wood part height distributions
    treedata = cylinder_height_distribution(treedata, cylinder, ind)

    # Wood part zenith direction distributions
    treedata = cylinder_distribution(treedata, cylinder, 'Zen')

    # Wood part azimuth direction distributions
    treedata = cylinder_distribution(treedata, cylinder, 'Azi')

    # ---------------------------------------------------------------------
    # BRANCH DISTRIBUTIONS
    # ---------------------------------------------------------------------
    # Branch order distributions
    treedata = branch_order_distribution(treedata, branch)

    # Branch diameter distributions
    treedata = branch_distribution(treedata, branch, 'Dia')

    # Branch height distribution
    treedata = branch_distribution(treedata, branch, 'Hei')

    # Branch angle distribution
    treedata = branch_distribution(treedata, branch, 'Ang')

    # Branch azimuth distribution
    treedata = branch_distribution(treedata, branch, 'Azi')

    # Branch zenith distribution
    treedata = branch_distribution(treedata, branch, 'Zen')

    # ---------------------------------------------------------------------
    # Convert to single precision
    # ---------------------------------------------------------------------
    for key in treedata:
        treedata[key] = np.float32(treedata[key])

    # ---------------------------------------------------------------------
    # Display and plotting
    # ---------------------------------------------------------------------
    if inputs['disp'] == 2:
        # Generate units for displaying the treedata
        Names = list(treedata.keys())
        Units = [''] * len(Names)
        for i, name in enumerate(Names):
            if not inputs["Tria"] and name.startswith('CrownVolumeAlpha'):
                m = i+1
            elif inputs["Tria"] and name.startswith('TriaTrunkLength'):
                m = i+1
            if name.startswith('DBH'):
                Units[i] = 'm'
            elif name.endswith('ume'):
                Units[i] = 'L'
            elif name.endswith('ght'):
                Units[i] = 'm'
            elif name.endswith('gth'):
                Units[i] = 'm'
            elif name.startswith('vol'):
                Units[i] = 'L'
            elif name.startswith('len'):
                Units[i] = 'm'
            elif name.endswith('rea'):
                Units[i] = 'm^2'
            elif name.startswith('loc'):
                Units[i] = 'm'
            elif name.endswith('aConv'):
                Units[i] = 'm^2'
            elif name.endswith('aAlpha'):
                Units[i] = 'm^2'
            elif name.endswith('eConv'):
                Units[i] = 'm^3'
            elif name.endswith('eAlpha'):
                Units[i] = 'm^3'
            elif name.endswith('Ave'):
                Units[i] = 'm'
            elif name.endswith('Max'):
                Units[i] = 'm'

        # Display treedata
        print('------------')
        print('  Tree attributes:')
        for i in range(m):
            name = Names[i]
            if name == 'DBHtri':
                print('  -----')
                print('  Tree attributes from triangulation:')
            print(f'  {name} = {treedata[name]} {Units[i]}')
        print('  -----')

    
        

    # Plot distributions
    Q = {'treedata': treedata}

    # Figure 6: Stem taper and cylinder distributions
    plt.figure(6+iter*10)
    plt.subplot(2, 4, 1)
    plt.plot(treedata['StemTaper'][0], treedata['StemTaper'][1], '-b')
    plt.title('Stem taper')
    plt.xlabel('Distance from base (m)')
    plt.ylabel('Diameter (m)')
    plt.grid(True)

    plt.subplot(2, 4, 2)
    PlottingUtils.plot_distribution(Q, 6+iter*10, 0, 0, 'VolCylDia')

    plt.subplot(2, 4, 3)
    PlottingUtils.plot_distribution(Q, 6+iter*10, 0, 0, 'AreCylDia')

    plt.subplot(2, 4, 4)
    PlottingUtils.plot_distribution(Q, 6+iter*10, 0, 0, 'LenCylDia')

    plt.subplot(2, 4, 5)
    PlottingUtils.plot_distribution(Q, 6+iter*10, 0, 0, 'VolBranchOrd')

    plt.subplot(2, 4, 6)
    PlottingUtils.plot_distribution(Q, 6+iter*10, 0, 0, 'LenBranchOrd')

    plt.subplot(2, 4, 7)
    PlottingUtils.plot_distribution(Q, 6+iter*10, 0, 0, 'AreBranchOrd')

    plt.subplot(2, 4, 8)
    PlottingUtils.plot_distribution(Q, 6+iter*10, 0, 0, 'NumBranchOrd')
    plt.tight_layout()

    # Figure 7: Cylinder distributions
    plt.figure(7+iter*10)
    plt.subplot(3, 3, 1)
    PlottingUtils.plot_distribution(Q, 7+iter*10, 0, 0, 'VolCylHei')

    plt.subplot(3, 3, 2)
    PlottingUtils.plot_distribution(Q, 7+iter*10, 0, 0, 'AreCylHei')

    plt.subplot(3, 3, 3)
    PlottingUtils.plot_distribution(Q, 7+iter*10, 0, 0, 'LenCylHei')

    plt.subplot(3, 3, 4)
    PlottingUtils.plot_distribution(Q, 7+iter*10, 0, 0, 'VolCylZen')

    plt.subplot(3, 3, 5)
    PlottingUtils.plot_distribution(Q, 7+iter*10, 0, 0, 'AreCylZen')

    plt.subplot(3, 3, 6)
    PlottingUtils.plot_distribution(Q, 7+iter*10, 0, 0, 'LenCylZen')

    plt.subplot(3, 3, 7)
    PlottingUtils.plot_distribution(Q, 7+iter*10, 0, 0, 'VolCylAzi')

    plt.subplot(3, 3, 8)
    PlottingUtils.plot_distribution(Q, 7+iter*10, 0, 0, 'AreCylAzi')

    plt.subplot(3, 3, 9)
    PlottingUtils.plot_distribution(Q, 7+iter*10, 0, 0, 'LenCylAzi')
    plt.tight_layout()
    # plt.show()
    
    # Figure 8: Branch distributions
    plt.figure(8+iter*10)
    plt.subplot(3, 4, 1)
    PlottingUtils.plot_distribution(Q, 8+iter*10, 1, 0, 'VolBranchDia', 'VolBranch1Dia')

    plt.subplot(3, 4, 2)
    PlottingUtils.plot_distribution(Q, 8+iter*10, 1, 0, 'AreBranchDia', 'AreBranch1Dia')

    plt.subplot(3, 4, 3)
    PlottingUtils.plot_distribution(Q, 8+iter*10, 1, 0, 'LenBranchDia', 'LenBranch1Dia')

    plt.subplot(3, 4, 4)
    PlottingUtils.plot_distribution(Q, 8+iter*10, 1, 0, 'NumBranchDia', 'NumBranch1Dia')

    plt.subplot(3, 4, 5)
    PlottingUtils.plot_distribution(Q, 8+iter*10, 1, 0, 'VolBranchHei', 'VolBranch1Hei')

    plt.subplot(3, 4, 6)
    PlottingUtils.plot_distribution(Q, 8+iter*10, 1, 0, 'AreBranchHei', 'AreBranch1Hei')

    plt.subplot(3, 4, 7)
    PlottingUtils.plot_distribution(Q, 8+iter*10, 1, 0, 'LenBranchHei', 'LenBranch1Hei')

    plt.subplot(3, 4, 8)
    PlottingUtils.plot_distribution(Q, 8+iter*10, 1, 0, 'NumBranchHei', 'NumBranch1Hei')

    plt.subplot(3, 4, 9)
    PlottingUtils.plot_distribution(Q, 8+iter*10, 1, 0, 'VolBranchAng', 'VolBranch1Ang')

    plt.subplot(3, 4, 10)
    PlottingUtils.plot_distribution(Q, 8+iter*10, 1, 0, 'AreBranchAng', 'AreBranch1Ang')

    plt.subplot(3, 4, 11)
    PlottingUtils.plot_distribution(Q, 8+iter*10, 1, 0, 'LenBranchAng', 'LenBranch1Ang')

    plt.subplot(3, 4, 12)
    PlottingUtils.plot_distribution(Q, 8+iter*10, 1, 0, 'NumBranchAng', 'NumBranch1Ang')
    plt.tight_layout()
    # Figure 9: Branch zenith and azimuth distributions
    plt.figure(9+iter*10)
    plt.subplot(2, 4, 1)
    PlottingUtils.plot_distribution(Q, 9+iter*10, 1, 0, 'VolBranchZen', 'VolBranch1Zen')

    plt.subplot(2, 4, 2)
    PlottingUtils.plot_distribution(Q, 9+iter*10, 1, 0, 'AreBranchZen', 'AreBranch1Zen')

    plt.subplot(2, 4, 3)
    PlottingUtils.plot_distribution(Q, 9+iter*10, 1, 0, 'LenBranchZen', 'LenBranch1Zen')

    plt.subplot(2, 4, 4)
    PlottingUtils.plot_distribution(Q, 9+iter*10, 1, 0, 'NumBranchZen', 'NumBranch1Zen')

    plt.subplot(2, 4, 5)
    PlottingUtils.plot_distribution(Q, 9+iter*10, 1, 0, 'VolBranchAzi', 'VolBranch1Azi')

    plt.subplot(2, 4, 6)
    PlottingUtils.plot_distribution(Q, 9+iter*10, 1, 0, 'AreBranchAzi', 'AreBranch1Azi')

    plt.subplot(2, 4, 7)
    PlottingUtils.plot_distribution(Q, 9+iter*10, 1, 0, 'LenBranchAzi', 'LenBranch1Azi')

    plt.subplot(2, 4, 8)
    PlottingUtils.plot_distribution(Q, 9+iter*10, 1, 0, 'NumBranchAzi', 'NumBranch1Azi')
    plt.tight_layout()
    if inputs['plot'] > 1:
        plt.show()
        figures = [plt.figure(6+iter*10), plt.figure(7+iter*10), plt.figure(8+iter*10), plt.figure(9+iter*10)]
        treedata['figures'] = figures
    else:
        figures = [plt.figure(6+iter*10), plt.figure(7+iter*10), plt.figure(8+iter*10), plt.figure(9+iter*10)]
        treedata['figures'] = figures
    return treedata, triangulation
def dbh_cylinder(treedata, trunk, Trunk, cylinder, ind):
    """
    
    Calculate the diameter at breast height (DBH) from cylinder data.
    Args:
        treedata (dict): Dictionary to store tree data.
        trunk (numpy.ndarray): Point cloud of the trunk.
        Trunk (numpy.ndarray): Boolean array indicating trunk cylinders.
        cylinder (dict): Dictionary containing cylinder data (radius, length, start, axis).
        ind (numpy.ndarray): Indices of the cylinders.
    Returns:
        (dict): Updated tree data with DBH attributes.

    """
    # Convert Trunk to a boolean array if not already
    Trunk = np.asarray(Trunk).astype(bool)
    T = ind[Trunk]  # Get indices of trunk cylinders
    n = len(T)
    
    # Part 1: Determine DBHqsm from cumulative cylinder lengths
    if n == 0:
        treedata['DBHqsm'] = 0
        treedata['DBHcyl'] = 0
        return treedata
    
    lengths = cylinder['length'][T]
    cum_sum = np.cumsum(lengths)
    mask = cum_sum >= 1.3
    if np.any(mask):
        i = np.argmax(mask)  # First index where cumsum >=1.3
    else:
        i = n - 1  # Use last cylinder if sum never reaches 1.3
    DBHqsm = 2 * cylinder['radius'][T[i]]
    treedata['DBHqsm'] = DBHqsm
    
    # Part 2: Fit cylinder to points between 1.1m and 1.5m
    start = cylinder['start'][0]  # First cylinder's start coordinates
    axis = cylinder['axis'][0]    # First cylinder's axis
    V = trunk - start  # Translate points
    h = np.dot(V, axis)  # Heights along the axis
    
    # Filter points between 1.1m and 1.5m
    IJ = (h > 1.1) & (h < 1.5)
    if np.sum(IJ) > 100:
        T_points = trunk[IJ]
        # Initial cylinder parameters from the identified trunk cylinder
        cyl0 = {
            'radius': cylinder['radius'][T[i]],
            'axis': cylinder['axis'][T[i]],
            'start': cylinder['start'][T[i]]
        }
        
        cyl = LSF.least_squares_cylinder(T_points, cyl0)
        
        # Check conditions
        radius_cyl = 2 * cyl['radius']
        RadiusOK = (0.8 * DBHqsm < radius_cyl < 1.2 * DBHqsm)
        axis_alignment = np.abs(np.dot(cylinder['axis'][T[i]], cyl['axis'])) > 0.9
        if RadiusOK and axis_alignment and cyl['conv'] and cyl['rel']:
            treedata['DBHcyl'] = radius_cyl
        else:
            treedata['DBHcyl'] = DBHqsm
    else:
        treedata['DBHcyl'] = DBHqsm
    
    return treedata

# Requires the alphashape library

def crown_measures(treedata, cylinder, branch):
    """
    
    Calculate crown measures using alphashape concave hull

    Args:
        treedata (dict): Dictionary to store tree data.
        cylinder (dict): Dictionary containing cylinder data (radius, length, start, axis, branch).
        branch (dict): Dictionary containing branch data (order, volume, length, diameter, height, angle, zenith, azimuth).
    Returns:
        (dict, numpy.ndarray): Updated tree data with crown measures and vertical profile spreads.
    """
    # Extract cylinder properties
    Axe = cylinder['axis']
    Len = cylinder['length']
    Sta = cylinder['start']
    Tip = Sta + Len[:, np.newaxis] * Axe  # Tips of the cylinders
    nc = len(Len)
    
    # Generate point clouds from the cylinder model
    P = np.zeros((5 * nc*10, 3))  # Four mid points on the cylinder surface
    t = 0
    for i in range(nc):
        U, V = Utils.orthonormal_vectors(Axe[i, :])
        U = cylinder['radius'][i] * U
        if cylinder['branch'][i] == 0:
            # For stem cylinders, generate more points
            R = Utils.rotation_matrix(Axe[i, :], np.pi / 12)
            for k in range( 4):
                M = Sta[i, :] + (k / 4) * Len[i] * Axe[i, :]
                for j in range(12):
                    if j > 0:
                        U = R @ U
                    t += 1
                    P[t-1, :] = M + U.T
        else:
            M = Sta[i, :] + 0.5 * Len[i] * Axe[i, :]
            R = Utils.rotation_matrix(Axe[i, :], np.pi / 4)
            for j in range(4):
                if j > 0:
                    U = R @ U
                t += 1
                P[t-1, :] = M + U.T
    P = P[:t+1, :]
    P = P[~np.isnan(P[:, 0]), :]
    P = np.vstack((P, Sta, Tip))
    P = np.unique(P, axis=0)

    # Vertical profiles (layer diameters/spreads), mean:
    bot = np.min(P[:, 2])
    top = np.max(P[:, 2])
    Hei = top - bot
    if Hei > 10:
        m = 20
    elif Hei > 2:
        m = 10
    else:
        m = 5
    spreads = np.zeros((m, 18))
    for j in range(m):
        I = (P[:, 2] >= bot + (j * Hei / m)) & (P[:, 2] < bot + ((j + 1) * Hei / m))
        X = np.unique(P[I, :], axis=0)
        if X.shape[0] > 5:
            hull = ConvexHull(X[:, :2])
            K = hull.vertices
            A = hull.volume
            # Compute center of gravity for the convex hull
            x = X[K, 0]
            y = X[K, 1]
            CX = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * A)
            CY = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * A)
            V = X[:, :2] - np.array([CX, CY])
            ang = np.arctan2(V[:, 1], V[:, 0]) + np.pi
            I = np.argsort(ang)
            ang =np.sort(ang)
            L = np.sqrt(np.sum(V ** 2, axis=1))
            L = L[I]
            for i in range(18):
                I = (ang >= (i * np.pi / 18)) & (ang < ((i + 1) * np.pi / 18))
                L1 = np.max(L[I]) if np.any(I) else 0
                J = (ang >= (i * np.pi / 18 + np.pi)) & (ang < ((i + 1) * np.pi / 18 + np.pi))
                L2 = np.max(L[J]) if np.any(J) else 0
                spreads[j, i] = L1 + L2

    # Crown diameters (spreads), mean and maximum:
    X = np.unique(P[:, :2], axis=0)
    hull = ConvexHull(X)
    K = hull.vertices
    A = hull.volume
    x = X[K, 0]
    y = X[K, 1]
    CX = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * A)
    CY = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * A)
    V = Tip[:, :2] - np.array([CX, CY])
    ang = np.arctan2(V[:, 1], V[:, 0]) + np.pi
    
    I = np.argsort(ang)
    ang = np.sort(ang)
    L = np.sqrt(np.sum(V ** 2, axis=1))
    L = L[I]
    S = np.zeros(18)
    for i in range(18):
        I = (ang >= (i * np.pi / 18)) & (ang < ((i + 1) * np.pi / 18))
        L1 = np.max(L[I]) if np.any(I) else 0
        J = (ang >= (i * np.pi / 18 + np.pi)) & (ang < ((i + 1) * np.pi / 18 + np.pi))
        L2 = np.max(L[J]) if np.any(J) else 0
        S[i] = L1 + L2
    treedata['CrownDiamAve'] = np.mean(S)
    MaxDiam = 0
    for i in range(len(x)):
        V = X - np.array([x[i], y[i]])
        L = np.max(np.sqrt(np.sum(V ** 2, axis=1)))
        if L > MaxDiam:
            MaxDiam = L
    treedata['CrownDiamMax'] = MaxDiam

    # Crown areas from convex hull and alpha shape:
    treedata['CrownAreaConv'] = A
    alp = max(0.5, treedata['CrownDiamAve'] / 10)
    shp = alphashape(X, alp)
    treedata['CrownAreaAlpha'] = shp.area

    # Crown base
    dbh = treedata['DBHcyl']
    nb = len(branch['order'])
    HL = np.zeros(nb)  # Horizontal reach
    branches1 = np.where(branch['order'] == 1)[0]
    nb = len(branches1)
    nc = Sta.shape[0]
    ind = np.arange(nc)
    for i in range(nb):
        C = ind[cylinder['branch'] == branches1[i]]
        if len(C) > 0:
            base = Sta[C[0], :]
            C = C[-1]
            tip = Sta[C, :] + Len[C] * Axe[C, :]
            V = tip[:2] - base[:2]
            HL[branches1[i]] = np.sqrt(V @ V.T) / dbh * 2
    M = min(10, np.median(HL))

    # Sort the branches according to their heights
    Hei = branch['height'][branches1]
    SortOrd = np.argsort(Hei)
    branches1 = branches1[SortOrd]

    # Search the first/lowest branch
    d = min(0.05, 0.05 * dbh)
    b = 0
    if nb > 1:
        i = 0
        while i < nb-1:
            i += 1
            if branch['diameter'][branches1[i]] > d and HL[branches1[i]] > M:
                b = branches1[i]
                break
        if i == nb and nb >1:
            b = branches1[0]

    if b > 0:
        # Search all the children of the first major branch
        nb = len(branch['parent'])
        Ind = np.arange(nb)
        chi = Ind[branch['parent'] == b]
        B = [b]
        while len(chi) > 0:
            B.extend(chi)
            n = len(chi)
            C = []
            for i in range(n):
                C.extend(Ind[branch['parent'] == chi[i]])
            chi = C

        # Define crown base height from the ground
        BaseHeight = np.max(Sta[:, 2])
        for i in range(len(B)):
            C = ind[cylinder['branch'] == B[i]]
            ht = np.min(Tip[C, 2])
            hb = np.min(Sta[C, 2])
            h = min(hb, ht)
            if h < BaseHeight:
                BaseHeight = h
        treedata['CrownBaseHeight'] = BaseHeight - Sta[0, 2]

        # Crown length and ratio
        treedata['CrownLength'] = treedata['TreeHeight'] - treedata['CrownBaseHeight']
        treedata['CrownRatio'] = treedata['CrownLength'] / treedata['TreeHeight']

        # Crown volume from convex hull and alpha shape
        I = P[:, 2] >= BaseHeight
        X = P[I, :]
        hull = ConvexHull(X)
        treedata['CrownVolumeConv'] = hull.volume
        alp = max(0.5, treedata['CrownDiamAve'] / 5)
        adj_X = X.copy()
        adj_X[:,0] = X[:,0]-min(X[:,0])
        adj_X[:,1] = X[:,1]-min(X[:,1])
        adj_X[:,2] = X[:,2]-min(X[:,2])
        max_alp = alp*5
        shp = None
        while alp<=max_alp:
            try:
                shp = alphashape(adj_X, alp)
                break
            except:
                alp+=max_alp/10
        #shp = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(adj_X)), alp)
        #shp = o3d.t.geometry.TriangleMesh.from_legacy(shp).fill_holes().to_legacy()
        #shp = alphashape(adj_X, alp)
        if shp is not None:
            treedata['CrownVolumeAlpha'] = abs(shp.volume)#abs(shp.get_volume())
        else:
            treedata['CrownVolumeAlpha'] = float('inf')
    else:
        # No branches
        treedata['CrownBaseHeight'] = treedata['TreeHeight']
        treedata['CrownLength'] = 0
        treedata['CrownRatio'] = 0
        treedata['CrownVolumeConv'] = 0
        treedata['CrownVolumeAlpha'] = 0

    return treedata, spreads



def triangulate_stem(treedata, cylinder, branch, trunk):
    """
    
    Calculate stem using triangulation method
    Args:
        treedata (dict): Dictionary to store tree data.
        cylinder (dict): Dictionary containing cylinder data (start, radius, length, axis, branch).
        branch (dict): Dictionary containing branch data (diameter, height, order).
        trunk (numpy.ndarray): Point cloud of the trunk.
    Returns:
        (dict, dict): Updated tree data with triangulation attributes and triangulation data."""
    raise NotImplementedError("Triangulation is currently disabled.")
    Sta = cylinder['start']
    Rad = cylinder['radius']
    Len = cylinder['length']
    DBHqsm = treedata['DBHqsm']

    # Determine the first major branch (over 10% of dbh or the maximum diameter branch)
    nb = len(branch['diameter'])
    ind = np.where(branch['order'] == 1)[0]
    I = np.argsort(branch['height'][ind])
    ind = ind[I]
    n = len(ind)
    b = 0
    while b < n and branch['diameter'][ind[b]] < 0.1 * DBHqsm:
        b += 1
    b = ind[b] if b < n else np.argmax(branch['diameter'])

    # Determine suitable cylinders up to the first major branch but keep the stem diameter above one quarter (25%) of dbh
    C = 0
    nc = Sta.shape[0]
    while C < nc and cylinder['branch'][C] < b:
        C += 1
    n = np.sum(cylinder['branch'] == 1)
    i = 1
    while i < n and Sta[i, 2] < Sta[C, 2] and Rad[i] > 0.125 * DBHqsm:
        i += 1
    CylInd = max(i, 2)
    TrunkLenTri = Sta[CylInd, 2] - Sta[0, 2]

    EmptyTriangulation = False
    # Calculate the volumes
    if trunk.shape[0] > 1000 and TrunkLenTri >= 1:
        # Set the parameters for triangulation
        # Compute point density, which is used to increase the triangle size if the point density is very small
        PointDensity = np.zeros(CylInd - 1)
        for i in range(CylInd - 1):
            I = (trunk[:, 2] >= Sta[i, 2]) & (trunk[:, 2] < Sta[i + 1, 2])
            PointDensity[i] = np.pi * Rad[i] * Len[i] / np.sum(I)
        PointDensity = PointDensity[PointDensity < np.inf]
        d = np.max(PointDensity)

        # Determine minimum triangle size based on dbh
        if DBHqsm > 1:
            MinTriaHeight = 0.1
        elif DBHqsm > 0.50:
            MinTriaHeight = 0.075
        elif DBHqsm > 0.10:
            MinTriaHeight = 0.05
        else:
            MinTriaHeight = 0.02
        TriaHeight0 = max(MinTriaHeight, 4 * np.sqrt(d))

        # Select the trunk point set used for triangulation
        I = trunk[:, 2] <= Sta[CylInd, 2]
        Stem = trunk[I, :]

        # Do the triangulation
        triangulation = {}
        l = 0
        while not triangulation and l < 4 and CylInd > 2:
            l += 1
            TriaHeight = TriaHeight0
            TriaWidth = TriaHeight
            k = 0
            while not triangulation and k < 3:
                k += 1
                j = 0
                while not triangulation and j < 5:
                    triangulation = curve_based_triangulation(Stem, TriaHeight, TriaWidth)
                    j += 1
                # Try different triangle sizes if necessary
                if not triangulation and k < 3:
                    TriaHeight += 0.03
                    TriaWidth = TriaHeight
            # Try different length of stem sections if necessary
            if not triangulation and l < 4 and CylInd > 2:
                CylInd -= 1
                I = trunk[:, 2] <= Sta[CylInd, 2]
                Stem = trunk[I, :]

        if triangulation:
            triangulation['cylind'] = CylInd
            # Dbh from triangulation
            Vert = triangulation['vert']
            h = Vert[:, 2] - triangulation['bottom']
            I = np.argmin(np.abs(h - 1.3))
            H = h[I]
            I = np.abs(h - H) < triangulation['triah'] / 2
            V = Vert[I, :]
            V = np.roll(V, -1, axis=0) - V
            d = np.sqrt(np.sum(V ** 2, axis=1))
            treedata['DBHtri'] = np.sum(d) / np.pi
            # Volumes from the triangulation
            treedata['TriaTrunkVolume'] = triangulation['volume']
            TrunkVolMix = treedata['TrunkVolume'] - \
                          1000 * np.pi * np.sum(Rad[:CylInd - 1] ** 2 * Len[:CylInd - 1]) + \
                          triangulation['volume']
            TrunkAreaMix = treedata['TrunkArea'] - \
                           2 * np.pi * np.sum(Rad[:CylInd - 1] * Len[:CylInd - 1]) + \
                           triangulation['SideArea']
            treedata['MixTrunkVolume'] = TrunkVolMix
            treedata['MixTotalVolume'] = TrunkVolMix + treedata['BranchVolume']
            treedata['TriaTrunkArea'] = triangulation['SideArea']
            treedata['MixTrunkArea'] = TrunkAreaMix
            treedata['MixTotalArea'] = TrunkAreaMix + treedata['BranchArea']
            treedata['TriaTrunkLength'] = TrunkLenTri
        else:
            EmptyTriangulation = True
    else:
        EmptyTriangulation = True

    if EmptyTriangulation:
        print('  No triangulation model produced')
        triangulation = {
            'vert': np.zeros((0, 3)),
            'facet': np.zeros((0, 3)),
            'fvd': np.zeros(0),
            'volume': 0,
            'SideArea': 0,
            'BottomArea': 0,
            'TopArea': 0,
            'bottom': 0,
            'top': 0,
            'triah': 0,
            'triaw': 0,
            'cylind': 0
        }
        treedata['DBHtri'] = DBHqsm
        treedata['TriaTrunkVolume'] = treedata['TrunkVolume']
        treedata['TriaTrunkArea'] = treedata['TrunkArea']
        treedata['MixTrunkVolume'] = treedata['TrunkVolume']
        treedata['MixTrunkArea'] = treedata['TrunkArea']
        treedata['MixTotalVolume'] = treedata['TotalVolume']
        treedata['MixTotalArea'] = treedata['TotalArea']
        treedata['TriaTrunkLength'] = 0

    return treedata, triangulation



def cylinder_distribution(treedata, cyl, dist):
    """
    Compute volume, area, and length distributions of wood parts based on cylinder diameter,
    zenith, or azimuth direction.

    Args:
        treedata (dict): Dictionary to store the results.
        cyl (dict): Dictionary containing cylinder data (radius, axis, length).
        dist (str): Distribution type ('Dia', 'Zen', or 'Azi').

    Returns:
        (dict): Updated dictionary with distribution results.
    """
    # Extract cylinder properties
    radius = cyl['radius']
    axis = cyl['axis']
    length = cyl['length']

    # Determine parameters based on distribution type
    if dist == 'Dia':
        Par = radius  # Diameter distribution
        n = int(np.ceil(200 * np.max(radius)))  # Number of bins
        a = 0.005  # Diameter bin size (1 cm classes)
    elif dist == 'Zen':
        Par = 180 / np.pi * np.arccos(axis[:, 2])  # Zenith angle distribution
        n = 18  # Number of bins
        a = 10  # Zenith bin size (10-degree classes)
    elif dist == 'Azi':
        Par = 180 / np.pi * np.arctan2(axis[:, 1], axis[:, 0]) + 180  # Azimuth angle distribution
        n = 36  # Number of bins
        a = 10  # Azimuth bin size (10-degree classes)
    else:
        raise ValueError("Invalid distribution type. Use 'Dia', 'Zen', or 'Azi'.")

    # Initialize distribution array
    CylDist = np.zeros((3, n))

    # Compute volume, area, and length for each bin
    for i in range(n):
        K = (Par >= (i * a)) & (Par < ((i + 1) * a))  # Filter cylinders in the current bin
        CylDist[0, i] = 1000 * np.pi * np.sum(radius[K] ** 2 * length[K])  # Volume in liters
        CylDist[1, i] = 2 * np.pi * np.sum(radius[K] * length[K])  # Area in square meters
        CylDist[2, i] = np.sum(length[K])  # Length in meters

    # Store results in treedata
    treedata[f'VolCyl{dist}'] = CylDist[0, :]
    treedata[f'AreCyl{dist}'] = CylDist[1, :]
    treedata[f'LenCyl{dist}'] = CylDist[2, :]

    return treedata


def cylinder_height_distribution(treedata, cylinder, ind):
    """
    Compute volume, area, and length distributions of cylinders as a function of height.

    Args:
        treedata (dict): Dictionary to store the results.
        cylinder (dict): Dictionary containing cylinder data (radius, length, axis, start).
        ind (numpy.ndarray): Indices of cylinders to consider.

    Returns:
        (dict): Updated dictionary with height distribution results.
    """
    # Extract cylinder properties
    Rad = cylinder['radius']
    Len = cylinder['length']
    Axe = cylinder['axis']
    Start = cylinder['start']

    # Compute end points of cylinders
    End = Start + Len[:, np.newaxis] * Axe

    # Determine maximum height and initialize distributions
    MaxHei = int(np.ceil(treedata['TreeHeight']))
    treedata['VolCylHei'] = np.zeros(MaxHei)
    treedata['AreCylHei'] = np.zeros(MaxHei)
    treedata['LenCylHei'] = np.zeros(MaxHei)

    # Compute base and top heights relative to the bottom
    bot = np.min(Start[:, 2])
    B = Start[:, 2] - bot  # Base heights
    T = End[:, 2] - bot    # Top heights

    # Iterate over height bins
    for j in range(1, MaxHei + 1):
        # Define height bin ranges
        I1 = (B >= (j - 2)) & (B < (j - 1))  # Base below this bin
        J1 = (B >= (j - 1)) & (B < j)        # Base in this bin
        K1 = (B >= j) & (B < (j + 1))        # Base above this bin
        I2 = (T >= (j - 2)) & (T < (j - 1))  # Top below this bin
        J2 = (T >= (j - 1)) & (T < j)        # Top in this bin
        K2 = (T >= j) & (T < (j + 1))        # Top above this bin

        # Identify cylinders in different cases
        C1 = ind[J1 & J2]  # Base and top in this bin
        C2 = ind[J1 & K2]  # Base in this bin, top above
        C3 = ind[J1 & I2]  # Base in this bin, top below
        C4 = ind[I1 & J2]  # Base in bin below, top in this
        C5 = ind[K1 & J2]  # Base in bin above, top in this

        # Compute volume, area, and length for each case
        # Case 1: Base and top in this bin
        v1 = 1000 * np.pi * np.sum(Rad[C1] ** 2 * Len[C1])
        a1 = 2 * np.pi * np.sum(Rad[C1] * Len[C1])
        l1 = np.sum(Len[C1])

        # Case 2: Base in this bin, top above
        r2 = (j - B[C2]) / (T[C2] - B[C2])  # Relative portion in this bin
        v2 = 1000 * np.pi * np.sum(Rad[C2] ** 2 * Len[C2] * r2)
        a2 = 2 * np.pi * np.sum(Rad[C2] * Len[C2] * r2)
        l2 = np.sum(Len[C2] * r2)

        # Case 3: Base in this bin, top below
        r3 = (B[C3] - j + 1) / (B[C3] - T[C3])  # Relative portion in this bin
        v3 = 1000 * np.pi * np.sum(Rad[C3] ** 2 * Len[C3] * r3)
        a3 = 2 * np.pi * np.sum(Rad[C3] * Len[C3] * r3)
        l3 = np.sum(Len[C3] * r3)

        # Case 4: Base in bin below, top in this
        r4 = (T[C4] - j + 1) / (T[C4] - B[C4])  # Relative portion in this bin
        v4 = 1000 * np.pi * np.sum(Rad[C4] ** 2 * Len[C4] * r4)
        a4 = 2 * np.pi * np.sum(Rad[C4] * Len[C4] * r4)
        l4 = np.sum(Len[C4] * r4)

        # Case 5: Base in bin above, top in this
        r5 = (j - T[C5]) / (B[C5] - T[C5])  # Relative portion in this bin
        v5 = 1000 * np.pi * np.sum(Rad[C5] ** 2 * Len[C5] * r5)
        a5 = 2 * np.pi * np.sum(Rad[C5] * Len[C5] * r5)
        l5 = np.sum(Len[C5] * r5)

        # Sum contributions from all cases
        treedata['VolCylHei'][j - 1] = v1 + v2 + v3 + v4 + v5
        treedata['AreCylHei'][j - 1] = a1 + a2 + a3 + a4 + a5
        treedata['LenCylHei'][j - 1] = l1 + l2 + l3 + l4 + l5

    return treedata


def branch_distribution(treedata, branch, dist):
    """
    Compute volume, area, length, and number of branches as a function of branch diameter,
    height, angle, zenith, or azimuth.

    Parameters:
        treedata (dict): Dictionary to store the results.
        branch (dict): Dictionary containing branch data (order, volume, area, length, diameter, height, angle, zenith, azimuth).
        dist (str): Distribution type ('Dia', 'Hei', 'Ang', 'Zen', or 'Azi').

    Args:
        (dict): Updated dictionary with branch distribution results.
    """
    # Extract branch properties (excluding the first element, as in MATLAB's 2:end)
    BOrd = branch['order'][1:]
    BVol = branch['volume'][1:]
    BAre = branch['area'][1:]
    BLen = branch['length'][1:]
    if len(BOrd) == 0:
        treedata[f'VolBranch{dist}'] = np.array([0])
        treedata[f'VolBranch1{dist}'] = np.array([0])
        treedata[f'AreBranch{dist}'] = np.array([0])
        treedata[f'AreBranch1{dist}'] = np.array([0])
        treedata[f'LenBranch{dist}'] = np.array([0])
        treedata[f'LenBranch1{dist}'] = np.array([0])
        treedata[f'NumBranch{dist}'] = np.array([0])
        treedata[f'NumBranch1{dist}'] = np.array([0])
        return treedata
    # Determine parameters based on distribution type
    if dist == 'Dia':
        Par = branch['diameter'][1:]  # Diameter distribution
        n = int(np.ceil(100 * np.max(Par)))  # Number of bins
        a = 0.005  # Diameter bin size (1 cm classes)
    elif dist == 'Hei':
        Par = branch['height'][1:]  # Height distribution
        n = int(np.ceil(treedata['TreeHeight']))  # Number of bins
        a = 1  # Height bin size (1 m classes)
    elif dist == 'Ang':
        Par = branch['angle'][1:]  # Angle distribution
        n = 18  # Number of bins
        a = 10  # Angle bin size (10-degree classes)
    elif dist == 'Zen':
        Par = branch['zenith'][1:]  # Zenith angle distribution
        n = 18  # Number of bins
        a = 10  # Zenith bin size (10-degree classes)
    elif dist == 'Azi':
        Par = branch['azimuth'][1:] + 180  # Azimuth angle distribution
        n = 36  # Number of bins
        a = 10  # Azimuth bin size (10-degree classes)
    else:
        raise ValueError("Invalid distribution type. Use 'Dia', 'Hei', 'Ang', 'Zen', or 'Azi'.")

    # Handle empty case
    if n == 0:
        n = 1  # Ensure at least one bin

    # Initialize distribution array
    BranchDist = np.zeros((8, n))

    # Compute distributions for each bin
    for i in range(n):
        I = (Par >= (i * a)) & (Par < ((i + 1) * a))  # Filter branches in the current bin
        BranchDist[0, i] = np.sum(BVol[I])  # Volume (all branches)
        BranchDist[1, i] = np.sum(BVol[I & (BOrd == 1)])  # Volume (1st-order branches)
        BranchDist[2, i] = np.sum(BAre[I])  # Area (all branches)
        BranchDist[3, i] = np.sum(BAre[I & (BOrd == 1)])  # Area (1st-order branches)
        BranchDist[4, i] = np.sum(BLen[I])  # Length (all branches)
        BranchDist[5, i] = np.sum(BLen[I & (BOrd == 1)])  # Length (1st-order branches)
        BranchDist[6, i] = np.sum(I)  # Number (all branches)
        BranchDist[7, i] = np.sum(I & (BOrd == 1))  # Number (1st-order branches)

    # Store results in treedata
    treedata[f'VolBranch{dist}'] = BranchDist[0, :]
    treedata[f'VolBranch1{dist}'] = BranchDist[1, :]
    treedata[f'AreBranch{dist}'] = BranchDist[2, :]
    treedata[f'AreBranch1{dist}'] = BranchDist[3, :]
    treedata[f'LenBranch{dist}'] = BranchDist[4, :]
    treedata[f'LenBranch1{dist}'] = BranchDist[5, :]
    treedata[f'NumBranch{dist}'] = BranchDist[6, :]
    treedata[f'NumBranch1{dist}'] = BranchDist[7, :]

    return treedata

import numpy as np

def branch_order_distribution(treedata, branch):
    """
    Compute volume, area, length, and number of branches as a function of branch order.

    Arg:
        treedata (dict): Dictionary to store the results.
        branch (dict): Dictionary containing branch data (order, volume, area, length).

    Returns:
        (dict): Updated dictionary with branch order distribution results.
    """
    # Determine the maximum branch order
    BO = int(np.max(branch['order']))

    # Initialize distribution array
    BranchOrdDist = np.zeros((BO, 4))

    # Compute distributions for each branch order
    for i in range(1, BO + 1):
        I = branch['order'] == i  # Filter branches of the current order
        BranchOrdDist[i - 1, 0] = np.sum(branch['volume'][I])  # Volume
        BranchOrdDist[i - 1, 1] = np.sum(branch['area'][I])    # Area
        BranchOrdDist[i - 1, 2] = np.sum(branch['length'][I])  # Length
        BranchOrdDist[i - 1, 3] = np.sum(I)                    # Number of branches

    # Store results in treedata
    treedata['VolBranchOrd'] = BranchOrdDist[:, 0]
    treedata['AreBranchOrd'] = BranchOrdDist[:, 1]
    treedata['LenBranchOrd'] = BranchOrdDist[:, 2]
    treedata['NumBranchOrd'] = BranchOrdDist[:, 3]

    return treedata