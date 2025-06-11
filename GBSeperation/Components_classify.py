import math
import numpy as np
from Eigen_transform import svd_eigen, pca_transform
from LS_circle import circleFitError

def components_classify(pcd, components, path_list, t_linearity=0.92, t_error=0.2, split_interval=0.2):
    """
    Classify the components as wood clusters with salient cylindrical/linear characteristic and
    others, follow a path-based correction script.

    Parameters
    ----------
    pcd : array
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).
    components : Two-dimensional list?
        connected components of the edge-cut graph G.
    path_list : dict
        Dictionary of nodes that comprises the path of every node in G to
        base_id node.
    t_linearity : float
        The threshold of linearity.
    t_error : float
        The threshold of cylindrical fitting relative error.
    Returns
    -------
    classify_components : list
        Record the classification information and elements of each connected component.
    """
    # Record the classification information and elements of each connected component.
    classify_components = []
    # Mark the component index where each point is located.
    components_idx = np.zeros(pcd.shape[0], dtype=int)
    for i, (component) in enumerate(components):
        component = list(component)
        c = classify_info(pcd, component, path_list, t_linearity, t_error, split_interval)
        classify_components.append([c, component])
        for elm in component:
            components_idx[elm] = i
    # Correct the misclassified wood clusters.
    itera_num = 0
    while(itera_num < 3):
        for i, (classify_component) in enumerate(classify_components):
            c = classify_component[0]
            if (c != 0):
                # The shortest path from any point in the cluster to the root point.
                path = path_list[classify_component[1][0]]
                for j in range(len(path) - 1, -1, -1):
                    path_elm = path[j]
                    pre_c = classify_components[components_idx[path_elm]][0]
                    if (components_idx[path_elm] != i and pre_c != 0):
                        if (c > pre_c):
                            classify_components[i][0] = 0
                        break
        itera_num += 1
    return classify_components

def classify_info(pcd, component, path_list, t_linearity, t_error, split_interval):
    """
    Get the classification information of a point cluster.

    Parameters
    ----------
    pcd : array
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).
    component : list
        A connected component of the edge-cut graph G.
    path_list : dict
        Dictionary of nodes that comprises the path of every node in G to
        base_id node.
    t_linearity : float
        The threshold of linearity.
    t_error : float
        The threshold of cylindrical fitting relative error.
    Returns
    -------
    classify_result : float
         <0:linear_wood_cluster,also the -linearity;
         >0:cylinder_wood_cluster,also the fitting radius;
         0:other_cluster.
    """
    # size filter
    if(len(component)<max(10, pcd.shape[0]/20000)):
        return 0

    # calculate the approximate axial based on path direction.
    axial = np.zeros(3)
    for i in component:
        if(len(path_list[i])>1):
            dire = pcd[path_list[i][-1]]-pcd[path_list[i][-2]]
            axial += dire
    if(np.linalg.norm(axial) == 0):
        return 0
    axial = axial / np.linalg.norm(axial)

    # calculate the axial-based eigenvalues and eigenvectors.
    cluster = pcd[component]
    centroid, evals, evecs = svd_eigen(cluster)
    eigenUpdate(axial, evals, evecs)

    evecs[0] = axial
    evecs[1] = [0, -axial[2], axial[1]]
    evecs[2] = np.cross(evecs[0], evecs[1])
    evecs[1] = evecs[1] / np.linalg.norm(evecs[1])
    evecs[2] = evecs[2] / np.linalg.norm(evecs[2])

    points_transformed = pca_transform(cluster, centroid, evecs)
    min_z = np.min(points_transformed[:, 0])
    max_z = np.max(points_transformed[:, 0])
    # dimension filter
    if(max_z-min_z<(1-0.25)*split_interval or max_z-min_z>(1+0.25)*split_interval):
        return 0

    curve = evals[2] / evals.sum()
    FitError, r = circleFitError(points_transformed[:, 1:])
    if(FitError<t_error and curve > 0.01):
        return r
    linearity = evals[0]/evals.sum()
    if(linearity>t_linearity):
        return -linearity
    return 0

# calculate the angular distance between the approximate axial and the eigenvector.
def getAngle3D(v1, v2):
    v1 /= math.sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2])
    v2 /= math.sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2])
    rad = (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
    if(rad<-1):
        rad = -1
    elif(rad>1):
        rad = 1
    d_normal = math.fabs(math.acos(rad))
    d_normal = min(d_normal, math.pi - d_normal)
    return d_normal

# calculate the axial-based eigenvalues and eigenvectors.
def eigenUpdate(axial, evals, evecs):
    idx = 0
    min_angle = getAngle3D(evecs[idx], axial)
    for i, (evec) in enumerate(evecs):
        angle = getAngle3D(evec, axial)
        if(angle<min_angle):
            idx = i
            min_angle = angle
    temp_eval = evals[idx]
    temp_evec = evecs[idx]
    while (idx > 0):
        evals[idx] = evals[idx-1]
        evecs[idx] = evecs[idx-1]
        idx -= 1
    evals[0] = temp_eval
    evecs[0] = temp_evec
