from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import numpy as np
def clouds_matching(classify_cloud, reference_cloud, tolerance=0.00001):

    """
        Find the common points in both result_cloud and reference_cloud.

        Parameters
        ----------
        classify_cloud : array
            Three-dimensional (m x n) array of a point cloud, where the
            coordinates are represented in the columns (n) and the points are
            represented in the rows (m).
        reference_cloud : array
            Three-dimensional (m x n) array of a point cloud, where the
            coordinates are represented in the columns (n) and the points are
            represented in the rows (m).
        tolerance : float
            The max distance of two points which be seen as the same point.

        Returns
        -------
        matching_count : int
        The total number of common points in both result_cloud and reference_cloud.
        matching_mask : array
        Boolean mask where 'True' represents matching points.

        """
    # Initializing NearestNeighbors search and searching for all 'knn'
    # neighboring points arround each point in 'arr'.
    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean',
                            leaf_size=15, n_jobs=-1).fit(reference_cloud)
    distances, indices = nbrs.kneighbors(classify_cloud)
    matching_count = 0
    matching_mask = np.zeros(classify_cloud.shape[0], dtype=bool)
    for i, (dis) in enumerate(distances):
        if(dis[0]<tolerance):
            matching_count += 1
            matching_mask[i] = True
    return matching_count, matching_mask

def evaluate_indicators(classify_wood, classify_leaf, reference_wood, reference_leaf, components=False):

    """
        Get the evaluating indicators of classification result based on the manual reference data.

        Parameters
        ----------
        classify_wood : array
            3d array of classified wood point cloud.
        classify_leaf : array
            3d array of classified leaf point cloud.
        reference_wood : array
            3d array of reference wood point cloud.
        reference_leaf : array
            3d array of reference leaf point cloud.
        components : bool
            Option to select if function should output component of classify result.

        Returns
        -------
        components_points : array
            5d array of point cloud with component label.
            the first 3 columns represent three dimensional coordinates.
            in 4th column, 0 represent classify_leaf, 1 represent classify_wood.
            in 5th column, 0:correct wood, 1:false wood, 2:correct leaf, 3:false leaf

        """

    print("classification indicators:")
    classified_points = classify_wood.shape[0] + classify_leaf.shape[0]
    print("count of classified points:", classified_points)
    reference_points = reference_wood.shape[0] + reference_leaf.shape[0]
    print("count of reference points:", reference_points)
    print("count of classified wood:", classify_wood.shape[0])
    print("count of reference wood:", reference_wood.shape[0])
    wood_true, matching_mask_wood = clouds_matching(classify_wood, reference_wood)
    print("count of wood_true:", wood_true)
    wood_false = classify_wood.shape[0] - wood_true
    print("count of wood_false:", wood_false)
    print("count of classified leaf:", classify_leaf.shape[0])
    print("count of reference leaf:", reference_leaf.shape[0])
    leaf_true, matching_mask_leaf = clouds_matching(classify_leaf, reference_leaf)
    print("count of leaf_true:", leaf_true)
    leaf_false = classify_leaf.shape[0] - leaf_true
    print("count of leaf_false:", leaf_false)
    P_wood = wood_true / classify_wood.shape[0]
    print("Precision of wood:", P_wood)
    R_wood = wood_true / reference_wood.shape[0]
    print("Recall of wood:", R_wood)
    F1_wood = 2 * P_wood * R_wood / (P_wood + R_wood)
    print("F1 of wood:", F1_wood)
    P_leaf = leaf_true / classify_leaf.shape[0]
    print("Precision of leaf:", P_leaf)
    R_leaf = leaf_true / reference_leaf.shape[0]
    print("Recall of leaf:", R_leaf)
    F1_leaf = 2 * P_leaf * R_leaf / (P_leaf + R_leaf)
    print("F1 of leaf:", F1_leaf)
    Accuracy = (wood_true + leaf_true) / classified_points
    print("Accuracy:", Accuracy)
    pe = ((wood_true + leaf_false) * (wood_true + wood_false) + (wood_false + leaf_true) * (
                leaf_false + leaf_true)) / pow(classified_points, 2)
    Kappa = (Accuracy - pe) / (1 - pe)
    print("Kappa:", Kappa)

    if(components == True):
        wood_true_points = classify_wood[matching_mask_wood]
        label = np.zeros((wood_true_points.shape[0], 2))
        label[:, 0] = 1
        label[:, 1] = 0
        wood_true_points = np.column_stack((wood_true_points, label))

        wood_false_points = classify_wood[~matching_mask_wood]
        label = np.zeros((wood_false_points.shape[0], 2))
        label[:, 0] = 1
        label[:, 1] = 1
        wood_false_points = np.column_stack((wood_false_points, label))

        leaf_true_points = classify_leaf[matching_mask_leaf]
        label = np.zeros((leaf_true_points.shape[0], 2))
        label[:, 0] = 0
        label[:, 1] = 2
        leaf_true_points = np.column_stack((leaf_true_points, label))

        leaf_false_points = classify_leaf[~matching_mask_leaf]
        label = np.zeros((leaf_false_points.shape[0], 2))
        label[:, 0] = 0
        label[:, 1] = 3
        leaf_false_points = np.column_stack((leaf_false_points, label))

        components_points = np.row_stack((wood_true_points, wood_false_points, leaf_true_points, leaf_false_points))

        return components_points

