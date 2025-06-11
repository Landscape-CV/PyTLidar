import numpy as np

def extract_final_wood(pcd, base_id, path_dis, path_list, init_wood_ids, G, max_iter=100):

    """
    Final wood points extraction by region growth.

    Parameters
    ----------
    pcd : array
        Three-dimensional point cloud of a single tree.
    base_id : int
        Index of base id (root) in the graph.
    path_dis : list
        Shortest path distance from all nodes in G to root node.
    path_list : dict
        Dictionary of nodes that comprises the path of every node in G to
        root node.
    init_wood_ids : list
        Index of init wood points.
    G : networkx graph
        The original graph construed on single tree point cloud.
    max_iter : int
        The max number of iterations in growing.

    Returns
    -------
    wood_mask : array
        Boolean mask where 'True' represents wood points.

    """

    """
    Downwards search the wood points located in irregular places (such as bifurcation, curved branch,
    leaf-surrounded branch, broken branch).
    """

    # detect every int_wood_ids shortest path toward root.
    temp_ids = []
    for i in init_wood_ids:
        for ids in path_list[i]:
            temp_ids.append(ids)
    current_idx = np.unique(temp_ids)

    final_wood_mask = np.zeros(pcd.shape[0], dtype=bool)
    final_wood_mask[current_idx] = True

    # Looping while there are still indices in current_idx to process.
    iteration = 0
    while (len(current_idx) > 0) & (iteration < max_iter):
        temp_idx = []
        for i in current_idx:
            for key, value in G[i].items():
                if(final_wood_mask[key] == False
                        and path_dis[key] <= path_dis[i]):
                    final_wood_mask[key] = True
                    temp_idx.append(key)
        # Obtaining an unique array of points currently being processed.
        current_idx = np.array(temp_idx)
        # Increasing one iteration step.
        iteration += 1

    """
    Neighborhood smoothing.
    """
    idx_base = np.arange(pcd.shape[0], dtype=int)
    wood_ids = idx_base[final_wood_mask]
    for i in wood_ids:
        if (i == base_id):
            continue
        pre_dis = path_dis[i] - path_dis[path_list[i][-2]]
        for key, value in G[i].items():
            if (final_wood_mask[key] == False
                    and value['weight'] < 2 * pre_dis):
                final_wood_mask[key] = True

    """
    extract the tree stump points.
    """
    stump = []
    for key, value in G[base_id].items():
        stump.append(key)
    for i, (point) in enumerate(pcd):
        if (point[2] < pcd[base_id][2]):
            stump.append(i)
    final_wood_mask[stump] = True

    return final_wood_mask
