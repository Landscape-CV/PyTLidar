import math
import numpy as np
from Components_classify import getAngle3D
from Components_classify import components_classify
import networkx as nx
from Visualization import graph_cluster, graph_cluster2, show_clusters

def extract_init_wood(pcd, G, base_id, path_dis, path_list, split_interval=[0.1,0.2,0.3,0.5,1],
                     max_angle=np.pi):
    """
    Clustering by cut the edges of graph G based on shortest path length and single edge length.

    Parameters
    ----------
    G : networkx graph
        NetworkX graph object from which to split.
    path_dis : dictionary
        the key is the point ID, the value is the shortest path length.
    split_interval : float
        split interval on shortest path length.
    max_angle : float
        The max acceptable spatial angle of two vectors.
    Returns
    -------

    """

    # precursor distance/direction-based segmentation.
    print("cut edges...")
    remove_edge_list = []
    for (u, v, d) in G.edges(data=True):
        if (u == base_id or v == base_id):
            continue
        pre_u_dis = path_dis[u] - path_dis[path_list[u][-2]]
        pre_v_dis = path_dis[v] - path_dis[path_list[v][-2]]
        pre_u_vec = pcd[u] - pcd[path_list[u][-2]]
        pre_v_vec = pcd[v] - pcd[path_list[v][-2]]
        if (d['weight'] > 2 * min(pre_u_dis, pre_v_dis)
                or getAngle3D(pre_u_vec, pre_v_vec) > max_angle):
            remove_edge_list.append([u, v])
    G.remove_edges_from(remove_edge_list)

    # clusters = graph_cluster(pcd, G)
    # show_clusters(clusters)

    # multi-scale segmentation.
    interval_dicts = []
    for i in range(len(split_interval)):
        interval_dicts.append({})
    for id, dis in path_dis.items():
        for i, (interval) in enumerate(split_interval):
            f = math.floor(dis / interval)
            if f in interval_dicts[i]:
                interval_dicts[i][f].append(id)
            else:
                interval_dicts[i][f] = [id]

    init_wood_ids = []
    for i, (interval_dict) in enumerate(interval_dicts):
        print("interval:", split_interval[i])
        components = []
        for key, value in interval_dict.items():
            sub_G = G.subgraph(value)
            for component in nx.connected_components(sub_G):
                components.append(component)
        print("components:", len(components))

        # clusters = graph_cluster2(pcd, components)
        # show_clusters(clusters)

        # recognition of wood clusters with linear/cylindrical shape in a individual scale.
        classify_components = components_classify(pcd, components, path_list, t_linearity=0.9,
                                                  t_error=0.2, split_interval=split_interval[i])
        for classify_component in classify_components:
            if (classify_component[0] != 0):
                for elm in classify_component[1]:
                    init_wood_ids.append(elm)

    init_wood_ids = np.unique(init_wood_ids)
    return init_wood_ids