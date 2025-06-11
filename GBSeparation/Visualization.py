import numpy as np
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt

def show_graph(pcd, G):
    edges = np.asarray(G.edges)
    points = o3d.utility.Vector3dVector(pcd)
    lines = o3d.utility.Vector2iVector(edges)
    line_set = o3d.geometry.LineSet(points, lines)
    line_set.paint_uniform_color((0.5, 0.5, 0.5))
    o3d.visualization.draw_geometries([line_set])

def show_pcd(pcd):
    points = o3d.utility.Vector3dVector(pcd)
    pcd = o3d.geometry.PointCloud(points)
    pcd.paint_uniform_color((1, 0, 0))
    o3d.visualization.draw_geometries([pcd])

# visualization and save the classify results.
def show_save_pcd_fmt(wood, leaf, save_path):
    wood_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(wood))
    leaf_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(leaf))
    wood_pcd.paint_uniform_color([184 / 255, 134 / 255, 11 / 255])
    leaf_pcd.paint_uniform_color([34 / 255, 139 / 255, 34 / 255])
    o3d.visualization.draw_geometries([wood_pcd, leaf_pcd])

    output_pcd = wood_pcd + leaf_pcd
    o3d.io.write_point_cloud(save_path, output_pcd)

def sp_graph(path_list, root_id):
    # Initializing graph.
    sp_G = nx.Graph()
    for key, value in path_list.items():
        if(key != root_id):
            sp_G.add_edge(key, value[-2])
    return sp_G

def graph_cluster(pcd, G):
    label = np.zeros((pcd.shape[0], 1))
    for i, (component) in enumerate(nx.connected_components(G)):
        component = list(component)
        label[component] = i
    clusters = np.column_stack((pcd, label))
    return clusters

def graph_cluster2(pcd, components):
    label = np.zeros((pcd.shape[0], 1))
    for i, (component) in enumerate(components):
        component = list(component)
        label[component] = i
    clusters = np.column_stack((pcd, label))
    return clusters

def show_clusters(clusters):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(clusters[:, :3]))
    labels = clusters[:, 3]
    color_num = 10
    print(f"point cloud has {labels.max() + 1} clusters")
    colors = plt.get_cmap("tab20")((labels % color_num)/color_num)
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])
