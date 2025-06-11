import datetime
import numpy as np
import open3d as o3d
import networkx as nx
from Graph_Path import array_to_graph, extract_path_info
from LS_circle import getRootPt
from ExtractInitWood import extract_init_wood
from ExtractFinalWood import extract_final_wood
from Accuracy_evaluation import evaluate_indicators
from Visualization import show_graph, sp_graph, show_pcd

# load single tree point cloud.
tree_file = 'tree_48_49'
pcd = o3d.io.read_point_cloud(f'{tree_file}.pcd')
pcd = np.asarray(pcd.points)

# Please ensure that the growth direction of the tree is parallel to the Z coordinate axis.
treeHeight = np.max(pcd[:, 2])-np.min(pcd[:, 2])

# fit the root point.
root, fit_seg = getRootPt(pcd, lower_h=0.0, upper_h=0.2)
pcd = np.append(pcd, root, axis=0)
root_id = pcd.shape[0]-1
print("root_ID:", root_id)
# show_pcd(pcd)

# construct networkx Graph.
print(str(datetime.datetime.now()) + ' | >>>constructing networkx Graph...')
G = array_to_graph(pcd, root_id, kpairs=3, knn=300, nbrs_threshold=treeHeight/30, nbrs_threshold_step=treeHeight/60)

# # save/read already constructed Graph to reduce processing time.
# nx.write_gpickle(G, 'E:\\folder\\G.gpickle')
# G = nx.read_gpickle('E:\\folder\\G.gpickle')

print(">>>connected components of constructed Graph: ", nx.number_connected_components(G))
# show_graph(pcd, G)

# extract path info information from graph
print(str(datetime.datetime.now()) + ' | >>>extracting shortest path information...')
path_dis, path_list = extract_path_info(G, root_id, return_path=True)
# show_graph(pcd, sp_graph(path_list, root_id))

# extract initial wood points.
print(str(datetime.datetime.now()) + ' | >>>extracting initial wood points...')
init_wood_ids = extract_init_wood(pcd, G, root_id, path_dis, path_list,
                                  split_interval=[0.1, 0.2, 0.3, 0.5, 1], max_angle=0.25*np.pi)

# extract final wood points.
print(str(datetime.datetime.now()) + ' | >>>extracting final wood points...')
final_wood_mask = extract_final_wood(pcd, root_id, path_dis, path_list, init_wood_ids, G)

# remove the inserted root point and extract wood/leaf points by mask index.
final_wood_mask[-1] = False
wood = pcd[final_wood_mask]
final_wood_mask[-1] = True
leaf = pcd[~final_wood_mask]

# write separation result with .txt format.
np.savetxt(f'wood_points_{tree_file}.txt', wood, fmt='%1.6f')
np.savetxt(f'leaf_points_{tree_file}.txt', leaf, fmt='%1.6f')    





