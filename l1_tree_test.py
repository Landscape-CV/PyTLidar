# import open3d as o3d
# import numpy as np

# from pc_skeletor import Dataset

# downloader = Dataset()
# trunk_pcd_path, branch_pcd_path = downloader.download_semantic_tree_dataset()

# pcd_trunk = o3d.io.read_point_cloud(trunk_pcd_path)
# pcd_branch = o3d.io.read_point_cloud(branch_pcd_path)
# pcd = pcd_trunk + pcd_branch
# %%
from pc_skeletor import LBC, SLBC

import networkx as nx
import matplotlib.pyplot as plt


from Utils.Utils import load_point_cloud
from Utils.plot_tools import  ResultsPlotter
import numpy as np



# pc = load_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\test_output\CC#20.las")

lbc = LBC(point_cloud=r"G:\Projects\TreeCanopyLidar\PyTLidar\_experimentation\test_output\segment_0.pcd",
          down_sample=0.008, init_contraction = 10,max_contraction=2048)
lbc.extract_skeleton()



# lbc.skeleton_graph
# lbc.skeleton
# print(lbc.skeleton)

# numpy = o3d.geometry.PointCloud

points_np = np.asarray(lbc.contracted_point_cloud)   # Shape: (N, 3)


lbc.extract_topology()

# plt.figure(figsize=(6, 6))
# nx.draw(lbc.skeleton_graph, with_labels=True, node_size=800, node_color="skyblue")
# plt.show()

# Debug/Visualization
lbc.visualize()
lbc.export_results('./output')
lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            steps=300,
            output='./output')

