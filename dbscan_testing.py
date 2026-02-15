from Utils.Utils import load_point_cloud
# from sklearn import _distributor_init


# from sklearn.cluster import DBSCAN
# import numpy as np

from Utils.plot_tools import SimplePlotter

simple_plotter = SimplePlotter()


pc = load_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\results\MVP_tiles_testing_noise_removal_2026-02-14_18-16-45\segment_5.0_before_treeqsm.xyz")

# clustering = DBSCAN(eps=3, min_samples=2).fit(pc)
# labels = clustering.labels_

# print("labels", labels, type(labels))


# combined_data = np.concatenate((pc, labels), axis=1)

# simple_plotter.add_point_cloud(combined_data)

# simple_plotter.show()


##### OPEN 3d DBSCAN

import open3d as o3d
import numpy as np

# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)

# pcd = o3d.io.read_point_cloud("cloud.ply")
pts = np.asarray(pc)


for epsilon in [0.05]:
    labels = np.array(
        pcd.cluster_dbscan(eps=epsilon, min_points=50, print_progress=True)
    )

    # max_label = labels.max()
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # print(pc.shape)
    # print(labels.shape)
    combined_data = np.concatenate((pc, labels[:,np.newaxis]), axis=1)

    mask = combined_data[:, 3] > 0

    simple_plotter.add_point_cloud(combined_data[mask])

    simple_plotter.show(f"Epsilon {epsilon}")
    # simple_plotter.plotter.save_graphic(f"Epsilon_{epsilon}.svg")

## PDAL Connected components instead of dbscan. I 

# they require building pdal cpp lib. Not worth it probably. 

## Sklearn Method

# from sklearn.cluster import OPTICS, DBSCAN
# import numpy as np


# pc = load_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\results\MVP_tiles_testing_noise_removal_2026-02-14_18-16-45\segment_5.0_before_treeqsm.xyz")

# clustering = OPTICS().fit(pc)
# labels = clustering.labels_

# print("labels", labels, type(labels))


# combined_data = np.concatenate((pc, labels), axis=1)

# simple_plotter.add_point_cloud(combined_data)

# simple_plotter.show()


# ### Connected components 3d

