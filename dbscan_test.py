import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from Utils.Utils import load_point_cloud
from Utils.plot_tools import ResultsPlotter

mean = np.array([5.731535729999999749e+05, 2.840115856000000145e+06,-24])
plotter = ResultsPlotter(mean)

pc = load_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\local_testing\wood_intenisty_filter_30000.clone.extract.las.xyz")

plotter.add_point_cloud_np(pc, mean)
plotter.show()
exit()
# Run DBSCAN
db = DBSCAN(eps=2, min_samples=50).fit(pc)

# Cluster labels (-1 means noise)
labels = db.labels_
print("Labels:", labels)

data = np.concat((pc, labels))
plotter.add_point_cloud_np(data, mean)
plotter.show()

