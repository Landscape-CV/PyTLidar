import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from Utils.Utils import load_point_cloud
from Utils.plot_tools import ResultsPlotter

class TwigDiameterCalculator: 
    def __init__(self):
        




# mean = np.array([5.731535729999999749e+05, 2.840115856000000145e+06,-24])
# plotter = ResultsPlotter(mean)

pc = load_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\test_output\smaller.las")


# Run DBSCAN
db = DBSCAN(eps=0.1, min_samples=10).fit(pc)

# Cluster labels (-1 means noise)
labels = db.labels_
print("Labels:", labels)
print(pc.shape)
labels = labels[:, np.newaxis]
print(labels.shape)
data = np.concat((pc, labels), axis=1)
np.savetxt("new_cloud.txt", data)

