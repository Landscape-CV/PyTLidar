from ecomodel import Ecomodel, ecomodel_tile
from datetime import datetime
import os
from Utils.plot_tools import ResultsPlotter

import argparse
parser = argparse.ArgumentParser(description="Process LiDAR point cloud data for tree and leaf metrics.")
# parser.add_argument("tile_file_path", )


if __name__ == "__main__":
    # args = parser.parse_args()

    # tile_file_path = r"G:\Projects\TreeCanopyLidar\PyTLidar\Dataset\Tiles\tile_00005_1_0.laz"
    # # tile_file_path = r"G:\Projects\TreeCanopyLidar\PyTLidar\Dataset\Tiles\tile_573150_2840110.laz"
    # # tile_file_path = r"G:\Projects\TreeCanopyLidar\PyTLidar\Dataset\Tiles\other\tile_573130_2840100.laz"
    # tile_name = os.path.basename(tile_file_path).split(".")[0]
    # now = datetime.now()
    # timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # results_folder = f"results/{tile_name:_<25}{timestamp}"

    # os.makedirs(results_folder, exist_ok=True)
    # ecomodel_tile(tile_file_path, results_folder)

    folder = r"G:\Projects\TreeCanopyLidar\PyTLidar\results\tile_00005_1_0___________2025-12-04_21-48-45"
    mean = [0,0,-1.75]
    plotter = ResultsPlotter(mean)
    for x in range(100):
        try:
            plotter.add_point_cloud_file(f"{folder}/segment_{x}.0_leaves_removed.xyz", mean)
        except:
            pass
    plotter.add_cylinders(f"{folder}/cylinders.txt", mean)
    plotter.show()
