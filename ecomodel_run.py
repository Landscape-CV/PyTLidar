from ecomodel import Ecomodel, ecomodel_tile, ecomodel_all_tiles
from datetime import datetime
import os


import argparse
parser = argparse.ArgumentParser(description="Process LiDAR point cloud data for tree and leaf metrics.")
# parser.add_argument("tile_file_path", )


if __name__ == "__main__":
    args = parser.parse_args()

    # tile_file_path = r"G:\Projects\TreeCanopyLidar\Datasets\oblong_tile\573090_2840160_oblong.las"
    # tile_file_path = r"G:\Projects\TreeCanopyLidar\Datasets\tiles_leaves_removed\tile_oblong_leaves_removed.xyz"
    tile_file_path = r"G:\Projects\TreeCanopyLidar\Datasets\tiles_leaves_removed\tile_573210_2840120_leaves_removed.xyz"
    # tile_file_path = r"G:\Projects\TreeCanopyLidar\Datasets\Rush7\Tiled_better\gpv1wf_14501596e03675889s_20120504_aa2f0_r06cd_p300khz_x01_smaller_0_4.las"
    # tile_file_path = r"G:\Projects\TreeCanopyLidar\Datasets\MVP_tiles\other\tile_573120_2840120.laz" # Noisy one with trunk in the middle. 
    # tile_file_path = r"G:\Projects\TreeCanopyLidar\Datasets\MVP_tiles\tile_573150_2840110.laz" #  Two trees one that works well.
    tile_name = os.path.basename(tile_file_path).split(".")[0]
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    results_folder = f"results/MVP_tiles_testing_noise_removal_{timestamp}"

    os.makedirs(results_folder, exist_ok=True)

    ecomodel_tile(tile_file_path, results_folder)

    # folder_path = r"G:\Projects\TreeCanopyLidar\Datasets\MVP_tiles\two_trees"
    # ecomodel_all_tiles(folder_path, results_folder)