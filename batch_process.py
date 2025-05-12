from treeqsm import treeqsm
import os
import sys
import numpy as np
import pandas as pd
from tools.define_input import define_input
from Utils.Utils import load_point_cloud
import warnings
warnings.filterwarnings('ignore')
def batched(folder, intensity_threshold = 0):
    """
    Batch process a folder of LAS or LAZ files.
    
    Parameters:
    folder : str
        Path to the folder containing LAS or LAZ files.
    intensity_threshold : int, optional
        Intensity threshold for filtering points. Default is 0.
    """ 
    # List all files in the folder
    files = os.listdir(folder)
    files = [f for f in files if f.endswith('.las') or f.endswith('.laz')]
    if len(files) == 0:
        print("No LAS or LAZ files found in the folder.")
        return
    # Define input for each tree
    clouds = []
    for i, file in enumerate(files):
        point_cloud = load_point_cloud(os.path.join(folder, file), intensity_threshold)
        if point_cloud is not None:
            point_cloud = point_cloud - np.mean(point_cloud,axis = 0)
            clouds.append(point_cloud)
    inputs = define_input(clouds,1,1,1)
    for i, input_params in enumerate(inputs):
        input_params['name'] = files[i]
        input_params['savemat'] = 0
        input_params['savetxt'] = 0
        input_params['plot'] = 0
        
    # Process each tree
    for i, input_params in enumerate(inputs):
        print(f"Processing tree {i + 1}...")
        treeqsm(clouds[i],input_params)

if __name__== "__main__":
    try:
        batched(sys.argv[1], sys.argv[2])
        
    except IndexError:
        try:
            batched("/Users/johnhagood/Documents/LiDAR/segmented_trees/",45000)
        except FileNotFoundError:
            batched(r'C:\Users\johnh\Documents\LiDAR\segmented_trees',45000)