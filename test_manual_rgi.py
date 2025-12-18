from SegmentRGI.SegmentRGI import classify_wood_leaf
from Utils.Utils import load_point_cloud
from pathlib import Path
from tempfile import TemporaryDirectory
from plyfile import PlyData, PlyElement
import subprocess
import numpy as np
import open3d as o3d

import os

import argparse

parser = argparse.ArgumentParser(description="Test RGI classification on a point cloud segment.")
# parser.add_argument("tree_path", type=str, help="Path to the point cloud file")


# classifier parameters exposed as command line options
def _str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1", "y")

parser.add_argument("--noise_percentile", type=int, default=5, help="Noise percentile for RGI classifier")
parser.add_argument("--angle_deg", type=float, default=15.0, help="Angle threshold in degrees")
parser.add_argument("--curv_thresh", type=float, default=0.07, help="Curvature threshold")
parser.add_argument("--resid_thresh", type=float, default=0.05, help="Residual threshold")
parser.add_argument("--k", type=int, default=30, help="Neighborhood k for surface fitting")
parser.add_argument("--minClusterSize", type=int, default=3, help="Minimum cluster size")
parser.add_argument("--maxClusterSize", type=int, default=100000, help="Maximum cluster size")
parser.add_argument("--smoothMode", type=_str2bool, default=False, help="Smooth mode (true/false)")
parser.add_argument("--useResidualTest", type=_str2bool, default=True, help="Use residual test (true/false)")
parser.add_argument("--useCurvatureTest", type=_str2bool, default=True, help="Use curvature test (true/false)")



class RGI_test:
    def __init__(self):
        pass


    def metric_gatherer(self, point_cloud_intensity):
        """
        Gathers the metrics that I am interested in. 

        Different trials of the input params. 

        data_dict: first value in list is default, next 3 what we are going to test. 
        """

        # tree_path = r"G:\Projects\TreeCanopyLidar\Datasets\segmented_pines\Pinus elliottii\tree_0099.laz"

        intensity_threshold = 0

        file_name = os.path.basename(tree_path).split(".")[0]

        inputs = {
            # "noise_percentile": [0, 2, 4],
            # "angle_deg": [1,2,3,4,5,6,7,8,9,10],
            # "curv_thresh": [0.07] ,
            # "resid_thresh": [0.05, 0.07, 0.1], 
            "k": [10,20,30,40,50,60, 70,80,90,100],
            # "minClusterSize": [ 3, 6, 9],
            # "maxClusterSize": [200000],
            # "smoothMode": [False],
            # "useResidualTest": [False],
            # "useCurvatureTest": [False],
        }

        noise_percentile=5,
        angle_deg=15.0, 
        curv_thresh=0.07, 
        resid_thresh=0.05, 
        k=30,
        minClusterSize=3,
        maxClusterSize=100000,
        smoothMode=True,
        useResidualTest=True,
        useCurvatureTest=True,

        np.savetxt(f"rgi_tuning/{file_name}_with_leaves.xyz",
            point_cloud_intensity[:, 0:4], delimiter=',')

        for key, value_list in inputs.items():
            for value in value_list:

                print(f"####### {file_name}_with_no_leaves__{key}_{value}.xyz")

                kw = {
                    "noise_percentile": 5,
                    "angle_deg": 6, 
                    "curv_thresh": 0.07, 
                    "resid_thresh": 0.05, 
                    "k": 30,
                    "minClusterSize": 150,
                    "maxClusterSize": 100000,
                    "smoothMode": True,
                    "useResidualTest": True,
                    "useCurvatureTest": True,
                }
                kw[key] = value


                try:
                # write as space-separated XYZ (downstream code expects whitespace-delimited files)
                    wood_mask, leaf_mask = self.classify_wood_leaf_on_array(point_cloud_intensity[:, 0:4], **kw)

                    # # Combine classification result with intensity threshold
                    # intensity_mask = point_cloud[:, 3] > intensity_threshold
                    # wood_mask = np.logical_or(wood_mask, intensity_mask)
                    # leaf_mask = np.logical_and(leaf_mask, ~intensity_mask) 

                    # Filter tree_cloud to retain only wood
                    # tree_cloud = tree_cloud[wood_mask]
                    
                    # write as space-separated XYZ (downstream code expects whitespace-delimited files)
                    np.savetxt(f"rgi_tuning/{file_name}_with_no_leaves__{key}_{value}.xyz", point_cloud_intensity[:, 0:4][wood_mask], delimiter=',')
                        
                except Exception as e:
                    print(e)
                    print(f"[WARNING] classify_wood_leaf() failed on segment: {file_name}_with_no_leaves__{key}_{value}.xyz")


    def classify_one(self, point_cloud_intensity):
        """
        
        """
        tree_cloud = point_cloud_intensity[:, 0:4]
        
        wood_mask, leaf_mask = self.classify_wood_leaf_on_array(tree_cloud, **{"smoothMode": False, "angle_deg": 3, "minClusterSize": 5})

        branches_cloud = tree_cloud[wood_mask]
        np.savetxt("rgi_tuning/branches_cloud.xyz", branches_cloud)

        # subprocess.run(["G:\Software\CloudCompare\CloudCompare.exe", "-o", r"G:\Projects\TreeCanopyLidar\PyTLidar\rgi_tuning\branches_cloud.xyz"])


        


    def classify_wood_leaf_on_array(self,tree_cloud, 
                        noise_percentile = 5,
                        angle_deg=15.0, 
                        curv_thresh=0.07, 
                        resid_thresh=0.05, 
                        k=30,
                        minClusterSize = 3,
                        maxClusterSize = 100000,
                        smoothMode = False,
                        useResidualTest = True,
                        useCurvatureTest = True,):
        """
        Run classify_wood_leaf() directly on an in-memory NumPy point cloud array.
        Saves the temporary segment, classifies it, and rebuilds boolean masks.
        """
        

        with TemporaryDirectory() as tmpdir:
            tmp_ply = Path(tmpdir) / "segment.ply"
            tmp_results = Path(tmpdir) / "results"
            tmp_results.mkdir(exist_ok=True)

            # Save current segment to temporary PLY
            vertex_dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('Intensity', 'f4')]
            structured = np.zeros(tree_cloud.shape[0], dtype=vertex_dtype)
            structured['x'] = tree_cloud[:, 0]
            structured['y'] = tree_cloud[:, 1]
            structured['z'] = tree_cloud[:, 2]
            structured['Intensity'] = tree_cloud[:, 3]

            ply_elements = PlyElement.describe(structured, 'vertex')
    
            PlyData([ply_elements]).write(str(tmp_ply))

            # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tree_cloud[:, :3]))
            # o3d.io.write_point_cloud(str(tmp_ply), pcd)


            # Run existing classification pipeline
            classify_wood_leaf(str(tmp_ply), save_dir=str(tmp_results), show_plots=False, 
                                    noise_percentile=noise_percentile,
                                    angle_deg=angle_deg,
                                    curv_thresh=curv_thresh,
                                    resid_thresh=resid_thresh,
                                    k=k,
                                    minClusterSize=minClusterSize,
                                    maxClusterSize=maxClusterSize,
                                    smoothMode=smoothMode,
                                    useResidualTest=useResidualTest,
                                    useCurvatureTest=useCurvatureTest,)

            # Read back the classified clouds
            wood_file = tmp_results / "segment_wood.ply"
            leaf_file = tmp_results / "segment_leaves.ply"
            if not wood_file.exists() or not leaf_file.exists():
                raise FileNotFoundError("Wood/Leaf outputs not found after classification.")

            wood_pcd = o3d.io.read_point_cloud(str(wood_file))
            leaf_pcd = o3d.io.read_point_cloud(str(leaf_file))
            tree_coords = np.asarray(tree_cloud[:, :3])
            wood_coords = np.asarray(wood_pcd.points)
            leaf_coords = np.asarray(leaf_pcd.points)

            def build_mask(sub_coords):
                mask = np.isin(
                    tree_coords.view([('', tree_coords.dtype)] * 3),
                    sub_coords.view([('', sub_coords.dtype)] * 3)
                )
                return mask.squeeze()

            wood_mask = build_mask(wood_coords)
            leaf_mask = build_mask(leaf_coords)
            return wood_mask, leaf_mask


if __name__ == "__main__":
    args = parser.parse_args()
    # tree_path = args.tree_path
    tree_path = r"G:\Projects\TreeCanopyLidar\PyTLidar\Dataset\Tiles\tile_00005_1_0_ground_removed_intensity42000.laz"
    basename = os.path.basename(tree_path)

    test = RGI_test()
    point_cloud, point_cloud_intensity = load_point_cloud(tree_path, full_data=True)

    # build kwargs from parsed args to pass into the classifier
    kw = {
        "noise_percentile": 0,
        "angle_deg":7, 
        "curv_thresh":0.07, 
        "resid_thresh":0.05, 
        "k":100,
        "minClusterSize" : 40,
        "maxClusterSize" : 100000,
        "smoothMode" : True,
        "useResidualTest" : True,
        "useCurvatureTest" : True,
    }


    # Best values so far: 
    # angle_deg = 6 deg. 

    tree_cloud = point_cloud_intensity[:, 0:4]
    # test.metric_gatherer(tree_cloud)
    
    wood_mask, leaf_mask = test.classify_wood_leaf_on_array(tree_cloud, **kw)

    
    branches_cloud = tree_cloud[wood_mask]
    os.makedirs("rgi_tuning", exist_ok=True)
    np.savetxt(f"results/wood_{basename}.xyz", branches_cloud)