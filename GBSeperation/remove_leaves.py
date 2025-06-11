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
import laspy
import matplotlib.pyplot as plt


def load_point_cloud(file_path, intensity_threshold = 0, full_data = False):
    """
    Load a point cloud from LAS or LAZ files.

    Parameters:
    file_path : str
        Path to the LAS or LAZ file.

    Returns:
    point_cloud : ndarray
        Nx3 matrix of point coordinates (x, y, z).
    """
    if ".xyz" in file_path:
        # Load point cloud from an XYZ file
        point_data = np.loadtxt(file_path, dtype=np.float64)
        if point_data.shape[1] == 3:
            point_cloud = point_data
        elif point_data.shape[1] == 4:
            I = point_data[:, 3] > intensity_threshold
            point_cloud = point_data[I, :3]
        else:
            raise ValueError("Unsupported format in XYZ file.")
        return point_cloud if not full_data else (point_cloud, point_data)
    with laspy.open(file_path) as las:
        point_data = las.read()
        point_data = np.vstack((point_data.x, point_data.y, point_data.z,point_data.intensity)).T.astype('float64')
        I = point_data[:,3]>intensity_threshold
        point_data = point_data[I]
        point_cloud = point_data[:,0:3]
    return point_cloud if not full_data else (point_cloud,point_data)


class LeafRemover:
    def __init__(self):
        self.knn = 300
        self.kpairs = 3
        
        
    def process(self, xyz_point_cloud):
        """
        From GBSeperation (https://zenodo.org/records/6837613)
        
        Parameters: 
            xyzi_point_cloud: Nx4 array of XYZ and intensity values
        Returns: 
            wood: Nx4 array of XYZ and intensity values representing wood
            leaf: Nx4 array of XYZ and intensity values representing leaves
        """
        # Please ensure that the growth direction of the tree is parallel to the Z coordinate axis.

        # Strip intensity values for GBSeperation Algorithm
        # xyz_point_cloud = xyzi_point_cloud[:,0:3]

        print("xyz_point_cloud", xyz_point_cloud.shape)
        # print("xyzi_point_cloud", xyzi_point_cloud.shape)

        # print(xyz_point_cloud)
        # exit()

        treeHeight = np.max(xyz_point_cloud[:, 2])-np.min(xyz_point_cloud[:, 2])

        # fit the root point.
        root, fit_seg = getRootPt(xyz_point_cloud, lower_h=0.0, upper_h=0.2)
        xyz_point_cloud = np.append(xyz_point_cloud, root, axis=0)
        root_id = xyz_point_cloud.shape[0]-1
        print("root_ID:", root_id)
        # show_pcd(pcd)

        # construct networkx Graph.
        print(str(datetime.datetime.now()) + ' | >>>constructing networkx Graph...')
        G = array_to_graph(xyz_point_cloud, root_id, kpairs=self.kpairs, knn=self.knn, nbrs_threshold=treeHeight/30, nbrs_threshold_step=treeHeight/60)

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
        init_wood_ids = extract_init_wood(xyz_point_cloud, G, root_id, path_dis, path_list,
                                        split_interval=[0.1, 0.2, 0.3, 0.5, 1], max_angle=0.25*np.pi)

        # extract final wood points.
        print(str(datetime.datetime.now()) + ' | >>>extracting final wood points...')
        final_wood_mask = extract_final_wood(xyz_point_cloud, root_id, path_dis, path_list, init_wood_ids, G)

        print(final_wood_mask.shape)

        # remove the inserted root point and extract wood/leaf points by mask index.
        final_wood_mask[-1] = False
        wood = xyz_point_cloud[final_wood_mask]
        final_wood_mask[-1] = True
        leaf = xyz_point_cloud[~final_wood_mask]

        return wood, leaf

if __name__ == "__main__":
    
    tree_file = "tree_1"
    filename = f"{tree_file}.las"

    point_cloud_array = load_point_cloud(filename)

    print("point_cloud_array", point_cloud_array.shape)

    leaf_remover = LeafRemover()

    wood, leaf = leaf_remover.process(point_cloud_array)

    # write separation result with .txt format.
    np.savetxt(f'wood_points_{tree_file}.txt', wood, fmt='%1.6f')
    np.savetxt(f'leaf_points_{tree_file}.txt', leaf, fmt='%1.6f') 

    fig = plt.figure()
    plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(wood[:,0], wood[:,1], wood[:,2], c=wood[:,2], cmap='viridis', s=1)

    fig.savefig('Plot.png')


# # load single tree point cloud.
# tree_file = 'tree_48_49'
# pcd = o3d.io.read_point_cloud(f'{tree_file}.pcd')
# pcd = np.asarray(pcd.points) # XYZ coordinates. 

# print(pcd)
# exit()



   





