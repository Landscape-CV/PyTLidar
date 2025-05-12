import Utils.Utils as Utils
import numpy as np
import os
import torch 
import laspy
from sklearn import linear_model
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from main_steps.cover_sets import cover_sets
from main_steps.segments import segments
from main_steps.cluster import connect_covers,segment_point_cloud
import time
import cProfile
import pstats
import open3d as o3d
import trimesh
from alphashape import alphashape
class Ecomodel:
    
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        self._raw_tiles = []
        self.min_x = float('inf')
        self.min_y = float('inf')
        self.min_z = float('inf')
        self.max_x = float('-inf')
        self.max_y = float('-inf')
        self.max_z = float('-inf')
        self.mean = np.zeros(3)
    def add_tile(self, tile):
        
        self.min_x = min(self.min_x, tile.min_x)
        self.min_y = min(self.min_y, tile.min_y)
        self.min_z = min(self.min_z, tile.min_z)
        self.max_x = max(self.max_x, tile.max_x)
        self.max_y = max(self.max_y, tile.max_y)
        self.max_z = max(self.max_z, tile.max_z)
        self._raw_tiles.append(tile)

    def normalize_raw_tiles(self):
        """
        Normalize the point cloud  to have zero mean and unit variance.
        Parameters: 
                None
        Returns:        
                numpy.ndarray: Normalized point cloud, shape (n_points, 3).
        """
        means = np.zeros(3)
        N =0
        for tile in self._raw_tiles:
            means+=np.mean(tile.cloud, axis=0)*len(tile.cloud)
            N+=len(tile.cloud)
        self.mean = means/N
        for tile in self._raw_tiles:    
            tile.cloud = tile.cloud - self.mean
            tile.point_data[:, 0:3] = tile.point_data[:, 0:3] - self.mean
            
            
        
    def filter_ground(self,tile_list, band_size = 0.1, threshold = 20,offset = 0.2):
        """
        Filter out ground points from the point cloud P.
        Parameters: 
                band_size (float): Size of the bands to find ground.
                threshold (float): number of points per square xy unit.
        Returns:        
                numpy.ndarray: Filtered point cloud, shape (n_points, 3).
        """
        print("Filtering Ground")
        new_min_z = float('inf')
        for tile in tile_list:
            if tile == 0 or  not tile.contains_ground:
                continue
            tile.numpy()
            z_range = tile.max_z - tile.min_z
            x_range = tile.max_x - tile.min_x
            y_range = tile.max_y - tile.min_y
            # prev_len = 1
            max_band_points = 0
            max_band = None
            for i in range(int(z_range/band_size+1)):
                
                band_min = tile.min_z + i * band_size
                band_max = tile.min_z + (i + 1) * band_size
                mask = (tile.cloud[:, 2] >= band_min) & (tile.cloud[:, 2] < band_max)
                band = tile.cloud[mask]
                if len(band) >(x_range*y_range*threshold):
                    max_band = band
                    max_band_points = len(band)
                    break
                if len(band) > max_band_points:
                    max_band = band
                    max_band_points = len(band)
            if len(band) != max_band_points:
                band = max_band  
            using_pt = False
            if type(band) == torch.Tensor:
                using_pt = True
                band = band.cpu().numpy()
            ground_start = min
            line = linear_model.RANSACRegressor(random_state = 0)
            band = band[band[:,2] < float(band_max)-.2*float(band_size)]
            
            line.fit(band[:,0:2],band[:,2])
            if using_pt:
                ground_line = torch.Tensor(line.predict(tile.get_cloud_as_array()[:, 0:2])).to(self.device)
                I = tile.cloud[:, 2] > ground_line+offset
            else:
                I = tile.cloud[:, 2] > line.predict(tile.cloud[:, 0:2]) + offset
            # I = tile.cloud[:, 2] > band_max+offset
            point_data = tile.point_data[I]
            tile.cloud = tile.cloud[I]
            new_min_z = min(new_min_z, tile.cloud[:, 2].min())
            tile.point_data = point_data
            # tile.to_xyz("filtered.xyz")
            
        self.min_z = new_min_z
            

    def segment_trees(self, intensity_threshold= 0):
        """
        Cluster the point cloud P
        Parameters: 
                min_points (int): Minimum number of points in a cluster.
        Returns:
                numpy.ndarray: Clustered point cloud, shape (n_points, 3).
        """         
        
        inputs = {'PatchDiam1': 0.15, 'BallRad1':.15, 'nmin1': 10}
        # inputs = {'PatchDiam1': 0.1, 'BallRad1':.125, 'nmin1': 5}
        
        cover_set_adjust = 0 
        for i,tile in enumerate(self.tiles.flatten()):
            if tile == 0:
                continue
            
            print("Segmenting tile: ",i)

            intensity_mask = tile.point_data[:,3]>intensity_threshold
            tile.point_data = tile.point_data[intensity_mask]
            tile.cloud = tile.cloud[intensity_mask]
            # tile.cover_sets = tile.cover_sets[intensity_mask]
            # tile.segment_labels = tile.segment_labels[intensity_mask]
            # tile.cluster_labels = tile.cluster_labels[intensity_mask]


            print("Create Cover Sets")
            start = time.time()
            cover = cover_sets(tile.get_cloud_as_array(), inputs, qsm =False, device = self.device, full_point_data = tile.point_data)
            if len(cover['sets']) == 0:
                print("No cover sets found")
                continue
            
            labels = cover['sets']
            
            mask = labels >-1
            tile.cloud = tile.cloud[mask]
            tile.point_data = tile.point_data[mask]
            labels = labels[mask]
            tile.cover_sets=labels
            cover_set_adjust=len(tile.cover_sets)

            if len(labels) == 0:
                print("No cover sets found")
                continue
            print("Time to create cover sets:",time.time()-start)

            print("Segment Cloud")
            start = time.time()
            segment_point_cloud(tile)
            print("Time to segment cloud:",time.time()-start)
            
            # tile.cluster_labels = labels

            

            

            print("Writing File")
            tile.to_xyz(f"clustered_{i}.xyz", True)
            
    
    def calc_volumes(self):
        print("Calculating volumes")
        start = time.time()
        for i,tile in enumerate(self.tiles.flatten()):
            if tile == 0:
                continue
            for label in torch.unique(tile.cluster_labels):
                shape_def_start = time.time()
               
                
                shape = alphashape(tile.cloud[tile.cluster_labels == label].cpu().numpy(), alpha=0.1)
                print("Volume of object : {}".format( shape.volume))
                print("Time to calculate volume:",time.time()-shape_def_start)
            

        print("Time to calculate volumes:",time.time()-start)
            

    def subdivide_tiles(self, cube_size = 1, meter_conversion = 1):
        """
        Subdivides the point cloud P into smaller cubes of size cube_size.
        Parameters: 
                
                cube_size (float): Size of the cubes to subdivide the point cloud into.
                meter_conversion (float): Conversion factor to convert the cube size to meters.
                Returns:        
                numpy.ndarray: Subdivided point cloud, shape (n_points, 3).
        """
        if meter_conversion != 1:
            cube_size = cube_size / meter_conversion
        
        if type(self._raw_tiles[0].cloud) == np.ndarray:
            data_type = 'numpy'
        elif type(self._raw_tiles[0].cloud) == torch.Tensor:
            data_type = 'torch'
        # Calculate the number of cubes in each direction   
        X = np.ceil((self.max_x - self.min_x) / cube_size).astype(int)
        Y = np.ceil((self.max_y - self.min_y) / cube_size).astype(int)
        Z = np.ceil((self.max_z - self.min_z) / cube_size).astype(int)
        MinX = self.min_x-self.mean[0]
        MinY = self.min_y-self.mean[1]
        MinZ = self.min_z-self.mean[2]
        

        
        subdivided = np.zeros(shape = (X,Y)).astype(object)
        print(f"Subdividing into {X} x {Y}  prisms with area {cube_size} m")
        for i in range(X):
            
            for j in range(Y):
                
                k=0
                # for k in range(Z):
                    
                if data_type == 'numpy':
                    # Calculate the coordinates of the cube
                    cube_min = np.array([MinX + i * cube_size, MinY + j * cube_size, MinZ ])
                    cube_max = np.array([MinX + (i + 1) * cube_size, MinY + (j + 1) * cube_size, MinZ + (Z) * cube_size])
                    points = np.zeros(shape = (0,4))
                    for tile in self._raw_tiles:
                        # tile.to_xyz("normalized_tile.xyz")
                        mask = np.all((tile.cloud >= cube_min) & (tile.cloud <= cube_max), axis=1)
                        points = np.concatenate([points, tile.point_data[mask]])


                else:
                    # Calculate the coordinates of the cube
                   
                    cube_min = torch.tensor([MinX + i * cube_size, MinY + j * cube_size, MinZ ]).to(torch.float32).to(self.device)
                    cube_max = torch.tensor([MinX + (i + 1) * cube_size, MinY + (j + 1) * cube_size, MinZ + Z * cube_size]).to(torch.float32).to(self.device)
                    
                

                    points = torch.zeros(size=(0,4)).to(self.device)#need to check for number of data points
                    for tile in self._raw_tiles:

                        tile.to(tile.device)
                        mask = torch.all((tile.cloud >= cube_min) & (tile.cloud <= cube_max), axis=1)
                        points = torch.concatenate([points, tile.point_data[mask]])
                if len(points)>0:
                    if k ==0:

                        subdivided[i, j] =Tile(points[:,0:3],points,True)
                        
                    else:
                        subdivided[i, j] =Tile(points[:,0:3],points)
                        

                       
                    # break#only do one tile for now
            #     break#only do one tile for now
            # break#only do one tile for now
        self.tiles = subdivided
        self._raw_tiles = []
        return subdivided
    
    def recombine_tiles(self):
        """
        """


        tiles = self.tiles.flatten()
        for tile in tiles:
            tile.numpy()
        base_tile = tiles[0]
        base_tile.cloud = np.concatenate([tile.cloud for tile in tiles])
        base_tile.point_data = np.concatenate([tile.point_data for tile in tiles])
        base_tile.segment_labels = np.concatenate([tile.segment_labels for tile in tiles])
        base_tile.cover_sets = np.concatenate([tile.cover_sets for tile in tiles])
        base_tile.cluster_labels = np.concatenate([tile.cluster_labels for tile in tiles])
            
        


        self._raw_tiles = [base_tile]
        self.tiles = None
        self.min_x = float(base_tile.cloud[:,0].min())+self.mean[0]
        self.min_y = float(base_tile.cloud[:,1].min())+self.mean[1]
        self.min_z = float(base_tile.cloud[:,2].min())+self.mean[2]
        self.max_x = float(base_tile.cloud[:,0].max())+self.mean[0]
        self.max_y = float(base_tile.cloud[:,1].max())+self.mean[1]
        self.max_z = float(base_tile.cloud[:,2].max())+self.mean[2]


    
    @staticmethod 
    def combine_las_files(folder,ecomodel, intensity_threshold = 0):
        """
        Combine multiple LAS or LAZ files into a single point cloud.
        Parameters:     
                folder (str): Path to the folder containing pre-tiled LAS or LAZ files.
        Returns:    
                np.array: Combined point cloud.
        
        """

        files = os.listdir(folder)
        files = [f for f in files if f.endswith('.las') or f.endswith('.laz')]
        if len(files) == 0:
            print("No LAS or LAZ files found in the folder.")
            return
        # Define input for each tree
        for i, file in enumerate(files):
            print(i)
            point_cloud, point_data = Utils.load_point_cloud(os.path.join(folder, file), intensity_threshold,True)
            if point_cloud is not None:
                
                ecomodel.add_tile(Tile(point_cloud,point_data,True))
        return ecomodel

    def denoise(self):
        """
        Denoise the point cloud P.
        Parameters: 
                None
        Returns:        
                numpy.ndarray: Denoised point cloud, shape (n_points, 3).
        """
        for tile in self._raw_tiles:
            if tile == 0:
                continue
            # tile.to_xyz("tile.xyz")
            
            tile.o3d_cloud = o3d.geometry.PointCloud()
            tile.o3d_cloud.points = o3d.utility.Vector3dVector(tile.cloud)
           
            print("DBSCAN Cleanup")
            start = time.time()
            labels = np.array(tile.o3d_cloud.cluster_dbscan(eps=0.5, min_points=20, print_progress=True))

            mask = labels != -1
            tile.cloud = tile.cloud[mask]
            tile.point_data = tile.point_data[mask]
            print("Time to DBSCAN:",time.time()-start)


class Tile:

    def __init__(self, cloud, point_data = None,contains_ground = False):
        self.cluster_labels = np.zeros(len(cloud))
        self.device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cloud = cloud# torch.from_numpy(cloud.astype('float32')).to(self.device)
        self.point_data = point_data if point_data is not None else cloud
        # self.point_data =self.get_point_data_as_array()
        if type(self.cloud) == np.ndarray:
            
            self.min_x = float(np.min(cloud[:, 0]))
            self.min_y = float(np.min(cloud[:, 1]))
            self.min_z = float(np.min(cloud[:, 2]))
            self.max_x = float(np.max(cloud[:, 0]))
            self.max_y = float(np.max(cloud[:, 1]))
            self.max_z = float(np.max(cloud[:, 2]))
        elif type(self.cloud) == torch.Tensor:
            self.min_x = float(torch.min(cloud[:, 0]))
            self.min_y = float(torch.min(cloud[:, 1]))
            self.min_z = float(torch.min(cloud[:, 2]))
            self.max_x = float(torch.max(cloud[:, 0]))
            self.max_y = float(torch.max(cloud[:, 1]))
            self.max_z = float(torch.max(cloud[:, 2]))
        self.contains_ground = contains_ground
        self.cover_sets = np.zeros(len(cloud))-1
        self.cluster_labels = np.zeros(len(cloud))-1
        self.segment_labels = np.zeros(len(cloud))-1
    

    def to_xyz(self, file_path, with_clusters = False, with_intensity = False):
        """
        Save the point cloud to a XYZ file.
        Parameters: 
                file_path (str): Path to save the XYZ file.
        Returns:        
                None
        """
        cloud = self.get_cloud_as_array()
        point_data = self.get_point_data_as_array()
        cluster_labels = self.get_cluster_labels_as_array()
        segment_labels = self.get_segment_labels_as_array()
        if not with_clusters and not with_intensity:
            np.savetxt(file_path, cloud, delimiter=',')
        elif with_clusters and not with_intensity:
            np.savetxt(file_path, np.column_stack([cloud,segment_labels,self.cover_sets]), delimiter=',')
        elif with_intensity and not with_clusters:
            np.savetxt(file_path, point_data, delimiter=',')
        elif with_clusters and with_intensity:
            np.savetxt(file_path, np.column_stack([point_data,cluster_labels]), delimiter=',')
    
    def get_point_data_as_array(self):
        """
        Convert point data to a numpy array.
        Parameters: 
                None
        Returns:        
                numpy.ndarray: Point data as a numpy array.
        """
        
        if type(self.point_data) == laspy.LasData:
            return np.vstack((self.point_data.x,self.point_data.y,self.point_data.z,self.point_data.intensity)).T.astype('float64')
        elif type(self.point_data) == torch.Tensor:
            return self.point_data.cpu().numpy().astype('float32')
        else:
            return self.point_data.astype('float64')

    def get_cluster_labels_as_array(self):
        """
        Convert point cloud to a numpy array.
        Parameters: 
                None
                """
        if type(self.cluster_labels) == np.ndarray:
            return self.cluster_labels
        elif type(self.cluster_labels) == torch.Tensor:
            return self.cluster_labels.cpu().numpy().astype('float64')      

    def get_segment_labels_as_array(self):
        """
        Convert point cloud to a numpy array.
        Parameters: 
                None
                """
        if type(self.segment_labels) == np.ndarray:
            return self.segment_labels
        elif type(self.segment_labels) == torch.Tensor:
            return self.segment_labels.cpu().numpy().astype('float64')    
    def get_cloud_as_array(self):
        """
        Convert point cloud to a numpy array.
        Parameters: 
                None
                """
        if type(self.cloud) == np.ndarray:
            return self.cloud
        elif type(self.cloud) == torch.Tensor:
            return self.cloud.cpu().numpy().astype('float64')
    
    def numpy(self):
        if torch.is_tensor(self.cloud):
            self.cloud = self.cloud.cpu().numpy()
        if torch.is_tensor(self.point_data):
            self.point_data = self.point_data.cpu().numpy()
    
    def plot(self):
        """
        Plot the point cloud.
        Parameters: 
                None
        Returns:        
                None
        """
        if type(self.cloud) == np.ndarray:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(projection='3d')
            subset = np.random.choice(np.arange(len(self.cloud)), size=10000, replace=False)
            subset = self.cloud[subset]
            ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2])
            plt.show()
        elif type(self.cloud) == torch.Tensor:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.cloud[:, 0].cpu().numpy(), self.cloud[:, 1].cpu().numpy(), self.point_data[:, 2].cpu().numpy())
            plt.show()   
   
    def get_intensity_normalized_data(self,factor):
        """
        Normalize the point cloud intensity data.
        Parameters: 
                None
        Returns:        
                numpy.ndarray: Normalized point cloud intensity data.
        """

        if type(self.point_data) == torch.Tensor:
            return self.point_data[:, 3] / 65535.0*factor
        else:
            return np.column_stack([self.cloud,self.point_data[:, 3] / 65535.0*factor])

    def to(self, device):
        if type(self.cloud) == np.ndarray:
            self.cloud = torch.from_numpy(self.cloud.astype('float32')).to(self.device)
        if type(self.point_data) == laspy.LasData:
            self.point_data = torch.from_numpy(np.vstack([self.point_data.x,self.point_data.y,self.point_data.z,self.point_data.intensity]).T.astype('float32')).to(self.device)
        elif type(self.point_data) == np.ndarray:
            self.point_data = torch.from_numpy(self.point_data.astype('float32')).to(self.device)

        if type(self.cluster_labels) == np.ndarray:
            self.cluster_labels= torch.from_numpy(self.cluster_labels.astype('int')).to(self.device)
        if type(self.cover_sets) == np.ndarray:
            self.cover_sets = torch.from_numpy(self.cover_sets.astype('int')).to(self.device)
        if type(self.segment_labels) == np.ndarray:
            self.segment_labels = torch.from_numpy(self.segment_labels.astype('int')).to(self.device)

    def numpy(self):
        if type(self.cloud) == torch.Tensor:
            self.cloud = self.cloud.cpu().numpy()
        if type(self.point_data) == laspy.LasData:
            self.point_data = np.vstack([self.point_data.x,self.point_data.y,self.point_data.z,self.point_data.intensity]).T
        elif type(self.point_data) == torch.Tensor:
            self.point_data = self.point_data.cpu().numpy()

        if type(self.cluster_labels) == torch.Tensor:
            self.cluster_labels= self.cluster_labels.cpu().numpy()
        if type(self.cover_sets) == torch.Tensor:
            self.cover_sets = self.cover_sets.cpu().numpy()
        if type(self.segment_labels) == torch.Tensor:
            self.segment_labels = self.segment_labels.cpu().numpy()

    def concat(self,tile):
        """
            Combines two tiles together. WARNING: Using this with two tiles that have not had the same operations done to them can result in corrupted data
        """
        #TODO: Above warning can be fixed by initializing with array of -1s the length of the cloud
        self.numpy()
        tile.numpy()
        self.cloud = np.concatenate([self.cloud,tile.cloud]) if len(tile.cloud)>0 else self.cloud
        self.point_data = np.concatenate([self.point_data,tile.point_data]) if len(tile.point_data) else self.point_data
        self.segment_labels = np.concatenate([self.segment_labels,tile.segment_labels]) if len(tile.segment_labels) else self.segment_labels
        self.cover_sets = np.concatenate([self.cover_sets,tile.cover_sets]) if len(tile.cover_sets)>0 else self.cover_sets
        self.cluster_labels = np.concatenate([self.cluster_labels,tile.cluster_labels]) if len(tile.cluster_labels)>0 else self.cluster_labels

    
        
    


if __name__ == "__main__":
    # Example usage
    folder = r'/Users/johnhagood/Documents/LiDAR/tiled_scans'
    model = Ecomodel()
    combined_cloud = Ecomodel.combine_las_files(folder,model)
    combined_cloud.filter_ground(combined_cloud._raw_tiles,.5)
    combined_cloud.normalize_raw_tiles()
    for tile in combined_cloud._raw_tiles:
        tile.to(tile.device)
    # combined_cloud.denoise()
    combined_cloud.subdivide_tiles(cube_size = 3)
    combined_cloud.filter_ground(combined_cloud.tiles.flatten())
    combined_cloud.recombine_tiles()
    for tile in combined_cloud._raw_tiles:
        tile.to(tile.device)
    combined_cloud.subdivide_tiles(cube_size = 15)



    
    

    # cProfile.run("combined_cloud.cluster()",filename="results.txt",sort=1)
    # stats = pstats.Stats('results.txt')
    # stats.sort_stats('tottime')
    # stats.reverse_order()
    # stats.print_stats()
    combined_cloud.segment_trees()
    # combined_cloud.calc_volumes()
    # subdivided_cloud = combined_cloud.subdivide_tiles(cube_size = 10)
    # print(subdivided_cloud)
