import warnings
# from skimage.morphology import skeletonize
# import sknw
warnings.filterwarnings('ignore')
import Utils.Utils as Utils
import numpy as np
import os
import torch 
import laspy
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from sklearn.cluster import k_means
import matplotlib.pyplot as plt
from TreeQSMSteps.cover_sets import cover_sets
from TreeQSMSteps.segments import segments
from TreeQSMSteps.correct_segments import correct_segments
from TreeQSMSteps.tree_sets import tree_sets
from TreeQSMSteps.relative_size import relative_size
from Utils.TreeSegmentation import segment_point_cloud
from TreeQSMSteps.cylinders import cylinders
from TreeQSMSteps.point_model_distance import point_model_distance
from Utils.define_input import define_input
from plotting.cylinders_line_plotting import cylinders_line_plotting
from plotting.point_cloud_plotting import point_cloud_plotting
from plotting.cylinders_plotting import cylinders_plotting
from plotting.qsm_plotting import qsm_plotting
import TreeQSMSteps.LSF as LSF
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation 
from scipy.spatial.distance import cdist
import time
import cProfile
import pstats
import open3d as o3d
import trimesh
from alphashape import alphashape
import pickle
import dotenv
from GBSeparation.remove_leaves import LeafRemover
from robpy.covariance import DetMCD,FastMCD
from sklearn.covariance import MinCovDet
import CSF

from Utils.RobustCylinderFitting import RobustCylinderFitterEcomodel

dotenv.load_dotenv()

class Ecomodel:
    """The Ecomodel class processes and creates html and pdf reports of a large scale point cloud. 

    A processing pipeline extracts the information from raw lidar data of trees. 
    """
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
        Normalize the point cloud to ensure numerical stability.
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
            if tile.terrain_model is not None:
                tile.cloud = Utils.subtract_terrain(tile.cloud,tile.terrain_model,grid_size=tile.grid_size)
            
            tile.cloud[:,:2] = tile.cloud[:,:2] - self.mean[:2]
            
            tile.point_data[:, 0:3] = tile.cloud
        

            
            
    def filter_ground(self,tile_list, band_size = 0.1, threshold = 20,offset = 0.2): 
        csf = CSF.CSF()
        new_min_z = float('inf')
        for tile in tile_list:
            if tile == 0 or not tile.contains_ground:
                continue
            tile.numpy()
            # prameter settings
            csf.params.bSloopSmooth = False
            csf.params.cloth_resolution = 0.3
            

            csf.setPointCloud(tile.cloud)
            ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
            non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
            csf.do_filtering(ground, non_ground)
            non_ground_mask = np.array(non_ground)
            ground_mask = np.array(ground)
            tile.ground = tile.cloud[ground_mask]
            tile.point_data = tile.point_data[non_ground_mask]
            tile.cloud = tile.cloud[non_ground_mask]
            new_min_z = min(new_min_z, tile.cloud[:, 2].min())
        self.min_z = new_min_z

    # Keeping for now, but overwriting with CSF
    # 
    def filter_below_ground(self,tile_list, band_size = 0.1, threshold = 100,offset = 0.2):

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
            if tile == 0 or not tile.contains_ground:
                continue
            tile.numpy()
            z_range = tile.max_z - tile.min_z
            x_range = tile.max_x - tile.min_x
            y_range = tile.max_y - tile.min_y
            # prev_len = 1
            max_band_points = 0
            max_band = None
            for i in range(int(z_range/band_size)):
                
                band_min = tile.min_z + i * band_size
                band_max = tile.min_z + (i + 1) * band_size
                mask = (tile.cloud[:, 2] >= band_min) & (tile.cloud[:, 2] < band_max)
                num_points = np.sum(mask)
                if num_points >(x_range*y_range*threshold):
                    max_band = mask
                    max_band_points = num_points
                    break
                if num_points > max_band_points:
                    max_band = mask
                    max_band_points = num_points
            band = tile.cloud[max_band] 

            if type(band) == torch.Tensor:
                band = band.cpu().numpy()
            
            # I = tile.cloud[:, 2] > (band_min + offset)
            I = tile.cloud[:, 2] > (band_max+offset)

            tile.cloud = tile.cloud[I]
            tile.point_data = tile.point_data[I]
            # if len(tile.cloud)
            new_min_z = min(new_min_z, tile.cloud[:, 2].min())
            
            # tile.to_xyz("filtered.xyz")
            
        self.min_z = new_min_z
            
    def get_terrain_model(self,tile_list,grid_size = 0.1):
        """
        Get the terrain model from the point cloud P.
        Parameters: 
                None
        Returns:        
                numpy.ndarray: Terrain model, shape (n_points, 3).
        """
        for tile in tile_list:
            #random subset of ground points to speed up processing
            if tile == 0 or tile.ground is None:
                continue

            if len(tile.ground)>10000:
                indices = np.random.choice(len(tile.ground), size=10000, replace=False)
                ground_points = tile.ground[indices]
            # surface = Utils.get_surface_points(ground_points, grid_size)

            # ground_points = ground_points-np.mean(ground_points,axis=0)#normalize ground points to improve numerical stability
            surface = Utils.rasterize_cloud(ground_points, grid_size)
            surface = Utils.fill_raster_gaps(surface)

            tile.terrain_model = surface
            tile.grid_size = grid_size

    def segment_trees(self, intensity_threshold= 0):
        """
        Segments the point cloud into groups from a single tree
        Parameters: 
                min_points (int): Minimum number of points in a cluster.
        Returns:
                numpy.ndarray: Clustered point cloud, shape (n_points, 3).
        """         
        
        # inputs = {'PatchDiam1': 0.08, 'BallRad1':.08, 'nmin1': 15}
        inputs = {'PatchDiam1': 0.15, 'BallRad1':.15, 'nmin1': 25}
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
            #settings for Missouri data
            # segment_point_cloud(tile,base_height=.75, connect_ambiguous_points=True, fix_overlapping_segments=False,base_dist_multiplier=1.2,max_dist=.17,combine_nearby_bases=False,initial_size_limit=100000,min_height =.1)
            
            segment_point_cloud(tile,min_height=.1,connect_using_midpoint=False,base_height=.3,base_dist_multiplier=2.5,connect_ambiguous_points=True,fix_overlapping_segments=False)
            mask = tile.segment_labels >-2#filters out points that could not be connected, ideal will segment better and this will be uneccesary
            tile.cloud = tile.cloud[mask]
            tile.point_data = tile.point_data[mask]
            tile.segment_labels=tile.segment_labels[mask]
            tile.cover_sets =tile.cover_sets[mask]
            print("Time to segment cloud:",time.time()-start)
            
            # tile.cluster_labels = labels

            

            

            print("Writing File")
            tile.to_xyz(f"clustered_{i}.xyz", True)

            
    def get_qsm_segments(self,intensity_threshold = 40000):
        """
        Get the modeled cylinder and QSM segments from the point cloud P.
        Parameters: 
                intensity_threshold (float): Intensity threshold for filtering.
        Returns:        
                numpy.ndarray: Segmented point cloud, shape (n_points, 3).
        """
        
        max_segment = 0
        for i,tile in enumerate(self.tiles.flatten()):
            if tile == 0:
                continue
            
            tile.numpy()
            
            

            
            tile.cluster_labels = np.array([-2]*len(tile.cloud))
            start = time.time()
            range_mask = np.arange(len(tile.cluster_labels))
            for segment in np.unique(tile.segment_labels):
                
                if segment == -1:
                    continue
                
                # mask = (tile.segment_labels == segment) & (tile.point_data[:,3] >intensity_threshold)
                mask = (tile.segment_labels == segment)
                if len(tile.cloud[mask]) < 100:
                    print(f"Segment {segment} too small")
                    tile.cluster_labels[mask] = -2
                    continue

                
                

                tree_cloud = tile.cloud[mask]
                print("Segment: ",segment)
                inputs = {'PatchDiam1': 0.02, 'BallRad1':.02, 'nmin1': 5}
                cover = cover_sets(tree_cloud, inputs, qsm =False, device = self.device, full_point_data = tile.point_data)
                if len(cover['sets']) == 0:
                    print("No cover sets found"),
                    continue
                
                labels = cover['sets']
                
                noise_mask = labels >-1
                tree_cloud = tree_cloud[noise_mask]
                if len(tree_cloud) < 100:
                    print(f"Segment {segment} too small after noise removal")
                    tile.segment_labels[mask] = -1
                    continue

                ##Code to downsample and remove leaves
                # inputs = {'PatchDiam1': 0.01, 'BallRad1':.01, 'nmin1': 1}
                # cover = cover_sets(tree_cloud, inputs, qsm =False, device = self.device, full_point_data = tile.point_data)
                # if len(cover['sets']) == 0:
                #     print("No cover sets found")
                #     continue
                
                # labels = cover['sets']
                
                # cover_mask = labels >-1
                # tree_cloud = tree_cloud[cover_mask]
                # labels = torch.tensor(labels[cover_mask])
                
                # num_masks = torch.max(labels)+1
                # dim = 3

                # center_points = torch.zeros((num_masks, dim), device=tile.point_data.device)
                # center_points.scatter_reduce_(
                # 0, 
                # labels.unsqueeze(-1).expand(-1, dim), 
                # torch.tensor(tree_cloud,dtype=torch.float32), 
                # reduce='mean',
                # include_self=False
                # )
                # np.savetxt(f"tree_{i}_{segment}.xyz",center_points,delimiter=',')
                # center_points = center_points.cpu().numpy().astype(np.float64)
                # LR = LeafRemover()
                # wood_mask,leaf_mask = LR.process(tree_cloud, True)
                
                # wood_mask = np.isin(labels,np.where(wood_mask)[0])
                # leaf_mask = np.isin(labels,np.where(leaf_mask)[0])
                # intensity_mask = tile.point_data[mask][:,3]>intensity_threshold

                # #Only remove leaves if intensity threshold is not met
                # wood_mask = np.logical_or(wood_mask,intensity_mask)
                # leaf_mask = np.logical_and(leaf_mask,~intensity_mask)
                # tree_cloud = tree_cloud[wood_mask]
                # np.savetxt(f"tree_{i}_{segment}_no_leaves.xyz",tree_cloud,delimiter=',')
                # if len(tree_cloud) < 100:
                #     print(f"Segment {segment} too small after leaf removal")
                #     tile.segment_labels[mask] = -1
                #     continue
                try:
                    qsm_input = define_input(tree_cloud,1,1,1)[0]
                except np.linalg.LinAlgError as e:
                    print(f"Unable to find axis for segment {segment}")
                    tile.segment_labels[mask] = -1
                qsm_input['PatchDiam1'] = 0.025
                qsm_input['PatchDiam2Min'] = 0.05
                qsm_input['PatchDiam2Max'] = 0.08
                qsm_input['BallRad1'] = 0.03
                qsm_input['BallRad2'] = 0.09
                qsm_input['nmin1'] = 5
                print("Cover sets")
                cover1 = cover_sets(tree_cloud, qsm_input)
                print("Tree sets")
                cover1, Base, Forb = tree_sets(tree_cloud, cover1, qsm_input)
                print("Segments")
                segment1 = segments( cover1, Base, Forb,qsm=False)#Running with QSM false, this pre-splits segments
                #Second round of QSM, opting not to do this for now
                # print("Correct")
                # segment1 =correct_segments(tree_cloud,cover1,segment1,qsm_input,0,1,1)#
                # RS = relative_size(tree_cloud, cover1, segment1)
                # print("Cover 2")
                # cover1 = cover_sets(tree_cloud, qsm_input, RS)
                # print("Tree Set 2")
                # cover1, Base, Forb = tree_sets(tree_cloud, cover1, qsm_input, segment1)
                # print("Segment 2")
                # segment1 = segments(cover1, Base, Forb)



                # segs = [np.concatenate(seg).astype(np.int64) for seg in segment1["segments"]]
                segs = segment1["SegmentArray"]


                cover = cover1["sets"]
                # S = np.argsort(cover)
                # tree_cloud = tree_cloud[S]
                # cover = cover[S]
                
                I = np.argsort(cover)
                cover = cover
                # tree_mask = range_mask[mask][noise_mask][wood_mask]
                tree_mask = range_mask[mask][noise_mask]
                tile.point_data[tree_mask] = tile.point_data[tree_mask][I]
                tree_cloud= tree_cloud[I]
                neg_mask = cover ==-1
                num_indices = np.bincount(cover[~neg_mask])
                num_indices = np.concatenate([np.array([np.sum(neg_mask)]),num_indices])
                segs = np.concatenate([np.array([-2,]),segs])
                
                
                cloud_segments= np.repeat(segs, num_indices) 
                new_cloud_segments = cloud_segments.copy()
                cloud_range_mask = np.arange(len(cloud_segments))
                
                
                
                
               
                
                for seg in np.unique(cloud_segments):
                    seg_mask = cloud_segments == seg
                    segment_cloud = tree_cloud[seg_mask]
                    
                    if len(segment_cloud)<30:
                        continue
                    
                    try:
                        axis =Utils.get_axis(segment_cloud)
                    except:
                        continue
                    

                    lexsort_indices = np.lexsort((segment_cloud[:, 2], segment_cloud[:, 1], segment_cloud[:, 0]),axis=0,)
                    #Comments are optional method for sorting segments to find subsegments
                    # lexsort_indices = Utils.get_axis_sort(segment_cloud,axis)
                    # rotated_cloud = Utils.rotate_cloud(segment_cloud,axis)
                    # rotated_cloud = rotated_cloud[lexsort_indices]
                    segment_cloud = segment_cloud[lexsort_indices]


                    sub_segments = Utils.split_segments(segment_cloud,6,15)
                    # sub_segments = Utils.split_segments(rotated_cloud,6,15)
                    while np.sum(sub_segments)>len(sub_segments)/6:

                        ss_idx = sub_segments.astype(bool)
                        sub_segments = sub_segments[np.argsort(lexsort_indices)]
                        I =np.where(sub_segments==1)[0]
                        # J = np.where(sub_segments==0)[0]
                        
                        cluster_mask = cloud_range_mask[seg_mask][I]
                        new_cloud_segments[cluster_mask] = np.max(new_cloud_segments)+1
                        segment_cloud = tree_cloud[seg_mask][I]
                        seg_mask= cluster_mask
                        lexsort_indices = np.lexsort((segment_cloud[:, 2], segment_cloud[:, 1], segment_cloud[:, 0]),axis=0)
                        
                        segment_cloud = segment_cloud[lexsort_indices]
                        
      
                        sub_segments = Utils.split_segments(segment_cloud,6,15)
                        # rotated_cloud= rotated_cloud[ss_idx]
                        # lexsort_indices = np.argsort(rotated_cloud[:, 2])
                        # rotated_cloud = rotated_cloud[lexsort_indices]
                        # sub_segments = Utils.split_segments(rotated_cloud,6,15)


                cloud_segments = new_cloud_segments+max_segment
                trunk = new_cloud_segments ==0
                    

                
                max_segment = cloud_segments.max()+max_segment
                # # mask_cluster_labels = np.zeros(np.sum(mask))-1
                # # mask_cluster_labels[wood_mask] = cloud_segments
                # cluster_mask = range_mask[mask][noise_mask][wood_mask]
                # trunk_mask = range_mask[mask][noise_mask][wood_mask][trunk]
                cluster_mask = range_mask[mask][noise_mask]
                trunk_mask = range_mask[mask][noise_mask][trunk]
                
                tile.cluster_labels[cluster_mask] = cloud_segments
                tile.cluster_labels[trunk_mask] = -3
                tile.trunk_points[trunk_mask]= 1
                tile.cloud[tree_mask] = tree_cloud
                
                # tile.cloud[cluster_mask] =segment_cloud

                # tile.point_data[cluster_mask] =tile.point_data[mask][I][S]
                # tile.cluster_labels[mask] = mask_cluster_labels
                
            print(f"Time to create QSMs in tile {i}:",time.time()-start) 
            # tile.to_xyz(f"clustered_{i}.xyz", True,True)
                # tile.to_xyz(f"clustered_{i}.xyz", True)
                # print("Writing File")
                # print("Writing File")
            # tile.to_xyz(f"clustered_{i}.xyz", True)


    def adjust_location(self):
        """
        Adjust the location of the point cloud P.
        Parameters: 
                None
        Returns:        
                numpy.ndarray: Adjusted point cloud, shape (n_points, 3).
        """
        for tile in self.tiles.flatten():
            if tile == 0:
                continue
            tile.cloud = tile.cloud + self.mean
            tile.point_data[:, 0:3] = tile.point_data[:, 0:3] + self.mean
            tile.min_x = float(tile.cloud[:, 0].min())
            tile.min_y = float(tile.cloud[:, 1].min())
            tile.min_z = float(tile.cloud[:, 2].min())
            tile.max_x = float(tile.cloud[:, 0].max())
            tile.max_y = float(tile.cloud[:, 1].max())
            tile.max_z = float(tile.cloud[:, 2].max())
            tile.cylinder_starts = tile.cylinder_starts + self.mean
            tile.cylinder_axes = tile.cylinder_axes + self.mean




    def get_voxel(self,min_x,min_y,min_z,voxel_size=1,fidelity =.3):
        """
        Get the data from the point cloud P.
        Parameters: 
                min_x (float): Minimum x coordinate of the point cloud.
                min_y (float): Minimum y coordinate of the point cloud.
                min_z (float): Minimum z coordinate of the point cloud.
                voxel_size (float): Size of the voxels to use for the cylinder fitting.
        Returns:        
                numpy.ndarray: Cylinders, shape (n_cylinders, 3).
        """
        for i,tile in enumerate(self._raw_tiles):
            if tile == 0:
                continue
          
            cube_min = np.array([min_x, min_y, min_z])
            cube_max = np.array([min_x + voxel_size, min_y + voxel_size, min_z + voxel_size])
            point_mask = np.all((tile.cloud>=cube_min) & (tile.cloud <= cube_max),axis=1)



            
            
            
            labels = tile.cluster_labels[point_mask]
            tile.reset_cylinders()
            self.calc_volumes(tile,np.unique(labels),cube_min,cube_max)
            labels = tile.cluster_labels[point_mask]
            mask = np.ones(len(tile.cylinder_starts),dtype = bool)#np.all((tile.cylinder_starts >= cube_min) & (tile.cylinder_starts <= cube_max), axis=1)
            cylinder_starts = tile.cylinder_starts[mask]
            cylinder_radii = tile.cylinder_radii[mask]
            cylinder_axes = tile.cylinder_axes[mask]
            cylinder_lengths = tile.cylinder_lengths[mask]
            # branch_labels = tile.branch_labels[mask]
            # branch_orders = tile.branch_orders[mask]
            
            cloud = tile.cloud[point_mask]
            cylinder ={"start": cylinder_starts, "radius": cylinder_radii, "axis": cylinder_axes, "length": cylinder_lengths}#, "branch": branch_labels, "BranchOrder": branch_orders}
            # cylinder = {}
            
            cyl_plot = qsm_plotting(cloud,tile.cover_sets[point_mask],labels,return_html=False,subset = True, fidelity=fidelity,marker_size=1)

            return cylinder, cyl_plot
    
    def calc_volumes(self,tile, segments,min_bound,max_bound):
        """
        Get cylinder information
        """
        print("Calculating volumes")

        start_time = time.time()
        shapes = []
        pcd = o3d.geometry.PointCloud()
        
        range_mask = np.arange(len(tile.cluster_labels))
        mcd = FastMCD()

        fitter = RobustCylinderFitterEcomodel()
        for label in np.unique(segments):
            if label <0:
                continue
            shape_def_start = time.time()
            seg_mask = tile.cluster_labels==label

            Q0 = tile.cloud[seg_mask]
            voxel_mask = np.all((Q0< max_bound) & (Q0 >min_bound),axis = 1)
            Q0 = Q0[voxel_mask]
            
            clustering = DBSCAN(eps=.05, min_samples=5).fit(Q0)
            db_mask = clustering.labels_ != -1
            Q0 = Q0[db_mask]
            pcd.points = o3d.utility.Vector3dVector(Q0)

            
            try:
                obb = pcd.get_oriented_bounding_box()
            except:
                continue

            
            dist =cdist(Q0,Q0)
            np.fill_diagonal(dist,1.0)
            avg_closest_point_dist = np.mean(np.min(dist,axis = 1))
            if avg_closest_point_dist > .02: #can make this a parameter
                tile.cluster_labels[seg_mask]=-2
                continue

            if np.min(obb.extent) <.01 and np.max(obb.extent)>.05:
                tile.cluster_labels[seg_mask]=-1
                continue

            if len(Q0)<5:
                    continue
            
            cylinder_params = fitter.fit(Q0)
            if cylinder_params is None:
                continue
            else:
                start, axis, r, l = cylinder_params

            rotvec = Rotation.from_rotvec(axis)
            rotated_cloud = rotvec.as_matrix() @ Q0.T
            rotated_cloud = rotated_cloud.T
            # rotated_cloud = rotvec.apply(Q0)
            I = np.argsort(rotated_cloud[:,2])
            bot = I[:int(len(I)*.1)]
            t = I[int(len(I)*.9):]
            bottom = Q0[bot]
            top = Q0[t]
            start = np.mean(bottom,axis=0)
            end = np.mean(top,axis=0)
            l = np.linalg.norm(end-start)
            # start_idx = np.argmin(rotated_cloud[:,2])
            # start = Q0[start_idx]

            tile.cylinder_starts = np.concatenate([tile.cylinder_starts,np.array([start])])
            tile.cylinder_axes = np.concatenate([tile.cylinder_axes,np.array([axis])])
            tile.cylinder_lengths = np.append(tile.cylinder_lengths,l)
            tile.cylinder_radii = np.append(tile.cylinder_radii,r)
            
            
            
           
            
               
            
        print("Time to calculate volumes:",time.time()-start_time)
            

    def subdivide_tiles(self, cube_size = 1, meter_conversion = 1):
        """
        Subdivides the point cloud P into smaller tiles of size cube_size. Maintains full z height
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
                    cube_min = np.array([MinX + i * cube_size, MinY + j * cube_size])
                    cube_max = np.array([MinX + (i + 1) * cube_size, MinY + (j + 1) * cube_size])
                    points = np.zeros(shape = (0,4))
                    for tile in self._raw_tiles:
                        # tile.to_xyz("normalized_tile.xyz")
                        mask = np.all((tile.cloud[:,:2] >= cube_min) & (tile.cloud[:,:2] <= cube_max), axis=1)
                        points = np.concatenate([points, tile.point_data[mask]])


                else:
                    # Calculate the coordinates of the cube
                   
                    cube_min = torch.tensor([MinX + i * cube_size, MinY + j * cube_size]).to(torch.float32).to(self.device)
                    cube_max = torch.tensor([MinX + (i + 1) * cube_size, MinY + (j + 1) * cube_size]).to(torch.float32).to(self.device)
                    
                

                    points = torch.zeros(size=(0,4)).to(self.device)#need to check for number of data points
                    for tile in self._raw_tiles:

                        tile.to(tile.device)
                        mask = torch.all((tile.cloud[:,:2] >= cube_min) & (tile.cloud[:,:2] <= cube_max), axis=1)
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
        Inverse of subdivide_tiles, recombines the tiles into a single tile.
        """


        tiles = self.tiles.flatten()
        valid_tiles = []
        for tile in tiles:
            if tile == 0:
                continue
            tile.numpy()
            valid_tiles.append(tile)
        tiles = valid_tiles
        base_tile = tiles[0]
        base_tile.cloud = np.concatenate([tile.cloud for tile in tiles])
        base_tile.point_data = np.concatenate([tile.point_data for tile in tiles])
        base_tile.segment_labels = np.concatenate([tile.segment_labels for tile in tiles])
        base_tile.cover_sets = np.concatenate([tile.cover_sets for tile in tiles])
        base_tile.cluster_labels = np.concatenate([tile.cluster_labels for tile in tiles])
        base_tile.trunk_points =np.concatenate([tile.trunk_points for tile in tiles])
        # base_tile.cylinder_starts = np.concatenate([tile.cylinder_starts for tile in tiles])
        # base_tile.cylinder_axes = np.concatenate([tile.cylinder_axes for tile in tiles])
        # base_tile.cylinder_lengths = np.concatenate([tile.cylinder_lengths for tile in tiles])
        # base_tile.cylinder_radii = np.concatenate([tile.cylinder_radii for tile in tiles])
        # base_tile.branch_labels = np.concatenate([tile.branch_labels for tile in tiles])
        # base_tile.branch_orders = np.concatenate([tile.branch_orders for tile in tiles])

            
        


        self._raw_tiles = [base_tile]
        self.tiles = None
        self.min_x = float(base_tile.cloud[:,0].min())+self.mean[0]
        self.min_y = float(base_tile.cloud[:,1].min())+self.mean[1]
        self.min_z = float(base_tile.cloud[:,2].min())+self.mean[2]
        self.max_x = float(base_tile.cloud[:,0].max())+self.mean[0]
        self.max_y = float(base_tile.cloud[:,1].max())+self.mean[1]
        self.max_z = float(base_tile.cloud[:,2].max())+self.mean[2]

    
           
    
    @staticmethod 
    def combine_las_files(folder,ecomodel, intensity_threshold = 0) -> 'Ecomodel':
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

    def remove_duplicate_points(self):
        for tile in self.tiles.flatten():
            tile.numpy()
            tile.remove_duplicate_points()


    def denoise(self,grid_size = .1, min_points = 10, resolution =.05):
        """
        Denoise the point cloud by subdividing into a X x Y tiles and creating a voronoi partition using cover_sets()
          with a patch diameter of resolution and atleast min_points in each patch.

        """
        for tile in self.tiles.flatten():
            if tile == 0:
                continue
            tile.numpy()
            print("Denoising tile")
            minX = np.min(tile.cloud[:,0])
            minY = np.min(tile.cloud[:,1])
            maxX = np.max(tile.cloud[:,0])
            maxY = np.max(tile.cloud[:,1])
            X = np.ceil((maxX-minX) / grid_size).astype(int)
            Y = np.ceil((maxY-minY) / grid_size).astype(int)
            
            mask = np.zeros(len(tile.cloud),dtype=bool)
            for i in range(X):  
                for j in range(Y):
                    print(f"Processing grid {i},{j} of {X},{Y}")
                    cube_min = np.array([minX + i * grid_size, minY + j * grid_size])
                    cube_max = np.array([minX + (i + 1) * grid_size, minY + (j + 1) * grid_size])
                    point_mask = np.all((tile.cloud[:, :2] >= cube_min) & (tile.cloud[:, :2] <= cube_max), axis=1)
                    if np.sum(point_mask) < min_points:
                        continue
                    sub_cloud = tile.cloud[point_mask]
                    inputs = {'PatchDiam1': resolution, 'BallRad1':resolution+.01, 'nmin1': min_points}
                    cover = cover_sets(sub_cloud, inputs)
                    labels = cover['sets']
                    sub_mask = labels >-1
                    mask[point_mask] = np.logical_or(mask[point_mask],sub_mask)
            tile.cloud = tile.cloud[mask]
            tile.point_data = tile.point_data[mask]
            tile.cover_sets = tile.cover_sets[mask]
            tile.cluster_labels = tile.cluster_labels[mask]
            tile.segment_labels = tile.segment_labels[mask]
            tile.trunk_points = tile.trunk_points[mask]
            
            

            
        

    def pickle(self,name):
        """
        Save the point cloud to a pickle file.
        Parameters: 
                name (str): Path to save the pickle file.
        Returns:        
                None
        """
        with open(name, 'wb') as f:
            pickle.dump(self, f)
    @staticmethod
    def unpickle(name):
        """
        Load the tile object from a pickle file.
        Parameters: 
                name (str): Path to load the pickle file from.
        Returns:        
                Ecomodel: Loaded point cloud.
        """
        with open(name, 'rb') as f:
            return pickle.load(f)


class Tile:
    """A Tile represents a subset of the lidar scan data, 
    and stores the attributes relavent to that tile specifically (cylinders, leafs, etc. )

    A point cloud tile contains N points and B branch segments.

    Attributes: 
        device (str): Device used for matrix operations
        cloud (Nx3 matrix): point cloud representing x, y, z points of tile
        point_data (NxD matrix): point data as well as location (intensity, labels etc)
        min_x (float): Minimum x value among all N points 
        min_y
        min_z
        max_x
        max_y
        max_z
        contains_ground (bool): Boolean if the tile current contains ground points
        cover_sets (Nx1 array): Array representing which cover set label is given to points 
        cluster_labels (Nx1 array): Array representing the labels given to each branch segment where
            -2 = Point not considered as part of a branch segment
            -3 = Point is apart of a trunk

        segment_labels (Nx1 array): Array represeting the labels given to each tree in a tile where
            -1 = Not a tree
        trunk_points (Nx1 array): Array representing whether point is a part of the trunk or not.
        cylinder_starts (Bx3 matrix): Matrix representing the cylinder start points in space of B branch segments 
        cylinder_radii (Bx1 array): Array representing the radii of all B branch segments
        cylinder_axes (Bx3 matrix): Matrix representing the unit vector axis of each B branch segment
        cylinder_lengths (Bx1 array): Array representing the length of each B branch segments.  
        branch_labels: UNUSED
        branch_orders: UNUSED
    """

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
        self.ground =None
        self.terrain_model = None
        self.cover_sets = np.zeros(len(cloud))-1
        self.cluster_labels = np.zeros(len(cloud))-1
        self.segment_labels = np.zeros(len(cloud))-1
        self.trunk_points = np.zeros(len(cloud))-1
        self.cylinder_starts = np.empty((0,3))
        self.cylinder_radii = np.array([])
        self.cylinder_axes = np.empty((0,3))
        self.cylinder_lengths = np.array([])
        self.branch_labels = np.array([])
        self.branch_orders = np.array([])
        
    def remove_duplicate_points(self):
        cloud, mask = np.unique(self.cloud,return_index = True, axis=0,)
        self.point_data = self.point_data[mask]
        self.cloud = self.point_data[:,:3]
        self.cover_sets = self.cover_sets[mask]
        self.cluster_labels = self.cluster_labels[mask]
        self.segment_labels = self.segment_labels[mask]
        self.trunk_points =self.trunk_points[mask]

    
    def reset_cylinders(self):
        self.cylinder_starts = np.empty((0,3))
        self.cylinder_radii = np.array([])
        self.cylinder_axes = np.empty((0,3))
        self.cylinder_lengths = np.array([])
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
            print(np.unique(segment_labels))
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
        if type(self.cylinder_starts) == np.ndarray:
            self.cylinder_starts = torch.from_numpy(self.cylinder_starts.astype('float32')).to(self.device)
        if type(self.cylinder_axes) == np.ndarray:
            self.cylinder_axes = torch.from_numpy(self.cylinder_axes.astype('float32')).to(self.device)
        if type(self.cylinder_lengths) == np.ndarray:
            self.cylinder_lengths = torch.from_numpy(self.cylinder_lengths.astype('float32')).to(self.device)
        if type(self.cylinder_radii) == np.ndarray:
            self.cylinder_radii = torch.from_numpy(self.cylinder_radii.astype('float32')).to(self.device)
        if type(self.branch_labels) == np.ndarray:
            self.branch_labels = torch.from_numpy(self.branch_labels.astype('int')).to(self.device)
        if type(self.branch_orders) == np.ndarray:
            self.branch_orders = torch.from_numpy(self.branch_orders.astype('int')).to(self.device)   

    def numpy(self):
        """
        Convert all attributes of the tile to numpy arrays.
        """
        if type(self.cloud) == torch.Tensor:
            self.cloud = self.cloud.cpu().numpy().astype('float64')
        if type(self.point_data) == laspy.LasData:
            self.point_data = np.vstack([self.point_data.x,self.point_data.y,self.point_data.z,self.point_data.intensity]).T
        elif type(self.point_data) == torch.Tensor:
            self.point_data = self.point_data.cpu().numpy().astype('float64')

        if type(self.cluster_labels) == torch.Tensor:
            self.cluster_labels= self.cluster_labels.cpu().numpy().astype('float64')
        if type(self.cover_sets) == torch.Tensor:
            self.cover_sets = self.cover_sets.cpu().numpy().astype('float64')
        if type(self.segment_labels) == torch.Tensor:
            self.segment_labels = self.segment_labels.cpu().numpy().astype('float64')
        if type(self.cylinder_starts) == torch.Tensor:
            self.cylinder_starts = self.cylinder_starts.cpu().numpy().astype('float64')
        if type(self.cylinder_axes) == torch.Tensor:
            self.cylinder_axes = self.cylinder_axes.cpu().numpy().astype('float64')
        if type(self.cylinder_lengths) == torch.Tensor:
            self.cylinder_lengths = self.cylinder_lengths.cpu().numpy().astype('float64')
        if type(self.cylinder_radii) == torch.Tensor:
            self.cylinder_radii = self.cylinder_radii.cpu().numpy().astype('float64')
        if type(self.branch_labels) == torch.Tensor:
            self.branch_labels = self.branch_labels.cpu().numpy().astype('float64')
        if type(self.branch_orders) == torch.Tensor:
            self.branch_orders = self.branch_orders.cpu().numpy().astype('float64')
        if type(self.trunk_points) ==torch.Tensor:
            self.trunk_points =self.trunk_points.cpu().numpy().astype('float64')
        
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
        self.trunk_points =np.concatenate([self.trunk_points,tile.trunk_points]) if len(tile.trunk_points)>0 else self.trunk_points



    
def process_entire_pointcloud(combined_cloud: Ecomodel):
    """
    Processes the point cloud and extracts tree and leaf metrics. Stores data in 'X' files.  

    Arguments:
        pointcloud: Ecomodel object. I think it should be a (Nx3) numpy array of points representing entire island

    Returns:
        None
    """
    # Combine tiles
    # subdivide all tiles
    # Remove ground
    # normalize tiles
    

    
    # combined_cloud._raw_tiles[0].to_xyz("raw_tiles_0.xyz")
    # combined_cloud.subdivide_tiles(cube_size = 3)
    # combined_cloud.tiles[8, 4].to_xyz("raw_tiles_sub_0.xyz")
    # combined_cloud.filter_ground(combined_cloud.tiles.flatten(),.5)

    # Processing Step for All tiles, removes ground, 

    # combined_cloud.normalize_raw_tiles()
    # for tile in combined_cloud._raw_tiles:
    #     tile.to(tile.device)
    # # combined_cloud.denoise()
    # combined_cloud.subdivide_tiles(cube_size = 3)
    # combined_cloud.filter_ground(combined_cloud.tiles.flatten())
    # combined_cloud.tiles[0, 0].to_xyz("removed_gound_etc.xyz")
    # combined_cloud.recombine_tiles()
    # for tile in combined_cloud._raw_tiles:
    #     tile.to(tile.device)
    # combined_cloud.pickle("test_model.pickle")
    # combined_cloud = Ecomodel.unpickle("test_model.pickle")
    # combined_cloud.subdivide_tiles(cube_size = 15)
    

    # combined_cloud = Ecomodel.unpickle("test_model_ground_removed.pickle")
    # combined_cloud.subdivide_tiles(cube_size = 15)
    # combined_cloud.tiles[0,0].to_xyz('FromJohn.xyz')


    # ---------------

    # combined_cloud = Ecomodel.combine_las_files(folder,model)

    # combined_cloud.filter_ground(combined_cloud._raw_tiles,.5)
    # combined_cloud.normalize_raw_tiles()
    
    # for tile in combined_cloud._raw_tiles:
    #     tile.to(tile.device)
    
    
    # combined_cloud.subdivide_tiles(cube_size = 3)
    # combined_cloud.filter_ground(combined_cloud.tiles.flatten())
    # combined_cloud.recombine_tiles()

    # for tile in combined_cloud._raw_tiles:
    #     tile.to(tile.device)
    # combined_cloud.pickle("test_model_ground_removed.pickle")
    # combined_cloud = Ecomodel.unpickle("test_model_ground_removed.pickle")
    # combined_cloud.subdivide_tiles(cube_size = 15)
    # combined_cloud.remove_duplicate_points()
    # combined_cloud.pickle("test_model_pre_segmentation.pickle")
    # combined_cloud = Ecomodel.unpickle("test_model_pre_segmentation.pickle")

    # combined_cloud.segment_trees()
    # combined_cloud.pickle("test_model_trees_segmented.pickle")
    # combined_cloud.unpickle("test_model_trees_segmented.pickle")
    # combined_cloud.get_qsm_segments()
    combined_cloud = Ecomodel.unpickle("test_model_post_qsm_correct_segments.pickle")
    combined_cloud.recombine_tiles()
    cylinder,base_plot = combined_cloud.get_voxel(-2,-2,-3,2,fidelity = .6)
    base_plot.write_html("results/segment_test_plot_no_continuation.html")
    cylinders_line_plotting(cylinder, scale_factor=1,file_name="test_plot",base_fig=base_plot)
    


if __name__ == "__main__":
    # folder = r"C:\Users\johnh\Documents\LiDAR\tiled_scans"
    folder = r'/Users/johnhagood/Documents/LiDAR/tiled_scans'
#     # model = Ecomodel()
#     # combined_cloud = Ecomodel.combine_las_files(folder,model)
#     # process_entire_pointcloud(Ecomodel())
#     # Example usage
    # folder = os.environ.get("DATA_FOLDER_FILEPATH") + "tiled_scans"
    # model = Ecomodel()
    # combined_cloud = Ecomodel.combine_las_files(folder,model)
    # combined_cloud.subdivide_tiles(cube_size = 15)
    # combined_cloud.remove_duplicate_points()
    # combined_cloud.recombine_tiles()
    # combined_cloud.filter_below_ground(combined_cloud._raw_tiles,0.5)
    
    # combined_cloud.filter_ground(combined_cloud._raw_tiles)
    # combined_cloud.pickle("test_model_.pickle")
    # combined_cloud = Ecomodel.unpickle("test_model_.pickle")
    # combined_cloud.get_terrain_model(combined_cloud._raw_tiles,1)
    # combined_cloud.normalize_raw_tiles()
    
    
    # for tile in combined_cloud._raw_tiles:
    #     tile.to(tile.device)
    
    

    # print("filtered")

    # combined_cloud.pickle("test_model_ground_removed.pickle")
    combined_cloud = Ecomodel.unpickle("test_model_ground_removed.pickle")
    combined_cloud.subdivide_tiles(cube_size = 15)
    # combined_cloud.denoise(grid_size =3,min_points=10,resolution=.1)
    # combined_cloud.remove_duplicate_points()

    
    combined_cloud.segment_trees()
    combined_cloud.pickle("test_model_trees_segmented.pickle")
    combined_cloud = Ecomodel.unpickle("test_model_trees_segmented.pickle")
    combined_cloud.get_qsm_segments(40000)
    combined_cloud.pickle("test_model_post_qsm_correct_segments.pickle")
    combined_cloud = Ecomodel.unpickle("test_model_post_qsm_correct_segments.pickle")
    combined_cloud.recombine_tiles()
    # Palm
    # cylinder,base_plot = combined_cloud.get_voxel(-15,-3,-3,5,fidelity = .3)
    # # Small Voxel
    # cylinder,base_plot = combined_cloud.get_voxel(-11,1,-1,3,fidelity = 1)
    # # Large Voxel
    cylinder,base_plot = combined_cloud.get_voxel(-2,-2,-3,2,fidelity = .6)
    base_plot.write_html("results/segment_test_plot_no_continuation.html")
    cylinders_line_plotting(cylinder, scale_factor=1,file_name="test_plot",base_fig=base_plot)
    # cylinders_plotting(cylinder,base_fig=base_plot)
    # combined_cloud.calc_volumes()
    # subdivided_cloud = combined_cloud.subdivide_tiles(cube_size = 10)
    # print(subdivided_cloud)