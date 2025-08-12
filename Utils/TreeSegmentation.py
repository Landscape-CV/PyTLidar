
import numpy as np
# from main_steps.segments import segments
import torch
import open3d as o3d
import numba 
from igraph import Graph


def segment_point_cloud(tile, max_dist = .16, base_height = .3, layer_size =.3, min_base_dist =.2, connect_ambiguous_points = True, fix_overlapping_segments = True,combine_nearby_bases =True,base_dist_multiplier=2,initial_size_limit =1000,min_height = 0):
    """Shortest-Path Segmentation of Point Cloud Utilizing a Voronoi Cover Set and "Scan-Line" Graph Generation

    Args:
        tile (np.Array): point cloud tile
        max_dist (float, optional): distance between neighboring points. Defaults to .16.
        base_height (float, optional): height of lowest layer determining base locations. Defaults to .3.
        layer_size (float, optional): size of non-base layers. Defaults to .3.
        min_base_dist (float, optional): minimum distance between bases (Only used with combine_nearby_bases). Defaults to .2.
        connect_ambiguous_points (bool, optional): If True, connect points that aren't clearly in a cluster to nearest segment. Defaults to True.
        fix_overlapping_segments (bool, optional): If True, separates segments that are overlapping. Defaults to True.
        combine_nearby_bases (bool, optional): If True, combines bases that are withing min_base_dist. Defaults to True.
        base_dist_multiplier (int, optional): optional multiplier for max_dist to be used on base layer. Defaults to 2.
        initial_size_limit (int, optional): Size limit of clusters grown on each layer. Defaults to 1000.
        min_height (int, optional): minimum height of found segments. Defaults to 0.
    """
    
    
    
    tile.to(tile.device)
    tile.cover_sets = torch.Tensor(tile.cover_sets).to(tile.device).to(int)
    I = torch.argsort(tile.cover_sets)
    tile.cover_sets = tile.cover_sets[I]
    tile.point_data = tile.point_data[I]
    tile.cloud = tile.cloud[I]


    # unique_masks, inverse_indices = torch.unique(tile.cover_sets, return_inverse=True)

    num_masks = torch.max(tile.cover_sets)+1#torch.bincount(tile.cover_sets)
    dim = tile.point_data.size(1)


    # #representative points for each cover set
    # center_points = torch.zeros((num_masks, dim), device=tile.point_data.device)
    # center_points.scatter_reduce_(
    # 0, 
    # tile.cover_sets.unsqueeze(-1).expand(-1, dim), 
    # tile.point_data, 
    # reduce='mean',
    # include_self=False
    # )
    min_points = torch.zeros((num_masks, dim), device=tile.point_data.device)
   
    min_points.scatter_reduce_(
    0, 
    tile.cover_sets.unsqueeze(-1).expand(-1, dim), 
    tile.point_data, 
    reduce='min',
    include_self=False
    )
    center_points = min_points.clone()
    
    
    
    cloud = center_points[:,:3].cpu().numpy()
    min_Z = np.min(cloud[:,2])
    I = (cloud[:,2]-min_Z)<base_height
    cloud = cloud[I]
    included_cover_sets = np.where(I)

    
    # cloud = tile.get_cloud_as_array()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    # pcd = pcd.voxel_down_sample(voxel_size=.03)
   
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    segments = np.zeros(len(cloud))-1

    not_explored = np.ones(len(cloud),dtype = bool)
    segment_num = 0
    
    neighbors = {}
    base = 0
    

        
  



    full_segments = np.zeros(len(center_points))-1


    I = torch.argsort(center_points[:,2])#sort points by z-index
    center_points = center_points[I]
    I = I.cpu().numpy()
    sorted_indices = I.copy() #Save original order of points

    cloud = center_points[:,:3].cpu().numpy()
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(cloud)
    full_pcd_tree = o3d.geometry.KDTreeFlann(full_pcd)
    min_Z = np.min(cloud[:,2])
    I = (cloud[:,2]-min_Z)<base_height
    
    prev_base_height = min_height
    network = Graph()
    network.add_vertices(len(center_points))
    print("Build Networks")
    size_limit = initial_size_limit
    multiplier = base_dist_multiplier

    #"Scan-Line" Method: Add to graph one layer at a time
    #Graph 1: Used to cluster points on each layer
    #Graph 2: (Full PCD) Actual full graph to connect clusters and perform shortest-path calculations
    while base_height+min_Z-1<torch.max(tile.cloud[:,2]):
        
        I = ((cloud[:,2]-min_Z)<base_height) & (prev_base_height<(cloud[:,2]-min_Z))
        
        cloud = cloud[I]
        included_cover_sets = np.where(I)
        segments = np.zeros(len(cloud))-1

        not_explored = np.ones(len(cloud),dtype = bool)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
    
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        segments,segment_num = add_layer(pcd_tree,pcd,segments,not_explored,segment_num,max_dist*multiplier,network,full_pcd_tree,full_pcd,included_cover_sets[0],size_limit)
        full_segments[included_cover_sets] = segments
        if prev_base_height ==min_height:#First layer is tree boles

            tree_bases = np.unique(segments)
        cloud = center_points[:,:3].cpu().numpy()
        prev_base_height = base_height
        base_height+=layer_size
        size_limit = 10
        multiplier =1
        

    segments = full_segments.copy()
    full_not_explored = np.ones(len(center_points),dtype = bool)
    cloud = center_points[:,:3].cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    filtered_tree_bases = []

    for base in tree_bases:#Only utilize base if base cluster is large enough
        base_set = center_points[segments ==base]
        if  len(base_set[:,2])>5:
            filtered_tree_bases.append(base)

    if combine_nearby_bases:#Combine bases that are within a certain distance, this helps for trees that have non-trunk segments that dip into base layer
        filtered_tree_bases=combine_close_bases(segments,center_points,filtered_tree_bases,min_base_dist)
        filtered_tree_bases = filtered_tree_bases.cpu().numpy()
        mod_filtered_tree_bases = []
        for base in tree_bases:
            base_set = center_points[segments ==base]
            # if  len(base_set[:,2]<min_Z+.3)>1:
            if  len(base_set[:,2])>5:
                mod_filtered_tree_bases.append(base)
        filtered_tree_bases=mod_filtered_tree_bases


    
    print("Connect Segments")
    segments,not_explored = connect_segments(pcd_tree,pcd,segments,full_not_explored,filtered_tree_bases,max_dist*2,network,False,True)#Shortest Path to lowest point of tree
    print("Connect More Segments")
    segments,not_explored = connect_segments(pcd_tree,pcd,segments,not_explored,filtered_tree_bases,max_dist,network,False,False)#Shortest path to average point of tree
    if connect_ambiguous_points:
        print("Connect Final Segments")
        segments,not_explored = connect_segments(pcd_tree,pcd,segments,not_explored,filtered_tree_bases,max_dist*1.5,network,True,True)#Shortest path to min point of tree -- allow connections to clusters that are not adjacent to assigned
    if fix_overlapping_segments:
        print("Fix Overlap")
        segments = fix_overlap(segments,center_points,network)
    print(len(segments))
    unassigned_sets = np.where(~np.isin(segments,filtered_tree_bases))
    segments[unassigned_sets]=-1

    print(np.unique(segments))
    I = torch.argsort(tile.cover_sets)
    tile.cover_sets = tile.cover_sets[I]
    tile.point_data = tile.point_data[I]
    tile.cloud = tile.cloud[I]
    num_indices = torch.bincount(tile.cover_sets.to(int))
    segments = segments[np.argsort(sorted_indices)]
    segments=torch.tensor(segments,device=tile.device,dtype=int)
    if len(num_indices) < len(segments):
        num_indices = torch.cat([num_indices, torch.zeros(len(segments)-len(num_indices), device=tile.device,dtype=int)-1])
    if len(segments)<len(num_indices):
        segments = torch.cat([segments, torch.zeros(len(num_indices)-len(segments), device=tile.device,dtype=int)-1])
    
    tile.segment_labels= torch.repeat_interleave(segments, num_indices)
    tile.cover_sets = tile.cover_sets.cpu().numpy()
    tile.numpy()
    print(np.unique(tile.segment_labels))
    # tile.cluster_labels = segments
    # tile.cloud = cloud
    

# @numba.jit(forceobj=True)
def add_layer(pcd_tree,pcd,segments,not_explored,segment_num,max_dist,network:Graph,full_pcd_tree,full_pcd,included_sets,size_limit = 100,graph_multiplier=2):
    """Assigns initial clusters to points within new layer of point cloud
    Also builds edges for shortest path calculations in the full point cloud graph

    Args:
        pcd_tree (o3d.KDTreeFlann): KDTree to build this layer on
        pcd (np.Array): Point cloud of layer
        segments (np.Array): Empty array of len(pcd) to assign cluster labels to
        not_explored (np.Array): Boolean array to track which point have been explored
        segment_num (int): current cluster label (incremented as new clusters are formed)
        max_dist (float): cluster max_dist variable used to determine neighbors
        network (Graph): Graph to add edges to for shortest path calculations
        full_pcd_tree (o3d.KDTreeFlann): KDTree built upon across layers to determine neighbors 
        full_pcd (np.Array): Full point cloud
        included_sets (np.array): indices of full point cloud found in layer
        size_limit (int, optional): maximum number of points in cluser. Defaults to 100.
        graph_multiplier (int, optional): multiplier to determine neighborhood for full graph. Defaults to 2.

    Returns:
        Segments: cluster assignments
        Segment_num: current cluster label (incremented as new clusters are formed)
        
    """
    K=20
    edges = []
    weights = []

    while any(not_explored):
        
        # print(f"segment: {segment_num}, remaining: {sum(not_explored)}")
        base = np.min(np.where(not_explored))
        not_explored[base]=False
        
        k,points,dist = pcd_tree.search_hybrid_vector_3d(pcd.points[base],max_dist,K)
        k,graph_points,dist =full_pcd_tree.search_hybrid_vector_3d(full_pcd.points[included_sets[base]],max_dist*graph_multiplier,K)
        edges.extend(make_edges(included_sets[base],list(graph_points)))
        
        weights.extend(list(dist))
        
        points.pop(0)
        segments[base] = segment_num
        p_arr = np.array(points)
        I = not_explored[p_arr]
        points = p_arr[I]
        points = o3d.utility.IntVector(points)
        point_count = len(points)
        while len(points)>0 and point_count<size_limit:
            
            next_point = points.pop()
            if not_explored[next_point]:
                segments[next_point]=segment_num
                k,new_points,dist = pcd_tree.search_hybrid_vector_3d(pcd.points[next_point],max_dist,K)
                k,graph_points,dist =full_pcd_tree.search_hybrid_vector_3d(full_pcd.points[included_sets[next_point]],max_dist*graph_multiplier,K)
                edges.extend(make_edges(included_sets[next_point],list(graph_points)))
                weights.extend(list(dist))
                not_explored[next_point]=False
                
                

                new_points = np.setdiff1d(new_points,points)
                I = not_explored[new_points]
                new_points = new_points[I]
                points.extend(new_points)
                point_count = point_count + len(new_points)

                
            
        segment_num= segment_num+1
    network.add_edges(edges,{"weight":weights})
    return segments,segment_num

@numba.jit(nopython=True)
def make_edges(source,target):
    """Creates edge list to be added to graph

    Args:
        source (int): index of source node
        target (list): indices of target nodes

    Returns:
        list: list of edges to be added to graph
    """
    edges= []
    for node in target:
        edges.append((source,node))
    return edges
# @numba.jit(forceobj=True)
def connect_segments(pcd_tree,pcd,segments,not_explored,tree_bases,max_dist,network,search_non_connecting,min_point=False):
    """Connects clusters to tree bases utilizing shortest path to base point

    Args:
        pcd_tree (KDTreeFlann): point Cloud KDTree
        pcd (np.Array): tile point cloud
        segments (np.Array): cluster assignments of each point
        not_explored (np.array): boolean array indicating which points have not been explored
        tree_bases (list): list cluster labels corresponding to tree bases
        max_dist (float): distance to determine adjacency between points
        network (iGraph.Graph): Graph to traverse for shortest path calculations
        search_non_connecting (bool): Indicates if non-adjacent clusters should be connected to tree bases
        min_point (bool, optional): Determines if lowest point will be used for tree base representation. Defaults to False -- if False, average point of tree base will be used.

    Returns:
        (tuple): returns assigned segments and not_expanded array indicating which points were not expanded to a cluster
    """
    not_expanded = np.zeros(len(not_explored),dtype=bool)

    point_data = np.array(pcd.points)
    tree_base_points = []
    if min_point:
       for base in tree_bases:
            tree_base_points.append(np.where(segments == base)[0][point_data[np.where(segments == base)[0]][:,2].argmin()])
    else:
        for base in tree_bases:
            # lexord = (point_data[np.where(segments == base)][:,0],point_data[np.where(segments == base)][:,1],point_data[np.where(segments == base)][:,2])
            base_set = point_data[np.where(segments == base)]
            base_estimate = np.lexsort((base_set[:,0],base_set[:,1],base_set[:,2]))[len(base_set)//2]##POTENTIAL BUG/Improvement: axis=0 on lexsort may be preferable
            tree_base_points.append(np.where(segments == base)[0][base_estimate])
        # tree_base_points.append(np.where(segments == base)[0][point_data[np.where(segments == base)][:,2].argmin()])

    tree_base_points=np.array(tree_base_points,dtype=int)
    tree_bases=np.array(tree_bases,dtype = int)
    

    while any(not_explored):
        
        # print(f"segment: {segment_num}, remaining: {sum(not_explored)}")
        base = np.min(np.where(not_explored))
        not_explored[base]=False
        if segments[base] in tree_bases:
            continue
        k,points,_ = pcd_tree.search_radius_vector_3d(pcd.points[base],max_dist)
        
        
        segs = np.unique(segments[points])
        
        if len(segs)==1:
            not_expanded[base]=True
            continue
        

        else:
            
            base_seg = segments[base]
            tree_base_seg =np.intersect1d(segs,tree_bases).astype(int)
            if len(tree_base_seg)==1:
                base_seg = tree_base_seg[0]
            elif len(tree_base_seg)==0:
                if not search_non_connecting:
                    not_expanded[base]=True
                    continue
                else:
                    euc_dist = np.sqrt(np.array([(pcd.points[idx]- pcd.points[base])**2 for idx in tree_base_points]).sum(axis=1))
                    top = np.argsort(euc_dist)[0]
                    path_dist=np.array(network.distances(base,tree_base_points[top],weights='weight'))[0]
                    if np.min(path_dist)==np.inf:
                        not_expanded[base]=True
                        continue
                    # base_seg=tree_bases[np.argmin(path_dist)]
                    base_seg=tree_bases[top]
            else:

                base_idx = np.where(np.isin(tree_bases, tree_base_seg))[0]

                # euc_dist = np.sqrt(np.array([(pcd.points[idx]- pcd.points[base])**2 for idx in base_idx]).sum(axis=1))
                # base_seg=tree_base_seg[np.argmin(euc_dist)]
                path_dist=np.array(network.distances(base,tree_base_points[base_idx],weights='weight'))[0]
                base_seg=tree_base_seg[np.argmin(path_dist)]
            # for seg in segs:
            #     if seg not in tree_bases:
            if segments[base]==-1:
                continue
            segments[segments==segments[base] ] = base_seg
        

                
            
        
        
    return segments,not_expanded

def combine_close_bases(segments,center_points,bases, bound = .1):
    """combines bases that are within a certain distance of each other into one base

    Args:
        segments (np.array): cluster labels
        center_points (Tensor): representative points
        bases (list): tree base cluster labels
        bound (float, optional): minimal distance for bases to be. Defaults to .1.

    Returns:
        
    """
    bases = torch.tensor(bases,dtype=int,device=center_points.device)
    again =True
    while again:
        
        new_bases = bases.clone()
        changed=torch.zeros(size=(len(bases),),dtype=bool,device=center_points.device)
        bounds = get_bounds(bases,segments,center_points)
        

        for i in range(len(bases)):
            if not changed[i]:
                base = center_points[segments ==bases[i].cpu().numpy()]
                min_y = torch.min(base[:,1])
                min_x = torch.min(base[:,0])
                min_z = torch.min(base[:,2])
                max_y = torch.max(base[:,1])
                max_x = torch.max(base[:,0])
                max_z = torch.max(base[:,2])
                seg_bound = torch.tensor([min_x,min_y,min_z,max_x,max_y,max_z],device=center_points.device)
                overlap = get_overlap(bounds,seg_bound)
                segments[np.isin(segments,bases[overlap].cpu().numpy())]=bases[i].cpu().numpy()
                new_bases[overlap] = bases[i]
                changed+=overlap
        if not torch.all(new_bases == bases):
            again =True
        else:
            again=False
        bases =torch.unique(new_bases).clone()
    return torch.unique(new_bases)


def get_overlap(bounds,seg_bound):
    """Determines if a segment overlaps with a given other segment.

    Args:
        bounds (Tensor): min_x,min_y,min_z,max_x,max_y,max_z for each segment
        seg_bound (Tensor): min_x,min_y,min_z,max_x,max_y,max_z for the segment to check overlap with
    Returns:
        Tensor: bools indicating if the segment overlaps with the given segment

    """
    #Corners (0,0,0) and (1,1,1)
    overlap = torch.all(bounds[:,:3]<seg_bound[3:],axis=1)&torch.all(bounds[:,:3]>seg_bound[:3],axis=1) | torch.all(bounds[:,3:]<seg_bound[3:],axis=1)&torch.all(bounds[:,3:]>seg_bound[:3],axis=1)
    
    #Corner (0,0,1)
    overlap =overlap | torch.all(torch.column_stack([bounds[:,0],bounds[:,1],bounds[:,5]])<seg_bound[3:])&torch.all(torch.column_stack([bounds[:,0],bounds[:,1],bounds[:,5]])>seg_bound[:3],axis=1)
    #Corner (0,1,1)
    overlap =overlap | torch.all(torch.column_stack([bounds[:,0],bounds[:,4],bounds[:,5]])<seg_bound[3:])&torch.all(torch.column_stack([bounds[:,0],bounds[:,4],bounds[:,5]])>seg_bound[:3],axis=1)
    #Corner (0,1,0)
    overlap =overlap | torch.all(torch.column_stack([bounds[:,0],bounds[:,4],bounds[:,2]])<seg_bound[3:])&torch.all(torch.column_stack([bounds[:,0],bounds[:,4],bounds[:,2]])>seg_bound[:3],axis=1)
    #Corner (1,1,0)
    overlap =overlap | torch.all(torch.column_stack([bounds[:,3],bounds[:,4],bounds[:,2]])<seg_bound[3:])&torch.all(torch.column_stack([bounds[:,3],bounds[:,4],bounds[:,2]])>seg_bound[:3],axis=1)
    #Corner (1,0,0)
    overlap =overlap | torch.all(torch.column_stack([bounds[:,3],bounds[:,1],bounds[:,2]])<seg_bound[3:])&torch.all(torch.column_stack([bounds[:,3],bounds[:,1],bounds[:,2]])>seg_bound[:3],axis=1)
    #Corner (1,0,1)
    overlap =overlap | torch.all(torch.column_stack([bounds[:,3],bounds[:,1],bounds[:,5]])<seg_bound[3:])&torch.all(torch.column_stack([bounds[:,3],bounds[:,1],bounds[:,5]])>seg_bound[:3],axis=1)
    
    return overlap

def get_bounds(bases,segments,center_points):
    """Calculates bounds for each segment

    Args:
        bases (list): Base cluster labels
        segments (np.Array): segment labels for each point
        center_points (Tensor): Point locations

    Returns:
        Tensor: bounds for each segment in the form of min_x,min_y,min_z,max_x,max_y,max_z
    """
    _bases = torch.tensor(bases,dtype=int,device=center_points.device)
    bounds = torch.zeros((len(bases),6),device=center_points.device)
    for i in range(len(bases)):
        base = center_points[segments ==_bases[i].cpu().numpy()]
        min_y = torch.min(base[:,1])
        min_x = torch.min(base[:,0])
        min_z = torch.min(base[:,2])
        max_y = torch.max(base[:,1])
        max_x = torch.max(base[:,0])
        max_z = torch.max(base[:,2])
        bounds[i] = torch.tensor([min_x,min_y,min_z,max_x,max_y,max_z],device=center_points.device)
    return bounds


def fix_overlap(segments,center_points,network):
    """Adjusts overlapping segments by reassigning points to the nearest base segment

    Args:
        segments (np.Array): array of segment labels
        center_points (Tensor): point locations
        network (Graph): Graph to use for shortest path calculations

    Returns:
        np.Array: adjusted segments
    """
    bases = np.unique(segments)
    bounds = get_bounds(bases,segments,center_points)
    base_mins = get_minimums(segments,center_points)

    for i in range(len(bases)):
        base = center_points[segments ==bases[i]]
        min_y = torch.min(base[:,1])
        min_x = torch.min(base[:,0])
        min_z = torch.min(base[:,2])
        max_y = torch.max(base[:,1])
        max_x = torch.max(base[:,0])
        max_z = torch.max(base[:,2])
        seg_bound = torch.tensor([min_x,min_y,min_z,max_x,max_y,max_z],device=center_points.device)
        overlap = get_overlap(bounds,seg_bound)
        if sum(overlap)>1:
            for j in range(len(bases)):
                test_bases = [base_mins[i],base_mins[j]]
                point_data = center_points[segments ==bases[i]]

                I = torch.all(point_data[:,:3]<bounds[i,:3],axis=1) & torch.all(point_data[:,:3]>bounds[i,3:],axis=1) 
                for k, point in enumerate(point_data[I]):
                    path_dist=np.array(network.distances(point,test_bases,weights='weight'))[0]
                    base_seg=test_bases[np.argmin(path_dist)]
                    segments[segments==bases[i]][k]=base_seg

                    

                   

    return segments

def get_minimums(segments,center_points):
    """Find lowest point of each segment

    Args:
        segments (np.Array): segment labels for each point
        center_points (Tensor): point locations

    Returns:
        np.Array: Index of minimum point for each segment
    """
    minimums = np.zeros((len(np.unique(segments)),3))
    for i,base in enumerate(np.unique(segments)):
        point_data = center_points[segments ==base]
        minimums[i]=np.where(segments == base)[0][point_data[:,2].argmin()]
    return minimums
        
            









                
                

            
            

                        
                            