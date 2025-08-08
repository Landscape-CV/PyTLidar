import open3d as o3d
import numpy as np
import math

class RegionGrowing:
    def __init__(self):
        self.pcd = None
        self.normals = None
        self.curvatures = None
        self.k_neighbors = 30
        self.TAngle = 15.0  # degrees
        self.curvatureThreshold = 0.1
        self.residualThreshold = 0.05
        self.smoothMode = True
        self.useCurvatureTest = True
        self.useResidualTest = True
        self.minClusterSize = 100
        self.maxClusterSize = 100000
        self.Clusters = []
        self.pcd_tree = None
        self.NPt = 0

    def SetDataThresholds(self, pcd, angle_deg=15.0, curv_thresh=0.1, resid_thresh=0.05, k=30):
        self.pcd = pcd
        self.NPt = len(pcd.points)
        self.k_neighbors = k
        self.TAngle = angle_deg
        self.curvatureThreshold = curv_thresh
        self.residualThreshold = resid_thresh
        self.pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    def compute_curvatures(self):
        curvatures = np.zeros(self.NPt)
        for i in range(self.NPt):
            [_, idx, _] = self.pcd_tree.search_knn_vector_3d(self.pcd.points[i], self.k_neighbors)
            neighbors = np.asarray(self.pcd.points)[idx, :]
            cov = np.cov(neighbors.T)
            eigvals = np.sort(np.abs(np.linalg.eigvalsh(cov)))
            curvature = eigvals[0] / np.sum(eigvals)
            curvatures[i] = curvature
        return curvatures

    def angle_between_normals(self, n1, n2):
        dot = np.clip(np.abs(np.dot(n1, n2)), -1.0, 1.0)
        return math.acos(dot) * 180.0 / math.pi

    def point_to_plane_residual(self, pt, pt_seed, normal_seed):
        diff = pt_seed - pt
        return np.abs(np.dot(normal_seed, diff))

    def validate_point(self, initial_seed, current_seed, neighbor_idx):
        is_seed = True

        normal_current = self.normals[current_seed]
        normal_seed = self.normals[initial_seed]
        normal_neighbor = self.normals[neighbor_idx]

        angle = self.angle_between_normals(normal_current if self.smoothMode else normal_seed, normal_neighbor)
        if angle > self.TAngle:
            return False, False

        if self.useCurvatureTest and self.curvatures[neighbor_idx] > self.curvatureThreshold:
            is_seed = False

        if self.useResidualTest:
            pt_seed = np.asarray(self.pcd.points[current_seed])
            pt_neigh = np.asarray(self.pcd.points[neighbor_idx])
            residual = self.point_to_plane_residual(pt_neigh, pt_seed, normal_current)
            if residual > self.residualThreshold:
                is_seed = False

        return True, is_seed

    def RGKnn(self):
        if len(self.pcd.normals) < self.NPt:
            self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.k_neighbors))
        self.normals = np.asarray(self.pcd.normals)
        self.curvatures = self.compute_curvatures()

        processed = np.full(self.NPt, False)
        available = set(range(self.NPt))
        residual_pairs = [(self.curvatures[i], i) for i in range(self.NPt)]
        residual_pairs.sort()

        for _, seed in residual_pairs:
            if processed[seed]:
                continue

            region = [seed]
            queue = [seed]
            processed[seed] = True

            i = 0
            while i < len(queue):
                curr = queue[i]
                [_, idx, _] = self.pcd_tree.search_knn_vector_3d(self.pcd.points[curr], self.k_neighbors)
                for j in idx[1:]:
                    if processed[j]:
                        continue
                    valid, is_seed = self.validate_point(seed, curr, j)
                    if not valid:
                        continue
                    region.append(j)
                    processed[j] = True
                    if is_seed:
                        queue.append(j)
                i += 1

            if self.minClusterSize <= len(region) <= self.maxClusterSize:
                self.Clusters.append(region)

    def ReLabeles(self):
        labels = np.zeros(self.NPt)
        for i, cluster in enumerate(self.Clusters):
            for idx in cluster:
                labels[idx] = i + 1
        return labels
