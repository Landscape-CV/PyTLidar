import math
import numpy as np
import open3d as o3d
from numba import njit,prange


class RegionGrowing:
    """
    Optimized Region Growing KNN segmentation.

    Key speedups over the original:
    1. All K-nearest neighbours are precomputed in ONE batched sklearn call
       instead of N individual Open3D KDTree queries in the loops.
    2. Curvature computation is vectorized with numpy (no Python loop).
    3. Normal angle and residual checks use precomputed arrays and numpy ops.
    4. The region-growing loop itself is still sequential (inherently so),
       but it never touches a KDTree again after the precomputation step.
    """

    def __init__(self):
        self.pcd = None
        self.normals = None
        self.curvatures = None
        self.k_neighbors = 30
        self.TAngle = 15.0
        self.curvatureThreshold = 0.1
        self.residualThreshold = 0.05
        self.smoothMode = True
        self.useCurvatureTest = True
        self.useResidualTest = True
        self.minClusterSize = 100
        self.maxClusterSize = 100000
        self.Clusters = []
        self.NPt = 0

        # precomputed arrays (filled in _precompute)
        self._all_neighbor_idx = None  # (N, k) int array
        self._pts = None  # (N, 3) float array

    def SetDataThresholds(self, pcd, angle_deg=15.0, curv_thresh=0.1,
                          resid_thresh=0.05, k=30):
        self.pcd = pcd
        self.NPt = len(pcd.points)
        self.k_neighbors = k
        self.TAngle = angle_deg
        self.curvatureThreshold = curv_thresh
        self.residualThreshold = resid_thresh

    # ------------------------------------------------------------------
    # Precompute everything that would otherwise be done inside hot loops
    # ------------------------------------------------------------------
    def _precompute(self):
        self._pts = np.asarray(self.pcd.points, dtype=np.float64)

        # --- normals (Open3D batch, already fast) ----------------------
        if len(self.pcd.normals) < self.NPt:
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(
                    knn=self.k_neighbors))
        self.normals = np.asarray(self.pcd.normals, dtype=np.float64)

        # --- precompute all KNN using Open3D KDTree (same as original) --
        # Queried once upfront so the region-growing loop never touches
        # the KDTree again. Using O3D's tree (not sklearn) preserves the
        # exact neighbor ordering the original algorithm uses.
        print(f"[RGKnn] Precomputing {self.k_neighbors}-NN for {self.NPt} points …")
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        self._all_neighbor_idx = np.empty((self.NPt, self.k_neighbors), dtype=np.int64)
        for i in range(self.NPt):
            [_, idx, _] = pcd_tree.search_knn_vector_3d(self.pcd.points[i], self.k_neighbors)
            self._all_neighbor_idx[i] = idx
        # shape: (N, k)  — column 0 is the point itself

        # --- vectorized curvature --------------------------------------
        print("[RGKnn] Computing curvatures …")
        self.curvatures = self._compute_curvatures_numba(self._pts, self._all_neighbor_idx, self.k_neighbors)
        #self.curvatures = self._compute_curvatures_vectorized()

    @njit(parallel=True)
    def _compute_curvatures_numba(pts, idx, k):
        N = pts.shape[0]
        curvatures = np.empty(N, dtype=np.float64)

        for i in prange(N):  # parallel loop over all points
            # gather neighbours
            neighbours = np.empty((k, 3), dtype=np.float64)
            for j in range(k):
                neighbours[j] = pts[idx[i, j]]

            # mean centre
            mean = np.zeros(3, dtype=np.float64)
            for j in range(k):
                mean += neighbours[j]
            mean /= k
            for j in range(k):
                neighbours[j] -= mean

            # 3x3 covariance matrix
            cov = np.zeros((3, 3), dtype=np.float64)
            for j in range(k):
                for r in range(3):
                    for c in range(3):
                        cov[r, c] += neighbours[j, r] * neighbours[j, c]
            cov /= (k - 1)

            # eigenvalues (numba supports np.linalg.eigvalsh on fixed small matrices)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.abs(eigvals)
            curvatures[i] = eigvals[0] / (eigvals[0] + eigvals[1] + eigvals[2] + 1e-12)

        return curvatures

    def _compute_curvatures_vectorized(self, chunk_size=10000):
        """
        Compute curvatures in chunks to avoid allocating N×k×3 array at once.
        chunk_size controls the memory/speed tradeoff — lower = less RAM.
        """
        pts = self._pts  # (N, 3)
        idx = self._all_neighbor_idx  # (N, k)
        N = self.NPt
        curvatures = np.empty(N, dtype=np.float64)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_idx = idx[start:end]  # (chunk, k)

            neigh_pts = pts[chunk_idx]  # (chunk, k, 3)
            means = neigh_pts.mean(axis=1, keepdims=True)
            centred = neigh_pts - means  # (chunk, k, 3)

            cov = np.einsum('nki,nkj->nij', centred, centred) / (self.k_neighbors - 1)
            eigvals = np.linalg.eigvalsh(cov)  # (chunk, 3)
            eigvals = np.abs(eigvals)
            curvatures[start:end] = eigvals[:, 0] / (eigvals.sum(axis=1) + 1e-12)

        return curvatures

    # ------------------------------------------------------------------
    # Region growing — same algorithm, no KDTree inside the loop
    # ------------------------------------------------------------------
    def RGKnn(self):
        self._precompute()

        pts = self._pts
        normals = self.normals
        curv = self.curvatures
        idx_mat = self._all_neighbor_idx  # (N, k)

        cos_thresh = math.cos(math.radians(self.TAngle))

        processed = np.zeros(self.NPt, dtype=bool)

        # sort seeds by curvature ascending (same as original)
        seed_order = np.argsort(curv)

        for seed in seed_order:
            if processed[seed]:
                continue

            region = [seed]
            queue = [seed]
            processed[seed] = True

            qi = 0
            while qi < len(queue):
                curr = queue[qi]
                qi += 1

                n_curr = normals[curr]
                n_seed = normals[seed]
                pt_curr = pts[curr]

                # neighbours of curr (precomputed, skip index 0 = self)
                neighbours = idx_mat[curr, 1:]

                for nb in neighbours:
                    if processed[nb]:
                        continue

                    n_nb = normals[nb]

                    # --- angle test ------------------------------------
                    ref_normal = n_curr if self.smoothMode else n_seed
                    dot = abs(float(np.dot(ref_normal, n_nb)))
                    dot = min(dot, 1.0)
                    if dot < cos_thresh:  # angle > TAngle
                        continue

                    is_seed = True

                    # --- curvature test --------------------------------
                    if self.useCurvatureTest and curv[nb] > self.curvatureThreshold:
                        is_seed = False

                    # --- residual test ---------------------------------
                    if self.useResidualTest:
                        diff = pt_curr - pts[nb]
                        residual = abs(float(np.dot(n_curr, diff)))
                        if residual > self.residualThreshold:
                            is_seed = False

                    region.append(nb)
                    processed[nb] = True
                    if is_seed:
                        queue.append(nb)

            if self.minClusterSize <= len(region) <= self.maxClusterSize:
                self.Clusters.append(region)

        print(f"[RGKnn] Done. {len(self.Clusters)} clusters found.")

    def ReLabeles(self):
        labels = np.zeros(self.NPt, dtype=np.int32)
        for i, cluster in enumerate(self.Clusters):
            labels[cluster] = i + 1
        return labels