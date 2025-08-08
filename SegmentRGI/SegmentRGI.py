import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import RegionGrowing as RG
import os


def save_cloud(input_path, save_dir, base_name, suffix, cloud, indices, ext):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{base_name}_{suffix}{ext}")
    if len(cloud.points) > 0:
        o3d.io.write_point_cloud(out_path, cloud)
    else:
        print(f"[WARNING] No {suffix} points found — not saved")


def load_point_cloud(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".pcd", ".ply"]:
        pcd_t = o3d.t.io.read_point_cloud(input_path)
        intensity_key = "scalar_Intensity" if "scalar_Intensity" in pcd_t.point else "Intensity"
        if intensity_key not in pcd_t.point:
            raise ValueError(f"No intensity field found in {input_path}")
        return pcd_t, intensity_key, ext
    else:
        raise ValueError(f"Unsupported format {ext}. Only .pcd and .ply are supported.")

def classify_wood_leaf(input_path, save_dir="", show_plots=False):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    pcd_t, intensity_key, ext = load_point_cloud(input_path)

    # print("[DEBUG] Available fields:", list(pcd_t.point))
    full_intensity = pcd_t.point[intensity_key].numpy().flatten()

    # --- Adaptive Noise Filter ---
    noise_percentile = 5
    threshold_noise = np.percentile(full_intensity, noise_percentile)
    print(f"[FILTER] Removing intensity < {threshold_noise:.2f} (bottom {noise_percentile}%)")

    valid_mask = full_intensity >= threshold_noise
    noise_mask = ~valid_mask
    pcd_filtered = pcd_t.select_by_mask(valid_mask)
    pcd = pcd_filtered.to_legacy()
    intensity = full_intensity[valid_mask]
    original_indices = np.nonzero(valid_mask)[0]

    # --- Region Growing Clustering ---
    start_rg = time.time()
    RGKNN = RG.RegionGrowing()
    RGKNN.SetDataThresholds(pcd, angle_deg=15.0, curv_thresh=0.07, resid_thresh=0.05, k=30)
    RGKNN.minClusterSize = 3
    RGKNN.maxClusterSize = 100000
    RGKNN.smoothMode = True
    RGKNN.useResidualTest = True
    RGKNN.useCurvatureTest = True
    RGKNN.RGKnn()
    end_rg = time.time()
    print(f"[TIMING] Region Growing took {end_rg - start_rg:.2f} seconds")

    labels_filtered = RGKNN.ReLabeles()
    labels_full = -1 * np.ones(len(full_intensity), dtype=np.int32)
    labels_full[original_indices] = labels_filtered
    labels = labels_full

    # --- Trunk Analysis ---
    start_class = time.time()
    largest_cluster_id = np.argmax([len(c) for c in RGKNN.Clusters])
    largest_cluster_indices = RGKNN.Clusters[largest_cluster_id]
    trunk_intensities = np.array([intensity[i] for i in largest_cluster_indices])
    trunk_median = np.median(trunk_intensities)
    # buffer = 5000
    # trunk_min = trunk_median - buffer
    # trunk_max = trunk_median + buffer
    q1 = np.percentile(trunk_intensities, 25)
    trunk_min = q1
    trunk_max = np.inf
    print(f"[TRUNK INTENSITY] min={trunk_min:.2f} median={trunk_median:.2f} max={trunk_max:.2f}")

    # --- Classification ---
    wood_clusters, leaf_clusters = set(), set()
    for cid, indices in enumerate(RGKNN.Clusters):
        ci_intensity = np.array([intensity[i] for i in indices])
        ci_median = np.median(ci_intensity)
        if cid == largest_cluster_id or trunk_min <= ci_median:
            wood_clusters.add(cid)
        else:
            leaf_clusters.add(cid)

    # --- Color Assignment ---
    colors = np.zeros((len(pcd.points), 3))
    for cid in wood_clusters:
        colors[RGKNN.Clusters[cid]] = [1, 0, 0]  # red
    for cid in leaf_clusters:
        colors[RGKNN.Clusters[cid]] = [0, 1, 0]  # green
    labels_filtered = labels[original_indices]
    colors[labels_filtered < 0] = [1, 1, 1]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    end_class = time.time()
    print(f"[TIMING] Wood/Leaf classification took {end_class - start_class:.2f} seconds")

    # --- Save Results ---
    wood_indices = [original_indices[i] for cid in wood_clusters for i in RGKNN.Clusters[cid]]
    leaf_indices = [original_indices[i] for cid in leaf_clusters for i in RGKNN.Clusters[cid]]
    noise_indices = np.where(labels < 0)[0]

    pcd_full = pcd_t.to_legacy()
    colors_full = np.zeros((len(pcd_full.points), 3))
    colors_full[wood_indices] = [1, 0, 0]
    colors_full[leaf_indices] = [0, 1, 0]
    colors_full[noise_indices] = [1, 1, 1]
    pcd_full.colors = o3d.utility.Vector3dVector(colors_full)

    print(f"\n[FINAL COUNTS]")
    print(f"  Wood points: {len(wood_indices)}")
    print(f"  Leaf points: {len(leaf_indices)}")
    # print(f"  Noise points: {len(noise_indices)}")

    save_cloud(input_path, save_dir, base_name, "wood", pcd_full.select_by_index(wood_indices), wood_indices, ext)
    save_cloud(input_path, save_dir, base_name, "leaves", pcd_full.select_by_index(leaf_indices), leaf_indices, ext)
    # save_cloud(input_path, save_dir, base_name, "noise", pcd_full.select_by_index(noise_indices), noise_indices, ext)

    # --- Optional Plots ---
    if show_plots:
        plt.figure(figsize=(8, 4))
        plt.hist(full_intensity, bins=100, color='gray', edgecolor='black')
        plt.axvline(threshold_noise, color='blue', linestyle='--', label='Noise Threshold')
        plt.title("Input Intensity Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Point Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.hist(trunk_intensities, bins=100, color='saddlebrown', edgecolor='black')
        plt.axvline(trunk_min, color='green', linestyle='--', label='Wood Threshold Min')
        plt.axvline(trunk_max, color='red', linestyle='--', label='Wood Threshold Max')
        plt.title("Trunk Intensity Distribution")
        plt.xlabel("Intensity")
        plt.ylabel("Point Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Main to use
if __name__ == "__main__":
    classify_wood_leaf("Dataset/tree_0025.ply", save_dir="Results", show_plots=False)

# if __name__ == "__main__":
#     import glob
#
#     input_folder = "Dataset"
#     save_dir = "Results"
#     extensions = ("*.ply", "*.pcd")
#
#     files = []
#     for ext in extensions:
#         files.extend(glob.glob(os.path.join(input_folder, ext)))
#
#     print(f"[INFO] Found {len(files)} files to process.")
#
#     for filepath in files:
#         print(f"\n[PROCESSING] {filepath}")
#         try:
#             classify_wood_leaf(filepath, save_dir=save_dir, show_plots=False)
#         except Exception as e:
#             print(f"[ERROR] Failed to process {filepath}: {e}")

