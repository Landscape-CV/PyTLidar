from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EcomodelConfig:
    # ── I/O ──────────────────────────────────────────────────────────────────
    input_folder: str = ""
    results_folder: str = "results"
    cylinder_filename: str = "ecomodel_cylinder_data"

    # ── Tile subdivision ─────────────────────────────────────────────────────
    cube_size: float = 10.0
    meter_conversion: float = 1.0

    # ── Ground filtering (CSF) ───────────────────────────────────────────────
    csf_band_size: float = 0.1
    csf_threshold: int = 20
    csf_offset: float = 0.2
    remove_under_ground: bool = True

    # ── Terrain model ────────────────────────────────────────────────────────
    terrain_grid_size: float = 0.1

    # ── Optional cleanup / denoise ───────────────────────────────────────────
    use_denoise: bool = False
    denoise_grid_size: float = 0.1
    denoise_min_points: int = 10
    denoise_resolution: float = 0.05

    # ── Duplicate removal ────────────────────────────────────────────────────
    remove_duplicates: bool = False

    # ── Tree segmentation ────────────────────────────────────────────────────
    segment_intensity_threshold: int = 0
    save_clusters: bool = False

    # ── QSM ──────────────────────────────────────────────────────────────────
    qsm_intensity_threshold: int = 40000
    save_leaf_removal_output: bool = False
    run_qsm: bool = True
    run_leaf_removal: bool = True

    # ── Visualisation ────────────────────────────────────────────────────────
    create_cylinder_plot: bool = False

    # ── Cover set placeholders ───────────────────────────────────────────────
    # Stored here for GUI/config completeness. These are not yet fully wired
    # through all Ecomodel methods unless explicitly passed downstream.
    patch_diam1: float = 0.15
    ball_rad1: float = 0.15
    nmin1: int = 25
    patch_diam2_min: float = 0.05
    patch_diam2_max: float = 0.08
    ball_rad2: float = 0.09

    # ── RGI placeholders ─────────────────────────────────────────────────────
    # Stored here for GUI/config completeness. These are not yet fully wired
    # into classify_wood_leaf() unless explicitly forwarded by the pipeline.
    rgi_noise_percentile: int = 0
    rgi_angle_deg: float = 7.0
    rgi_curv_thresh: float = 0.07
    rgi_resid_thresh: float = 0.05
    rgi_k: int = 100
    rgi_min_cluster_size: int = 40
    rgi_max_cluster_size: int = 100000
    rgi_smooth_mode: bool = True
    rgi_use_residual_test: bool = True
    rgi_use_curvature_test: bool = True

    # ── Checkpointing ────────────────────────────────────────────────────────
    checkpoint_name: str = "ecomodel_checkpoint"
    checkpoint_folder: str = "pickle"
    use_checkpoint: bool = False
    save_checkpoint: bool = False
    checkpoint_resume_file: str = ""   # full path to .pickle file to resume from