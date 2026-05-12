from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

from gui.config import EcomodelConfig


class PipelineStopped(Exception):
    """Raised when the user clicks Stop — not a real error."""


from gui.results_io import (
    find_run_dirs,
    make_run_dir,
    save_cover_sets_snapshot,
    save_point_fields_snapshot,
    save_point_cloud_snapshot,
    save_segment_labels_snapshot,
    write_run_metadata,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def run_ecomodel_pipeline(
    config: EcomodelConfig,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    tile_update_callback: Optional[Callable[[str, str, int, str], None]] = None,
):
    """
    Execute the Ecomodel pipeline.

    PIPELINE ORDER
    ──────────────
      1. combine_las_files              – load LAS/LAZ tiles into eco._raw_tiles
      2. subdivide_tiles (first pass)   – create coarse tiles for deduplication
      3. remove_duplicate_points        – remove duplicate points per tile
      4. recombine_tiles                – merge coarse tiles back into eco._raw_tiles
      5. filter_ground                  – CSF ground filtering on raw tiles
      6. get_terrain_model              – build DEM on raw tiles
      7. normalize_raw_tiles            – apply terrain subtraction / XY centering
      8. move raw tiles to device       – convert raw tiles to torch if needed
      9. subdivide_tiles (second pass)  – create tiles for segmentation / QSM
     10. denoise (optional)             – remove sparse local noise on subdivided tiles
     11. segment_trees                  – tree instance segmentation on eco.tiles
     12. reset_terrain                  – restore coordinates before QSM extraction
     13. get_qsm_segments_*             – run QSM with or without leaf removal
     14. recombine_tiles                – merge processed tiles back to eco._raw_tiles
     15. get_all_cylinders              – export cylinder array
     16. create_cylinder_plot (optional)– create plot output

    Processing Flow
    ───────────────
    The pipeline alternates between operating on:
        • eco._raw_tiles  (full point cloud)
        • eco.tiles       (subdivided tiles)

    Key design constraint:
        Some operations require full tiles (ground filtering, terrain model),
        while others require spatial subdivision (segmentation, QSM).

    Therefore:
        raw_tiles → subdivide → tiles → recombine → raw_tiles → subdivide → tiles
    """

    from ecomodel import Ecomodel

    # ── Create a dedicated output subfolder for this run ──────────────────
    run_dir = make_run_dir(config.results_folder, config.cylinder_filename)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Base steps: load, dedup, ground filter, terrain, normalize,
    #             device, subdivide, segment, reset terrain, recombine, export = 11
    total_steps = 11
    if config.use_denoise:
        total_steps += 1
    if config.run_qsm:
        total_steps += 1
    if config.create_cylinder_plot:
        total_steps += 1

    def _log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    def _progress(current: int, label: str = "") -> None:
        if progress_callback:
            progress_callback(current, total_steps, label)

    def _check_stop() -> None:
        if should_stop and should_stop():
            raise PipelineStopped("Pipeline stopped by user.")

    def _tile_update(name: str, status: str, cyl_count: int = -1, path: str = "") -> None:
        if tile_update_callback:
            tile_update_callback(name, status, cyl_count, path)

    # Scalar field names for columns 3+ of tile.point_data (e.g. "Intensity",
    # "Return Number", …).  Populated after step 1 loads the first LAS file,
    # or restored from the checkpoint sidecar on resume.  Stored as a mutable
    # list so the _save_checkpoint closure can read the current value.
    _las_field_names: list = []

    def _save_checkpoint(tag: str) -> None:
        if not getattr(config, "save_checkpoint", False):
            return
        checkpoint_name = f"{config.checkpoint_name}_{tag}.pickle"
        eco.pickle(checkpoint_name, config.checkpoint_folder)
        _log(f"[Checkpoint] Saved: {checkpoint_name}\n")
        # Write resume metadata sidecar so the loader doesn't have to guess
        # step numbers from the filename + current config.
        import json as _json
        _sidecar_path = Path(config.checkpoint_folder) / f"{config.checkpoint_name}_{tag}.resume.json"
        _sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(_sidecar_path, "w") as _f:
                _json.dump({
                    "resume_after_step": step,
                    "tag": tag,
                    "use_denoise": config.use_denoise,
                    "total_steps": total_steps,
                    "las_field_names": list(_las_field_names),
                }, _f, indent=2)
        except Exception:
            pass  # non-critical — fallback to filename parsing on load

    def _should_run(step_num: int) -> bool:
        return step_num > _resume_after_step

    step = 0

    # Create or load model
    if config.use_checkpoint:
        # Determine full path to the checkpoint file
        resume_file = getattr(config, "checkpoint_resume_file", "").strip()
        if resume_file:
            ckpt_path = Path(resume_file)
        else:
            # Fallback: construct from name + folder (no tag suffix)
            ckpt_name = config.checkpoint_name
            if not ckpt_name.endswith(".pickle"):
                ckpt_name = f"{ckpt_name}.pickle"
            ckpt_path = Path(config.checkpoint_folder) / ckpt_name

        # Try reading sidecar metadata first (written by _save_checkpoint).
        _sidecar_path = ckpt_path.with_suffix(".resume.json")
        _resume_after_step = 0
        _checkpoint_tag = "start"
        if _sidecar_path.exists():
            import json as _json
            try:
                with open(_sidecar_path) as _f:
                    _sidecar = _json.load(_f)
                _resume_after_step = _sidecar["resume_after_step"]
                _checkpoint_tag = _sidecar.get("tag", "sidecar")
                # Restore scalar field names so the snapshot at the end of this
                # resumed run is labelled correctly without re-loading the LAS.
                _las_field_names.extend(_sidecar.get("las_field_names", []))
            except Exception:
                pass  # fall through to filename parsing

        # Fallback: parse tag from filename (for checkpoints saved before sidecar support).
        # Step numbers must match the actual pipeline counter, which depends on
        # use_denoise and run_qsm being the same as when the checkpoint was saved.
        if _resume_after_step == 0:
            _d = 1 if config.use_denoise else 0      # denoise adds one step
            _STAGE_STEPS = {
                "post_normalize":     5,
                # post_segmentation is now saved after reset_terrain (one step
                # past segment_trees) so the resume lands directly before QSM.
                "post_segmentation":  9 + _d,         # reset_terrain = step 9 (no denoise) or 10
                "post_qsm":           10 + _d,        # QSM = one step after reset_terrain
            }
            _stem = ckpt_path.stem
            for _tag, _snum in _STAGE_STEPS.items():
                if _stem.endswith(f"_{_tag}"):
                    _resume_after_step = _snum
                    _checkpoint_tag = _tag
                    break

        _log(f"Loading checkpoint '{ckpt_path.name}' from '{ckpt_path.parent}'...\n")
        eco = Ecomodel.unpickle(ckpt_path.name, str(ckpt_path.parent))
        _original_results_folder = eco.results_folder  # where the previous run wrote files
        eco.results_folder = str(run_dir)   # redirect outputs into this run's folder
        _log(f"Checkpoint loaded. Resuming after step {_resume_after_step} ({_checkpoint_tag}).\n")
    else:
        eco = Ecomodel(results_folder=str(run_dir))
        _original_results_folder = None
        _resume_after_step = 0
        _checkpoint_tag = ""

    _check_stop()

    # Enumerate LAS files for tile_update status signals.
    # When resuming from a checkpoint the original input folder may not exist
    # — the Ecomodel already has all its data.  Gracefully fall back to an
    # empty list; tile_update signals are informational only.
    import os as _os
    try:
        _las_files = sorted(
            f for f in _os.listdir(config.input_folder)
            if f.lower().endswith(('.las', '.laz'))
        ) if config.input_folder else []
    except OSError:
        _las_files = []

    # Step 1: load raw tiles
    step += 1
    if _should_run(step):
        _log(f"Step {step}/{total_steps} – Loading LAS/LAZ tiles...\n")
        _progress(step, "Loading tiles")

        eco = Ecomodel.combine_las_files(config.input_folder, eco)
        if eco is None or len(eco._raw_tiles) == 0:
            raise FileNotFoundError(f"No LAS/LAZ files found in '{config.input_folder}'.")

        # Capture the field names discovered during LAS loading so they can
        # be written to the checkpoint sidecar and used at snapshot time.
        from utils.utils import get_last_las_field_names as _get_fields
        _las_field_names.clear()
        _las_field_names.extend(_get_fields())
        _log(f"Loaded {len(eco._raw_tiles)} raw tile(s).  "
             f"Scalar fields: {', '.join(_las_field_names)}\n")
        for _fname in _las_files:
            _tile_update(_fname, "Pending")
    else:
        _log(f"Step {step}/{total_steps} – Skipped (checkpoint: {_checkpoint_tag})\n")
        _progress(step, "Skipped")
        for _fname in _las_files:
            _tile_update(_fname, "Pending")   # populate table even when resuming

    _check_stop()

    # Step 2: optional duplicate removal on raw tiles
    step += 1
    if _should_run(step):
        _log(f"Step {step}/{total_steps} – Removing duplicate points...\n")
        _progress(step, "Deduplicating")

        if config.remove_duplicates:
            for tile in eco._raw_tiles:
                if tile is None or not hasattr(tile, "remove_duplicate_points"):
                    continue
                tile.remove_duplicate_points()
            _log("Duplicate removal complete.\n")
        else:
            _log("Skipping duplicate removal.\n")
    else:
        _log(f"Step {step}/{total_steps} – Skipped (checkpoint: {_checkpoint_tag})\n")
        _progress(step, "Skipped")

    _check_stop()

    # Step 3: ground filtering on raw tiles
    step += 1
    if _should_run(step):
        _log(f"Step {step}/{total_steps} – Filtering ground (CSF)...\n")
        _progress(step, "Ground filtering")

        eco.filter_ground(
            tile_list=eco._raw_tiles,
            band_size=config.csf_band_size,
            threshold=config.csf_threshold,
            offset=config.csf_offset,
            remove_under_ground=config.remove_under_ground,
        )
    else:
        _log(f"Step {step}/{total_steps} – Skipped (checkpoint: {_checkpoint_tag})\n")
        _progress(step, "Skipped")

    _check_stop()

    # Step 4: terrain model on raw tiles
    step += 1
    if _should_run(step):
        _log(f"Step {step}/{total_steps} – Building terrain model...\n")
        _progress(step, "Terrain model")

        eco.get_terrain_model(
            tile_list=eco._raw_tiles,
            grid_size=config.terrain_grid_size,
        )
    else:
        _log(f"Step {step}/{total_steps} – Skipped (checkpoint: {_checkpoint_tag})\n")
        _progress(step, "Skipped")

    _check_stop()

    # Step 5: normalize raw tiles
    step += 1
    if _should_run(step):
        _log(f"Step {step}/{total_steps} – Normalizing raw tiles...\n")
        _progress(step, "Normalizing")

        eco.normalize_raw_tiles()
        _save_checkpoint("post_normalize")
    else:
        _log(f"Step {step}/{total_steps} – Skipped (checkpoint: {_checkpoint_tag})\n")
        _progress(step, "Skipped")

    _check_stop()

    # Step 6: move raw tiles to device
    step += 1
    if _should_run(step):
        _log(f"Step {step}/{total_steps} – Preparing tiles on device...\n")
        _progress(step, "Preparing tensors")

        for tile in eco._raw_tiles:
            if tile is None or not hasattr(tile, "to"):
                continue
            tile.to(tile.device)
    else:
        _log(f"Step {step}/{total_steps} – Skipped (checkpoint: {_checkpoint_tag})\n")
        _progress(step, "Skipped")

    _check_stop()

    # Step 7: subdivide once
    step += 1
    if _should_run(step):
        _log(f"Step {step}/{total_steps} – Subdividing tiles (cube={config.cube_size}m)...\n")
        _progress(step, "Subdividing")

        eco.subdivide_tiles(
            cube_size=config.cube_size,
            meter_conversion=config.meter_conversion,
        )
    else:
        _log(f"Step {step}/{total_steps} – Skipped (checkpoint: {_checkpoint_tag})\n")
        _progress(step, "Skipped")

    _check_stop()

    # Step 8: optional denoise on subdivided tiles
    if config.use_denoise:
        step += 1
        if _should_run(step):
            _log(f"Step {step}/{total_steps} – Denoising subdivided tiles...\n")
            _progress(step, "Denoising")

            eco.denoise(
                grid_size=config.denoise_grid_size,
                min_points=config.denoise_min_points,
                resolution=config.denoise_resolution,
            )
        else:
            _log(f"Step {step}/{total_steps} – Skipped (checkpoint: {_checkpoint_tag})\n")
            _progress(step, "Skipped")

        _check_stop()

    # Step 9: tree segmentation
    step += 1
    if _should_run(step):
        _log(f"Step {step}/{total_steps} – Segmenting trees...\n")
        _progress(step, "Tree segmentation")

        eco.segment_trees(
            intensity_threshold=config.segment_intensity_threshold,
            save_clusters=config.save_clusters,
        )
    else:
        _log(f"Step {step}/{total_steps} – Skipped (checkpoint: {_checkpoint_tag})\n")
        _progress(step, "Skipped")

    _check_stop()

    # Step 10: restore terrain
    # Saved as "post_segmentation" here (not after segment_trees) so that
    # restoring this checkpoint lands directly at QSM — reset_terrain is
    # cheap bookkeeping and there is no point re-doing segmentation to get it.
    step += 1
    if _should_run(step):
        _log(f"Step {step}/{total_steps} – Restoring terrain coordinates...\n")
        _progress(step, "Restoring terrain")

        eco.reset_terrain()
        _save_checkpoint("post_segmentation")
    else:
        _log(f"Step {step}/{total_steps} – Skipped (checkpoint: {_checkpoint_tag})\n")
        _progress(step, "Skipped")

    _check_stop()

    # Step 11: optional QSM + RGI
    if config.run_qsm:
        step += 1
        if _should_run(step):
            _log(f"Step {step}/{total_steps} – Running QSM + RGI leaf separation...\n")
            _progress(step, "QSM + leaf separation")

            eco.get_qsm_segments_rgi(
                intensity_threshold=config.qsm_intensity_threshold,
                save_leaf_removal_output=config.save_leaf_removal_output,
            )
            _save_checkpoint("post_qsm")
        else:
            _log(f"Step {step}/{total_steps} – Skipped (checkpoint: {_checkpoint_tag})\n")
            _progress(step, "Skipped")

        _check_stop()

    # Step 12: recombine
    step += 1
    _log(f"Step {step}/{total_steps} – Recombining tiles...\n")
    _progress(step, "Recombining")

    eco.recombine_tiles()

    # ── Helper: secondary LAS read for extra scalar fields ───────────────────
    # point_data is kept at exactly (N, 4) throughout the pipeline so that
    # ecomodel's torch subdivide_tiles never sees unexpected column counts.
    # To expose all LAS dimensions (return number, GPS time, scan angle, …)
    # in the viewer we re-open the original files *after* the pipeline ends,
    # match each surviving snapshot point back to its LAS row by world-XY
    # coordinate, and extract every available dimension.
    #
    # Matching strategy
    # -----------------
    #  • snap_cloud is in normalised space; world XY = snap_cloud[:, :2] + eco.mean[:2]
    #  • Both snap world XY and las.x / las.y are float64.
    #  • Round to nearest 1 mm so that floating-point round-trip errors
    #    (≈ 0.2 nm) never cause false mismatches.
    #  • Sort LAS rows by (xi_mm, yi_mm) structured key; use np.searchsorted
    #    to locate each snapshot point in O(N log N).
    def _read_extra_las_fields(
        las_folder: str,
        snap_cloud_f64,           # (N, 3) float64 normalised coords
        eco_mean_f64,             # sequence of 3 floats (mean XYZ)
    ) -> dict:
        """
        Return a dict {display_name: float32 array of length N} for every
        LAS dimension found in las_folder, aligned row-for-row with
        snap_cloud_f64.  Returns {} on any error or if las_folder is absent.
        """
        import numpy as _np2
        try:
            import laspy as _lp
        except ImportError:
            _log("[Snapshot] laspy not available — skipping extra LAS fields.\n")
            return {}
        import os as _os2

        n = len(snap_cloud_f64)
        if n == 0:
            return {}

        # Reconstruct world XY at float64, round to nearest mm
        _mxy = _np2.asarray(eco_mean_f64[:2], dtype=_np2.float64)
        _wxy = snap_cloud_f64[:, :2].astype(_np2.float64) + _mxy
        # Pack (x_mm, y_mm) as a structured int64 dtype for lexicographic sort
        _snap_keys = _np2.empty(n, dtype=[("xi", _np2.int64), ("yi", _np2.int64)])
        _snap_keys["xi"] = _np2.round(_wxy[:, 0] * 1000).astype(_np2.int64)
        _snap_keys["yi"] = _np2.round(_wxy[:, 1] * 1000).astype(_np2.int64)

        result: dict = {}
        _filled: dict = {}   # bool mask: True = row already written

        # Dimensions to skip (raw integer coordinate columns already in cloud).
        # laspy may expose these as uppercase "X","Y","Z" or lowercase "x","y","z".
        _skip_dims = {"X", "Y", "Z", "x", "y", "z"}

        try:
            _fnames = sorted(
                f for f in _os2.listdir(las_folder)
                if f.lower().endswith((".las", ".laz"))
            ) if las_folder else []
        except OSError:
            return {}

        for _fn in _fnames:
            try:
                _fpath = _os2.path.join(las_folder, _fn)
                _las = _lp.read(_fpath)
                _nl = len(_las.x)
                if _nl == 0:
                    continue

                # Build (xi_mm, yi_mm) keys for every LAS point
                _lk = _np2.empty(_nl, dtype=[("xi", _np2.int64), ("yi", _np2.int64)])
                _lk["xi"] = _np2.round(
                    _np2.asarray(_las.x, dtype=_np2.float64) * 1000
                ).astype(_np2.int64)
                _lk["yi"] = _np2.round(
                    _np2.asarray(_las.y, dtype=_np2.float64) * 1000
                ).astype(_np2.int64)

                # Sort LAS keys; searchsorted snap keys into sorted order
                _ord = _np2.argsort(_lk, order=("xi", "yi"), kind="stable")
                _slk = _lk[_ord]
                _pos = _np2.searchsorted(_slk, _snap_keys)
                _pos = _np2.minimum(_pos, _nl - 1)
                _matched = _slk[_pos] == _snap_keys    # bool (N,)
                _las_rows = _ord[_pos]                 # LAS row index per snap point

                if not _np2.any(_matched):
                    del _las, _lk, _ord, _slk
                    continue

                # Extract all scalar dimensions from this file
                for _dim in _las.point_format.dimension_names:
                    if _dim in _skip_dims:
                        continue
                    # Human-readable display name: "return_number" → "Return Number"
                    _dname = " ".join(
                        w.capitalize() for w in _dim.replace("_", " ").split()
                    )
                    try:
                        _col = _np2.asarray(getattr(_las, _dim), dtype=_np2.float32)
                    except Exception:
                        continue

                    if _dname not in result:
                        result[_dname]  = _np2.zeros(n, dtype=_np2.float32)
                        _filled[_dname] = _np2.zeros(n, dtype=_np2.bool_)

                    # Only write rows not yet filled (first file that contains a
                    # matching point wins — avoids overwriting with zeros from
                    # a tile file that doesn't cover that point).
                    _to_fill = _matched & ~_filled[_dname]
                    result[_dname][_to_fill]  = _col[_las_rows[_to_fill]]
                    _filled[_dname][_to_fill] = True

                del _las, _lk, _ord, _slk

            except Exception as _exc:
                _log(f"[Snapshot] Warning: could not read fields from '{_fn}': {_exc}\n")
                continue

        # Drop fields where no point was matched (all zeros from initialisation)
        return {k: v for k, v in result.items() if _np2.any(_filled.get(k, _np2.zeros(1, dtype=_np2.bool_)))}

    # ── Save subsampled snapshots ─────────────────────────────────────────────
    # save_point_cloud_snapshot returns the indices it used so we can pass
    # exactly the same rows to save_segment_labels_snapshot — guaranteeing
    # the two files are aligned row-for-row.
    if eco._raw_tiles:
        import numpy as _np
        _snap_tile   = eco._raw_tiles[0]
        _snap_cloud  = _snap_tile.cloud
        _snap_labels = _snap_tile.segment_labels
        _snap_cover  = getattr(_snap_tile, "cover_sets", None)
        if isinstance(_snap_cloud, _np.ndarray) and len(_snap_cloud) > 0:
            _idx = save_point_cloud_snapshot(run_dir, _snap_cloud)
            if isinstance(_snap_labels, _np.ndarray) and len(_snap_labels) == len(_snap_cloud):
                save_segment_labels_snapshot(run_dir, _snap_labels, indices=_idx)
            if isinstance(_snap_cover, _np.ndarray) and len(_snap_cover) == len(_snap_cloud):
                save_cover_sets_snapshot(run_dir, _snap_cover, indices=_idx)

            # ── Scalar fields ────────────────────────────────────────────────
            # Seed from pipeline's in-memory point_data col 3 (intensity).
            # Then augment with every other LAS dimension via a secondary read
            # of the original files (avoids widening point_data which would
            # break torch subdivide_tiles).
            _fields: dict = {}
            _snap_pd = getattr(_snap_tile, "point_data", None)
            if (isinstance(_snap_pd, _np.ndarray) and _snap_pd.ndim == 2
                    and len(_snap_pd) == len(_snap_cloud)):
                # col 3 is always intensity; use the name from _las_field_names
                # if available, otherwise default to "Intensity".
                _int_name = _las_field_names[0] if _las_field_names else "Intensity"
                _fields[_int_name] = _snap_pd[:, 3].astype(_np.float32)

            # Secondary read: all other dimensions from the original LAS files.
            # Gracefully skipped if input_folder is unavailable (checkpoint resume
            # from a machine that no longer has the source data).
            _eco_mean = (
                [float(v) for v in eco.mean]
                if hasattr(eco, "mean") and eco.mean is not None
                else [0.0, 0.0, 0.0]
            )
            try:
                _extra = _read_extra_las_fields(
                    config.input_folder, _snap_cloud, _eco_mean
                )
                # Merge: pipeline intensity takes precedence over secondary read
                for _k, _v in _extra.items():
                    if _k not in _fields:
                        _fields[_k] = _v
                if _extra:
                    _log(
                        f"[Snapshot] Extra LAS fields saved: "
                        f"{', '.join(_extra.keys())}\n"
                    )
            except Exception as _exc:
                _log(f"[Snapshot] Secondary LAS read failed (non-critical): {_exc}\n")

            if _fields:
                save_point_fields_snapshot(run_dir, _fields, indices=_idx)

    _check_stop()

    # Step 13: export cylinders
    step += 1
    _log(f"Step {step}/{total_steps} – Exporting cylinder data...\n")
    _progress(step, "Exporting cylinders")

    eco.get_all_cylinders(filename=config.cylinder_filename)

    _out_path = str(run_dir / f"{config.cylinder_filename}.txt")
    # Count from the exported file — tile.cylinder_radii after recombine_tiles()
    # only holds the first subdivided tile's data (concat lines are commented out).
    try:
        import numpy as _np
        _cyl_count = int(len(_np.loadtxt(_out_path)))
    except Exception:
        _cyl_count = -1
    _point_count = int(len(eco._raw_tiles[0].cloud)) if eco._raw_tiles else 0
    _cloud_mean = [float(v) for v in eco.mean] if hasattr(eco, "mean") and eco.mean is not None else [0.0, 0.0, 0.0]
    write_run_metadata(run_dir, config, _cyl_count, _point_count, cloud_mean=_cloud_mean)
    for _fname in _las_files:
        _tile_update(_fname, "Done", _cyl_count, _out_path)

    _check_stop()

    # If resuming from a checkpoint, the _leaves_removed.xyz files from QSM
    # live in the original run folder.  Copy them so create_cylinder_plot finds them.
    if _original_results_folder and Path(_original_results_folder).is_dir():
        import shutil as _shutil
        for _f in Path(_original_results_folder).glob("*_leaves_removed*.xyz"):
            _dest = run_dir / _f.name
            if not _dest.exists():
                _shutil.copy2(_f, _dest)
                _log(f"[Checkpoint] Copied {_f.name} from previous run.\n")

    # Step 14: optional plot
    if config.create_cylinder_plot:
        step += 1
        _log(f"Step {step}/{total_steps} – Creating cylinder plot...\n")
        _progress(step, "Visualization")

        eco.create_cylinder_plot(config.cylinder_filename)

        _check_stop()

    _log("Pipeline complete.\n")
    _progress(total_steps, "Done")
    return eco