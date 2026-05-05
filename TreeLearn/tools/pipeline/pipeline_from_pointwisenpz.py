import os
import pickle
import argparse
import numpy as np
from tree_learn.util import (
    get_config,
    get_coords_within_shape, get_hull_buffer, get_hull, get_cluster_means,
    propagate_preds, save_treewise, load_data, save_data, make_labels_consecutive,
    propagate_preds_hash_full, propagate_preds_hash_vox
)

NON_TREES_LABEL_IN_GROUPING = 0


def pipeline_from_pointwise(config, pointwise_results_path=None):
    """Resume the TreeLearn pipeline from already-computed pointwise results.

    Loads pointwise_results.npz (saved by run_treelearn_pipeline when
    save_cfg.save_pointwise is True) and runs everything from the
    visualization-save / post-processing stage onward.

    Args:
        config: Munch config object (same one used for the original run).
        pointwise_results_path (str, optional): explicit path to
            pointwise_results.npz.  If None, the path is inferred from
            config.forest_path the same way the original pipeline builds it.
    """

    # ------------------------------------------------------------------ paths
    plot_name = os.path.basename(config.forest_path)[:-4]
    base_dir = os.path.dirname(os.path.dirname(config.forest_path))
    voxelized_data_dir = os.path.join(base_dir, f'forest_voxelized{config.sample_generation.voxel_size}')
    results_dir_name = getattr(config.save_cfg, 'results_dir', 'results')
    results_dir = os.path.join(base_dir, results_dir_name)
    pointwise_dir = os.path.join(results_dir, 'pointwise_results')

    if pointwise_results_path is None:
        pointwise_results_path = os.path.join(pointwise_dir, 'pointwise_results.npz')

    # --------------------------------------------------------------- load npz
    print(f'Loading pointwise results from {pointwise_results_path}')
    data = np.load(pointwise_results_path, allow_pickle=True)

    coords                      = data['coords']
    offset_predictions          = data['offset_predictions']
    offset_labels               = data['offset_labels']
    semantic_prediction_logits  = data['semantic_prediction_logits']
    semantic_labels             = data['semantic_labels']
    instance_labels             = data['instance_labels']
    backbone_feats              = data['backbone_feats']
    input_feats                 = data['input_feats']
    instance_preds              = data['instance_preds'].copy()   # mutable copy

    # xyz_mean was subtracted before saving tiles; reload it so we can add it
    # back at the end.  It is stored alongside the npz when save_pointwise=True,
    # otherwise re-compute from coords (approximation).
    xyz_mean_path = os.path.join(pointwise_dir, 'xyz_mean.npy')
    if os.path.exists(xyz_mean_path):
        xyz_mean = np.load(xyz_mean_path)
    else:
        print('Warning: xyz_mean.npy not found – inferring from coords (may differ slightly).')
        xyz_mean = np.mean(coords, axis=0).astype(np.float64)

    # ----------------------------------------- save visualization LAZ files
    if config.save_cfg.save_pointwise:
        os.makedirs(pointwise_dir, exist_ok=True)

        # -- snapshot 1: high-confidence trunk-like points (initial clustering)
        verticality = input_feats[:, -1]
        verticality_mask = verticality >= config.grouping.tau_vert
        offset_mask = np.abs(offset_predictions[:, 2]) <= config.grouping.tau_off
        sem_mask = instance_preds != NON_TREES_LABEL_IN_GROUPING
        mask = verticality_mask & offset_mask & sem_mask
        cluster_coords = coords[mask] + offset_predictions[mask]
        cluster_coords = np.hstack([cluster_coords, instance_preds[mask].reshape(-1, 1)])
        save_data(cluster_coords, 'laz', 'cluster_coords_initial', pointwise_dir)

        # -- snapshot 2: all tree points after remaining-point assignment
        cluster_coords = coords + offset_predictions
        cluster_coords = cluster_coords[instance_preds != NON_TREES_LABEL_IN_GROUPING]
        cluster_coords = np.hstack([
            cluster_coords,
            instance_preds[instance_preds != NON_TREES_LABEL_IN_GROUPING].reshape(-1, 1)
        ])
        save_data(cluster_coords, 'laz', 'cluster_coords', pointwise_dir)

    # ------------------------------------------------- remove outer points
    if config.shape_cfg.outer_remove:
        hull_buffer_large_path = os.path.join(pointwise_dir, 'hull_buffer_large.pkl')
        if os.path.exists(hull_buffer_large_path):
            import geopandas as gpd
            hull_buffer_large = gpd.read_file(hull_buffer_large_path)
        else:
            hull_buffer_large = get_hull_buffer(
                coords[:, :2], config.shape_cfg.alpha,
                buffersize=config.shape_cfg.outer_remove
            )
        mask_coords_within_hull_buffer_large = get_coords_within_shape(coords, hull_buffer_large)
        masks_inner_coords = np.logical_not(mask_coords_within_hull_buffer_large)

        coords                      = coords[masks_inner_coords]
        semantic_prediction_logits  = semantic_prediction_logits[masks_inner_coords]
        semantic_labels             = semantic_labels[masks_inner_coords]
        offset_predictions          = offset_predictions[masks_inner_coords]
        offset_labels               = offset_labels[masks_inner_coords]
        instance_labels             = instance_labels[masks_inner_coords]
        instance_preds              = instance_preds[masks_inner_coords]
        input_feats                 = input_feats[masks_inner_coords]

        instance_preds[instance_preds != NON_TREES_LABEL_IN_GROUPING], _ = \
            make_labels_consecutive(
                instance_preds[instance_preds != NON_TREES_LABEL_IN_GROUPING], start_num=1
            )

    # -------------------------------- hull / edge info for treewise saving
    if config.save_cfg.save_treewise:
        cluster_means = get_cluster_means(
            coords[instance_preds != NON_TREES_LABEL_IN_GROUPING]
            + offset_predictions[instance_preds != NON_TREES_LABEL_IN_GROUPING],
            instance_preds[instance_preds != NON_TREES_LABEL_IN_GROUPING]
        )
        hull = get_hull(coords[:, :2], config.shape_cfg.alpha)
        cluster_means_within_hull = get_coords_within_shape(cluster_means, hull)

        hull_buffer_small = get_hull_buffer(
            coords[:, :2], config.shape_cfg.alpha,
            buffersize=config.shape_cfg.buffer_size_to_determine_edge_trees
        )
        mask_coords_at_edge = get_coords_within_shape(coords, hull_buffer_small)
        instance_preds_at_edge = np.unique(instance_preds[mask_coords_at_edge])
        instance_preds_at_edge = np.delete(
            instance_preds_at_edge,
            np.where(instance_preds_at_edge == NON_TREES_LABEL_IN_GROUPING)
        )
        insts_not_at_edge = np.ones(len(cluster_means_within_hull))
        insts_not_at_edge[instance_preds_at_edge - 1] = 0
        insts_not_at_edge = insts_not_at_edge.astype('bool')

    # -------------------------------- propagate to original / voxelized
    if config.save_cfg.return_type == 'original':
        print(f'{plot_name}: Propagating predictions to original points')
        coords_to_return = load_data(config.forest_path)[:, :3]
        hash_mapping_path = os.path.join(voxelized_data_dir, f'{plot_name}_hash_mapping.pkl')
        with open(hash_mapping_path, 'rb') as f:
            hash_mapping = pickle.load(f)
        preds_to_return, not_yet_propagated = propagate_preds_hash_full(
            coords, instance_preds, coords_to_return, hash_mapping
        )
    elif config.save_cfg.return_type == 'voxelized':
        print(f'{plot_name}: Propagating predictions to voxelized points')
        voxelized_forest_path = os.path.join(voxelized_data_dir, f'{plot_name}.npz')
        coords_to_return = load_data(voxelized_forest_path)[:, :3]
        preds_to_return, not_yet_propagated = propagate_preds_hash_vox(
            coords, instance_preds, coords_to_return
        )
    else:  # 'voxelized_and_filtered'
        coords_to_return = coords
        preds_to_return = instance_preds
        not_yet_propagated = np.zeros(len(coords_to_return), dtype=bool)

    if config.shape_cfg.outer_remove:
        mask_coords_to_return_within_hull_buffer_large = get_coords_within_shape(
            coords_to_return, hull_buffer_large
        )
        masks_inner_coords_to_return = np.logical_not(mask_coords_to_return_within_hull_buffer_large)
        coords_to_return    = coords_to_return[masks_inner_coords_to_return]
        preds_to_return     = preds_to_return[masks_inner_coords_to_return]
        not_yet_propagated  = not_yet_propagated[masks_inner_coords_to_return]

    if not_yet_propagated.any():
        preds_to_return[not_yet_propagated] = propagate_preds(
            coords, instance_preds, coords_to_return[not_yet_propagated], n_neighbors=5
        )

    # ------------------------------------------------- re-add xyz_mean
    coords_to_return = coords_to_return.astype(np.float64) + xyz_mean

    # ----------------------------------------------------------------- save
    print(f'{plot_name}: Saving results')
    full_dir = os.path.join(results_dir, 'full_forest')
    os.makedirs(full_dir, exist_ok=True)

    for save_format in config.save_cfg.save_formats:
        save_data(
            np.hstack([coords_to_return, preds_to_return.reshape(-1, 1)]),
            save_format, plot_name, full_dir
        )

    if config.save_cfg.save_treewise:
        trees_dir = os.path.join(results_dir, 'individual_trees')
        os.makedirs(trees_dir, exist_ok=True)
        save_treewise(
            coords_to_return, preds_to_return,
            cluster_means_within_hull, insts_not_at_edge,
            'las', trees_dir, NON_TREES_LABEL_IN_GROUPING
        )

    print(f'{plot_name}: Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('pipeline_from_pointwise')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--pointwise_results', type=str, default=None,
                        help='explicit path to pointwise_results.npz (optional)')
    args = parser.parse_args()
    config = get_config(args.config)
    pipeline_from_pointwise(config, pointwise_results_path=args.pointwise_results)
