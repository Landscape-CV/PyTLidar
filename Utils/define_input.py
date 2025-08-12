"""
Python adaptation and extension of TREEQSM:

Estimates required parameters for TreeQSM reconstruction from tree point clouds.


% -----------------------------------------------------------
% This file is part of TREEQSM.
%
% TREEQSM is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% TREEQSM is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with TREEQSM.  If not, see <http://www.gnu.org/licenses/>.
% -----------------------------------------------------------


Version: 0.0.1
Date: 18 Jan 2025
Authors: Fan Yang, John Hagood, Amir Hossein Alikhah Mishamandani
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import numpy as np
try:
    from Utils import compute_patch_diam,distances_to_line,load_point_cloud,optimal_parallel_vector
except ImportError:
    from .Utils import compute_patch_diam,distances_to_line,load_point_cloud,optimal_parallel_vector
from numba.typed import Dict

def define_input(clouds, nPD1=1, nPD2Min=1, nPD2Max=1):
    """
    Estimates required parameters for TreeQSM reconstruction from tree point clouds.

    Parameters:
    clouds : ndarray, str, or list of str
        Point cloud of a tree (Nx3 array), or the filename(s) of LAS/LAZ files.
    nPD1 : int, optional
        Number of parameter values for PatchDiam1. Default is 1.
    nPD2Min : int, optional
        Number of parameter values for PatchDiam2Min. Default is 1.
    nPD2Max : int, optional
        Number of parameter values for PatchDiam2Max. Default is 1.

    Returns:
    list of dict:
        List of input structures with the estimated parameter values for each tree.
    """
    

    # Handle different input types (single file, multiple files, or array)
    if isinstance(clouds, list):  # Multiple LAS/LAZ files
        nt = len(clouds)
        cloud_list = clouds#[load_point_cloud(file) for file in clouds]
    elif isinstance(clouds, str):  # Single LAS/LAZ file
        cloud_list = [load_point_cloud(clouds)]
        nt = 1
    else:  # Single point cloud as ndarray
        cloud_list = [clouds]
        nt = 1
    inputs = []
    for i in range(nt):
        P = cloud_list[i]
        if P is None or P.shape[0] == 0:
            print(f"Skipping empty point cloud for tree {i + 1}")
            continue
                
        input_params = {'name': f"Tree_{i + 1}"}

        # Tree height estimation
        Hb = np.min(P[:, 2])
        Ht = np.max(P[:, 2])
        tree_height = Ht - Hb
        Hei = P[:, 2] - Hb

        # Section selection near the base
        hSecTop = min(4, 0.1 * tree_height)
        hSecBot = 0.02 * tree_height
        section_indices = (Hei > hSecBot) & (Hei < hSecTop)
        StemBot = P[section_indices]

        # Estimate stem axis
        axis_point = np.mean(StemBot, axis=0)
        V = StemBot - axis_point
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        axis_dir = optimal_parallel_vector(V)

        # Stem diameter estimation
        d, _, _, _ = distances_to_line(StemBot, axis_dir, axis_point)
        Rstem = np.median(d)

        # Point resolution
        hSec = hSecTop - hSecBot
        Res = np.sqrt((2 * np.pi * Rstem * hSec) / StemBot.shape[0])

        # Define PatchDiam parameters
        pd1 = Rstem / 3
        input_params['PatchDiam1'] = compute_patch_diam(pd1, nPD1)

        pd2 = Rstem / 6 * min(1, 20 / tree_height)
        input_params['PatchDiam2Min'] = compute_patch_diam(pd2, nPD2Min)

        pd3 = Rstem / 2.5
        input_params['PatchDiam2Max'] = compute_patch_diam(pd3, nPD2Max)

        # Define BallRad parameters
        ball_rads = []
        for pd in input_params['PatchDiam1']:
            ball_rad = max(pd + 1.5 * Res, min(1.25 * pd, pd + 0.025))
            ball_rads.append(ball_rad)
        input_params['BallRad1'] = ball_rads
        # input_params['BallRad1'] = [max(input_params['PatchDiam1'][-1] + 1.5 * Res,
        #                                min(1.25 * input_params['PatchDiam1'][-1],
        #                                    input_params['PatchDiam1'][-1] + 0.025))]

        ball_rad2s = []
        for pd in input_params['PatchDiam2Max']:
            ball_rad2 = max(pd + 1.25 * Res, min(1.2 * pd, pd + 0.025))
            ball_rad2s.append(ball_rad2)
        input_params['BallRad2'] = ball_rad2s
        # input_params['BallRad2'] = [max(input_params['PatchDiam2Max'][-1] + 1.25 * Res,
        #                                min(1.2 * input_params['PatchDiam2Max'][-1],
        #                                    input_params['PatchDiam2Max'][-1] + 0.025))]

        # Include additional fields as per MATLAB output
        input_params.update({
            'nmin1': 3, 'nmin2': 1, 'OnlyTree': 1, 'Tria': 0, 'Dist': 1,
            'MinCylRad': 0.0025, 'ParentCor': 1, 'TaperCor': 1, 'GrowthVolCor': 0,
            'GrowthVolFac': 1.5, 'tree': i + 1, 'model': 1, 'savemat': 1,
            'savetxt': 1, 'plot': 2, 'disp': 2
        })

        inputs.append(input_params)

    return inputs

