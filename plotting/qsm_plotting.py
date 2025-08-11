"""
Python adaptation and extension of TREEQSM:

Plots the given point cloud in 2D or 3D.


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
Date: 5 April 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np
try:
    from plotting.cylinders_plotting import cylinders_plotting
except ImportError:
    from .cylinders_plotting import cylinders_plotting
try:
    from ..Utils import Utils
except ImportError:
    import Utils.Utils as Utils

def qsm_plotting(points, cover_sets, segments, qsm=None, marker_size=3,return_html = True,subset=False,fidelity = .1):
    """
    Plots the given point cloud in 3D interactively using Plotly.

    Parameters:
    points : ndarray
        Nx3 NumPy array of point coordinates (x, y, z).
    qsm : treeqsm model
    marker_size : int
        Marker size for plotting.
    """
    if points.shape[1] != 3:
        raise ValueError("Points array must have 3 columns (x, y, z).")
    n = len(points)
    if type(cover_sets)==dict:
        if cover_sets['ball'] is []:
            cover_sets = np.zeros(n)
        if segments['segments'] is []:
            segments = np.zeros(n)
        segs = [np.concatenate(seg).astype(np.int64) for seg in segments["segments"]]
        segment_ids = Utils.assign_segments(points,segs,cover_sets["sets"])
        cover_set_ids = cover_sets["sets"]  
    else:
        segment_ids = segments
        cover_set_ids = cover_sets
        
    #print(segments['segments'])

    # create cover set and segment ids for each point OLD SLOWER METHOD
    # cover_set_ids = np.full(n, -1)  # cover set id each point belongs to, initialize with -1 (not assigned)
    # segment_ids = np.full(n, -1)  # segment id each point belongs to
    # #cover_set_segment_ids = np.full(len(cover_sets['ball']), -1)  # segment id each cover set belongs to

    # for i, elements in enumerate(segments['segments']):
    #     for j, seg_cover_indices in enumerate(elements):
    #         pts = []
    #         for cover_idx in seg_cover_indices:
    #             pts.extend(cover_sets['ball'][cover_idx.astype(int)])  # get all points in a segment
    #         #pts = np.concatenate([pt for pt in pts])
    #         if len(pts) > 0:
    #             segment_ids[pts] = i
    # for i, p_indices in enumerate(cover_sets['ball']):
    #     cover_set_ids[p_indices] = i  # assign cover set ID to those indices

    

    if subset:
        points = points.copy()
        segment_ids = segment_ids.copy()
        cover_set_ids = cover_set_ids.copy()
        # mask = segment_ids != -1
        # segment_ids = segment_ids[mask]
        # cover_set_ids = cover_set_ids[mask]
        # points = points[mask]
        I = np.random.permutation(np.arange(points.shape[0]))
        segment_ids = segment_ids[I[:int(points.shape[0] * fidelity)]]
        cover_set_ids = cover_set_ids[I[:int(points.shape[0] * fidelity)]]
        points = points[I[:int(points.shape[0] * fidelity)], :]
    leaf_mask = segment_ids == -1
    filtered_mask = segment_ids <0
    #print(cover_set_ids)
    #print(cover_sets['ball'])
    #print(len(cover_sets['ball']), len(segments['segments']), p_count)

    
    # trace = make_categorical_color_trace(points, segment_ids, "Segment", marker_size, visible=True, additional_labels=segment_ids)
    # ]
    #Create 5 traces, only one visible at a time
    traces = [
        # make_trace(points, points[:, 0], "X", marker_size, visible=False),
        # make_trace(points, points[:, 1], "Y", marker_size, visible=False),
        # make_trace(points, points[:, 2], "Z", marker_size, visible=True),  # Default visible
        make_categorical_color_trace(points, cover_set_ids, "Cover Set", marker_size, visible=False, additional_labels=cover_set_ids),
        make_categorical_color_trace(points, segment_ids, "Segment", marker_size, visible=True, additional_labels=segment_ids),
        make_categorical_color_trace(points[~leaf_mask], segment_ids[~leaf_mask], "Segment without leaves", marker_size, visible=True),
        make_categorical_color_trace(points[~filtered_mask], segment_ids[~filtered_mask], "Segment with full filtering", marker_size, visible=True)
    ]

    fig = go.Figure(data=traces)

    # Buttons to toggle traces
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    # dict(label="Color by X", method="update",
                    #      args=[{"visible": [True, False, False, False, False]},
                    #            {"title": "Point Cloud Colored by X"}]),
                    # dict(label="Color by Y", method="update",
                    #      args=[{"visible": [False, True, False, False, False]},
                    #            {"title": "Point Cloud Colored by Y"}]),
                    # dict(label="Color by Z", method="update",
                    #      args=[{"visible": [False, False, True, False, False]},
                    #            {"title": "Point Cloud Colored by Z"}]),
                    dict(label="Color by Cover Set", method="update",
                         args=[{"visible": [ True, False,False,False]},
                               {"title": "Point Cloud Colored by Cover Set"}]),
                    dict(label="Color by Segment", method="update",
                         args=[{"visible": [ False, True,False,False]},
                               {"title": "Point Cloud Colored by Segment"}]),
                    dict(label="Segment Without leaves", method="update",
                         args=[{"visible": [ False, False,True,False]},
                               {"title": "Point Cloud Colored by Segment"}]),
                    dict(label="Fully Filtered Segments", method="update",
                         args=[{"visible": [ False, False,False,True]},
                               {"title": "Point Cloud Colored by Segment"}]),
                ],
                direction="down",
                showactive=True,
                type="buttons",
                x=0.02,
                y=1.1,
                xanchor="left",
                yanchor="top"
            )
        ],
        title="Point Cloud Colored by Cover_set",
        title_x=0.5,  # Center the title
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    
    if return_html:
        filename = f"results/point_cloud_plot{qsm['rundata']['inputs']['name']}.html"
        fig.write_html(filename)  # save html to results folder
        return filename
    else:
        return fig
  


def make_categorical_color_trace(points, labels, label_type="Cover Set", marker_size=3, visible=True, additional_labels = None):
    """
    points: Nx3 array
    labels: N array of int values (e.g., cover set or segment IDs)
    label_type: str, "Cover Set" or "Segment"
    """
    unique_labels = np.unique(labels[labels >= 0])  # skip -1 or unassigned
    color_map = pc.qualitative.Plotly  # you can also try 'D3', 'Set3', 'Bold', etc.

    # Assign a color to each unique label
    label_to_color = {label: color_map[i % len(color_map)] for i, label in enumerate(unique_labels)}
    colors = np.array([label_to_color.get(label, 'gray') for label in labels])

    hover_text = [f"{label_type} ID: {labels[i]}" for i in range(len(labels))]

    # If we have additional labels (like cover set IDs or segment IDs), add them to hover text
    if additional_labels is not None:
        hover_text = [f"{label_type} ID: {labels[i]}, Group: {additional_labels[i]}" for i in range(len(labels))]

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=colors,
            opacity=0.8
        ),
        name=f"{label_type} Coloring",
        visible=visible,
        hoverinfo='text',  # Show only custom text
        hovertext=hover_text  # Set custom hover text
    )


# Helper to make a colored trace
def make_trace(points, values, label, marker_size, visible=False):
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=marker_size,
            color=values,
            colorscale="Jet",
            colorbar=dict(title=label) if visible else None,
            showscale=visible
        ),
        name=f"Color by {label}",
        visible=visible
    )