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
Date: 18 Jan 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import plotly.graph_objects as go
import plotly.offline
import numpy as np
def point_cloud_plotting(points, marker_size=3,fidelity=.1, subset = False, return_html = True):
    """
    Plots the given point cloud in 3D interactively using Plotly.

    Parameters:
    points : ndarray
        Nx3 NumPy array of point coordinates (x, y, z).
    marker_size : int
        Marker size for plotting.
    """
    if points.shape[1] != 3:
        raise ValueError("Points array must have 3 columns (x, y, z).")
    if subset:
        points = points.copy()
        I = np.random.permutation(np.arange(points.shape[0]))
        points = points[I[:int(points.shape[0] * fidelity)], :]


    # Create the 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=points[:, 2],  # Color by Z-coordinate for a heatmap effect
                    colorscale="Jet",  # Heatmap color scheme
                    colorbar=dict(title="Height (Z-axis)")
                ),
            )
        ]
    )

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
        title="Interactive 3D Point Cloud",
    )

    # Show the interactive plot
    fig.write_html("point_cloud_plot.html", include_plotlyjs=True, full_html=True,auto_open=False)
    
    if return_html:
        return "point_cloud_plot.html"
    else:
        return fig
    


