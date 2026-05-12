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
def point_cloud_plotting(points, marker_size=3, fidelity=.1, subset=False,
                         return_html=True, fields=None):
    """
    Plots the given point cloud in 3D interactively using Plotly.

    Parameters
    ----------
    points : ndarray
        Nx3 NumPy array of point coordinates (x, y, z).
    marker_size : int
        Marker size for plotting.
    subset : bool
        If True, randomly subsample to ``fidelity`` fraction of points.
    return_html : bool
        If True, write to disk and return the filename.
        If False, return the Plotly Figure object.
    fields : dict or None
        Optional mapping of ``{display_name: ndarray}`` for per-point scalar
        fields (e.g. ``{"Intensity": arr, "Return Number": arr}``).
        "Height (Z)" is always added automatically as the first option.
        When more than one option exists a dropdown is shown in the viewer.
    """
    if points.shape[1] != 3:
        raise ValueError("Points array must have 3 columns (x, y, z).")

    if subset:
        points = points.copy()
        I = np.random.permutation(np.arange(points.shape[0]))
        idx = I[:int(points.shape[0] * fidelity)]
        points = points[idx]
        if fields:
            fields = {k: v[idx] for k, v in fields.items()
                      if v is not None and len(v) > max(idx)}

    # ── Build ordered field dict ──────────────────────────────────────────────
    # "Height (Z)" is always available; extra fields follow in the order given.
    all_fields: dict = {"Height (Z)": points[:, 2]}
    if fields:
        for name, arr in fields.items():
            if arr is not None and len(arr) == len(points):
                all_fields[name] = arr

    # Default to Intensity if present, otherwise Z
    default_name = next(
        (k for k in all_fields if "intensity" in k.lower()),
        list(all_fields.keys())[0],
    )

    # Colorscales per field — intensity → Plasma, everything else → Viridis
    def _colorscale(name: str) -> str:
        return "Plasma" if "intensity" in name.lower() else "Viridis"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=all_fields[default_name],
                    colorscale=_colorscale(default_name),
                    colorbar=dict(title=dict(text=default_name, side="right")),
                    showscale=True,
                ),
            )
        ]
    )

    # ── Dropdown (only shown when > 1 field is available) ─────────────────────
    if len(all_fields) > 1:
        buttons = []
        for name, values in all_fields.items():
            buttons.append(dict(
                label=name,
                method="restyle",
                args=[{
                    "marker.color":            [values.tolist()],
                    "marker.colorscale":       [_colorscale(name)],
                    "marker.colorbar.title.text": [name],
                }],
            ))

        fig.update_layout(
            updatemenus=[dict(
                type="dropdown",
                buttons=buttons,
                direction="down",
                showactive=True,
                active=list(all_fields.keys()).index(default_name),
                pad={"r": 10, "t": 10},
                x=0.0,
                y=1.12,
                xanchor="left",
                yanchor="top",
            )],
            annotations=[dict(
                text="Colour by:",
                showarrow=False,
                x=0.0,
                y=1.16,
                xref="paper",
                yref="paper",
                align="left",
                font=dict(size=12),
            )],
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        title="Interactive 3D Point Cloud",
    )

    fig.write_html("point_cloud_plot.html", include_plotlyjs=True, full_html=True, auto_open=False)

    if return_html:
        return "point_cloud_plot.html"
    else:
        return fig
    


