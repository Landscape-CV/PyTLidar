import numpy as np
import plotly.graph_objects as go
import matplotlib
import random

def rotation_matrix_from_z(new_z):
    """Compute the rotation matrix to align the z-axis with the given direction vector."""
    new_z = new_z / np.linalg.norm(new_z)
    up = np.array([0.0, 0.0, 1.0])
    if np.allclose(new_z, up) or np.allclose(new_z, -up):
        up = np.array([1.0, 0.0, 0.0])
    new_x = np.cross(up, new_z)
    new_x = new_x / np.linalg.norm(new_x)
    new_y = np.cross(new_z, new_x)
    new_y = new_y / np.linalg.norm(new_y)
    return np.column_stack([new_x, new_y, new_z])


def create_cylinder(start, axis, radius, height, num_points):
    """
    Create a 3D cylinder using Plotly with specified start point, axis, and radius.

    Parameters:
        start (list or np.array): Array of starting points [x, y, z] of the cylinder.
        axis (list or np.array): Array of axis vectors [dx, dy, dz] defining the direction and length.
        radius (float): Array of radius of the cylinder.
        height (float): Array of height of the cylinder.
        num_points (int): Number of points to sample around the circumference and along the height.

    Returns:
        plotly.graph_objects.Figure: Figure containing the cylinder visualization.
    """
    # Compute cylinder height and direction
    #height = np.linalg.norm(axis)
    if height == 0:
        raise ValueError("Axis vector cannot be zero length.")
    direction = axis / height

    # Generate parametric grids for the surfaces
    theta = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, height, num_points)
    theta_grid, v_grid = np.meshgrid(theta, v)

    # Side surface
    x_side = radius * np.cos(theta_grid)
    y_side = radius * np.sin(theta_grid)
    z_side = v_grid

    # Bottom disk
    r_bottom = np.linspace(0, radius, num_points)
    theta_bottom = np.linspace(0, 2 * np.pi, num_points)
    r_grid_bottom, theta_grid_bottom = np.meshgrid(r_bottom, theta_bottom)
    x_bottom = r_grid_bottom * np.cos(theta_grid_bottom)
    y_bottom = r_grid_bottom * np.sin(theta_grid_bottom)
    z_bottom = np.zeros_like(x_bottom)

    # Top disk
    x_top = r_grid_bottom * np.cos(theta_grid_bottom)
    y_top = r_grid_bottom * np.sin(theta_grid_bottom)
    z_top = height * np.ones_like(x_top)

    # Combine all surfaces
    surfaces = [
        (x_side, y_side, z_side),
        (x_bottom, y_bottom, z_bottom),
        (x_top, y_top, z_top)
    ]

    return surfaces


def cylinders_plotting(cylinders, num_points=10,base_fig=None):
    """
    Plot multiple cylinders in a single 3D figure.

    Parameters:
        cylinders (list of dicts): Each dict contains:
            - 'start' (list/np.array): [x, y, z] start point.
            - 'axis' (list/np.array): [dx, dy, dz] direction vector.
            - 'radius' (float): Radius.
            - 'height' (float): Height.
        num_points (int): Discretization points for surfaces.

    Returns:
        plotly.graph_objects.Figure: Figure with all cylinders.
    """
    if base_fig is not None:
        fig = base_fig
    else:
        fig = go.Figure()

    for i in range(np.size(cylinders['radius'])):
        start = np.array(cylinders['start'][i])
        axis = np.array(cylinders['axis'][i])
        radius = cylinders['radius'][i]
        length = cylinders['length'][i]
        colors_dict = dict(matplotlib.colors.cnames.items())
        hex_colors = list(colors_dict.values())
        color = random.choice(hex_colors)
        # color = 'blue' # Default color

        # Generate cylinder surfaces
        surfaces = create_cylinder(start, axis, radius, length, num_points)

        # Rotate and translate each surface
        rot_matrix = rotation_matrix_from_z(axis / np.linalg.norm(axis))

        for x, y, z in surfaces:
            points = np.stack([x, y, z], axis=-1)
            rotated_points = np.dot(points, rot_matrix.T)
            translated_points = rotated_points + start

            x_t = translated_points[..., 0]
            y_t = translated_points[..., 1]
            z_t = translated_points[..., 2]

            fig.add_trace(go.Surface(
                x=x_t, y=y_t, z=z_t,
                colorscale=[[0, color], [1, color]],  # Uniform color
                showscale=False,
                opacity=0.8,

            ))

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    html_file = f"cylinder_plot_volume_.html"
    fig.write_html(html_file, include_plotlyjs=True, full_html=True)
    return fig


# Example Usage
if __name__ == "__main__":
    # Define multiple cylinders (each with start, axis, radius, optional color)
    cylinders = {
        'start': [
        [0, 0, 0],  # Cylinder 1 start
        [0, 0, 0],  # Cylinder 2 start
        [0, 0, 0],  # Cylinder 3 start
        [2, 2, 2]  # Cylinder 4 start
    ],

    'axis': [
        [3, 0, 0],  # Cylinder 1 axis (along x)
        [0, 3, 0],  # Cylinder 2 axis (along y)
        [0, 0, 3],  # Cylinder 3 axis (along z)
        [1, 1, 1]  # Cylinder 4 axis (diagonal)
    ],

    'radius': [1.0, 0.8, 0.6, 0.5],  # Radii for each cylinder

    'length': [1.0, 2.0, 3.0, 4.0]  # Colors for each cylinder
    }
    
    fig = cylinders_plotting(cylinders)
    #fig.show()
