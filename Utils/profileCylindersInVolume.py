import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
import pyproj
from pathlib import Path
import vtk

# ═══════════════════════════════════════════════════════════════════════════════
#  Vectorised ray-AABB slab intersection  (all N cylinders at once)
# ═══════════════════════════════════════════════════════════════════════════════

def rays_aabb_intersection(origins, directions, box_min, box_max):
    """
    Vectorised slab method: intersect N rays against one AABB simultaneously.

    Parameters
    ----------
    origins    : (N, 3)  ray start points
    directions : (N, 3)  ray direction unit vectors
    box_min    : (3,)    AABB lower corner
    box_max    : (3,)    AABB upper corner

    Returns
    -------
    t_entry  : (N,)     entry t (0 if origin inside box); inf on miss
    t_exit   : (N,)     exit  t; -inf on miss
    normals  : (N, 3)   outward entry normal (zero vector if origin inside box)
    hit      : (N,)     boolean mask

    Algorithm
    ---------
    For each axis i, compute the two slab intersection ts:
        t = (box_bound - origin) / direction
    IEEE 754: division by zero gives ±inf, which propagates correctly through
    min/max — no special-casing needed for axis-aligned rays.
    Entry = max of per-axis near ts;  exit = min of per-axis far ts.
    Hit iff entry <= exit.
    """
    # Per-axis slab entry/exit for all rays at once
    with np.errstate(divide='ignore', invalid='ignore'):
        t0 = (box_min - origins) / directions   # (N, 3)
        t1 = (box_max - origins) / directions   # (N, 3)

    t_near = np.minimum(t0, t1)                 # (N, 3)  entry per axis
    t_far  = np.maximum(t0, t1)                 # (N, 3)  exit  per axis

    t_entry = np.maximum(t_near.max(axis=1), 0.)    # (N,)
    t_exit  = t_far.min(axis=1)                     # (N,)

    hit = t_entry <= t_exit                         # (N,)

    # Normal: axis where t_near == t_entry, sign opposes ray direction
    entry_axis = np.argmax(t_near == t_entry[:, None], axis=1)  # (N,)
    idx = np.arange(len(origins))
    normals = np.zeros_like(origins)
    normals[idx, entry_axis] = -np.sign(directions[idx, entry_axis])

    # Origins inside the box: entry at t=0, no meaningful normal
    inside = np.all((origins >= box_min) & (origins <= box_max), axis=1)
    t_entry[inside] = 0.0
    normals[inside] = 0.0

    return t_entry, t_exit, normals, hit


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def cylinderDiameterHistogramInBox(cylinders, aabb,
                                    bins=5, plot=True, verbose=True):
    """
    Build a diameter histogram for cylinders whose axis intersects the AABB,
    weighted by the length of the axis segment inside the box.

    Parameters
    ----------
    cylinders : (N, 8) ndarray, list of dicts, or CSV filepath string
                Each row: [sx, sy, sz, radius, ax, ay, az, length]
    aabb      : tuple  ((min_x,min_y,min_z), (max_x,max_y,max_z))
    bins      : int    histogram bins
    plot      : bool   show matplotlib histogram
    verbose   : bool   print results table

    Returns
    -------
    results   : list of dicts:
                    index, diameter, radius, inter_length,
                    t_entry, t_exit, intersects, normal
    hist      : (counts, bin_edges)  weighted by intersection length
    """

    if not isinstance(cylinders, np.ndarray):
        cylinders = np.array([[*c['start'], c['radius'], *c['axis'], c['length']]
                               for c in cylinders])

    box_min = np.asarray(aabb[0], float)
    box_max = np.asarray(aabb[1], float)

    origins = cylinders[:, 0:3]  # (N, 3)
    radii = cylinders[:, 3]  # (N,)
    directions = cylinders[:, 4:7]  # (N, 3)
    lengths = cylinders[:, 7]  # (N,)

    # ── All N cylinders in one vectorised call ────────────────────────────────
    t_entry, t_exit, normals, hit = rays_aabb_intersection(
        origins, directions, box_min, box_max
    )

    # Clamp to physical cylinder extent [0, length]
    t_entry = np.maximum(t_entry, 0.)
    t_exit = np.minimum(t_exit, lengths)
    inter_lengths = np.where(hit, np.maximum(0., t_exit - t_entry), 0.)

    # ── Assemble results ──────────────────────────────────────────────────────
    results = [
        {
            'index': i + 1,
            'diameter': radii[i] * 2,
            'radius': radii[i],
            'inter_length': inter_lengths[i],
            't_entry': t_entry[i] if hit[i] else None,
            't_exit': t_exit[i] if hit[i] else None,
            'intersects': bool(hit[i]) and inter_lengths[i] > 0,
            'normal': normals[i],
        }
        for i in range(len(cylinders))
    ]

    if verbose:
        print(f"\n{'Cyl':>4}  {'Radius':>8}  {'Intersects':>10}  "
              f"{'t_entry':>9}  {'t_exit':>9}  {'Inter. Length':>14}")
        print("─" * 64)
        for r in results:
            if r['intersects']:
                print(f"  {r['index']:>2}  {r['radius']:>8.4f}  {'YES':>10}  "
                      f"{r['t_entry']:>9.4f}  {r['t_exit']:>9.4f}  {r['inter_length']:>14.4f}")
            else:
                print(f"  {r['index']:>2}  {r['radius']:>8.4f}  {'NO':>10}  "
                      f"{'—':>9}  {'—':>9}  {'0':>14}")

    radiuses = [r['radius']     for r in results if r['intersects']]
    weights   = [r['inter_length'] for r in results if r['intersects']]

    if not radiuses:
        print("\nNo cylinders intersect the AABB.")
        return results, (np.array([]), np.array([]))

    hist = np.histogram(radiuses, bins=bins, weights=weights)

    if plot:
        counts, edges = hist
        centres = (edges[:-1] + edges[1:]) / 2
        widths  =  edges[1:]  - edges[:-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(centres, counts, width=widths * 0.85,
               color='steelblue', edgecolor='white', linewidth=0.8)
        ax.set_xlabel("Radius", fontsize=12)
        ax.set_ylabel("Total intersection length", fontsize=12)
        ax.set_title("Cylinder radius histogram (weighted by axis length inside AABB)",
                     fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('../results/cylinder_radius_histogram.png', dpi=150)
        plt.show()
        print("Histogram saved.")

    return results, hist

def mergeCylinderFiles(directory):
    cylList=[]
    for filename in os.listdir(directory):
        tmp=np.loadtxt(os.path.join(directory,filename))
        if tmp.ndim==2 and tmp.shape[1]==8:
            cylList.append(tmp)
    return np.vstack(cylList)
def getAABB(queryPpoint,AABBSize):
    minPoint=np.array(queryPpoint)-AABBSize/2
    maxPoint=np.array(queryPpoint)+AABBSize/2
    return np.array([minPoint,maxPoint])

def cylinders_to_polydata(cyls):
    """
    Convert (N, 8) cylinder array to a single pv.PolyData of line segments.

    Each cylinder becomes one line segment from start to end.
    Radius is stored as point data so .tube() can vary width per cylinder.

    Parameters
    ----------
    cyls : (N, 8) ndarray  [sx,sy,sz, radius, ax,ay,az, length]

    Returns
    -------
    pv.PolyData with lines and 'radius' point array
    """
    starts  = cyls[:, 0:3]
    radii   = cyls[:, 3]
    axes    = cyls[:, 4:7]
    lengths = cyls[:, 7]
    ends    = starts + axes * lengths[:, None]      # (N, 3)

    N = len(cyls)

    # Interleave start/end points: [s0, e0, s1, e1, ...]
    points = np.empty((N * 2, 3))
    points[0::2] = starts
    points[1::2] = ends

    # VTK line connectivity: [2, start_idx, end_idx] per segment
    lines = np.empty((N, 3), dtype=int)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(0, N * 2, 2)
    lines[:, 2] = np.arange(1, N * 2, 2)

    pd = pv.PolyData()
    pd.points = points
    pd.lines  = lines.ravel()

    # Repeat radius for start and end point of each segment
    pd.point_data['radius']   = np.repeat(radii, 2)
    pd.point_data['diameter'] = np.repeat(radii * 2, 2)

    return pd


def build_aabb_mesh(aabb):
    """
    Build a pv.Box wireframe from an aabb tuple.

    Parameters
    ----------
    aabb : tuple  ((min_x,min_y,min_z), (max_x,max_y,max_z))

    Returns
    -------
    pv.Box
    """
    (x0, y0, z0), (x1, y1, z1) = aabb
    return pv.Box(bounds=(x0, x1, y0, y1, z0, z1))

def get_satellite_tile(x_min, y_min, x_max, y_max,
                        utm_epsg  = 32617,
                        pad_m     = 0,
                        cache_path = 'satellite_tile.npz'):
    """
    Return satellite tile image + Web Mercator extent.
    Loads from cache_path if it exists, otherwise fetches from Esri and saves.

    Parameters
    ----------
    x_min, y_min, x_max, y_max : float   UTM bounding box in metres
    utm_epsg   : int    EPSG code of your CRS (default 32617 = UTM 17N Florida)
    pad_m      : float  padding in metres around the bounding box
    cache_path : str    path to save/load the cached .npz tile

    Returns
    -------
    img      : (H, W, 4) RGBA uint8 array
    extent   : (x0_wm, x1_wm, y0_wm, y1_wm) in Web Mercator metres
    tf       : pyproj Transformer (UTM → Web Mercator)
    """
    from pyproj import Transformer
    tf = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:3857", always_xy=True)

    cache = Path(cache_path)

    if cache.exists():
        print(f"[minimap] Loading tile from cache: {cache}")
        data   = np.load(cache)
        img    = data['img']
        extent = tuple(data['extent'])
        return img, extent, tf

    # Not cached — fetch from Esri
    print("[minimap] Fetching satellite tile from Esri (saved to cache after)...")
    try:
        import contextily as ctx

        x0_wm, y0_wm = tf.transform(x_min - pad_m, y_min - pad_m)
        x1_wm, y1_wm = tf.transform(x_max + pad_m, y_max + pad_m)

        img, extent = ctx.bounds2img(x0_wm, y0_wm, x1_wm, y1_wm,
                                      source=ctx.providers.Esri.WorldImagery)

        np.savez(cache, img=img, extent=np.array(extent))
        print(f"[minimap] Tile cached to {cache}  (shape={img.shape})")
        return img, extent, tf

    except Exception as e:
        print(f"[minimap] Fetch failed: {e}")
        return None, None, tf


def visualize(cyls, aabb,

              tube_sides   = 10,

              color_by     = 'hit',
              color_hit    = 'tomato',
              color_miss   = 'steelblue',
              aabb_color   = 'yellow',
              aabb_opacity = 0.15,
              utm_epsg     = 32617,
              center_utm  = (573160.68, 2840102.11),
              map_pad_m    = 40,
              tile_cache   = 'satellite_tile.npz',
              window_size  = (1400, 900)):
    """
    Interactive 3D view of cylinders + AABB with a satellite minimap.

    Parameters
    ----------
    cyls         : (N, 8) ndarray  [sx,sy,sz, radius, ax,ay,az, length]
    aabb         : tuple           ((min_x,min_y,min_z), (max_x,max_y,max_z))
    utm_epsg     : int    EPSG code of your coordinate system (default 32617)
    tube_sides   : int    tube cross-section resolution
                          use 6 for 150k+ cylinders, 10-16 for smaller sets
    color_by     : 'hit'      — intersecting=color_hit, miss=color_miss
                   'diameter' — colormap by diameter
                   'single'   — uniform color_hit for all cylinders
    color_hit    : color for intersecting cylinders
    color_miss   : color for non-intersecting cylinders
    aabb_color   : AABB wireframe and face color
    aabb_opacity : AABB surface opacity (0 = wireframe only)
    map_pad_m    : metres of padding around cylinder extent for the tile
    tile_cache   : path to the .npz cache file for the satellite tile
    window_size  : (width, height) in pixels
    """
    box_min     = np.asarray(aabb[0], float)
    box_max     = np.asarray(aabb[1], float)
    aabb_center = (box_min + box_max) / 2

    # ── Intersection detection ────────────────────────────────────────────────
    t_entry, t_exit, _, hit = rays_aabb_intersection(
        cyls[:, 0:3], cyls[:, 4:7], box_min, box_max
    )
    t_entry   = np.maximum(t_entry, 0.)
    t_exit    = np.minimum(t_exit,  cyls[:, 7])
    inter_len = np.where(hit, np.maximum(0., t_exit - t_entry), 0.)
    hit_mask  = inter_len > 0

    print(f"Cylinders total : {len(cyls):,}")
    print(f"Intersecting    : {hit_mask.sum():,}")
    print(f"Non-intersecting: {(~hit_mask).sum():,}")

    # ── Tube mesh ─────────────────────────────────────────────────────────────
    pd = cylinders_to_polydata(cyls)
    print(f"Generating tubes (n_sides={tube_sides})...")
    tubes = pd.tube(scalars='radius', absolute=True, n_sides=tube_sides)
    print(f"Tube mesh: {tubes.n_points:,} pts, {tubes.n_cells:,} cells")

    cells_per_cyl = max(1, tubes.n_cells // len(cyls))
    cyl_ids = np.repeat(np.arange(len(cyls)), cells_per_cyl)
    if len(cyl_ids) < tubes.n_cells:
        cyl_ids = np.append(cyl_ids,
                            np.full(tubes.n_cells - len(cyl_ids), len(cyls) - 1))
    cyl_ids = cyl_ids[:tubes.n_cells]

    # ── AABB ──────────────────────────────────────────────────────────────────
    aabb_surface   = build_aabb_mesh(aabb)
    aabb_wireframe = aabb_surface.extract_all_edges()

    # ── Satellite tile (cached) ───────────────────────────────────────────────
    cx,cy=center_utm
    starts=cyls[:,0:3]
    z_map = starts[:, 2].min() - 50

    img, extent_wm, tf = get_satellite_tile(
        cx-map_pad_m, cy-map_pad_m, cx+map_pad_m, cy+map_pad_m,
        utm_epsg=utm_epsg,
        cache_path=tile_cache
    )
    have_tile = img is not None

    if have_tile:
        x0, x1, y0, y1 = extent_wm
        map_plane = pv.Plane(
            center=((x0 + x1) / 2, (y0 + y1) / 2, z_map),
            direction=(0, 0, 1),
            i_size=x1 - x0,
            j_size=y1 - y0,
            i_resolution=1,
            j_resolution=1,
        )
        map_texture = pv.Texture(img)

        cx_map, cy_map = tf.transform(aabb_center[0], aabb_center[1])
    else:
        # No tile — plain grey plane centred on center_utm
        map_plane = pv.Plane(
            center=(cx, cy, z_map),
            direction=(0, 0, 1),
            i_size=map_pad_m * 2,
            j_size=map_pad_m * 2,
            i_resolution=1,
            j_resolution=1,
        )
        map_texture = None

        cx_map, cy_map = aabb_center[0], aabb_center[1]


    aabb_marker = pv.PolyData(np.array([[cx_map, cy_map, z_map + 2]]))
    # ── Plotter ───────────────────────────────────────────────────────────────────
    pl = pv.Plotter(window_size=window_size)
    pl.set_background('black')

    # ── Main 3D view (layer 0, full window) ───────────────────────────────────────
    if color_by == 'hit':
        tubes.cell_data['hit'] = hit_mask[cyl_ids].astype(float)
        pl.add_mesh(tubes, scalars='hit', clim=[0, 1],
                    cmap=[color_miss, color_hit], show_scalar_bar=False)
    elif color_by == 'diameter':
        tubes.cell_data['diameter'] = cyls[cyl_ids, 3] * 2
        pl.add_mesh(tubes, scalars='diameter', cmap='viridis',
                    scalar_bar_args={'title': 'Diameter (m)', 'color': 'white'})
    else:
        pl.add_mesh(tubes, color=color_hit)

    pl.add_mesh(aabb_surface, color=aabb_color, opacity=aabb_opacity)
    pl.add_mesh(aabb_wireframe, color=aabb_color, line_width=2)
    pl.add_axes(color='white')
    pl.add_text('3D View  |  red = intersects AABB  |  blue = miss',
                font_size=9, color='white', position='upper_left')
    pl.add_text(f"Query point: X {round(aabb_center[0],2)} Y {round(aabb_center[1],2)} Z {round(aabb_center[2],2)} ",
                font_size=9, color='white', position='upper_right')
    # ── Minimap (layer 1, bottom-left inset, overlaid on top) ────────────────────
    minimap = vtk.vtkRenderer()
    minimap.SetViewport(0.0, 0.0, 0.27, 0.27)  # (x_min, y_min, x_max, y_max) normalised
    minimap.SetLayer(1)  # draw on top of the main view
    minimap.SetBackground(0.08, 0.08, 0.08)

    pl.ren_win.SetNumberOfLayers(2)
    pl.ren_win.AddRenderer(minimap)

    def _add_to_minimap(mesh, **kwargs):
        """Wrap pv.Actor so we can add PyVista meshes to a raw vtkRenderer."""
        actor = pv.Actor(mapper=pv.DataSetMapper(dataset=mesh))
        for k, v in kwargs.items():
            setattr(actor.prop, k, v)
        minimap.AddActor(actor)

    if have_tile:
        # Textured plane needs a texture actor
        tex_actor = vtk.vtkActor()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(map_plane)
        tex_actor.SetMapper(mapper)
        tex_actor.SetTexture(map_texture)
        minimap.AddActor(tex_actor)
    else:
        _add_to_minimap(map_plane, color='darkgray', opacity=0.8)


    _add_to_minimap(aabb_marker, color='yellow', point_size=14,
                    render_points_as_spheres=True)

    # Top-down camera for minimap
    minimap.ResetCamera()
    cam = minimap.GetActiveCamera()
    cam.SetPosition(cx_map, cy_map, 1e6)
    cam.SetFocalPoint(cx_map, cy_map, z_map)
    cam.SetViewUp(0, 1, 0)
    cam.ParallelProjectionOn()
    minimap.ResetCameraClippingRange()
    pl.view_xy()
    pl.camera.up = (0, 1, 0)
    pl.show()

#cylinders=mergeCylinderFiles(r'../Dataset/EcomodelCylinders_3_2_2025')
# queryPoint=[5.731325400760621997e+05,2.840063605150410440e+06,-2.509709902458836694e+01]
queryPoint=[5.731031857598150382e+05,2.840123000221467111e+06,-1.848659356832329337e+01]
AABBSize=5
AABB=getAABB(queryPoint,AABBSize)
#visualize(cylinders,AABB)
cylinders=np.loadtxt(r"C:\Users\kaipo\Documents\Dev\Dev2\PyTLidar\results_treelearn\retile_573088_2840115_1_0\retile_573088_2840115_1_0_cylinders.txt")
#cylinders=np.loadtxt(r"..\results\tile_573150_2840110.laz_cylinders.txt")
#cylinders=np.loadtxt(r"..\results\tile_573150_2840110.laz_cylinders_debug.txt")
#AABB=np.array([[-3,-3,-3],[2,2,2]])
visualize(cylinders,AABB,10)
# results,histogram=cylinderDiameterHistogramInBox(cylinders,AABB)
# #
# AABBCenter=[AABB[0][0]+(AABB[1][0]-AABB[0][0])/2,AABB[0][1]+(AABB[1][1]-AABB[0][1])/2,AABB[0][2]+(AABB[1][2]-AABB[0][2])/2]
# mesh = pv.Cube(center=AABBCenter,x_length=AABB[1][0]-AABB[0][0], y_length=AABB[1][1]-AABB[0][1], z_length=AABB[1][2]-AABB[0][2])
# pl = pv.Plotter()
# actor = pl.add_mesh(mesh, color='red', style='wireframe', line_width=4)
# points=[]
# labels=[]
# for i,cyl in enumerate(cylinders):
#     dir = cyl[4:7]
#     r = cyl[3]
#     len = cyl[7]
#     ori = cyl[0:3]
#     cent = ori + dir * len / 2
#     points.append(cent)
#     labels.append(results[i]['index'])
#     if results[i]['intersects']:
#         actor = pl.add_mesh(pv.Cylinder(center=cent,radius=r,direction=dir,height=len), color='green')
#     else:
#         actor = pl.add_mesh(pv.Cylinder(center=cent, radius=r, direction=dir, height=len), color='red')
# pl.add_axes_at_origin()
# pl.show()
