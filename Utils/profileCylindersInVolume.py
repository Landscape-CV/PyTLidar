import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

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


#cylinders=np.loadtxt(r"..\results\testCylinders.txt")
cylinders=np.loadtxt(r"..\results\tile_573150_2840110.laz_cylinders.txt")
#cylinders=np.loadtxt(r"..\results\tile_573150_2840110.laz_cylinders_debug.txt")
AABB=np.array([[-3,-3,-3],[2,2,2]])
results,histogram=cylinderDiameterHistogramInBox(cylinders,AABB)

AABBCenter=[AABB[0][0]+(AABB[1][0]-AABB[0][0])/2,AABB[0][1]+(AABB[1][1]-AABB[0][1])/2,AABB[0][2]+(AABB[1][2]-AABB[0][2])/2]
mesh = pv.Cube(center=AABBCenter,x_length=AABB[1][0]-AABB[0][0], y_length=AABB[1][1]-AABB[0][1], z_length=AABB[1][2]-AABB[0][2])
pl = pv.Plotter()
actor = pl.add_mesh(mesh, color='red', style='wireframe', line_width=4)
points=[]
labels=[]
for i,cyl in enumerate(cylinders):
    dir = cyl[4:7]
    r = cyl[3]
    len = cyl[7]
    ori = cyl[0:3]
    cent = ori + dir * len / 2
    points.append(cent)
    labels.append(results[i]['index'])
    if results[i]['intersects']:
        actor = pl.add_mesh(pv.Cylinder(center=cent,radius=r,direction=dir,height=len), color='green')
    else:
        actor = pl.add_mesh(pv.Cylinder(center=cent, radius=r, direction=dir, height=len), color='red')
# actor = pl.add_point_labels(
#     np.array(points),
#     labels,
#     italic=True,
#     font_size=10,
#     point_color='red',
#     point_size=10,
#     render_points_as_spheres=True,
#     always_visible=True,
#     shadow=True,
# )
pl.add_axes_at_origin()
pl.show()