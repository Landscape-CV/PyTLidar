"""
Alpha shape of a point set, built the same way as the alphashape package
(Delaunay simplices kept when their circumradius is under 1/alpha, the
boundary being the simplex faces that occur once). The circumradii are
computed for all simplices at once instead of one numpy call per simplex.
"""
import itertools
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, MultiLineString
from shapely.ops import polygonize, unary_union


def _circumradius(points):
    """Circumradius of one simplex, the way the alphashape package computes it."""
    num_rows = points.shape[0]
    A = np.block([[2 * np.dot(points, points.T), np.ones((num_rows, 1))],
                  [np.ones((1, num_rows)), np.zeros((1, 1))]])
    b = np.hstack((np.sum(points * points, axis=1), np.ones(1)))
    c = np.linalg.solve(A, b)[:-1]
    return np.linalg.norm(points[0, :] - np.dot(c, points))


def _circumradii(coords, simplices):
    """Circumradius of every simplex. NaN where the bordered system is singular."""
    pts = coords[simplices]
    n, N, K = pts.shape
    A = np.zeros((n, N + 1, N + 1))
    A[:, :N, :N] = 2 * np.matmul(pts, pts.transpose(0, 2, 1))
    A[:, :N, N] = 1.0
    A[:, N, :N] = 1.0
    b = np.ones((n, N + 1, 1))
    b[:, :N, 0] = np.sum(pts * pts, axis=2)
    try:
        c = np.linalg.solve(A, b)[:, :N, 0]
    except np.linalg.LinAlgError:
        # at least one degenerate simplex: do them one at a time so the others survive
        r = np.empty(n)
        for i in range(n):
            try:
                r[i] = _circumradius(pts[i])
            except np.linalg.LinAlgError:
                r[i] = np.nan
        return r
    d = pts[:, 0, :] - np.matmul(c[:, None, :], pts)[:, 0, :]
    return np.sqrt(np.sum(d * d, axis=1))


def alphashape(points, alpha):
    """
    Alpha shape (concave hull) of the points for the given alpha. Fewer than
    four points, or alpha <= 0, gives the convex hull. Returns a shapely
    geometry for 2D input and a trimesh mesh for 3D input.
    """
    if len(points) < 4 or alpha <= 0:
        if not isinstance(points, MultiPoint):
            points = MultiPoint(list(points))
        return points.convex_hull

    coords = np.array(points)
    simplices = Delaunay(coords).simplices
    radii = _circumradii(coords, simplices)
    keep = radii < 1.0 / alpha

    # Boundary faces are the ones met exactly once. The faces are kept as index
    # tuples in simplex vertex order, as in the alphashape package. The set
    # operations are written the same way as there too: the order the faces come
    # out of the set is the face order of the mesh, and the mesh volume is summed
    # in that order.
    edges = set()
    perimeter_edges = set()
    dim = coords.shape[-1]
    for point_indices in simplices[keep]:
        for edge in itertools.combinations(point_indices, r=dim):
            if edge not in edges:
                edges.add(edge)
                perimeter_edges.add(edge)
            else:
                perimeter_edges -= set(itertools.combinations(edge, r=len(edge)))

    if dim > 3:
        return perimeter_edges
    if dim == 3:
        import trimesh
        result = trimesh.Trimesh(vertices=coords, faces=list(perimeter_edges))
        trimesh.repair.fix_normals(result)
        return result

    m = MultiLineString([coords[np.array(edge)] for edge in perimeter_edges])
    return unary_union(list(polygonize(m)))
