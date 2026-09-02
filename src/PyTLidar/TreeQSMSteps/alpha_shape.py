"""
Alpha shape area and volume of a point set: the Delaunay simplices whose
circumradius is under the alpha radius, with the circumradii computed for all
simplices at once.
"""
import numpy as np
from scipy.spatial import Delaunay


def _circumradius(points):
    """Circumradius of one simplex from the bordered system of its vertices."""
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


def _kept_simplices(coords, alpha_radius):
    """Delaunay simplices whose circumradius is under the alpha radius."""
    simplices = Delaunay(coords).simplices
    radii = _circumradii(coords, simplices)
    return simplices[radii < alpha_radius]


def alpha_area(points, alpha_radius):
    """
    Area of the 2D alpha shape with the given alpha radius, as the sum of the
    kept triangle areas, which is what MATLAB's alphaShape area returns.
    """
    coords = np.asarray(points, dtype=np.float64)[:, :2]
    if len(coords) < 3:
        return 0.0
    if len(coords) < 4 or alpha_radius <= 0:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(coords).volume)
    tri = coords[_kept_simplices(coords, alpha_radius)]
    if len(tri) == 0:
        return 0.0
    a = tri[:, 1, :] - tri[:, 0, :]
    b = tri[:, 2, :] - tri[:, 0, :]
    return float(0.5 * np.sum(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])))


def alpha_volume(points, alpha_radius):
    """
    Volume enclosed by the outer boundary of the 3D alpha shape with the given
    alpha radius. Interior voids are filled, as MATLAB's alphaShape does with
    its HoleThreshold when the tree data is computed.
    """
    coords = np.asarray(points, dtype=np.float64)[:, :3]
    if len(coords) < 4:
        return 0.0
    if alpha_radius <= 0:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(coords).volume)
    tets = _kept_simplices(coords, alpha_radius)
    if len(tets) == 0:
        return 0.0
    # the four faces of every tetrahedron, wound so the normal points away from it
    opp = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    faces = tets[:, opp].reshape(-1, 3)
    fourth = np.repeat(tets, 4, axis=0)
    fourth = fourth[np.arange(len(fourth)), np.tile([0, 1, 2, 3], len(tets))]
    p0 = coords[faces[:, 0]]
    n = np.cross(coords[faces[:, 1]] - p0, coords[faces[:, 2]] - p0)
    inward = np.einsum("ij,ij->i", n, coords[fourth] - p0) > 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    # boundary faces occur once (faces and edges keyed as single integers)
    npts = np.int64(len(coords))
    key = np.sort(faces, axis=1).astype(np.int64)
    key = (key[:, 0] * npts + key[:, 1]) * npts + key[:, 2]
    _, first, counts = np.unique(key, return_index=True, return_counts=True)
    boundary = faces[first[counts == 1]]
    if len(boundary) == 0:
        return 0.0
    # connected components of the boundary surface, joined along shared edges
    edges = np.sort(np.concatenate([boundary[:, [0, 1]], boundary[:, [1, 2]], boundary[:, [2, 0]]]), axis=1).astype(np.int64)
    edges = edges[:, 0] * npts + edges[:, 1]
    face_of_edge = np.tile(np.arange(len(boundary)), 3)
    _, inv = np.unique(edges, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    faces_sorted = face_of_edge[order]
    parent = np.arange(len(boundary))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    starts = np.flatnonzero(np.r_[True, inv_sorted[1:] != inv_sorted[:-1]])
    ends = np.r_[starts[1:], len(inv_sorted)]
    for s, e in zip(starts, ends):
        root = find(faces_sorted[s])
        for f in faces_sorted[s + 1:e]:
            parent[find(f)] = root
    comp = np.array([find(i) for i in range(len(boundary))])
    # signed volume of each closed component; voids come out negative and are filled
    a = coords[boundary[:, 0]]
    b = coords[boundary[:, 1]]
    c = coords[boundary[:, 2]]
    signed = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    per_comp = np.bincount(comp, weights=signed, minlength=len(boundary))
    return float(np.sum(per_comp[per_comp > 0]))
