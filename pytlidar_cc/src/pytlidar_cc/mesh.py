"""Triangle mesh of a QSM's cylinders.

Pure numpy so it can be tested without CloudCompare. Each cylinder becomes a
ring of `facets` vertices at both ends and 2 * facets side triangles; the
rings are oriented by an orthonormal frame built from the cylinder axis.
"""

import numpy as np


def cylinder_mesh(starts, axes, lengths, radii, facets=8):
    """Mesh the cylinders as one vertex/triangle soup.

    starts, axes: (n, 3). lengths, radii: (n,). Returns (vertices, triangles,
    vertex_cylinder): vertices is (n * 2 * facets, 3) float64, triangles is
    (n * 2 * facets, 3) int 0-based indices into vertices, vertex_cylinder
    maps every vertex to the index of the cylinder it belongs to, for
    carrying per-cylinder values onto the mesh as scalar fields.
    """
    starts = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
    axes = np.asarray(axes, dtype=np.float64).reshape(-1, 3)
    lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    n = starts.shape[0]
    if n == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                np.zeros(0, dtype=np.int64))

    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = np.where(norms > 1e-12, axes / np.where(norms == 0, 1.0, norms),
                    [[0.0, 0.0, 1.0]])

    # Frame per cylinder: u, v span the plane of the end rings.
    ref = np.where(np.abs(axes[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]],
                   [[1.0, 0.0, 0.0]])
    u = np.cross(axes, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(axes, u)

    theta = 2.0 * np.pi * np.arange(facets) / facets
    ring = (u[:, None, :] * np.cos(theta)[None, :, None]
            + v[:, None, :] * np.sin(theta)[None, :, None]) * radii[:, None, None]
    base = starts[:, None, :] + ring
    top = (starts + axes * lengths[:, None])[:, None, :] + ring
    vertices = np.concatenate([base, top], axis=1).reshape(-1, 3)

    # Two triangles per side quad, indices local to one cylinder's block of
    # 2 * facets vertices (base ring first, then top ring).
    k = np.arange(facets)
    kn = (k + 1) % facets
    quad = np.concatenate([np.stack([k, kn, k + facets], axis=1),
                           np.stack([kn, kn + facets, k + facets], axis=1)])
    triangles = (quad[None, :, :]
                 + (np.arange(n) * 2 * facets)[:, None, None]).reshape(-1, 3)

    vertex_cylinder = np.repeat(np.arange(n), 2 * facets)
    return vertices, triangles, vertex_cylinder
