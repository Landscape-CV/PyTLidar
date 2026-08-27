"""Conversions from PyTLidar QSM results to CloudCompare objects."""

import sys

import numpy as np
import pycc

from .mesh import cylinder_mesh

FACETS = 8


def cloud_from_cc(cc_cloud):
    """Nx3 float64 array from a ccPointCloud."""
    return cc_cloud.points().astype(np.float64)


def results_to_cc(cc, source, qsm, offset, label=None):
    """Add one QSM to the scene and print the treedata summary.

    offset undoes the plugin's mean centring, added in float64 before the
    float32 cast. label distinguishes the groups of a multi-run job.
    """
    cylinder = qsm["cylinder"]
    starts = np.asarray(cylinder["start"], dtype=np.float64) + offset

    name = f"PyTLidar - {source.getName()}"
    if label:
        name = f"{name} ({label})"
    group = pycc.ccHObject(name)
    group.addChild(_cylinder_cloud(cylinder, starts, source))
    if starts.shape[0] > 0:
        group.addChild(_cylinder_ccmesh(cylinder, starts, source))
    tri = qsm.get("triangulation")
    if isinstance(tri, dict) and np.size(tri.get("vert", [])) > 0:
        group.addChild(_stem_ccmesh(tri, offset, source))
    cc.addToDB(group)
    cc.updateUI()
    _print_tree_data(qsm["treedata"], name)


def _make_cloud(name, xyz, source):
    cloud = pycc.ccPointCloud(np.ascontiguousarray(xyz[:, 0], dtype=np.float32),
                              np.ascontiguousarray(xyz[:, 1], dtype=np.float32),
                              np.ascontiguousarray(xyz[:, 2], dtype=np.float32))
    cloud.setName(name)
    cloud.copyGlobalShiftAndScale(source)
    return cloud


def _add_scalar_field(cloud, name, values):
    idx = cloud.addScalarField(name)
    sf = cloud.getScalarField(idx)
    sf.asArray()[:] = np.asarray(values, dtype=np.float32).reshape(-1)
    sf.computeMinAndMax()
    return idx


def _cylinder_cloud(cylinder, starts, source):
    """Cylinder start points with the per-cylinder attributes as scalar
    fields, the filterable view of the model."""
    cloud = _make_cloud("cylinders", starts, source)
    n = starts.shape[0]
    if n == 0:
        return cloud

    for name, key in (("radius", "radius"), ("length", "length"),
                      ("branch_order", "BranchOrder"), ("branch_id", "branch"),
                      ("surface_coverage", "SurfCov")):
        idx = _add_scalar_field(cloud, name, cylinder[key])
        if name == "radius":
            cloud.setCurrentDisplayedScalarField(idx)
    cloud.showSF(True)
    return cloud


def _cylinder_ccmesh(cylinder, starts, source):
    """Merged triangle mesh of all cylinders, the visual view of the model.
    branch_order rides on the vertices so the mesh can be coloured by it."""
    vertices, triangles, vertex_cylinder = cylinder_mesh(
        starts,
        np.asarray(cylinder["axis"], dtype=np.float64),
        np.asarray(cylinder["length"], dtype=np.float64),
        np.asarray(cylinder["radius"], dtype=np.float64),
        facets=FACETS,
    )
    vcloud = _make_cloud("vertices", vertices, source)
    idx = _add_scalar_field(vcloud, "branch_order",
                            np.asarray(cylinder["BranchOrder"])[vertex_cylinder])
    vcloud.setCurrentDisplayedScalarField(idx)

    mesh = pycc.ccMesh(vcloud)
    mesh.setName("cylinder mesh")
    if hasattr(mesh, "reserve"):
        # Some pycc builds require the triangle capacity up front.
        mesh.reserve(triangles.shape[0])
    for a, b, c in triangles:
        mesh.addTriangle(int(a), int(b), int(c))
    mesh.addChild(vcloud)
    return mesh


def _stem_ccmesh(tri, offset, source):
    """The stem triangulation (inputs Tria) as its own mesh, in the same frame
    as the cylinders."""
    vertices = np.asarray(tri["vert"], dtype=np.float64) + offset
    facets = np.asarray(tri["facet"], dtype=np.int64)
    vcloud = _make_cloud("stem vertices", vertices, source)
    mesh = pycc.ccMesh(vcloud)
    mesh.setName("stem mesh")
    if hasattr(mesh, "reserve"):
        mesh.reserve(facets.shape[0])
    for a, b, c in facets:
        mesh.addTriangle(int(a), int(b), int(c))
    mesh.addChild(vcloud)
    return mesh


def _print_tree_data(treedata, name):
    """Tree metrics to the CloudCompare console. tree_data casts every value
    to float32, so the counts need an explicit int."""
    lines = [
        f"--- {name} ---",
        f"  Total volume:     {float(treedata.get('TotalVolume', 0)):.3f} L",
        f"  Trunk volume:     {float(treedata.get('TrunkVolume', 0)):.3f} L",
        f"  Branch volume:    {float(treedata.get('BranchVolume', 0)):.3f} L",
        f"  Tree height:      {float(treedata.get('TreeHeight', 0)):.2f} m",
        f"  Trunk length:     {float(treedata.get('TrunkLength', 0)):.2f} m",
        f"  Branch length:    {float(treedata.get('BranchLength', 0)):.2f} m",
        f"  DBH (QSM):        {float(treedata.get('DBHqsm', 0)):.4f} m",
        f"  DBH (cylinder):   {float(treedata.get('DBHcyl', 0)):.4f} m",
        f"  Branches:         {int(treedata.get('NumberBranches', 0))}",
        f"  Max branch order: {int(treedata.get('MaxBranchOrder', 0))}",
        "---",
    ]
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
