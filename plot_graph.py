import networkx as nx
import numpy as np
import open3d as o3d

# 1. Build or load a NetworkX graph
G = nx.Graph()
G.add_edges_from([
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 2),
])

# 2. Get 3D positions for each node
# If you already have 3D coords, skip this and use your own dict {node: (x,y,z)}
pos = nx.spring_layout(G, dim=3, seed=0)  # 3D layout

# 3. Map nodes to indices and build point array
nodes = list(G.nodes())
node_index = {n: i for i, n in enumerate(nodes)}

points = np.array([pos[n] for n in nodes], dtype=float)  # shape (N, 3)

# 4. Build line index array from edges
lines = np.array([
    [node_index[u], node_index[v]]
    for u, v in G.edges()
], dtype=int)  # shape (M, 2)

# 5. Create Open3D LineSet
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines  = o3d.utility.Vector2iVector(lines)

# Optional: color edges
colors = np.tile(np.array([[0.2, 0.6, 1.0]]), (lines.shape[0], 1))
line_set.colors = o3d.utility.Vector3dVector(colors)

# 6. Visualize
o3d.visualization.draw_geometries([line_set])
