import pyvista as pv
from pyvista import examples
sphere = pv.Sphere(center=(0, 0, 1))
cube = pv.Cube()
plotter = pv.Plotter()
_ = plotter.add_mesh(
    sphere, color='grey', smooth_shading=True, label='Sphere'
)
_ = plotter.add_mesh(cube, color='r', label='Cube')
_ = plotter.add_legend(bcolor='w', face=None)
plotter.show()