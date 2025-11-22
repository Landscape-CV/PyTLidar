import json
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from .Utils import load_point_cloud

class ResultsPlotter:
    """
    Class contains methods to plot cylinder results using pyvista

    Usage: 
    >>> results = ResultsPlotter()
    >>> mean = np.array([ 573155.38910263, 2840116.02216773 ,-24.5])
    >>> results.add_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\tree_1_0.0_no_leaves_dtm.xyz", mean)
    >>> results.add_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\tree_1_1.0_no_leaves_dtm.xyz",  mean)
    >>> results.add_cylinders(r"results/File_first_link_stroud.txt", mean)
    >>> results.show() 
    """
    def __init__(self, grid_center, grid=True, terrain_style=True, parallel_projection=True, legend=True):
        self.plotter = pv.Plotter()

        # Create grid for reference to ground plane.
        if grid:
            grid = pv.Plane(center= grid_center, i_size=10, j_size=10, i_resolution=10, j_resolution=10)
            self.plotter.add_mesh(grid, color='lightgray', style='wireframe')

        # Camera settings:
        if terrain_style:
            self.plotter.enable_terrain_style()
        
        if parallel_projection:
            self.plotter.camera.SetParallelProjection(False)

        # Create Legend:
        if legend:
            top = 100
            left = 800
            self.plotter.add_text("Legend:", position=(left, top), font_size=12)
            self.plotter.add_text('Twig (r=0-2cm)', color= 'r', position=(left, top-20), font_size=12)
            self.plotter.add_text('Small Branch (r=2-5cm)', color= 'purple', position=(left, top-40), font_size=12)
            self.plotter.add_text('Medium Branch (r=5-10cm)', color= 'b', position=(left, top-60), font_size=12)
            self.plotter.add_text('Large Branch (r=10+cm)', color= 'orange', position=(left, top-80), font_size=12)

            self.plotter.add_axes(interactive=True)

    def _get_cylinder_mesh(self, cylinder_start, radius, axis, length):
        """
        returns mesh from ecomodel data
        
        Get the center from the start, length and axis. 
        """
        center = cylinder_start + (length/2) * axis
        mesh = pv.Cylinder(center=center, direction=axis, radius=radius[0], height=length[0])
        return mesh
    
    def _get_class(self, radius):
        """
        returns color and a string. 

        Group 1: 0-2cm
        Group 2: 2-5cm
        Group 3: 5-10cm
        Groups 4: 10+cm
        """

        if 0 < radius <= 0.02:
            return ('r', 'Twig (0-2cm)')
        elif 0.02 < radius <= 0.05:
            return ('purple', 'Small Branch (2-5cm)')
        elif 0.05 < radius <= 0.10:
            return( 'b', 'Medium Branch (5-10cm)')
        else:
            return( 'orange', 'Large Branch (10+cm)')
        
    def add_point_cloud_file(self, path_to_point_cloud, mean):
        """
        Adds a point cloud to the scene

        Args:
            path_to_point_cloud (str): path to point cloud file
            mean (np.array): Mean from ecomodel results to adjust point cloud
        """
        xyz, data = load_point_cloud(path_to_point_cloud, full_data=True)
        data = data[:, 3]
        xyz = xyz - mean
        pc = pv.PolyData(xyz)
        pc['intensity'] = data
        self.plotter.add_mesh(pc, scalars="intensity", point_size=2, cmap="viridis")

    def add_point_cloud_np(self, point_cloud_with_intensity, mean):
        """
        Adds a point cloud to the scene
        """
        xyz = point_cloud_with_intensity[:, :3]
        xyz = xyz - mean
        pc = pv.PolyData(xyz)
        pc['intensity'] = point_cloud_with_intensity[:, 3]
        self.plotter.add_mesh(pc, scalars="intensity", point_size=2, cmap="viridis")



    def add_cylinders(self, path_to_cylinder_data_file, mean):
        """
        Plots cylinders from ecomodel results

        Args:
            path_to_cylinder_data_file (str): path to cylinder data file
            mean (np.array): Mean from ecomodel results to adjust cylinders
        """
        cylinders = np.loadtxt(path_to_cylinder_data_file)
        cylinders[:, 0:3] = cylinders[:, 0:3] - mean 

        print("Total Number of Cylinders:", cylinders.shape[0])
        
        for cylinder_row in range(0, cylinders.shape[0]):
            cylinder_data = cylinders[cylinder_row]

            start = cylinder_data[0:3]
            radius = cylinder_data[3:4]
            axis = cylinder_data[4:7]
            length = cylinder_data[7:8]

            mesh = self._get_cylinder_mesh(start, radius, axis, length)
            
            color, label = self._get_class(radius)

            self.plotter.add_mesh(mesh, color=color, opacity=0.5)


    def add_box_coordinate(self, coordinate, box_size):
        """
        Plots a red box surrounding a point of interest

        Args:
            coordinate (np.array): coordinate of interest
            box_size (float): size of the box

        """
        box_mesh = pv.Cube(center = coordinate, x_length=box_size, y_length=box_size, z_length=box_size)
        self.plotter.add_mesh(box_mesh, style='wireframe', color='r', line_width=2)
        mesh = pv.Sphere(radius=0.05, center=coordinate)
        self.plotter.add_mesh(mesh, color='r')
        self.plotter.add_point_labels([coordinate], ["Lizard"], point_size=5, always_visible=True)

    def show(self):
        """
        Shows the plot
        """
        self.plotter.show()


class Datamaker:
    """
    Class which can be used to evaluate the number of cylinders in a voxel or create histograms. 
    """
    def __init__(self, data_file, mean):
        self.cylinders = np.loadtxt(data_file)
        print(mean)
        print("Before", self.cylinders[0, 0:3])
        self.cylinders[:, 0:3] = self.cylinders[:, 0:3] - mean 
        print("After", self.cylinders[0, 0:3])
        np.savetxt("mean_adjusted_cylinders.txt", self.cylinders)
        self.plotter = ResultsPlotter()

        self.mean = mean
        print(f"Loaded cylinders shape: {self.cylinders.shape}")
        print(f"First cylinder: {self.cylinders[0] if len(self.cylinders) > 0 else 'EMPTY'}")

    def get_class(self, radius):
        # Group 1: 0-2cm
        # Group 2: 2-5cm
        # Group 3: 5-10cm
        # Groups 4: 10+cm

        if 0 < radius <= 0.02:
            return "twig"
        elif 0.02 < radius <= 0.05:
            return "small"
        elif 0.05 < radius <= 0.10:
            return "med"
        else:
            return "large"

    def evaluate_coordinate(self, coordinate, voxel_size):

        # coordinate = np.array(coordinate) - np.array([ 5.73124678e+05,  2.84012414e+06, 0])

        half_voxel = voxel_size / 2
        x = coordinate[0]
        y = coordinate[1]
        z = coordinate[2]

        cube_min = np.array([x - half_voxel, y - half_voxel, z - half_voxel])
        cube_max = np.array([x + half_voxel, y + half_voxel, z + half_voxel])
        
        mask = np.all((self.cylinders[:,0:3] >= cube_min) & (self.cylinders[:,0:3] <= cube_max), axis=1)
        
        # print(mask.shape)
        # print(self.cylinders.shape)
        cylinder_data = self.cylinders
        self.plotter.add_cylinders(cylinder_data, coord_if_interest=np.array(coordinate), box_size=voxel_size)
        # Add point cloud to the visualization
        self.plotter.add_point_cloud_file(r"G:\Projects\TreeCanopyLidar\PyTLidar\Dataset\Tiles\tile_573150_2840110.laz", self.mean)
        self.plotter.show()
        
        
        # cylinder_data = self.cylinders[mask]
        #print(cylinder_data.shape)

        # self.plotter.plot(cylinder_data, coord_if_interest=np.array(coordinate), box_size=voxel_size)

        # self.plotter.show()
        # self.create_histogram(cylinder_data)

    def plot_point_cloud(self, plotter):
        """
        Plots point cloud alongside cylinders.
        """
        xyz, data = load_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\Dataset\Tiles\tile_573110_2840090.laz", full_data=True)
        data = data[:, 3]
        print(data.shape)

        xyz = xyz - self.mean

        plotter.plot(
            xyz,
            scalars=data,
            render_points_as_spheres=False,
            point_size=1,
            show_scalar_bar=True,
        )

    def create_histogram(self, cylinder_data):
        
        data = {"twig": 0, "small": 0, "med": 0, "large": 0}
        for idx in range(cylinder_data.shape[0]):
            
            radius = cylinder_data[idx, 3]
            branch_class =  self.get_class(radius)
            data[branch_class] += 1

        with open("Branch_out_classes.json", "w") as f:
            json.dump(data, f, indent=4)

        
        labels = list(data.keys())
        counts = list(data.values())

        # Plot as a bar chart (histogram-like)
        plt.bar(labels, counts, color='blue', edgecolor='black')


        # Add labels and title
        plt.xlabel('Branch Size')
        plt.ylabel('Frequency')
        plt.title('Area Branch Size')

        # Show plot
        plt.show()
        


    def filter_inside(self, bounding_box, line):
        """
        Returns subset of cylinders that are inside the box using the endpoints
        """



if __name__ == "__main__":
    # cyl_plotter = CylinderPlotter()
    # cyl_plotter.plot()

    # Line intersecting box code. would need to make some better python version or use that dudes. 
    # https://stackoverflow.com/questions/3235385/given-a-bounding-box-and-a-line-two-points-determine-if-the-line-intersects-t
    # [ 573135.74006211 2840102.76752775       0.        ]

    results = ResultsPlotter(parallel_projection=False)
    mean = np.array([ 573155.38910263, 2840116.02216773 ,-24.5])
    results.add_point_cloud_file(r"G:\Projects\TreeCanopyLidar\PyTLidar\tree_1_0.0_no_leaves_dtm.xyz", mean)
    results.add_point_cloud_file(r"G:\Projects\TreeCanopyLidar\PyTLidar\tree_1_1.0_no_leaves_dtm.xyz",  mean)
    results.add_cylinders(r"results/File_first_link_stroud.txt", mean)
    results.show()


    # datamaker = Datamaker("results/File_first_link_stroud.txt", np.array([ 573155.38910263, 2840116.02216773 ,      24]))
    # # # datamaker.evaluate_coordinate((0.442, -1.278, 0.58), 1)
    # datamaker.evaluate_coordinate((0.542, -2.278, 1.5), 1)



        # xyz, data = load_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\tree_1_0.0_no_leaves_dtm.xyz", full_data=True)
        # xyz2, data2 = load_point_cloud(r"G:\Projects\TreeCanopyLidar\PyTLidar\tree_1_1.0_no_leaves_dtm.xyz", full_data=True)