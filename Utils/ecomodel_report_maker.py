"""
Ecomodel Report Maker. 

This script generates a PDF report for the Ecomodel results. It processes the original point cloud data, 
the leaves removed point cloud, and the cylinders data to create visualizations and compile them into a report. 
The report includes the original point cloud, the leaves removed point cloud, the leaves removed with cylinders, 
and just the cylinders for each tile. The script uses the ReportLab library to create the PDF and the 
ResultsPlotter class to generate images from the point cloud data.

Usage:
1. Run Ecomodel_lite to generate the results in the specified results folder.
2. Update the `results_folder` and `original_folder` variables in the script to point
    to the correct directories.
3. Run this script to generate the PDF report.
"""


from reportlab.lib.pagesizes import letter, landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from Utils.plot_tools import ResultsPlotter
import numpy as np
from pathlib import Path
import os
from Utils.Utils import load_point_cloud



class ReportMaker:
    def __init__(self, filename):
        self.canvas = canvas.Canvas(filename, pagesize=landscape(A4))

    def add_orignal_point_cloud(self, point_cloud_image):
        self.canvas.drawImage(point_cloud_image, 0, 50, width=400, preserveAspectRatio=True)

    def add_point_cloud_leaves_removed(self, point_cloud_image):
        self.canvas.drawImage(point_cloud_image, 350, 50, width=400, preserveAspectRatio=True)

    def add_point_cloud_leaves_removed_cylinders(self, point_cloud_image):
        self.canvas.drawImage(point_cloud_image,0, -250, width=400, preserveAspectRatio=True)
    
    def add_just_cylinders(self, point_cloud_image):
        self.canvas.drawImage(point_cloud_image,350, -250, width=400, preserveAspectRatio=True)
        
    def save(self):
        self.canvas.save()

if __name__ == "__main__":
    results_folder = "results_lite_rush"
    original_folder = r"G:\Projects\TreeCanopyLidar\Datasets\Rush7\Tiled_better"
    report = ReportMaker("Ecomodel_Results_Rush2.pdf")
        
    for file in os.listdir(original_folder):
        laz_file = Path(original_folder, file)
        if laz_file.stem == "complete":
            continue
        data_present = False
        
        for tile_folder in os.listdir(results_folder):
            # print(tile_folder)
            folder_path = Path(results_folder, tile_folder)
            if laz_file.stem in tile_folder:
                print(f"Found matching folder for tile {laz_file.stem}: {tile_folder}")
                data_present = True
                break
        
        tile_name = laz_file.stem

        print(folder_path)
        if data_present:
            print(os.listdir(str(folder_path)))
            for data_file in os.listdir(str(folder_path)):
                print(data_file)
                data_path = f"{folder_path}/{data_file}"
                if "cylinders" in data_file:
                    cylinders_data = np.loadtxt(data_path)

                if "data" in data_file:
                    with open(data_path, 'r') as f:
                        mean_string = f.readline()
                        ground_z = float(f.readline())
                        actual_mean = np.array([float(mean_string.split(" ")[0]), float(mean_string.split(" ")[1]), float(mean_string.split(" ")[2])])

                if "leavesremoved" in data_file.split("_")[-1]:
                    # print(data_path)
                    try:
                        _, leaves_removed = load_point_cloud(data_path, full_data=True)
                    except Exception as e:
                        print(f"Error loading leaves removed data from {data_path}: {e}")
                        leaves_removed = np.array([])  # Set to empty array if loading fails
                    print("loaded removed.")

            
            
            # Normalize data for plotting. 
            if cylinders_data.size == 0  or leaves_removed.size == 0:
                ground_z = 0
                pc, pcdata = load_point_cloud(str(laz_file), full_data=True)
                actual_mean = np.mean(pcdata[:, :3], axis=0)
            else: 
                leaves_removed[:, :3] = leaves_removed[:, :3] - actual_mean
                cylinders_data[:,:3] = cylinders_data[:,:3] - actual_mean   
                # Only Leaves Removed
                plotter = ResultsPlotter(np.array([0,0,ground_z]), legend=False, off_screen=True)
                plotter.add_point_cloud_np_intensity(leaves_removed)
                path = plotter.get_image(f"{tile_name}_wood.jpg")
                report.add_point_cloud_leaves_removed(path)

                # Leaves Removed with Cylinders:
                plotter = ResultsPlotter(np.array([0,0,ground_z]), legend=False, off_screen=True)
                plotter.add_point_cloud_np_intensity(leaves_removed)
                plotter.add_cylinders(cylinders_data)
                path = plotter.get_image(f"{laz_file.stem}_brances_cylinders.jpg")
                report.add_point_cloud_leaves_removed_cylinders(path)

                # Just Cylinders
                plotter = ResultsPlotter(np.array([0,0,ground_z]), legend=False, off_screen=True)
                plotter.add_cylinders(cylinders_data)
                path = plotter.get_image(f"{laz_file.stem}_cylinders.jpg")
                report.add_just_cylinders(path)

        else:
            ground_z = 0
            pc, pcdata = load_point_cloud(str(laz_file), full_data=True)
            actual_mean = np.mean(pcdata[:, :3], axis=0)
            # print(actual_mean)
 
        # Add original Point Cloud
        print(f"Processing tile: {laz_file.stem}")  
        plotter = ResultsPlotter(np.array([0,0,ground_z]), legend=False, off_screen=True)
        # tracker = CameraTracker(plotter.plotter)
        pc, pcdata = load_point_cloud(str(laz_file), full_data=True)
        pcdata[:, :3] = pcdata[:, :3] - actual_mean
        plotter.add_point_cloud_np_intensity(pcdata)
        path = plotter.get_image(f"{laz_file.stem}_original.jpg")
        report.add_orignal_point_cloud(path)

        report.canvas.drawString(5 * inch, 8 * inch, f"Tile: {tile_name}")

        report.canvas.showPage()

        # break


        
    report.save()