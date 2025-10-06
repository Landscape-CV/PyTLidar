import subprocess

# Example: colorize a point cloud
subprocess.run([
    "G:\Software\CloudCompare\CloudCompare.exe",
    "-SILENT",
    "-O", "1_leaf_removal_afterCloud3.las",
    "-COLOR", "255", "0", "0",
    "-SAVE_CLOUDS"
])
