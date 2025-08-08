---
title: 'PyTLiDAR: A Python Package for Tree QSM Modeling from Terrestrial LiDAR Data'
tags:
  - Python
  - Ecology
  - Terrestrial Lidar Scan
  - Quantitative Structure Model
authors:
  - name: John Hagood
    orcid: 0009-0002-3894-4464
    affiliation: 1
    corresponding: true
  - name: Fan Yang
    orcid: 0009-0006-9787-2471
    affiliation: 1
    corresponding: true
  - name: Jeffery B. Cannon
    affiliation: 2
    corresponding: true
  - name: James Stroud
    affiliation: 3
    corresponding: true

affiliations:
  - name: Human Augmented Analytics Groups (HAAG), School of Computer Science, Georgia Institute of Technology, United States
    index: 1
  - name: The Jones Center at Ichauway, United States
    index: 2
  - name: School of Biological Sciences, Georgia Institute of Technology, United States
    index: 3

date: 2025-06-30
journal: JOSS
bibliography: paper.bib
---

# Summary

PyTLiDAR is an open-source Python package that reconstructs 3D tree Quantitative Structure Models (QSM) from Terresrial LiDAR Scan (TLS) data, providing a user-friendly alternative to the MATLAB-based [TreeQSM](https://github.com/InverseTampere/TreeQSM?tab=readme-ov-file) method [@TreeQSM]. 
PyTLiDAR provides an accessible, extensible, and GUI-driven workflow for researchers and practitioners in forestry, ecology, and 3D vegetation modeling. 
The package also integrates interactive visualization tools for inspecting model quality and derived tree metrics.

Key features of PyTLiDAR include:

-A full reimplementation of TreeQSM's core logic in Python

-A user-friendly GUI built with PyQt6

-Automated and manual configuration of model generation parameters, including patch diameter ranges

-Support for interactive 3D visualization of tree models and parameter tuning

-Batch or single-file processing

# Statement of Need

Terrestrial Laser Scanning (TLS) is an active remote sensing technology which uses infrared laser pulses to collect millions of points on the surface of objects, 
preserving spatial information and providing unprecedented detail on structural information. The technology is rapidly being adopted for diverse uses in forestry and ecology, 
as it is useful for estimating forest structure [@rs13122297], Above Ground Biomass (AGB) [@https://doi.org/10.1002/ecs2.70232], gap fraction and forest fuels [@fire6040151], crown shape [@10.1093/forestry/cpaa037], disturbance patterns [@cannon2024terrestrial], tree competition [@METZ2013275], physiology [@bg-12-1629-2015],and other ecological properties. 
To fully realize the potential of TLS for these applications, accurate and efficient reconstruction of QSMs from TLS point cloud data is essential [@f6114245].

The use of QSM software on point cloud data permits estimation of detailed components of branch architecture such as branch diameter, volume, and distribution along the trunk [@Lau2018],
providing detailed information for fine-scale estimates of AGB, canopy architecture, and more.
TreeQSM is a software that has been widely used in forestry and ecology for modeling tree structures from TLS point clouds [@TERRYN2020170]. 
However, its reliance on MATLAB makes it less accessible for users without a commercial license or familiarity with the MATLAB environment. Furthermore, the lack of a graphical interface makes the tool less user-friendly and its parameter tuning less efficient.

PyTLiDAR addresses these issues by providing a native Python implementation of TreeQSM’s core algorithms, 
wrapped in a streamlined graphical interface that allows researchers to visualize and evaluate their models. 
It promotes reproducible and exploratory research by offering transparent parameter control, open-source licensing, and seamless integration into Python-based analysis workflows. 
This work lowers the barrier for adoption of QSM modeling by removing the MATLAB dependency, enhancing accessibility for the broader open-source geospatial and ecological modeling community.

# Method

TreeQSM models individual trees from terrestrial LiDAR scans by covering the input point cloud with small, connected surface patches. 
These patches form the building blocks for reconstructing the tree’s global shape. The algorithm first identifies these surface patches using local geometric properties, 
then establishes neighbor relationships between adjacent patches. Based on neighbor relationships of the surface patches, the point cloud is segmented into individual branches, 
with parent-children relationships of branches recorded. Then each branch is approximated as a collection of connected cylinders of varying radius, length, and orientation. 
This cylinder-based representation offers a simple yet effective regularization of the complex tree structure, supporting downstream analyses such as stem volume estimation or structural trait extraction [@rs5020491] [@rs70404581].

# Software Architecture

PyTLiDAR is organized into several key modules: core QSM algorithms (treeqsm.py), batch processing utilities (treeqsm_batch.py), GUI components built with PyQt6, and visualization tools using Plotly. 
The software follows a modular design that allows researchers to either use the complete GUI application or integrate individual components into their own Python workflows. 

# Software Description

PyTLiDAR implements the same method stated above in Python, and uses [PyQt6](https://pypi.org/project/PyQt6/) to create an intuitive interface for parameter configuration and data processing (Figure 1). 
Upon launching the application, users are presented with fields to input or generate values for key modeling parameters, including the minimum, and maximum patch diameters. 
The application supports both numeric entry and automatic generation of value ranges based on user-defined parameter space. Also, an intensity threshold can be set to 
filter the point cloud data, helping to remove LiDAR returns due to noise or vegetation prior to modeling. 

Users may choose between batch processing of an entire directory of point cloud files or processing a single file. The GUI also includes options for displaying only the optimal model, 
based on selectable performance metrics such as 'all_mean_dis' (mean distance between point cloud and reconstructed model surface ), and provides a dropdown menu to choose the preferred metric.

![Software interface for user input and data selection. \label{fig:pc1}](figs/fig1.jpg){ width=80% }

After parameter and file selection, the software opens a new interface displaying data processing progress and allowing visualization of model outputs. 
Once the QSM reconstruction process is complete, PyTLiDAR provides interactive 3D visualization of the generated QSM using [plotly](https://plotly.com/) (Figure 2). 
Users can inspect the structural fidelity of the reconstructed model, including trunk and branch geometry, and compare different parameter configurations for best fit. 
This combination of visual feedback and customizable processing offers an efficient path toward accurate and transparent tree structure analysis. 
If running in batch mode, users may also set the number of parallel cores to utilize to run simultaneous processes.

![Software interface for processing and interactive visualization. \label{fig:pc1}](figs/fig2.jpg){ width=80% }

Users can also review the relevant morphological summeries of the QSM, including distribution of branch diameters, branch volume, surface area, 
and length with regard to diameter or order from stem, as with the original TreeQSM implementation.

![Tree QSM data display \label{fig:pc1}](figs/fig3.jpg){ width=80% }

If desired, both treeqsm.py and teeqsm_batch.py may be run directly from the command line using the following arguments:

  -intensity: filter point cloud based on values greater than the indicated intensity

  -normalize: recenter point cloud locations. Use this if your point cloud X, Y location values are very large (e.g., using UTM coordinates rather than a local coordinate system).

  -custominput: user sets specific patch diameters to test

  -ipd: initial patch diameter

  -minpd: min patch diameter

  -maxpd: maximum patch diameter

  -name: specifies a name for the current modeling run. This will be appended to the name generated by PyTLiDAR

  -outputdirectory: specifies the directory to put the "results" folder

  -numcores: specify number of cores to use to process files in parallel. Only valid in batched mode, Must be a single integer

  -optimum: specify an optimum metric to select best model to save [#need_more_description]

  -help: displays the run options

  -verbose: verbose mode, displays outputs from PyTLiDAR as it processes

  -h: displays the run options

  -v: verbose mode

This allows users to integrate the same functionality provided in the GUI into their own scripts with ease, whether those scripts are in python or not. 
Users who are using python can use the package directly and get the full functionality by importing treeqsm. 

[#add_benchmark_comparison]

# Availability and Installation

PyTLiDAR is available at this [GitHub repository](https://github.com/Landscape-CV/PyTLiDAR). The package requires Python 3.8+ and key dependencies including PySide6 and Plotly. 
Installation instructions and example datasets are provided in the repository documentation.

# Future Additions

While the initial release is focused on porting only TreeQSM, future additions to PyTLiDAR are planned. 

The first planned enhancement is to provide a novel pipeline for analyzing LiDAR scans of entire forest ecosystems to quantify vegetation structure at particular locations. 
This would allow users to load a series of LiDAR scan tiles and GPS observations of fauna and directly measure the environments, providing greater insights on components of habitat structural complexity. 

Other planned enhancements include functions provided to users for processing LiDAR point clouds, including but not limited to both established and novel methods to perform Ground 
Filtering, Tree Segmentation and Leaf/Wood separation. The intended goal for this package is to provide a single source for any user processing terrestrial LiDAR to perform every step of 
their analysis. 

# Acknowledgements

We acknowledge contributions from Amir Hossein Alikhah Mishamandani during the development of the package. This work also received high level guidance from Breanna Shi, Dori P., and thanks to
other staffs from the Human-Augmented Analytics Group to make this happen.

# References
