---
title: 'PyTLiDAR: A Python Package for Tree QSM Modeling from Terrestrial LiDAR Data'
tags:
  - Python
  - biology
  - lidar
  - qsm
authors:
  - name: John Hagood
    orcid: 0009-0002-3894-4464
    affiliation: 1
  - name: Fan Yang
    orcid: 0009-0006-9787-2471
    affiliation: 1
  - name: Jeffery Cannon
    affiliation: 2
  - name: James Stroud
    affiliation: 3
    corresponding: true

affiliations:
  - name: School of Computer Science, Georgia Institute of Technology, United States
    index: 1
  - name: Jones Center at Ichauway, United States
    index: 2
  - name: School of Biological Sciences, Georgia Institute of Technology, United States
    index: 3

date: 2025-06-30
journal: JOSS
bibliography: paper.bib
---

# Summary

PyTLiDAR is an open-source Python software package that ports the MATLAB-based [TreeQSM](https://github.com/InverseTampere/TreeQSM?tab=readme-ov-file) method [@TreeQSM] into Python, also providing an accessible, extensible, and GUI-driven alternative for researchers and practitioners in forestry, ecology, and 3D vegetation modeling. The software reconstructs Quantitative Structure Models (QSMs) of trees from Terrestrial LiDAR Scans (TLS) and provides interactive visualization tools for inspecting model quality and derived tree metrics.

Key features of PyTLiDAR include:

-A full reimplementation of TreeQSM's core logic in Python

-A user-friendly GUI built with PyQt6 for batch or single-file processing

-Automated and manual configuration of model generation parameters, including patch diameter ranges

-Support for interactive 3D visualization of tree models and parameter tuning

-Batch data processing

# Statement of need

Terrestrial Laser Scanning typically utilizes LiDAR systems to collect millions of points on the surface of objects and preserves spatial information. For estimating above ground biomass (AGB), gap fraction, crown shape, and other ecological properties, accurate and efficient tree QSM reconstruction from TLS point cloud data is essential [@f6114245].

TreeQSM has been widely used in forestry and ecology for modeling three-dimensional tree structures from TLS point clouds [@TERRYN2020170]. However, its reliance on MATLAB makes it less accessible for users without a commercial license or familiarity with the MATLAB environment. Furthermore, the lack of a graphical interface makes the tool less user-friendly and its parameter tuning less efficient.

PyTLiDAR addresses these issues by providing a native Python implementation of TreeQSM’s core algorithms, wrapped in a streamlined graphical interface that allows researchers to visualize and evaluate their models. It promotes reproducible and exploratory research by offering transparent parameter control, open-source licensing, and seamless integration into Python-based analysis workflows. This work lowers the barrier for adoption of QSM modeling by removing the MATLAB dependency, enhancing accessibility for the broader open-source geospatial and ecological modeling community.

# Method

TreeQSM models individual trees from terrestrial LiDAR scans by covering the input point cloud with small, connected surface patches. These patches form the building-bricks for reconstructing the tree’s global shape. Based on neighbor-relation of the cover sets, the point cloud is segmented into individual branches, with parent-children relationships recorded. Then each segment is approximated as a collection of connected cylinders of varying radius, length, and orientation. This cylinder-based representation offers a simple yet effective regularization of the complex tree structure, supporting downstream analyses such as stem volume estimation or structural trait extraction [@rs5020491] [@rs70404581].

# Software Description

PyTLiDAR implements the same method stated above in Python, and uses [PyQt6](https://pypi.org/project/PyQt6/) to create an intuitive interface for parameter configuration and data processing. Upon launching the application, users are presented with fields to input or generate values for key modeling parameters, including the minimum, and maximum patch diameters. The application supports both numeric entry and automatic generation of value ranges based on user-defined counts. Also, an intensity threshold can be set to filter the point cloud data, helping to remove noise and irrelevant data before modeling. 

Users may choose between batch processing of an entire directory of point cloud files or processing a single file. The GUI also includes options for showing only the optimal model, based on selectable performance metrics such as 'all_mean_dis', and provides a dropdown menu to choose the preferred metric.

![Software interface for user input and data selection. \label{fig:pc1}](figs/fig1.jpg){ width=80% }

After data selection, the software opens a new interface allows for data processing and visualization. Once the QSM reconstruction process is complete, PyTLiDAR provides interactive 3D visualization of the generated QSM using [plotly](https://plotly.com/). Users can inspect the structural fidelity of the reconstructed model, including trunk and branch geometry, and compare different parameter configurations for best fit. This combination of visual feedback and customizable processing offers an efficient path toward accurate and transparent tree structure analysis. If running in batch mode, users may also set the number of parallel cores to utilize to run simultaneous processes.

![Software interface for processing and interactive visualization. \label{fig:pc1}](figs/fig2.jpg){ width=80% }

Users can also review the relavant morphological data of the QSM, including stem diameters, branch volume, surface area, and length with regard to diameter or order from stem.

![Software interface for user input and data selection. \label{fig:pc1}](figs/fig3.jpg){ width=80% }

If desired, both treeqsm.py and teeqsm_batch.py may be run directly from the command line using the following arguments:

  -intensity: filter point cloud based on intensity

  -normalize: recenter point cloud locations. Use this if your point cloud location values are very large

  -custominput: user sets specific patch diameters to test

  -ipd: initial patch diameter

  -minpd: min patch diameter

  -maxpd: maximum patch diameter

  -name: specifies a name of the run. This will be appended to the name generated by TreeQSM

  -outputdirectory: specifies the directory to put the "results" folder

  -numcores: specify number of cores to use to process files in parallel. Only valid in batched mode, Must be a single integer

  -optimum: specify an optimum metric to select best model to save

  -help: displays the run options

  -verbose: verbose mode, displays outputs from TreeQSM as it processes

  -h: displays the run options

  -v: verbose mode

This allows users to integrate the same functionality provided in the GUI into their own scripts with ease, whether those scripts are in python or not. Additionally users who are using python can use the package directly and get the full functionality by importing treeqsm. 

# Future Additions

While the initial release is focused on porting only TreeQSM, future additions to PyTLiDAR are planned. 

The first enhancement is to provide a novel pipeline for analyzing LiDAR scans of entire ecosystems to provide insight into vegetation structure at particular locations. This would allow users to load a series of LiDAR scan tiles and GPS observations of fauna and directly measure the environments around where those particular specimens spend their time. This would provide researchers great insight into the daily lives of the specimen without requiring direct observation. 

Further enhancement will be to build out the functions provided to users for processing LiDAR point clouds, including but not limited to both established and novel methods to perform Ground Filtering, Tree Segmentation and Leaf/Wood separation. The intended goal for this package is to provide a single source for any user processing terrestrial LiDAR to perform every step of their analysis. 

# Acknowledgements

We acknowledge contributions from XXX, XXX, and Amir Hossein Alikhah Mishamandani during the development of the package. This work is enabled in part by funding from XXXXX.

# References
