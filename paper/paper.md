---
title: 'PyTLiDAR: A Python Framework for Tree QSM Modeling from Terrestrial LiDAR Data'
tags:
  - Python
  - biology
  - lidar
  - qsm
authors:
  - name: John Hagood
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Fan Yang
    orcid: 0009-0006-9787-2471
    affiliation: 1
  - name: TBD
    orcid: 0000-0000-0000-0000
    affiliation: 1

affiliations:
  - name: Georgia Institute of Technology, United States
    index: 1

date: 2025-06-14
journal: JOSS
bibliography: paper.bib
---

# Summary

PyTLiDAR is an open-source Python software package that ports the MATLAB-based TreeQSM algorithm into Python, also
providing an accessible, extensible, and GUI-driven alternative for researchers and practitioners in forestry, ecology, 
and 3D vegetation modeling. The software reconstructs Quantitative Structure Models (QSMs) of trees from terrestrial LiDAR 
scans and provides interactive visualization tools for inspecting model quality and derived tree metrics.

Key features of PyTLiDAR include:

-A full reimplementation of TreeQSM’s core logic in Python

-A user-friendly GUI built with PyQt6 for batch or single-file processing

-Automated and manual configuration of model generation parameters, including patch diameter ranges

-Support for interactive 3D visualization of tree models and parameter tuning

-Batch data processing

# Statement of need

TreeQSM has been widely used in forestry and ecology for modeling three-dimensional tree structures from terrestrial laser 
scanning (TLS) point clouds. However, its reliance on MATLAB creates a barrier for users without a commercial license or 
familiarity with the MATLAB environment. Furthermore, the lack of a graphical interface and real-time visualization options 
makes parameter tuning and model validation labor-intensive and opaque.

PyTLiDAR addresses these challenges by providing a native Python implementation of TreeQSM’s core algorithms, wrapped in a 
streamlined graphical interface that allows researchers to visualize and evaluate their models dynamically. It promotes 
reproducible and exploratory research by offering transparent parameter control, open-source licensing, and seamless 
integration into Python-based analysis workflows. This work lowers the barrier for adoption of QSM modeling by removing 
the MATLAB dependency, enhancing accessibility for the broader open-source geospatial and ecological modeling community.

# Software Description

PyTLiDAR is implemented in Python and uses PyQt6 to create an intuitive interface for parameter configuration and data 
processing. Upon launching the application, users are presented with fields to input or generate values for key modeling 
parameters, including the initial, minimum, and maximum patch diameters. The application supports both numeric entry and 
automatic generation of value ranges based on user-defined counts.

Users may choose between batch processing of an entire directory of point cloud files or processing a single file. An 
intensity threshold can be set to filter the point cloud data, helping to remove noise and irrelevant data before modeling. 
The GUI also includes options for showing only the optimal model, based on selectable performance metrics such as 
'all_mean_dis', and provides a dropdown menu to choose the preferred metric.

Once processing is complete, PyTLiDAR provides interactive 3D visualization of the generated QSM using plotly. Users can 
inspect the structural fidelity of the reconstructed model, including trunk and branch 
geometry, and compare different parameter configurations for best fit. This combination of visual feedback and customizable 
processing offers an efficient path toward accurate and transparent tree structure analysis.

# Acknowledgements

We acknowledge contributions from Amir Hossein Alikhah Mishamandani, and the
support from James T Stroud and Jeffery Cannon, during the development of the 
software and this project.

# References