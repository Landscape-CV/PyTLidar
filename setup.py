import setuptools



setuptools.setup(
    name="PyTLiDAR",
    version="0.0.1.12",
    author="John Hagood",

    url="https://github.com/Landscape-CV/PyTLiDAR",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=['PyTLiDAR.Utils', 'PyTLiDAR.plotting', 'PyTLiDAR.TreeQSMSteps'],
    py_modules=['PyTLiDAR.main', 'PyTLiDAR.treeqsm', 'PyTLiDAR.treeqsm_batch'],
    python_requires=">=3.8",
    install_requires=[
        'alphashape',
        'laspy[lazrs,laszip]',
        'matplotlib',
        'numba==0.61.2',
        'numpy>=2.0',
        'open3d',
        'pandas>=2.0',
        'plotly',
        'PySide6==6.8.3',
        'python-dotenv',
        'igraph',
        'scikit_learn',
        'scipy',
        'torch==2.6.0',
        'trimesh',
    ],
)