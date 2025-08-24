import setuptools



setuptools.setup(
    name="PyTLidar",
    version="0.0.1",
    author="John Hagood",

    url="https://github.com/Landscape-CV/PyTLidar",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=['PyTLidar.Utils', 'PyTLidar.plotting', 'PyTLidar.TreeQSMSteps'],
    py_modules=['PyTLidar.main', 'PyTLidar.treeqsm', 'PyTLidar.treeqsm_batch'],
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
        'scipy'
    ],
)