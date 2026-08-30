# pytlidar-cc

CloudCompare plugin that runs [PyTLidar](https://github.com/Landscape-CV/PyTLidar)
TreeQSM on selected point clouds and puts the models back into the scene.

## What you get

One group per fitted model, named after the source cloud:

- `cylinders`: the cylinder start points with `radius`, `length`,
  `branch_order`, `branch_id` and `surface_coverage` scalar fields.
- `cylinder mesh`: the cylinders as one triangle mesh, `branch_order` on its
  vertices for colouring.
- `stem mesh` when stem triangulation is on.
- The tree metrics printed to the console.

Everything is in the source cloud's frame and carries its global shift.

## Requirements

- CloudCompare 2.13 or newer with the PythonRuntime plugin. It ships in the
  Windows installer and in the OpenFields macOS builds
  (simulation.openfields.fr); otherwise build it from source.
- PyTLidar 1.0.6 or newer in the plugin's Python environment.

## Install

With the Python environment CloudCompare uses:

```
python -m pip install pytlidar-cc
```

## Usage

1. Select one or more point clouds (one segmented tree each).
2. Plugins, PyTLidar, Run PyTLidar.
3. Adjust the settings, OK.

Settings: generated or custom PatchDiam values (several values per parameter
runs every combination and the metric picks the best model), an optional
scalar field filter, stem triangulation, repeat runs and parallel workers,
and where the model files save.

## Notes

- Model files (the QSM npz, the cylinder, branch and treedata tables and a
  run_info.json per run) save into one folder per CloudCompare session,
  under Documents/PyTLidar/results by default; run numbers continue across
  jobs in a session.
- TreeQSM is stochastic. Repeat runs puts each model in the scene as its own
  group with a console summary that flags suspect fits; keep the tree you
  like.
- Fits run in worker processes: progress streams to the console and the job
  can be cancelled from the progress dialog. Parallel runs each use their
  own memory, so keep workers low for very large clouds.
- Segment the tree from the plot first (for example with the Treeiso plugin)
  and clean obvious noise.
- A running job holds off system idle sleep so fits continue with the
  screen locked; closing the lid still sleeps as normal.

## License

GPL-3.0-only, like PyTLidar.
