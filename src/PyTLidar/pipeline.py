"""
The steps every caller repeats around treeqsm: load a cloud, centre it, build the inputs
dict(s), run one model, or run a batch of models in worker processes. The GUI, the two CLIs and
any script use these; nothing here imports Qt.
"""
import os
import multiprocessing as mp
import numpy as np

if __package__:
    from .treeqsm import treeqsm, calculate_optimal
    from .Utils.define_input import define_input
    from .Utils.Utils import load_point_cloud, list_scalar_fields, list_las_scalar_fields
else:
    from treeqsm import treeqsm, calculate_optimal
    from Utils.define_input import define_input
    from Utils.Utils import load_point_cloud, list_scalar_fields, list_las_scalar_fields

__all__ = ["load_cloud", "centre", "build_inputs", "run_qsm", "run_batch", "calculate_optimal",
           "list_scalar_fields", "list_las_scalar_fields"]


def load_cloud(path, intensity_threshold=0.0, scalar_field="intensity", normalize_scalar=False):
    """Nx3 float64 point array from a .las, .laz, .ply or .xyz file, keeping points whose
    intensity is at least intensity_threshold. scalar_field names the field used as intensity
    (case-insensitive); normalize_scalar rescales it to 0-65535 before the threshold."""
    return load_point_cloud(path, intensity_threshold=float(intensity_threshold),
                            scalar_field=scalar_field, normalize_scalar=normalize_scalar)


def centre(P, xy=True, z=True):
    """Return a centred copy of P. xy subtracts the mean of every coordinate; z then moves the
    lowest point to height zero. The GUI does both; the CLI's --normalize does xy only."""
    P = np.asarray(P)
    if P.shape[0] == 0:
        return P.copy()
    P = P - np.mean(P, axis=0) if xy else P.copy()
    if z:
        P[:, 2] = P[:, 2] - np.min(P[:, 2], axis=0)
    return P


def build_inputs(clouds, n_patchdiam=(1, 1, 1), custom=None, names=None,
                 savemat=0, savetxt=1, plot=0, disp=0, savepdf=None):
    """One inputs dict per cloud, as a list.

    clouds: an Nx3 array or a list of them. Empty clouds are dropped, and so are their names, so
    the result lines up with the clouds that will actually run.
    n_patchdiam: how many PatchDiam1, PatchDiam2Min and PatchDiam2Max values define_input should
    generate. Ignored when custom is given.
    custom: (PatchDiam1, PatchDiam2Min, PatchDiam2Max) lists of values to test instead. BallRad1
    and BallRad2 are then set one centimetre above PatchDiam1 and PatchDiam2Max.
    names: model name per cloud; define_input's Tree_i when omitted.
    savemat, savetxt, plot and disp are always set. savepdf is set only when given, since
    define_input's default of 1 writes PDFs into results/.
    """
    if isinstance(clouds, np.ndarray):
        if clouds.shape[0] == 0:
            raise ValueError("The point cloud is empty, so there is nothing to build inputs for")
        clouds = [clouds]
    else:
        clouds = list(clouds)
    if names is not None and len(names) != len(clouds):
        raise ValueError("names must have one entry per cloud")
    keep = [i for i, c in enumerate(clouds) if np.asarray(c).shape[0] > 0]
    clouds = [clouds[i] for i in keep]
    if names is not None:
        names = [names[i] for i in keep]
    if not clouds:
        return []

    if custom is None:
        inputs = define_input(clouds, *n_patchdiam)
    else:
        pd1, pd2min, pd2max = custom
        inputs = define_input(clouds, 1, 1, 1)
        for inp in inputs:
            inp["PatchDiam1"] = list(pd1)
            inp["PatchDiam2Min"] = list(pd2min)
            inp["PatchDiam2Max"] = list(pd2max)
            inp["BallRad1"] = [d + .01 for d in inp["PatchDiam1"]]
            inp["BallRad2"] = [d + .01 for d in inp["PatchDiam2Max"]]

    for i, inp in enumerate(inputs):
        if names is not None:
            inp["name"] = names[i]
        inp["savemat"] = savemat
        inp["savetxt"] = savetxt
        inp["plot"] = plot
        inp["disp"] = disp
        if savepdf is not None:
            inp["savepdf"] = savepdf
    return inputs


def run_qsm(P, inputs, results_dir=None):
    """Run treeqsm on one cloud and return (models, cyl_htmls). Output files go under
    results_dir (the current directory when None). Raises RuntimeError if the model fails;
    treeqsm has already printed the traceback by then."""
    cwd = os.getcwd()
    try:
        models, cyl_htmls = treeqsm(P, inputs, 0, None, results_dir)
    finally:
        os.chdir(cwd)
    if isinstance(models, str) and models == "ERROR":
        raise RuntimeError(f"treeqsm failed on {inputs.get('name', 'the cloud')}")
    return models, cyl_htmls


def run_batch(clouds, inputs_list, n_workers=1, results_dir=None, on_message=print, on_result=None):
    """Run one treeqsm worker process per cloud, n_workers at a time, the way the batch CLI and
    the GUI always have: every worker gets its own queue, workers start in groups of n_workers and
    their results are read back in order.

    Returns a list aligned with inputs_list holding (models, cyl_htmls), or None for a tree that
    failed. on_result(i, models, cyl_htmls) is called as each tree finishes, which lets a caller
    deal with one model at a time instead of holding all of them. on_message receives progress
    lines; pass None to silence them.
    """
    if len(clouds) != len(inputs_list):
        raise ValueError("clouds and inputs_list must have the same length")
    n_workers = max(1, int(n_workers))
    ctx = mp.get_context("spawn")
    queues, procs = [], []
    for i, (cloud, inp) in enumerate(zip(clouds, inputs_list)):
        q = ctx.Queue()
        queues.append(q)
        procs.append(ctx.Process(target=treeqsm, args=(cloud, inp, i, q, results_dir)))

    results = [None] * len(inputs_list)
    for start in range(0, len(inputs_list), n_workers):
        group = range(start, min(start + n_workers, len(inputs_list)))
        for i in group:
            if on_message is not None:
                on_message(f"Processing {inputs_list[i]['name']}. This may take several minutes...\n")
            procs[i].start()
        for i in group:
            _, models, cyl_htmls = queues[i].get()
            procs[i].join()
            if isinstance(models, str) and models == "ERROR":
                if on_message is not None:
                    on_message(f"An error occured on file {inputs_list[i]['name']}. Please try again. "
                               "Consider checking the console and reporting the bug to us.\n")
                continue
            results[i] = (models, cyl_htmls)
            if on_result is not None:
                on_result(i, models, cyl_htmls)
    return results
