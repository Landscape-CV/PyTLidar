"""One TreeQSM fit per process, driven by the plugin.

python -m pytlidar_cc.worker <job_dir> <cloud_idx> <run_id>, with
cloud_<idx>.npy and settings.json in the job directory. Writes
result_<idx>_<run_id>.pkl minus the bulky point/cover/segment data; stdout
is relayed to CloudCompare's console. With saving on, the classic model
files and a run_info.json land in out_<idx>_<run_id>/.
"""

import json
import pickle
import shutil
import sys
import time
from pathlib import Path


def main(job_dir, cloud_idx, run_id):
    job = Path(job_dir)

    # The fit builds matplotlib figures; keep the worker headless.
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np

    from PyTLidar.pipeline import build_inputs, calculate_optimal, run_qsm

    settings = json.loads((job / "settings.json").read_text())
    P = np.load(job / f"cloud_{cloud_idx}.npy")
    print(f"fitting {P.shape[0]} points", flush=True)

    save = bool(settings.get("save"))
    name = f"{settings['names'][int(cloud_idx)]}_run{run_id}"
    common = dict(names=[name], savemat=1 if save else 0,
                  savetxt=1 if save else 0, plot=0, disp=2, savepdf=0)
    if settings["custom"] is not None:
        inputs = build_inputs(P, custom=tuple(settings["custom"]), **common)
    else:
        inputs = build_inputs(P, n_patchdiam=tuple(settings["n_patchdiam"]), **common)
    if not inputs:
        raise RuntimeError("The cloud is empty.")
    inputs[0]["Tria"] = 1 if settings.get("tria") else 0

    outdir = job / f"out_{cloud_idx}_{run_id}"
    outdir.mkdir(exist_ok=True)
    t0 = time.time()
    models, _ = run_qsm(P, inputs[0], results_dir=str(outdir))
    elapsed = time.time() - t0
    if not models:
        raise RuntimeError("TreeQSM fitted no cylinders to this cloud.")

    best = 0
    if len(models) > 1:
        best, _, _ = calculate_optimal(models, settings["metric"])
        best = int(best)
    model = models[best]

    if save:
        # flatten the results/ subfolder treeqsm creates
        sub = outdir / "results"
        if sub.is_dir():
            for f in sub.iterdir():
                shutil.move(str(f), outdir / f.name)
            sub.rmdir()
        inp = model["rundata"]["inputs"]
        td = model["treedata"]
        info = {
            "source": settings["names"][int(cloud_idx)],
            "run": int(run_id),
            "points": int(P.shape[0]),
            "PatchDiam1": float(np.ravel(inp["PatchDiam1"])[0]),
            "PatchDiam2Min": float(np.ravel(inp["PatchDiam2Min"])[0]),
            "PatchDiam2Max": float(np.ravel(inp["PatchDiam2Max"])[0]),
            "Tria": int(inp.get("Tria", 0)),
            "metric": settings["metric"],
            "TotalVolume_L": float(np.ravel(td.get("TotalVolume", 0))[0]),
            "elapsed_s": round(elapsed, 1),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            from importlib.metadata import version
            info["PyTLidar"] = version("PyTLidar")
        except Exception:
            pass
        (outdir / f"run_info_{name}.json").write_text(json.dumps(info, indent=1))

    for key in ("points", "cover", "segment"):
        model.pop(key, None)
    with open(job / f"result_{cloud_idx}_{run_id}.pkl", "wb") as f:
        pickle.dump(model, f, protocol=4)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
