"""CloudCompare PythonRuntime plugin entry point for PyTLidar TreeQSM.

Only pycc, numpy and the standard library import at module scope: an import
error here silently drops the plugin from the Plugins menu. Fits run in
worker processes using the selected environment's own python, because
sys.executable is CloudCompare itself in the embedded interpreter.
"""

import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pycc

_active_run = {}
# one results subfolder per CloudCompare session; run numbers continue across jobs
_session_stamp = None
_session_runs = {}


def _venv_python():
    for rel in ("bin/python", "bin/python3", "Scripts/python.exe"):
        candidate = Path(sys.prefix) / rel
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        f"No python executable found under {sys.prefix}. Select a virtual "
        "environment in the Python plugin settings.")


def _worker_env(workers):
    """Split the machine's threads between parallel workers so they do not
    fight over cores."""
    env = os.environ.copy()
    per = max(1, (os.cpu_count() or 4) // max(1, workers))
    for var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = str(per)
    return env


def _hold_idle_sleep(reason):
    """Keep the system from idle sleeping until the returned callable runs.

    Every path releases by itself if the process dies: macOS uses a named
    IOKit power assertion (caffeinate as fallback), Windows the thread
    execution state, Linux a systemd-inhibit tied to this pid. Best effort,
    a no-op where none of that works."""
    if sys.platform == "win32":
        try:
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            k32 = ctypes.windll.kernel32
            if k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
                return lambda: k32.SetThreadExecutionState(ES_CONTINUOUS)
        except Exception:
            pass
        return lambda: None
    if sys.platform.startswith("linux"):
        try:
            proc = subprocess.Popen(
                ["systemd-inhibit", "--what=idle:sleep", "--who=PyTLidar",
                 f"--why={reason}", "--mode=block",
                 "tail", f"--pid={os.getpid()}", "-f", "/dev/null"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return proc.terminate
        except OSError:
            return lambda: None
    if sys.platform != "darwin":
        return lambda: None
    try:
        import ctypes
        import ctypes.util

        iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))
        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        iokit.IOPMAssertionCreateWithName.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32)]
        utf8 = 0x08000100
        a_type = cf.CFStringCreateWithCString(
            None, b"PreventUserIdleSystemSleep", utf8)
        a_name = cf.CFStringCreateWithCString(None, reason.encode(), utf8)
        assertion = ctypes.c_uint32(0)
        err = iokit.IOPMAssertionCreateWithName(
            a_type, 255, a_name, ctypes.byref(assertion))
        cf.CFRelease(a_type)
        cf.CFRelease(a_name)
        if err == 0:
            return lambda: iokit.IOPMAssertionRelease(assertion)
    except Exception:
        pass
    try:
        proc = subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
        return proc.terminate
    except OSError:
        return lambda: None


def _run_stats(model):
    """(mean dist mm, trunk dist mm, mean surface coverage %) of one model."""
    pm = model.get("pmdistance")
    dist = trunk = float("nan")
    if isinstance(pm, dict):
        if pm.get("mean") is not None:
            dist = float(np.ravel(pm["mean"])[0]) * 1000.0
        if pm.get("TrunkMean") is not None:
            trunk = float(np.ravel(pm["TrunkMean"])[0]) * 1000.0
    cov = float("nan")
    sc = model.get("cylinder", {}).get("SurfCov")
    if sc is not None and np.size(sc) > 0:
        cov = float(np.mean(sc)) * 100.0
    return dist, trunk, cov


def _run_treeqsm():
    cc = pycc.GetInstance()
    if _active_run:
        raise RuntimeError("A PyTLidar job is already running; cancel it or "
                           "wait for it to finish.")
    clouds = [e for e in cc.getSelectedEntities()
              if isinstance(e, pycc.ccPointCloud)]
    if not clouds:
        raise RuntimeError("Select at least one point cloud.")

    from PyTLidar.pipeline import centre
    from PyTLidar.Utils import Utils as pt_utils

    from .adapter import cloud_from_cc, results_to_cc
    from .dialog import make_progress_dialog, QApplication, show_settings_dialog

    settings = show_settings_dialog(clouds, pt_utils.get_all_metrics())
    if settings is None:
        return

    runs = max(1, int(settings.get("runs", 1)))
    workers = max(1, int(settings.get("workers", 1)))
    global _session_stamp
    save = bool(settings.get("save_files"))
    dest = None
    if save:
        if _session_stamp is None:
            _session_stamp = time.strftime("%Y-%m-%d_%H%M")
        dest = Path(settings["results_dir"]) / _session_stamp
        dest.mkdir(parents=True, exist_ok=True)

    names = []
    for i, c in enumerate(clouds):
        n = re.sub(r"[^\w.-]+", "_", Path(c.getName()).stem) or f"cloud{i}"
        if n in names:
            n = f"{n}_{i}"
        names.append(n)

    job = Path(tempfile.mkdtemp(prefix="pytlidar_cc_"))
    (job / "settings.json").write_text(json.dumps({
        "custom": settings["custom"],
        "n_patchdiam": list(settings["n_patchdiam"]),
        "metric": settings["metric"],
        "tria": bool(settings.get("tria")),
        "save": save,
        "names": names,
    }))

    # A cloud that does not look like a tree is skipped, not fatal
    prepared = {}
    for ci, source in enumerate(clouds):
        P = cloud_from_cc(source)
        if settings["intensity_sf"] is not None:
            idx = source.getScalarFieldIndexByName(settings["intensity_sf"])
            if idx >= 0:
                keep = (source.getScalarField(idx).asArray()
                        >= settings["intensity_threshold"])
                P = P[keep]
        # define_input's stem estimation misbehaves on selections that are
        # not a tree.
        if P.shape[0] < 1000 or P[:, 2].max() - P[:, 2].min() < 1.5:
            sys.stdout.write(f"Skipping {source.getName()}: needs at least 1000 "
                             "points and 1.5 m of height.\n")
            continue
        # Centre on the mean to protect the float32 cylinder arrays from
        # precision loss; treeqsm returns z in the frame it is given.
        offset = P.mean(axis=0)
        P = centre(P, z=False)
        np.save(job / f"cloud_{ci}.npy", P)
        prepared[ci] = {"source": source, "offset": offset}
    if not prepared:
        shutil.rmtree(job, ignore_errors=True)
        raise RuntimeError("No usable point cloud in the selection.")

    python = _venv_python()
    env = _worker_env(workers)
    tasks = []
    for ci in prepared:
        start = _session_runs.get(names[ci], 0)
        tasks += [(ci, rid) for rid in range(start + 1, start + runs + 1)]
        _session_runs[names[ci]] = start + runs
    total = len(tasks)
    sys.stdout.write(f"Running PyTLidar: {len(prepared)} tree(s) x {runs} run(s), "
                     f"up to {min(workers, total)} at a time.\n")

    title = (clouds[0].getName() if len(clouds) == 1
             else f"{len(prepared)} trees")
    progress = make_progress_dialog(f"PyTLidar (TreeQSM): {title}")
    state = {"queue": tasks, "procs": {}, "buffers": {}, "done": 0,
             "failed": [], "summaries": {}, "cancelled": False,
             "progress": progress,
             "release_sleep": _hold_idle_sleep("PyTLidar TreeQSM fits")}
    _active_run.update(state)

    def tag(ci, rid):
        name = prepared[ci]["source"].getName()
        return f"{name} run {rid}" if (runs > 1 or rid > 1) else name

    def launch_next():
        while state["queue"] and len(state["procs"]) < workers:
            ci, rid = state["queue"].pop(0)
            proc = subprocess.Popen(
                [python, "-u", "-m", "pytlidar_cc.worker",
                 str(job), str(ci), str(rid)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            os.set_blocking(proc.stdout.fileno(), False)
            state["procs"][(ci, rid)] = proc
            sys.stdout.write(f"[{tag(ci, rid)}] started\n")

    def finish():
        # closing the dialog would emit canceled(), disconnect first
        try:
            progress.canceled.disconnect(cancel)
        except Exception:
            pass
        progress.allow_close()
        progress.close()
        try:
            if state["cancelled"]:
                for proc in state["procs"].values():
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        pass
                sys.stdout.write("PyTLidar job cancelled.\n")
                return
            if state["failed"]:
                names = ", ".join(tag(ci, rid) for ci, rid in state["failed"])
                sys.stdout.write(f"Failed fits: {names}, see the lines above.\n")
            if runs > 1:
                for ci, rows in sorted(state["summaries"].items()):
                    if len(rows) < 2:
                        continue
                    med_dist = float(np.nanmedian([r[3] for r in rows]))
                    med_trunk = float(np.nanmedian([r[4] for r in rows]))
                    sys.stdout.write(f"--- {prepared[ci]['source'].getName()} run summary ---\n")
                    for rid, vol, branches, dist, trunk, cov in sorted(rows):
                        mark = ""
                        if dist > 1.5 * med_dist or trunk > 1.5 * med_trunk:
                            mark = "  <- fit looks off, check before using"
                        sys.stdout.write(f"  run {rid}: volume {vol:.0f} L, branches "
                                         f"{branches}, mean dist {dist:.1f} mm, "
                                         f"coverage {cov:.0f}%{mark}\n")
            if dest is not None:
                sys.stdout.write(f"Model files saved to {dest}\n")
        finally:
            state["release_sleep"]()
            shutil.rmtree(job, ignore_errors=True)
            _active_run.clear()

    def drain(key, proc):
        # unbuffered workers can hand over partial lines; emit only whole ones
        ci, rid = key
        try:
            data = os.read(proc.stdout.fileno(), 65536)
        except (BlockingIOError, OSError):
            return
        if not data:
            return
        buf = state["buffers"].get(key, "") + data.decode("utf-8", errors="replace")
        *lines, state["buffers"][key] = buf.split("\n")
        for ln in lines:
            if ln.strip():
                sys.stdout.write(f"[{tag(ci, rid)}] {ln}\n")

    def poll_once():
        for key, proc in list(state["procs"].items()):
            ci, rid = key
            drain(key, proc)
            rc = proc.poll()
            if rc is not None:
                drain(key, proc)
                leftover = state["buffers"].pop(key, "")
                if leftover.strip():
                    sys.stdout.write(f"[{tag(ci, rid)}] {leftover}\n")
                del state["procs"][key]
                result = job / f"result_{ci}_{rid}.pkl"
                if state["cancelled"]:
                    pass
                elif rc == 0 and result.exists():
                    state["done"] += 1
                    # scene insertion must happen here on the main thread
                    with open(result, "rb") as f:
                        model = pickle.load(f)
                    label = f"run {rid}" if (runs > 1 or rid > 1) else None
                    results_to_cc(cc, prepared[ci]["source"], model,
                                  prepared[ci]["offset"], label=label)
                    td = model["treedata"]
                    dist, trunk, cov = _run_stats(model)
                    state["summaries"].setdefault(ci, []).append(
                        (rid, float(td.get("TotalVolume", 0)),
                         int(td.get("NumberBranches", 0)), dist, trunk, cov))
                    del model
                    if dest is not None:
                        outdir = job / f"out_{ci}_{rid}"
                        if outdir.is_dir():
                            for f in outdir.iterdir():
                                shutil.move(str(f), dest / f.name)
                else:
                    state["failed"].append(key)
                    sys.stdout.write(f"[{tag(ci, rid)}] failed (exit {rc})\n")
        if not state["cancelled"]:
            launch_next()
        finished = state["done"] + len(state["failed"])
        progress.setLabelText(
            f"PyTLidar (TreeQSM): {title}\n{finished}/{total} fits finished, "
            f"{len(state['procs'])} running")

    def cancel():
        state["cancelled"] = True
        state["queue"].clear()
        for proc in state["procs"].values():
            proc.kill()

    progress.canceled.connect(cancel)
    launch_next()
    # the runtime does not call Python back once this action returns, so
    # pump the event loop here until the job ends
    while (state["procs"] or state["queue"]) and not state["cancelled"]:
        poll_once()
        QApplication.processEvents()
        time.sleep(0.05)
    poll_once()
    finish()


class PyTLidarPlugin(pycc.PythonPluginInterface):
    def __init__(self):
        pycc.PythonPluginInterface.__init__(self)

    def getIcon(self):
        icon = Path(__file__).parent / "assets" / "icon.png"
        return str(icon.resolve()) if icon.exists() else ""

    def getActions(self):
        return [pycc.Action(name="Run PyTLidar", icon=self.getIcon(),
                            target=_run_treeqsm)]
