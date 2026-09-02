__all__ = ["load_cloud", "centre", "build_inputs", "run_qsm", "run_batch", "calculate_optimal",
           "list_scalar_fields", "list_las_scalar_fields"]


def __getattr__(name):
    # Lazy so that `import PyTLidar` stays cheap and `python -m PyTLidar.treeqsm` does not
    # import treeqsm a second time before running it.
    if name in __all__:
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
