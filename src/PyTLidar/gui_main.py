"""Entry point for pytlidar-gui. PySide6 is an optional extra."""
import importlib.util


def run():
    if importlib.util.find_spec("PySide6") is None:
        raise SystemExit("The PyTLidar GUI needs PySide6. Install it with: pip install PyTLidar[gui]")
    from .main import run as run_gui
    run_gui()


if __name__ == "__main__":
    run()
