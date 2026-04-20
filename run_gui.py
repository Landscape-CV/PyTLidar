import sys
from pathlib import Path

# Add the project root to sys.path so `from ecomodel import Ecomodel` and
# `from gui.xxx import ...` both resolve correctly, regardless of the
# working directory from which this script is launched.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PySide6.QtWidgets import QApplication
from gui.main_window import EcomodelMainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Ecomodel")
    app.setOrganizationName("PyTLidar")

    window = EcomodelMainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
