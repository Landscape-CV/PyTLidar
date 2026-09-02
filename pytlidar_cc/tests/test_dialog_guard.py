import pytest

from pytlidar_cc import dialog


def test_dialog_refuses_a_qt_without_an_application():
    # Outside CloudCompare there is no running QApplication, which is the same
    # state a foreign PySide6 is in inside CloudCompare.
    assert dialog.QApplication.instance() is None
    with pytest.raises(RuntimeError, match="not CloudCompare's own"):
        dialog.show_settings_dialog([], [])
    with pytest.raises(RuntimeError, match="not CloudCompare's own"):
        dialog.make_progress_dialog("x")


def test_guard_passes_with_an_application():
    app = dialog.QApplication.instance() or dialog.QApplication([])
    dialog.check_qt_binding()
    assert app is dialog.QApplication.instance()
