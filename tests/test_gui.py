import json
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtWidgets import QComboBox, QDialog, QLabel, QPushButton

import main


@pytest.fixture(autouse=True)
def local_settings_file(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "SETTINGS_PATH", tmp_path / "settings.json")


@pytest.fixture
def window(app, monkeypatch):
    # 构造实际 GUI，但禁止启动任何硬件线程。
    for cls in (main.CameraThread, main.LaplacianThread, main.ControlThread):
        monkeypatch.setattr(cls, "start", lambda self: None)
    widget = main.MainWindow(2, "COM12", "COM13")
    widget.control_thread.xy_ser = Mock(is_open=True)
    widget.control_thread.z_ser = Mock(is_open=True)
    widget.control_thread.busy = False
    widget.on_connections(True, True)
    widget.update_controls(False)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def find_button(window, text):
    # 方向键显示箭头，通过可访问名称识别轴和方向。
    return next(button for button in window.findChildren(QPushButton)
                if button.text() == text or button.accessibleName() == text)


def test_axis_units_and_negative_move(window):
    assert window.axis_inputs["X"].suffix() == " mm"
    assert window.axis_inputs["Z"].suffix() == " 步"
    window.axis_inputs["Y"].setValue(0.125)
    find_button(window, "Y− 单次").click()
    assert window.control_thread.commands.get_nowait() == ("move", "Y", -0.125)
    assert not find_button(window, "Z+ 单次").isEnabled()


def test_press_release_stops_and_does_not_queue_repeated_moves(window):
    button = find_button(window, "长按 Z+")
    button.setDown(True)
    button.pressed.emit()
    assert window.control_thread.commands.get_nowait() == ("jog", "Z", 1)
    assert button.isEnabled()
    assert not find_button(window, "X+ 单次").isEnabled()
    window.control_thread.jog_alive = 0
    window.activateWindow()
    window.keep_jog()
    button.released.emit()
    assert window.control_thread.cancel_motion.is_set()
    assert not window.jog_timer.isActive()
    assert window.control_thread.commands.empty()


@pytest.mark.parametrize("event_type", [QEvent.WindowDeactivate, QEvent.Leave, QEvent.MouseButtonRelease])
def test_lost_focus_leave_or_release_cancels_hold(window, event_type):
    button = find_button(window, "长按 X−")
    button.setDown(True)
    button.pressed.emit()
    assert window.control_thread.commands.get_nowait() == ("jog", "XY", (-1, 0))
    assert window.jog_button is button
    assert not window.control_thread.cancel_motion.is_set()
    watched = button if event_type == QEvent.Leave else window
    window.eventFilter(watched, QEvent(event_type))
    assert window.control_thread.cancel_motion.is_set()
    assert window.jog_button is None
    assert not window.jog_timer.isActive()


def test_existing_capture_buttons_keep_their_commands(window):
    for button, command in ((window.btn_save, "save"), (window.btn_auto, "auto"),
                            (window.btn_record_start, "record_start"),
                            (window.btn_record_stop, "record_stop")):
        button.click()
        assert window.camera_thread.commands.get_nowait() == command


def test_missing_axis_is_disabled_but_other_axis_works(window):
    window.on_connections(True, False)
    assert find_button(window, "X+ 单次").isEnabled()
    assert not find_button(window, "Z+ 单次").isEnabled()
    assert window.focus_label.isEnabled()


def test_focus_score_replaces_autofocus_button(window):
    assert not any("自动对焦" in button.text() for button in window.findChildren(QPushButton))
    window.laplacian_thread.focus_score_ready.emit(123.456)
    assert window.focus_label.text() == "123.46"
    assert window.control_thread.commands.empty()


def test_finished_hold_cannot_restart_without_a_new_press(window):
    button = find_button(window, "长按 Z+")
    button.pressed.emit()
    window.control_thread.commands.get_nowait()
    window.control_thread.busy = False
    window.update_controls(False)
    # 模拟仍按着鼠标移回按钮，而不是新的一次鼠标按下。
    button.pressed.emit()
    assert window.control_thread.commands.empty()


def test_close_keeps_window_alive_until_workers_finish(window, monkeypatch):
    monkeypatch.setattr(window.control_thread, "isRunning", lambda: True)
    assert not window.close()
    assert window.isVisible()
    assert not window.control_thread.running
    monkeypatch.setattr(window.control_thread, "isRunning", lambda: False)
    assert window.close()


def test_layout_puts_controls_left_status_above_image(window, app):
    app.processEvents()
    image_top = window.video_label.mapTo(window, window.video_label.rect().topLeft())
    status_top = window.status_label.mapTo(window, window.status_label.rect().topLeft())
    button_top = window.btn_stop.mapTo(window, window.btn_stop.rect().topLeft())
    assert button_top.x() < image_top.x()
    assert status_top.y() < image_top.y()
    assert window.video_label.width() > 600
    assert window.video_label.height() > 400


def test_close_requests_all_workers_stop(window):
    window.close()
    assert not window.control_thread.running
    assert not window.laplacian_thread.running
    assert window.camera_thread.commands.get_nowait() == "stop"
    assert window.control_thread.cancel_motion.is_set()


def test_startup_dialog_explains_units_and_can_skip_serial(app, monkeypatch):
    monkeypatch.setattr(main.list_ports, "comports", lambda: [])
    def accept_dialog():
        dialog = app.activeModalWidget()
        assert isinstance(dialog, QDialog)
        text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        assert "步数" in text and "mm" in text
        dialog.accept()
    QTimer.singleShot(0, accept_dialog)
    assert main.choose_settings() == (2, "", "")


def test_startup_rejects_duplicate_ports(app, monkeypatch):
    monkeypatch.setattr(main.list_ports, "comports", lambda: [])
    warnings = []
    def correct_ports(dialog, title, text):
        warnings.append(text)
        boxes = dialog.findChildren(QComboBox)
        boxes[2].setCurrentText("COM13")
        QTimer.singleShot(0, dialog.accept)
    monkeypatch.setattr(main.QMessageBox, "warning", correct_ports)
    def accept_dialog():
        dialog = app.activeModalWidget()
        boxes = dialog.findChildren(QComboBox)
        boxes[1].setCurrentText("COM12")
        boxes[2].setCurrentText("com12")
        dialog.accept()
    QTimer.singleShot(0, accept_dialog)
    assert main.choose_settings() == (2, "COM12", "COM13")
    assert len(warnings) == 1


def test_startup_remembers_last_entered_ports(app, monkeypatch):
    monkeypatch.setattr(main.list_ports, "comports", lambda: [])
    def enter_ports():
        dialog = app.activeModalWidget()
        boxes = dialog.findChildren(QComboBox)
        boxes[1].setCurrentText(" com12 ")
        boxes[2].setCurrentText("COM13")
        dialog.accept()
    QTimer.singleShot(0, enter_ports)
    assert main.choose_settings() == (2, "COM12", "COM13")
    assert json.loads(main.SETTINGS_PATH.read_text()) == {"xy_port": "COM12", "z_port": "COM13"}

    def check_remembered_ports():
        dialog = app.activeModalWidget()
        boxes = dialog.findChildren(QComboBox)
        assert boxes[1].currentText() == "COM12"
        assert boxes[2].currentText() == "COM13"
        dialog.reject()
    QTimer.singleShot(0, check_remembered_ports)
    assert main.choose_settings() is None
    assert json.loads(main.SETTINGS_PATH.read_text())["xy_port"] == "COM12"


@pytest.mark.parametrize("content", ["broken json", "[]", '{"xy_port": 12, "z_port": null}'])
def test_bad_saved_settings_do_not_prevent_startup(app, monkeypatch, content):
    monkeypatch.setattr(main.list_ports, "comports", lambda: [])
    main.SETTINGS_PATH.write_text(content)
    def accept_dialog():
        dialog = app.activeModalWidget()
        boxes = dialog.findChildren(QComboBox)
        assert boxes[1].currentText() == boxes[2].currentText() == ""
        dialog.accept()
    QTimer.singleShot(0, accept_dialog)
    assert main.choose_settings() == (2, "", "")
