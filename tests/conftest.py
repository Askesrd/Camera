import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QFontDatabase


@pytest.fixture(scope="session")
def app():
    application = QApplication.instance() or QApplication([])
    # Windows 的 offscreen 插件不自动载入系统字体，布局测试需显式提供。
    if not QFontDatabase().families():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
        application.setFont(QFont("Microsoft YaHei", 9))
    yield application


@pytest.fixture(autouse=True)
def forbid_real_hardware(monkeypatch):
    import serial
    import toupcam

    def forbidden(*args, **kwargs):
        raise AssertionError("离线测试不允许访问真实相机或串口")

    monkeypatch.setattr(serial, "Serial", forbidden)
    monkeypatch.setattr(toupcam.Toupcam, "Open", forbidden)
