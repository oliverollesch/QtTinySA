"""
Headless checks for the hardware-independent parts of the EMI map wizard.

The .ui wiring is checked by parsing the XML, and the serial-ownership rules
are checked against fake devices, so neither a QApplication nor a display nor
an analyser is needed.

Run from the src directory so that `modules` is importable:
    python -m pytest tests -q
"""

import os
import re
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.emi_map import (  # noqa: E402
    ENGINE_READ_TIMEOUT_S,
    PAGE_ORIGIN,
    PAGE_RESULTS,
    PAGE_SCAN,
    PAGE_SETUP,
    TRAVEL_WARNING,
    _MapSerial,
    engine_config,
    engine_scanner,
)

MODULES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules")
EMI_UI = os.path.join(MODULES, "emi_map.ui")
SPECTRUM_UI = os.path.join(MODULES, "spectrum.ui")


@pytest.fixture(scope="module")
def emi_widgets():
    root = ET.parse(EMI_UI).getroot()
    return {w.get("name"): w for w in root.iter("widget")}


def properties(widget):
    return {prop.get("name"): prop for prop in widget.findall("property")}


class TestMenuWiring:
    """The map is a Measurements entry beside the FCC test, not a preference."""

    def test_action_sits_in_the_measurements_menu(self):
        root = ET.parse(SPECTRUM_UI).getroot()
        menus = {w.get("name"): w for w in root.iter("widget") if w.get("class") == "QMenu"}
        entries = [a.get("name") for a in menus["menuMeasurements"].findall("addaction")]
        assert "actionEMIMap" in entries
        assert entries.index("actionEMIMap") == entries.index("actionFCCTest") + 1

    def test_action_label_matches_the_fcc_naming(self):
        root = ET.parse(SPECTRUM_UI).getroot()
        actions = {a.get("name"): a for a in root.iter("action")}
        text = properties(actions["actionEMIMap"])["text"].find("string").text
        assert text == "EMI Near-Field Map"


class TestDialogLayout:
    def test_all_four_pages_exist_in_order(self, emi_widgets):
        root = ET.parse(EMI_UI).getroot()
        stack = next(w for w in root.iter("widget") if w.get("name") == "wizardStack")
        pages = [w.get("name") for w in stack.findall("widget")]
        assert pages == ["pageSetup", "pageOrigin", "pageScan", "pageResults"]
        assert (PAGE_SETUP, PAGE_ORIGIN, PAGE_SCAN, PAGE_RESULTS) == (0, 1, 2, 3)

    def test_set_origin_starts_disabled(self, emi_widgets):
        prop = properties(emi_widgets["btnSetOrigin"]).get("enabled")
        assert prop is not None and prop.find("bool").text == "false"

    def test_travel_acknowledgement_checkbox_exists(self, emi_widgets):
        assert "chkTravelClear" in emi_widgets

    def test_no_widget_name_is_shadowed_by_the_qdialog_api(self, emi_widgets):
        # QUiLoader attaches children as attributes, so a widget called e.g.
        # "metric" would resolve to QPaintDevice.metric() instead
        from PySide6.QtWidgets import QDialog

        clashes = [name for name in emi_widgets if hasattr(QDialog, name)]
        assert clashes == []

    def test_no_z_or_home_controls_are_offered(self, emi_widgets):
        names = " ".join(emi_widgets).lower()
        assert "zmm" not in names and "probez" not in names
        assert "home" not in names


class TestWidgetReferences:
    """Every widget the controller touches must exist in the .ui file.

    The dialog is built by QUiLoader at runtime, so a typo here would only
    surface as an AttributeError in front of the operator.
    """

    def test_no_controller_reference_is_missing_from_the_ui(self, emi_widgets):
        source = open(
            os.path.join(MODULES, "emi_map.py"), encoding="utf-8"
        ).read()
        referenced = set(re.findall(r"(?:self\.ui|\bu)\.([A-Za-z_][A-Za-z0-9_]*)", source))
        # QDialog's own API plus the signals the controller connects to
        builtin = {"show", "raise_", "activateWindow", "hide", "rejected", "setEnabled"}
        declared = set(emi_widgets) | builtin
        assert referenced <= declared, sorted(referenced - declared)


class TestTravelWarning:
    def test_names_both_axes_and_the_distance(self):
        text = TRAVEL_WARNING.format(width=100, height=70)
        assert "+X by 100 mm" in text
        assert "+Y by 70 mm" in text
        assert "clear" in text.lower()


class _FakeTimer:
    def __init__(self, active=True):
        self.active = active
        self.interval = None

    def isActive(self):
        return self.active

    def stop(self):
        self.active = False

    def start(self, interval):
        self.active = True
        self.interval = interval


class _FakeUsb:
    def __init__(self):
        self.is_open = True
        self.timeout = None
        self.baudrate = 576000


class _FakeDevice:
    def __init__(self, sn=42, enabled=True, thread_running=False):
        self.sn = sn
        self.enabled = enabled
        self.threadRunning = thread_running
        self.usb = _FakeUsb()
        self.usbPort = "COM9"
        self.fifoTimer = _FakeTimer()
        self.fifo = _FakeQueue()
        self.aborted = False
        self.buffer_cleared = False

    def abort(self):
        self.aborted = True

    def clearBuffer(self):
        self.buffer_cleared = True


class _FakeQueue:
    def qsize(self):
        return 0

    def get_nowait(self):
        raise RuntimeError("empty")


class _FakeUsbInstr:
    def __init__(self, devices):
        self.devices = devices
        self.stopped = False

    def stop(self, restart=False):
        self.stopped = True


class TestSerialOwnership:
    def _owner(self, device):
        owner = _MapSerial(_FakeUsbInstr([device]))
        owner.adopt()
        return owner

    def test_take_stops_the_sweep_and_the_fifo(self):
        device = _FakeDevice()
        owner = self._owner(device)
        owner.take()
        assert owner.usb_instr.stopped is True
        assert device.fifoTimer.isActive() is False
        assert device.aborted is True

    def test_take_bounds_the_read_timeout_so_reads_can_be_cancelled(self):
        device = _FakeDevice()
        owner = self._owner(device)
        owner.take()
        # QtTinySA opens the analyser with no timeout, which would block forever
        assert device.usb.timeout == ENGINE_READ_TIMEOUT_S

    def test_release_restores_the_timeout_and_restarts_the_fifo(self):
        device = _FakeDevice()
        device.usb.timeout = 3
        owner = self._owner(device)
        owner.take()
        owner.release()
        assert device.usb.timeout == 3
        assert device.fifoTimer.isActive() is True

    def test_take_refuses_while_the_sweep_worker_is_still_reading(self):
        device = _FakeDevice(thread_running=True)
        owner = self._owner(device)
        with pytest.raises(RuntimeError):
            owner.take(wait_s=0.1)
        # A refused handover must leave the GUI's own scanner usable again
        assert device.fifoTimer.isActive() is True

    def test_release_is_safe_without_a_take(self):
        device = _FakeDevice()
        owner = self._owner(device)
        owner.release()
        assert device.usb.timeout is None

    def test_device_is_tracked_by_serial_number_not_index(self):
        first = _FakeDevice(sn=7)
        second = _FakeDevice(sn=9)
        instr = _FakeUsbInstr([first, second])
        owner = _MapSerial(instr)
        owner.adopt()
        instr.devices = [second, first]  # QtTinySA renumbers on reconnect
        assert owner.device() is first


class _StubWorkerHost:
    """Stands in for the wizard: take, run, release, no matter what happens."""

    def __init__(self, owner):
        self.owner = owner

    def run(self, body):
        self.owner.take()
        try:
            return body()
        finally:
            self.owner.release()


class TestReleaseOnException:
    def test_a_failing_scan_still_releases_the_port(self):
        device = _FakeDevice()
        owner = _MapSerial(_FakeUsbInstr([device]))
        owner.adopt()
        host = _StubWorkerHost(owner)

        def explode():
            raise engine_scanner.ScanAborted("cancelled")

        with pytest.raises(engine_scanner.ScanAborted):
            host.run(explode)

        assert device.fifoTimer.isActive() is True
        assert device.usb.timeout is None


class TestEngineConfig:
    """The wizard and the CLI must produce the same kind of ScanConfig."""

    def test_defaults_match_the_frozen_v1_recipe(self):
        config = engine_config.ScanConfig()
        assert config.tinysa.center_hz == 20_000_000
        assert config.tinysa.span_hz == 2_000_000
        assert config.tinysa.points == 101
        assert config.tinysa.samples_per_xy == 3
        assert config.metric == "peak"
        assert config.background_mode == "delta_db"
        assert config.printer.manage_z is False
        assert config.printer.home_xy is False

    def test_wizard_style_config_suppresses_the_engine_g92(self):
        # The Origin page already sent G92; a second one would move the map
        config = engine_config.ScanConfig(
            printer=engine_config.PrinterConfig(set_current_xy_as_origin=False)
        )
        config.validate()
        assert config.printer.set_current_xy_as_origin is False

    def test_managed_z_is_rejected(self):
        config = engine_config.ScanConfig(
            printer=engine_config.PrinterConfig(manage_z=True)
        )
        with pytest.raises(ValueError):
            config.validate()
