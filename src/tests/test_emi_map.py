"""
Headless checks for the hardware-independent parts of the EMI map wizard.

The .ui wiring is checked by parsing the XML, and the serial-ownership rules
are checked against fake devices, so neither a QApplication nor a display nor
an analyser is needed.

Run from the src directory so that `modules` is importable:
    python -m pytest tests -q
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import emi_map  # noqa: E402
from modules.emi_map import (  # noqa: E402
    ENGINE_READ_TIMEOUT_S,
    MODE_PAGES,
    MODE_PCB,
    MODE_RECTANGLE,
    PAGE_BOARD,
    PAGE_ORIGIN,
    PAGE_REGISTER,
    PAGE_RESULTS,
    PAGE_SCAN,
    PAGE_SETUP,
    TRAVEL_WARNING,
    _MapSerial,
    engine_board,
    engine_config,
    engine_printer,
    engine_registration,
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
    def test_every_page_exists_in_order(self, emi_widgets):
        # The two PCB pages sit between Setup and Origin, and each mode walks
        # its own subset of this stack; see MODE_PAGES.
        root = ET.parse(EMI_UI).getroot()
        stack = next(w for w in root.iter("widget") if w.get("name") == "wizardStack")
        pages = [w.get("name") for w in stack.findall("widget")]
        assert pages == [
            "pageSetup",
            "pageBoard",
            "pageRegister",
            "pageOrigin",
            "pageScan",
            "pageResults",
        ]
        assert (PAGE_SETUP, PAGE_BOARD, PAGE_REGISTER) == (0, 1, 2)
        assert (PAGE_ORIGIN, PAGE_SCAN, PAGE_RESULTS) == (3, 4, 5)

    def test_setup_form_scrolls_so_the_nav_buttons_stay_on_screen(self, emi_widgets):
        # Setup is too tall for a laptop; the form scrolls, Back/Next do not.
        assert "setupScroll" in emi_widgets
        assert emi_widgets["setupScroll"].get("class") == "QScrollArea"
        root = ET.parse(EMI_UI).getroot()
        scroll = next(w for w in root.iter("widget") if w.get("name") == "setupScroll")
        inside = {w.get("name") for w in scroll.iter("widget")}
        assert "btnNext" not in inside and "btnBack" not in inside
        assert "setupInner" in inside

    def test_the_register_column_scrolls_so_the_nav_buttons_stay_on_screen(
        self, emi_widgets
    ):
        # The jog pad, landmark table and profile group together outgrow a
        # laptop viewport, exactly as the Setup form does. The control column
        # must scroll, not crush, and must not swallow the plot or nav buttons.
        assert emi_widgets["registerScroll"].get("class") == "QScrollArea"
        assert emi_widgets["registerSplit"].get("class") == "QSplitter"
        root = ET.parse(EMI_UI).getroot()
        scroll = next(w for w in root.iter("widget") if w.get("name") == "registerScroll")
        inside = {w.get("name") for w in scroll.iter("widget")}
        assert {"grpJog", "grpProfile", "landmarkTable"} <= inside
        assert "chkTravelClearBoard" not in inside
        assert not {"btnNext", "btnBack", "regPlot"} & inside
        layout = next(
            lay
            for lay in scroll.iter("layout")
            if lay.get("name") == "registerLeft"
        )
        constraint = next(
            prop.find("enum").text
            for prop in layout.findall("property")
            if prop.get("name") == "sizeConstraint"
        )
        assert constraint.endswith("SetMinimumSize")

    def test_spare_height_goes_to_the_board_view_not_the_intro(self, emi_widgets):
        # Without this stretch the intro label grows and centres its text,
        # which is the blank band at the top of the Register page.
        root = ET.parse(EMI_UI).getroot()
        page = next(w for w in root.iter("widget") if w.get("name") == "pageRegister")
        assert page.find("layout").get("stretch") == "0,0,1"

    def test_set_origin_starts_disabled(self, emi_widgets):
        prop = properties(emi_widgets["btnSetOrigin"]).get("enabled")
        assert prop is not None and prop.find("bool").text == "false"

    def test_travel_acknowledgement_checkbox_exists(self, emi_widgets):
        assert "chkTravelClear" in emi_widgets
        assert "chkTravelClearBoard" in emi_widgets
        root = ET.parse(EMI_UI).getroot()
        scan = next(w for w in root.iter("widget") if w.get("name") == "pageScan")
        on_scan = {w.get("name") for w in scan.iter("widget")}
        assert "chkTravelClearBoard" in on_scan
        assert "btnStartScan" in on_scan

    def test_no_widget_name_is_shadowed_by_the_qdialog_api(self, emi_widgets):
        # QUiLoader attaches children as attributes, so a widget called e.g.
        # "metric" would resolve to QPaintDevice.metric() instead
        from PySide6.QtWidgets import QDialog

        clashes = [name for name in emi_widgets if hasattr(QDialog, name)]
        assert clashes == []

    def test_no_z_controls_are_offered(self, emi_widgets):
        # Probe height stays a hand-set, locked quantity in both modes
        names = " ".join(emi_widgets).lower()
        assert "zmm" not in names and "probez" not in names
        assert "homez" not in names

    def test_xy_homing_sits_behind_its_own_clearance_acknowledgement(self, emi_widgets):
        # PCB mode homes X and Y, which is a different hazard from the travel
        # sweep, so it gets its own acknowledgement rather than sharing one.
        assert "btnHomeXy" in emi_widgets and "chkHomeClear" in emi_widgets
        assert properties(emi_widgets["btnHomeXy"])["enabled"].find("bool").text == "false"
        home = properties(emi_widgets["chkHomeClear"])["text"].find("string").text
        travel = properties(emi_widgets["chkTravelClearBoard"])["text"].find("string").text
        assert "homing" in home.lower()
        assert "travel" in travel.lower()
        assert home != travel


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
        builtin = {
            "show",
            "raise_",
            "activateWindow",
            "hide",
            "rejected",
            "setEnabled",
            "screen",
            "resize",
            "move",
            "setMaximumSize",
            "width",
            "height",
            "frameGeometry",
        }
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


# --------------------------------------------------------------- PCB harness
# The wizard itself is exercised here, not just its helpers, because the PCB
# interlocks live in the controller. Only the two pyqtgraph canvases are
# stubbed out; every state transition below is the real one.

_SIGNALS = {
    "clicked",
    "stateChanged",
    "valueChanged",
    "currentTextChanged",
    "currentIndexChanged",
    "currentChanged",
    "toggled",
}


class _FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class _FakeWidget:
    """Enough of a Qt widget for the controller: a value, text, enablement."""

    def __init__(self, value=0.0, text="", checked=False):
        self._signals = {}
        self._value = value
        self._text = text
        self._checked = checked
        self.enabled = True
        self.visible = True
        self._tooltip = ""
        self.rows = 0
        self.cells = {}

    def __getattr__(self, name):
        if name in _SIGNALS:
            return self._signals.setdefault(name, _FakeSignal())
        return lambda *args, **kwargs: None

    def value(self):
        return self._value

    def text(self):
        return self._text

    def currentText(self):
        return self._text

    def currentRow(self):
        return getattr(self, "_current_row", -1)

    def toPlainText(self):
        return self._text

    def isChecked(self):
        return self._checked

    def isEnabled(self):
        return self.enabled

    def setText(self, text):
        self._text = text

    def setPlainText(self, text):
        self._text = text

    def setEnabled(self, enabled):
        self.enabled = enabled

    def setVisible(self, visible):
        self.visible = visible

    def setToolTip(self, text):
        self._tooltip = text

    def toolTip(self):
        return self._tooltip

    def isVisible(self):
        return self.visible

    def setChecked(self, checked):
        self._checked = checked

    def setCurrentText(self, text):
        self._text = text
        self.currentTextChanged.emit(text)

    def setRowCount(self, count):
        self.rows = count

    def setItem(self, row, column, item):
        self.cells[(row, column)] = item


class _FakeStack(_FakeWidget):
    def __init__(self):
        super().__init__()
        self.index = 0

    def currentIndex(self):
        return self.index

    def setCurrentIndex(self, index):
        self.index = index
        self.currentChanged.emit(index)


class _FakeMessageBox:
    """Records what the operator would have been told, and by which call."""

    warnings = []
    questions = []
    # The enum members the controller passes through to question()
    Yes = 0x4000
    No = 0x10000
    answer = No

    @classmethod
    def warning(cls, _parent, _title, message):
        cls.warnings.append(message)

    critical = warning

    @classmethod
    def question(cls, _parent, _title, message, *_args):
        """Declines unless a test opts in, so nothing destructive happens by default."""
        cls.questions.append(message)
        return cls.answer


class _FakePrinter:
    def __init__(self, position=(0.0, 0.0), home_error=None, xy_error=None):
        self.position = position
        self.home_error = home_error
        self.xy_error = xy_error
        self.sent = []

    def drain(self):
        pass

    def prepare(self):
        self.sent.append("prepare")

    def home_xy(self):
        if self.home_error is not None:
            raise self.home_error
        self.sent.append("G28 X Y")
        self.position = (0.0, 0.0)

    def set_origin(self):
        self.sent.append("G92 X0 Y0")

    def get_xy(self):
        if self.xy_error is not None:
            raise self.xy_error
        return self.position

    def move_xy(self, x_mm, y_mm):
        self.sent.append(f"G0 X{x_mm} Y{y_mm}")
        self.position = (x_mm, y_mm)


class _CountingSerial(_MapSerial):
    """Counts handovers, so analyser-last ordering can be asserted, not assumed."""

    def __init__(self):
        super().__init__(_FakeUsbInstr([_FakeDevice()]))
        self.take_calls = 0

    def take(self, wait_s=4.0):
        self.take_calls += 1
        return super().take(wait_s=wait_s)


class _HeadlessWizard(emi_map.EMIMapWizard):
    """The wizard minus its two pyqtgraph canvases, which need a display."""

    def _setup_plot(self):
        pass

    def _setup_board_plots(self):
        pass

    def _draw_board(self, plot, view):
        pass

    def _printer(self):
        return self.printer


def _checked_in_ui(widget):
    """The designer's default tick state, so the fake starts where the real one does."""
    prop = properties(widget).get("checked")
    return prop is not None and prop.find("bool").text == "true"


def _fake_ui():
    """A stand-in dialog built from the .ui itself, so it cannot drift from it."""
    root = ET.parse(EMI_UI).getroot()
    ui = type("FakeUi", (), {})()
    for widget in root.iter("widget"):
        setattr(ui, widget.get("name"), _FakeWidget(checked=_checked_in_ui(widget)))
    ui.wizardStack = _FakeStack()
    ui.rejected = _FakeSignal()
    numbers = {
        "printerBaud": 115200,
        "settleMs": 200,
        "widthMm": 100.0,
        "heightMm": 100.0,
        "stepMm": 10.0,
        "centreMhz": 20.0,
        "spanMhz": 2.0,
        "points": 101,
        "samplesPerXy": 3,
        "rbwKhz": 0.0,
        "attenDb": -1,
        "jogStep": 1.0,
    }
    texts = {
        "printerPort": "COM9",
        "outputRoot": "scans",
        "runLabel": "",
        "boardPath": "",
        "scanMode": MODE_RECTANGLE,
        "boardSide": "top",
        "metricBox": "peak",
        "spur": "auto",
        "backgroundMode": "delta_db",
    }
    for name, value in numbers.items():
        setattr(ui, name, _FakeWidget(value=value))
    for name, value in texts.items():
        setattr(ui, name, _FakeWidget(text=value))
    return ui


@pytest.fixture
def wizard(monkeypatch):
    monkeypatch.setattr(emi_map, "QMessageBox", _FakeMessageBox)
    _FakeMessageBox.warnings = []
    _FakeMessageBox.questions = []
    _FakeMessageBox.answer = _FakeMessageBox.No
    made = _HeadlessWizard(_fake_ui(), _FakeUsbInstr([_FakeDevice()]), None)
    made.printer = _FakePrinter()
    return made


class _Port:
    def __init__(self, device, description="", manufacturer=""):
        self.device = device
        self.description = description
        self.manufacturer = manufacturer


class _FakeCombo(_FakeWidget):
    """A combo that remembers its items, so the offered ports can be asserted."""

    def __init__(self, text=""):
        super().__init__(text=text)
        self.items = []
        self.index = -1

    def clear(self):
        self.items = []
        self.index = -1

    def addItem(self, label, data=None):
        self.items.append((label, data))

    def findData(self, data):
        for index, (_label, value) in enumerate(self.items):
            if value == data:
                return index
        return -1

    def findText(self, text):
        for index, (label, _value) in enumerate(self.items):
            if label == text:
                return index
        return -1

    def setCurrentIndex(self, index):
        self.index = index
        self._text = self.items[index][0]

    def setCurrentText(self, text):
        self.index = -1
        super().setCurrentText(text)

    def currentData(self):
        if 0 <= self.index < len(self.items):
            return self.items[self.index][1]
        return None


class _SilentPort:
    """A port that opens and then says nothing, like a Bluetooth profile."""

    is_open = True

    def __init__(self):
        self.closed = False

    def write(self, data):
        return len(data)

    def read(self, _size=1):
        return b""

    def close(self):
        self.closed = True


class _MarlinPort(_SilentPort):
    def __init__(self):
        super().__init__()
        self._out = bytearray(b"start\n")  # opening the port reboots the board

    def write(self, data):
        self._out += b"ok FIRMWARE_NAME:Marlin\n"
        return len(data)

    def read(self, size=1):
        chunk = bytes(self._out[:size])
        del self._out[:size]
        return chunk


class TestChoosingThePrinterPort:
    """Windows advertises every paired Bluetooth profile as a COM port. Opening
    one stalls, then answers nothing, so picking the wrong port looks exactly
    like a wizard that has hung rather than like the trivial mistake it is."""

    @pytest.fixture
    def ports(self, wizard, monkeypatch):
        combo = _FakeCombo()
        wizard.ui.printerPort = combo
        monkeypatch.setattr(
            emi_map.list_ports,
            "comports",
            lambda: [
                _Port("COM15", "Standard Serial over Bluetooth link (COM15)", "Microsoft"),
                _Port("COM9", "USB Serial Device (COM9)", "Microsoft"),
                _Port("COM11", "USB-SERIAL CH340 (COM11)", "wch.cn"),
            ],
        )
        return combo

    def test_bluetooth_links_are_never_offered(self, wizard, ports):
        wizard._refresh_ports()
        assert all(device != "COM15" for _, device in ports.items)

    def test_the_usb_serial_bridge_is_offered_first(self, wizard, ports):
        wizard._refresh_ports()
        assert ports.items[0] == ("COM11 - USB-SERIAL CH340 (COM11)", "COM11")

    def test_the_analyser_port_is_not_offered_as_the_printer(self, wizard, ports):
        wizard._refresh_ports()  # the stub analyser sweeps on COM9
        assert all(device != "COM9" for _, device in ports.items)

    def test_the_description_never_reaches_the_engine(self, wizard, ports):
        wizard._refresh_ports()
        ports.setCurrentIndex(0)
        assert wizard._printer_port() == "COM11"
        assert wizard.build_config().printer.port == "COM11"

    def test_a_hand_typed_port_still_works(self, wizard, ports):
        wizard._refresh_ports()
        ports.setCurrentText("COM7")
        assert wizard._printer_port() == "COM7"


class TestRefusingAPortThatIsNotAPrinter:
    @pytest.fixture(autouse=True)
    def quick(self, monkeypatch):
        monkeypatch.setattr(emi_map, "PRINTER_BOOT_S", 0.0)
        monkeypatch.setattr(emi_map, "PRINTER_GREET_S", 0.05)

    def test_a_silent_port_is_rejected_by_name(self, wizard):
        with pytest.raises(RuntimeError, match="COM7 did not answer"):
            wizard._greet_marlin(_SilentPort(), "COM7")

    def test_marlin_is_accepted_through_its_reboot_banner(self, wizard):
        wizard._greet_marlin(_MarlinPort(), "COM11")

    def test_a_refused_port_is_not_left_open(self, wizard, monkeypatch):
        opened = _SilentPort()
        monkeypatch.setattr(wizard, "_open_serial", lambda port, baud: opened)
        with pytest.raises(RuntimeError):
            wizard._open_printer()
        assert opened.closed
        assert wizard.printer_serial is None


def board_50x30(sha=""):
    """The fixture board. With ``sha`` it looks like it came from an archive."""
    report = None
    if sha:
        report = engine_board.ImportReport(
            source_name="job.tgz", source_sha256=sha, cad_units="MM", job_name="FIXTURE"
        )
    return engine_board.rectangular_board(50.0, 30.0, report=report)


def register(wizard, offsets=((0.0, 0.0), (50.0, 30.0)), machine_origin=(60.0, 40.0)):
    """Home, then record landmarks at given board points with a pure translation."""
    wizard.ui.chkHomeClear.setChecked(True)
    wizard._home_xy()
    for board_x, board_y in offsets:
        wizard._select_landmark(board_x, board_y)
        wizard.printer.position = (
            machine_origin[0] + board_x,
            machine_origin[1] + board_y,
        )
        wizard._record_landmark()


def register_points(wizard, pairs, machine_origin=(60.0, 40.0)):
    """Record exact board/machine pairs, bypassing the click snapping.

    Clicks snap to the nearest landmark, so two clicks a couple of millimetres
    apart collapse onto one corner; the separation gate is about pairs that are
    genuinely distinct and still too close.
    """
    wizard.ui.chkHomeClear.setChecked(True)
    wizard._home_xy()
    for board_x, board_y in pairs:
        wizard._landmarks.append(
            engine_registration.RegistrationPoint(
                board_x_mm=board_x,
                board_y_mm=board_y,
                machine_x_mm=machine_origin[0] + board_x,
                machine_y_mm=machine_origin[1] + board_y,
            )
        )
    wizard._refit_registration()


@pytest.fixture
def pcb_wizard(wizard):
    wizard.ui.scanMode.setCurrentText(MODE_PCB)
    wizard._board_model = board_50x30()
    wizard._apply_board_side()
    return wizard


class TestModeNavigation:
    def test_each_mode_walks_only_its_own_pages(self):
        assert PAGE_BOARD not in MODE_PAGES[MODE_RECTANGLE]
        assert PAGE_REGISTER not in MODE_PAGES[MODE_RECTANGLE]
        assert PAGE_ORIGIN not in MODE_PAGES[MODE_PCB]

    def test_rectangle_next_skips_the_board_pages(self, wizard):
        wizard.ui.wizardStack.setCurrentIndex(PAGE_SETUP)
        wizard._on_next()
        assert wizard.ui.wizardStack.currentIndex() == PAGE_ORIGIN

    def test_pcb_next_reaches_the_board_page_and_skips_the_origin(self, pcb_wizard):
        pcb_wizard.ui.wizardStack.setCurrentIndex(PAGE_SETUP)
        pcb_wizard._on_next()
        assert pcb_wizard.ui.wizardStack.currentIndex() == PAGE_BOARD
        pcb_wizard._on_next()
        assert pcb_wizard.ui.wizardStack.currentIndex() == PAGE_REGISTER
        register(pcb_wizard)
        pcb_wizard._on_next()
        assert pcb_wizard.ui.wizardStack.currentIndex() == PAGE_SCAN

    def test_back_returns_along_the_same_sequence(self, pcb_wizard):
        pcb_wizard.ui.wizardStack.setCurrentIndex(PAGE_REGISTER)
        pcb_wizard._on_back()
        assert pcb_wizard.ui.wizardStack.currentIndex() == PAGE_BOARD

    def test_next_names_the_rectangle_origin_page(self, wizard):
        wizard.ui.wizardStack.setCurrentIndex(PAGE_SETUP)
        wizard._update_nav()
        assert wizard.ui.btnNext.text() == "Next: set origin"

    def test_next_names_the_pcb_import_page(self, pcb_wizard):
        pcb_wizard.ui.wizardStack.setCurrentIndex(PAGE_SETUP)
        pcb_wizard._update_nav()
        assert pcb_wizard.ui.btnNext.text() == "Next: import board"

    def test_the_origin_page_says_it_is_not_the_pcb_path(self, emi_widgets):
        text = properties(emi_widgets["originIntro"])["text"].find("string").text
        assert "Rectangle mode only" in text
        assert "PCB aligned" in text

    def test_the_board_page_gates_next_until_a_board_exists(self, wizard):
        wizard.ui.scanMode.setCurrentText(MODE_PCB)
        wizard.ui.wizardStack.setCurrentIndex(PAGE_BOARD)
        wizard._update_nav()
        assert wizard.ui.btnNext.isEnabled() is False


class TestPcbScanConfig:
    def test_a_pcb_plan_puts_the_config_in_the_board_frame(self, pcb_wizard):
        register(pcb_wizard)
        config = pcb_wizard.build_config()
        plan = pcb_wizard._build_validated_plan(config)
        assert config.frame == "board"
        assert config.printer.set_current_xy_as_origin is False
        # The engine cross-checks the two, so agreement here is the contract
        engine_scanner._check_plan_matches_config(config, plan)

    def test_the_plan_covers_the_board_on_the_step_grid(self, pcb_wizard):
        register(pcb_wizard)
        plan = pcb_wizard._build_validated_plan(pcb_wizard.build_config())
        assert (plan.nx, plan.ny) == (6, 4)  # 50x30 mm on a 10 mm step, inclusive
        assert plan.point_count == 24
        assert plan.machine_bbox_mm() == pytest.approx((60.0, 40.0, 110.0, 70.0))

    def test_a_board_off_the_bed_is_refused_before_anything_moves(self, pcb_wizard):
        register(pcb_wizard, machine_origin=(200.0, 40.0))
        with pytest.raises(ValueError, match="machine limits"):
            pcb_wizard._build_validated_plan(pcb_wizard.build_config())


class TestRegistrationGates:
    def test_two_far_apart_landmarks_are_accepted(self, pcb_wizard):
        register(pcb_wizard)
        assert pcb_wizard._registration_problems() == []
        assert "Registration: PASS" in pcb_wizard.ui.regStatus.text()

    def test_a_click_near_a_corner_snaps_onto_that_corner(self, pcb_wizard):
        pcb_wizard._select_landmark(1.0, 1.5)
        assert pcb_wizard._selected_landmark[1:] == (0.0, 0.0)

    def test_close_landmarks_are_rejected_even_with_a_zero_residual(self, pcb_wizard):
        register_points(pcb_wizard, [(0.0, 0.0), (2.0, 0.0)])
        registration = pcb_wizard._registration
        # A rigid 2D fit has three degrees of freedom, so two points always fit
        assert registration.rms_mm == pytest.approx(0.0, abs=1e-9)
        assert pcb_wizard._registration_problems() != []
        report = pcb_wizard.ui.regStatus.text()
        assert "Registration: BLOCKED" in report
        assert "Separation:" in report and "FAIL" in report

    def test_a_rejected_registration_keeps_next_disabled(self, pcb_wizard):
        register_points(pcb_wizard, [(0.0, 0.0), (2.0, 0.0)])
        pcb_wizard.ui.wizardStack.setCurrentIndex(PAGE_REGISTER)
        pcb_wizard._update_nav()
        assert pcb_wizard.ui.btnNext.isEnabled() is False

    def test_a_third_landmark_is_recommended_but_never_required(self, pcb_wizard):
        register(pcb_wizard)
        assert "third landmark" in pcb_wizard.ui.landmarkHint.text()
        pcb_wizard.ui.wizardStack.setCurrentIndex(PAGE_REGISTER)
        pcb_wizard._update_nav()
        assert pcb_wizard.ui.btnNext.isEnabled() is True

    def test_a_landmark_recorded_without_jogging_is_refused(self, pcb_wizard):
        """Clicking a second corner while the probe still sits on the first is
        the easiest slip to make, and it silently poisons the fit."""
        register(pcb_wizard, offsets=((0.0, 0.0),))
        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._record_landmark()  # probe deliberately not moved
        assert len(pcb_wizard._landmarks) == 1
        assert any("Jog it onto" in text for text in _FakeMessageBox.warnings)

    def test_the_same_board_point_cannot_be_recorded_twice(self, pcb_wizard):
        register(pcb_wizard, offsets=((0.0, 0.0),))
        pcb_wizard._select_landmark(0.0, 0.0)
        pcb_wizard.printer.position = (99.0, 99.0)
        pcb_wizard._record_landmark()
        assert len(pcb_wizard._landmarks) == 1
        assert any("already landmark 1" in text for text in _FakeMessageBox.warnings)

    def test_landmarks_stuck_on_one_machine_point_say_exactly_that(self, pcb_wizard):
        """The generic 'spread them out' advice describes the wrong mistake."""
        pcb_wizard.ui.chkHomeClear.setChecked(True)
        pcb_wizard._home_xy()
        for board_x, board_y in ((0.0, 0.0), (50.0, 30.0), (0.0, 30.0)):
            pcb_wizard._landmarks.append(
                engine_registration.RegistrationPoint(
                    board_x_mm=board_x,
                    board_y_mm=board_y,
                    machine_x_mm=69.0,
                    machine_y_mm=135.0,
                )
            )
        pcb_wizard._refit_registration()
        report = pcb_wizard.ui.regStatus.text()
        assert "Registration: BLOCKED" in report
        assert "machine X 69.00, Y 135.00 mm" in report
        assert "never moved between them" in report

    def test_one_bad_landmark_can_be_dropped_without_losing_the_rest(self, pcb_wizard):
        register(pcb_wizard, offsets=((0.0, 0.0), (50.0, 30.0), (0.0, 30.0)))
        pcb_wizard.ui.landmarkTable._current_row = 1
        pcb_wizard._remove_landmark()
        assert [(p.board_x_mm, p.board_y_mm) for p in pcb_wizard._landmarks] == [
            (0.0, 0.0),
            (0.0, 30.0),
        ]
        assert pcb_wizard._registration.point_count == 2

    def test_a_coarse_jog_step_is_named_as_the_cause(self, pcb_wizard):
        """A sub-millimetre gate cannot be met by parking within 2.5 mm."""
        register(pcb_wizard)
        # 1% of the 58 mm baseline: exactly the error a 5 mm jog step leaves
        pcb_wizard._landmarks[1] = engine_registration.RegistrationPoint(
            board_x_mm=50.0,
            board_y_mm=30.0,
            machine_x_mm=112.0,
            machine_y_mm=70.0,
        )
        pcb_wizard.ui.jogStep._value = 5.0
        pcb_wizard._refit_registration()
        report = pcb_wizard.ui.regStatus.text()
        assert "jog step is 5 mm" in report
        assert "0.1 mm for the last millimetre" in report

    def test_a_fine_jog_step_draws_no_such_remark(self, pcb_wizard):
        register(pcb_wizard)
        pcb_wizard.ui.jogStep._value = 0.1
        pcb_wizard._refit_registration()
        assert "jog step" not in pcb_wizard.ui.regStatus.text()

    def test_recording_needs_a_homed_machine(self, wizard):
        wizard.ui.scanMode.setCurrentText(MODE_PCB)
        wizard._board_model = board_50x30()
        wizard._apply_board_side()
        wizard._select_landmark(0.0, 0.0)
        wizard._record_landmark()
        assert wizard._landmarks == []


class TestInvalidationCascade:
    def test_a_rectangle_origin_costs_the_pcb_its_homing(self, pcb_wizard):
        register(pcb_wizard)
        pcb_wizard._set_origin()
        # G92 moved Marlin's workspace, and stock Ender builds have no G92.1
        assert pcb_wizard.origin_set is True
        assert pcb_wizard._pcb_xy_homed is False
        assert pcb_wizard._landmarks == []
        assert pcb_wizard._registration is None

    def test_flipping_the_side_clears_the_fit_but_keeps_homing(self, pcb_wizard):
        register(pcb_wizard)
        plan = pcb_wizard._build_validated_plan(pcb_wizard.build_config())
        assert plan is pcb_wizard._plan
        pcb_wizard.ui.boardSide.setCurrentText("bottom")
        pcb_wizard._apply_board_side()
        assert pcb_wizard._board_view.side == "bottom"
        assert pcb_wizard._landmarks == []
        assert pcb_wizard._registration is None
        assert pcb_wizard._plan is None
        # Flipping the board does not move the Ender's frame
        assert pcb_wizard._pcb_xy_homed is True

    def test_changing_the_step_drops_the_plan_but_keeps_the_landmarks(self, pcb_wizard):
        register(pcb_wizard)
        pcb_wizard._build_validated_plan(pcb_wizard.build_config())
        pcb_wizard.ui.stepMm._value = 5.0
        pcb_wizard._on_area_changed()
        assert pcb_wizard._plan is None
        assert len(pcb_wizard._landmarks) == 2
        assert pcb_wizard._registration is not None

    def test_a_failed_home_leaves_nothing_standing(self, pcb_wizard):
        register(pcb_wizard)
        pcb_wizard.printer.home_error = engine_printer.PrinterTimeout("no ok")
        pcb_wizard._home_xy()
        assert pcb_wizard._pcb_xy_homed is False
        assert pcb_wizard._landmarks == []
        assert _FakeMessageBox.warnings

    def test_a_printer_reset_during_the_scan_unhomes_the_gui(self, pcb_wizard):
        register(pcb_wizard)
        # The reset happens on the worker, inside run_scan, so the type has to
        # survive the thread hop or the GUI keeps claiming the machine is homed
        pcb_wizard._on_printer_reset()
        assert pcb_wizard._pcb_xy_homed is False
        assert pcb_wizard._landmarks == []
        assert pcb_wizard._registration is None

    def test_the_worker_reports_a_reset_as_a_reset(self, monkeypatch):
        # The generic handler would flatten the exception into a string and the
        # GUI could not tell a lost machine frame from any other scan failure
        def explode(*_args, **_kwargs):
            raise engine_printer.PrinterReset("Marlin reset detected; home XY again")

        monkeypatch.setattr(emi_map.engine_scanner, "run_scan", explode)
        worker = emi_map.ScanWorker(engine_config.ScanConfig(), None, None)
        resets, failures = [], []
        worker.printer_reset.connect(lambda: resets.append(True))
        worker.failed.connect(failures.append)
        worker.run()
        assert resets == [True]
        assert failures and "PrinterReset" in failures[0]

    def test_switching_the_printer_port_unhomes_the_gui(self, pcb_wizard):
        register(pcb_wizard)
        pcb_wizard.ui.printerPort.setCurrentText("COM11")
        assert pcb_wizard._pcb_xy_homed is False
        assert pcb_wizard._landmarks == []


class TestAnalyserBorrowedLast:
    """Preflight refusals must never cost QtTinySA its sweep."""

    def _armed(self, wizard):
        wizard.serial = _CountingSerial()
        wizard.serial.adopt()
        return wizard

    def test_an_unchecked_clearance_blocks_the_start_without_taking_the_analyser(
        self, pcb_wizard
    ):
        register(pcb_wizard)
        wizard = self._armed(pcb_wizard)
        wizard._start_scan()
        assert wizard.serial.take_calls == 0
        assert wizard.thread is None
        assert any("clear" in text.lower() for text in _FakeMessageBox.warnings)

    def test_an_unhomed_machine_blocks_the_start_without_taking_the_analyser(
        self, pcb_wizard
    ):
        register(pcb_wizard)
        pcb_wizard._clear_pcb_homing()
        pcb_wizard.ui.chkTravelClearBoard.setChecked(True)
        wizard = self._armed(pcb_wizard)
        wizard._start_scan()
        assert wizard.serial.take_calls == 0
        assert any("Home X and Y" in text for text in _FakeMessageBox.warnings)

    def test_a_bad_registration_blocks_the_start_without_taking_the_analyser(
        self, pcb_wizard
    ):
        register_points(pcb_wizard, [(0.0, 0.0), (2.0, 0.0)])
        pcb_wizard.ui.chkTravelClearBoard.setChecked(True)
        wizard = self._armed(pcb_wizard)
        wizard._start_scan()
        assert wizard.serial.take_calls == 0
        assert any("Registration" in text for text in _FakeMessageBox.warnings)

    def test_an_unreachable_board_blocks_the_start_without_taking_the_analyser(
        self, pcb_wizard
    ):
        register(pcb_wizard, machine_origin=(200.0, 40.0))
        pcb_wizard.ui.chkTravelClearBoard.setChecked(True)
        wizard = self._armed(pcb_wizard)
        wizard._start_scan()
        assert wizard.serial.take_calls == 0
        assert any("machine limits" in text for text in _FakeMessageBox.warnings)

    def test_the_wizard_gives_the_analyser_back_when_the_worker_ends(self, wizard):
        # ScanWorker.run emits ended from a finally, so this one handler covers
        # success, cancellation and any exception run_scan raises alike
        device = _FakeDevice()
        device.usb.timeout = 3
        wizard.serial = _MapSerial(_FakeUsbInstr([device]))
        wizard.serial.adopt()
        wizard.serial.take()
        wizard._on_scan_ended()
        assert device.usb.timeout == 3
        assert device.fifoTimer.isActive() is True

    def test_the_happy_path_only_needs_the_clearance_tick(self, pcb_wizard):
        register(pcb_wizard)
        assert pcb_wizard._registration_problems() == []
        assert pcb_wizard._pcb_xy_homed is True

        with pytest.raises(RuntimeError, match="clear"):
            pcb_wizard._pcb_preflight()

        pcb_wizard.ui.chkTravelClearBoard.setChecked(True)
        pcb_wizard._pcb_preflight()  # no longer raises
        plan = pcb_wizard._build_validated_plan(pcb_wizard.build_config())
        assert plan.point_count == 24

    def test_start_stays_off_until_travel_is_confirmed_on_the_scan_page(self, pcb_wizard):
        register(pcb_wizard)
        pcb_wizard.ui.wizardStack.setCurrentIndex(PAGE_SCAN)
        pcb_wizard._update_nav()
        assert pcb_wizard.ui.btnStartScan.isEnabled() is False
        assert "clear" in pcb_wizard.ui.scanStatus.text().lower()
        pcb_wizard.ui.chkTravelClearBoard.setChecked(True)
        pcb_wizard._update_nav()
        assert pcb_wizard.ui.btnStartScan.isEnabled() is True
        assert pcb_wizard.ui.scanStatus.text() == "Ready to scan."

    def test_a_refused_start_is_written_on_the_scan_page(self, pcb_wizard):
        register(pcb_wizard)
        pcb_wizard._start_scan()
        assert "clear" in pcb_wizard.ui.scanStatus.text().lower()
        assert any("clear" in text.lower() for text in _FakeMessageBox.warnings)


class TestEndToEndBoardScan:
    """What the wizard builds must be what the engine accepts, run for real.

    Uses the engine's own fake transports, so no printer, analyser or display
    is involved; only the wizard-to-engine contract is under test.
    """

    def test_a_wizard_built_plan_scans_the_board_and_writes_the_overlays(
        self, pcb_wizard, tmp_path
    ):
        from EMI_Mapper import fakes

        register(pcb_wizard)
        pcb_wizard.ui.chkTravelClearBoard.setChecked(True)
        pcb_wizard.ui.outputRoot._text = str(tmp_path)

        config = pcb_wizard.build_config()
        plan = pcb_wizard._preflight(config)
        result = engine_scanner.run_scan(
            config,
            tinysa_transport=fakes.FakeTinySATransport(config.tinysa),
            printer_transport=fakes.FakePrinterTransport(),
            plan=plan,
        )

        assert result.plan.plan_id == plan.plan_id
        measured = result.power["dut"]
        assert measured.shape == (plan.ny, plan.nx)
        # Off-board cells are never driven to and stay NaN
        assert int(np.isfinite(measured).sum()) == plan.point_count
        for name in ("board_json", "board_png", "board_html", "heatmap_csv"):
            assert Path(result.files[name]).stat().st_size > 0
        # And the results page can offer the overlay off that key
        pcb_wizard.result = result
        assert pcb_wizard.result.files.get("board_html")


# ------------------------------------------------------- saved profiles
# A loaded profile fits perfectly by construction, so the registration maths
# alone would open the scan gate before anyone had looked at the board. These
# cover the verification state that stands between the two.


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    home = tmp_path / "emi_fixtures"
    monkeypatch.setattr(
        emi_map.engine_profiles, "profiles_dir", lambda: home, raising=True
    )
    return home


IDENTITY = {
    "fixtureId": "bed fixture A",
    "machineId": "ender3-01",
    "probeSetupId": "loop-6mm-holder-v2",
}


def declare_identity(wizard, **overrides):
    for name, value in {**IDENTITY, **overrides}.items():
        getattr(wizard.ui, name)._text = value


def save_current(wizard, name="board a", *, orientation_locked=False):
    """Save the registration the wizard currently holds, as the operator would."""
    wizard.ui.chkOrientationLocked.setChecked(orientation_locked)
    return emi_map.engine_profiles.save_profile(
        name,
        wizard._board_view,
        wizard._registration,
        fixture_id=wizard.ui.fixtureId.text(),
        machine_id=wizard.ui.machineId.text(),
        probe_setup_id=wizard.ui.probeSetupId.text(),
        orientation_locked=orientation_locked,
        printer_config=wizard._printer_config(),
        overwrite=True,
    )


def load_saved(wizard, name="board a"):
    """Seat the board, select the profile and load it, as the operator would."""
    wizard.ui.fixtureProfile = _FakeCombo()
    wizard.ui.fixtureProfile.addItem(name, name)
    wizard.ui.fixtureProfile.setCurrentIndex(0)
    wizard.ui.chkBoardSeated.setChecked(True)
    wizard._load_profile()


def prepared_profile(wizard, *, orientation_locked=False):
    """A homed wizard holding a freshly loaded, not yet verified, profile."""
    declare_identity(wizard)
    register(wizard)
    save_current(wizard, orientation_locked=orientation_locked)
    wizard._clear_landmarks_and_alignment()
    load_saved(wizard)
    return wizard


def verify_point(wizard, board_x, board_y, *, error=(0.0, 0.0)):
    """Select a point, drive to it, park the probe, and confirm the alignment."""
    wizard._select_landmark(board_x, board_y)
    wizard._move_now()
    if error != (0.0, 0.0):
        wizard.printer.position = (
            wizard.printer.position[0] + error[0],
            wizard.printer.position[1] + error[1],
        )
    wizard._confirm_alignment()


class TestProfileWidgets:
    def test_the_move_button_is_separate_from_the_plot_click(self, emi_widgets):
        # A stray click on a plot must never start the machine moving
        assert "btnMoveHere" in emi_widgets
        assert properties(emi_widgets["btnMoveHere"])["enabled"].find("bool").text == "false"

    def test_snapping_is_on_by_default(self, emi_widgets):
        prop = properties(emi_widgets["chkSnapLandmark"])["checked"]
        assert prop.find("bool").text == "true"

    def test_the_seating_checkbox_does_not_claim_stops_that_may_not_exist(
        self, emi_widgets
    ):
        text = properties(emi_widgets["chkBoardSeated"])["text"].find("string").text
        tip = properties(emi_widgets["chkBoardSeated"])["toolTip"].find("string").text
        assert "seated" in text.lower()
        assert "same position and orientation" in tip

    def test_the_orientation_lock_says_what_it_buys(self, emi_widgets):
        text = properties(emi_widgets["chkOrientationLocked"])["text"].find("string").text
        tip = properties(emi_widgets["chkOrientationLocked"])["toolTip"].find("string").text
        assert "orientation" in text.lower()
        assert "mechanically constrained" in tip
        assert "one verification point" in tip

    def test_saving_is_offered_as_new_and_as_an_update_separately(self, emi_widgets):
        assert "btnSaveAsProfile" in emi_widgets
        assert "btnUpdateProfile" in emi_widgets

    def test_the_four_jog_presets_reach_a_tenth_and_a_twentieth(self, emi_widgets):
        labels = [
            properties(emi_widgets[name])["text"].find("string").text
            for name in ("btnStep10", "btnStep1", "btnStep01", "btnStep005")
        ]
        assert labels == ["10", "1", "0.1", "0.05"]
        assert emi_map.JOG_STEP_PRESETS == (10.0, 1.0, 0.1, 0.05)

    def test_a_preset_only_has_to_be_within_the_spinbox_range(self, emi_widgets):
        minimum = float(
            properties(emi_widgets["jogStep"])["minimum"].find("double").text
        )
        assert min(emi_map.JOG_STEP_PRESETS) >= minimum


class TestExactClickSelection:
    def test_snapping_off_selects_the_point_that_was_clicked(self, pcb_wizard):
        pcb_wizard.ui.chkSnapLandmark.setChecked(False)
        pcb_wizard._select_landmark(21.5, 12.25)
        assert pcb_wizard._selected_landmark[1:] == (21.5, 12.25)

    def test_snapping_on_still_snaps(self, pcb_wizard):
        pcb_wizard._select_landmark(21.5, 12.25)
        assert pcb_wizard._selected_landmark[1:] != (21.5, 12.25)

    def test_a_click_off_the_board_selects_nothing(self, pcb_wizard):
        pcb_wizard.ui.chkSnapLandmark.setChecked(False)
        pcb_wizard._select_landmark(80.0, 12.0)
        assert pcb_wizard._selected_landmark is None
        assert "off the board outline" in pcb_wizard.ui.targetLabel.text()

    def test_a_click_on_the_drawn_edge_counts_as_on_the_board(self, pcb_wizard):
        pcb_wizard.ui.chkSnapLandmark.setChecked(False)
        inside = emi_map.CLICK_EDGE_TOLERANCE_MM / 2.0
        pcb_wizard._select_landmark(50.0 + inside, 15.0)
        assert pcb_wizard._selected_landmark is not None

    def test_an_off_board_point_can_never_become_a_landmark(self, pcb_wizard):
        register(pcb_wizard, offsets=((0.0, 0.0),))
        pcb_wizard.ui.chkSnapLandmark.setChecked(False)
        pcb_wizard._select_landmark(80.0, 12.0)
        pcb_wizard.printer.position = (999.0, 999.0)
        pcb_wizard._record_landmark()
        assert len(pcb_wizard._landmarks) == 1

    def test_toggling_snapping_drops_the_selection(self, pcb_wizard):
        pcb_wizard._select_landmark(0.0, 0.0)
        pcb_wizard.ui.chkSnapLandmark.setChecked(False)
        pcb_wizard._on_snap_changed()
        assert pcb_wizard._selected_landmark is None


class TestClickToMove:
    def test_the_target_is_previewed_in_machine_coordinates(self, pcb_wizard):
        register(pcb_wizard)
        pcb_wizard._select_landmark(50.0, 30.0)
        assert "machine X 110.00 Y 70.00" in pcb_wizard.ui.targetLabel.text()

    def test_moving_drives_to_the_mapped_machine_point(self, pcb_wizard):
        register(pcb_wizard)
        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._move_now()
        assert pcb_wizard.printer.position == (110.0, 70.0)

    def test_moving_needs_a_homed_machine(self, pcb_wizard):
        register(pcb_wizard)
        pcb_wizard._select_landmark(0.0, 0.0)
        # The fit survives, so homing is the only thing under test here
        pcb_wizard._pcb_xy_homed = False
        pcb_wizard._move_now()
        assert not any(c.startswith("G0 X60.0") for c in pcb_wizard.printer.sent)
        assert any("Home X and Y" in text for text in _FakeMessageBox.warnings)

    def test_moving_needs_an_alignment(self, pcb_wizard):
        pcb_wizard.ui.chkHomeClear.setChecked(True)
        pcb_wizard._home_xy()
        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._move_now()
        assert pcb_wizard.ui.btnMoveHere.isEnabled() is False
        assert pcb_wizard.printer.sent.count("prepare") == 1  # homing only

    def test_a_target_off_the_bed_is_refused_in_board_terms(self, pcb_wizard):
        register(pcb_wizard, machine_origin=(200.0, 40.0))
        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._move_now()
        assert not any(c.startswith("G0") for c in pcb_wizard.printer.sent)
        message = " ".join(_FakeMessageBox.warnings)
        assert "outside the machine range" in message
        assert "board (50.00, 30.00)" in message

    def test_no_move_ever_emits_z_or_g92(self, pcb_wizard):
        register(pcb_wizard)
        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._move_now()
        pcb_wizard._select_landmark(0.0, 0.0)
        pcb_wizard._move_now()
        moves = [c for c in pcb_wizard.printer.sent if c.startswith("G0")]
        assert moves
        assert not any("Z" in command for command in moves)
        assert not any(command.startswith("G92") for command in pcb_wizard.printer.sent)


class TestProfileRoundTrip:
    def test_a_loaded_profile_restores_the_transform(self, pcb_wizard, profile_home):
        declare_identity(pcb_wizard)
        register(pcb_wizard)
        before = pcb_wizard._registration.transform
        save_current(pcb_wizard)
        pcb_wizard._clear_landmarks_and_alignment()
        assert pcb_wizard._registration is None

        load_saved(pcb_wizard)
        assert pcb_wizard._registration is not None
        assert engine_registration.transforms_agree(
            pcb_wizard._registration.transform, before, [(0, 0), (50, 30)]
        )
        assert len(pcb_wizard._landmarks) == 2

    def test_loading_needs_the_board_seated(self, pcb_wizard, profile_home):
        declare_identity(pcb_wizard)
        register(pcb_wizard)
        save_current(pcb_wizard)
        pcb_wizard._clear_landmarks_and_alignment()
        pcb_wizard.ui.fixtureProfile = _FakeCombo()
        pcb_wizard.ui.fixtureProfile.addItem("board a", "board a")
        pcb_wizard.ui.fixtureProfile.setCurrentIndex(0)
        pcb_wizard._load_profile()
        assert pcb_wizard._profile is None
        assert any("Seat the board" in text for text in _FakeMessageBox.warnings)

    def test_a_refused_profile_leaves_the_current_alignment_alone(
        self, pcb_wizard, profile_home
    ):
        declare_identity(pcb_wizard)
        register(pcb_wizard)
        save_current(pcb_wizard)
        # The same board, flipped: a saved top-side transform cannot describe it
        pcb_wizard.ui.boardSide.setCurrentText("bottom")
        pcb_wizard._apply_board_side()
        register(pcb_wizard)
        kept = pcb_wizard._registration
        load_saved(pcb_wizard)
        assert pcb_wizard._registration is kept
        assert pcb_wizard._profile is None
        assert "refused" in pcb_wizard.ui.profileStatus.text().lower()

    def test_the_chooser_lists_what_was_saved(self, pcb_wizard, profile_home):
        declare_identity(pcb_wizard)
        register(pcb_wizard)
        save_current(pcb_wizard, "board a")
        pcb_wizard.ui.fixtureProfile = _FakeCombo()
        pcb_wizard._refresh_profiles()
        assert [label for label, _data in pcb_wizard.ui.fixtureProfile.items] == [
            "board a"
        ]


class TestVerificationGate:
    """The one statement software cannot make, so the one it must not assume."""

    def test_scanning_stays_blocked_until_the_alignment_is_confirmed(
        self, pcb_wizard, profile_home
    ):
        prepared_profile(pcb_wizard)
        # The maths is perfect: the saved points are exactly the fitted ones
        assert pcb_wizard._registration.rms_mm == pytest.approx(0.0, abs=1e-9)
        assert pcb_wizard._registration_problems() != []
        pcb_wizard.ui.wizardStack.setCurrentIndex(PAGE_REGISTER)
        pcb_wizard._update_nav()
        assert pcb_wizard.ui.btnNext.isEnabled() is False
        assert "UNVERIFIED" in pcb_wizard.ui.profileStatus.text()

    def test_two_confirmed_points_open_the_gate(self, pcb_wizard, profile_home):
        prepared_profile(pcb_wizard)
        verify_point(pcb_wizard, 0.0, 0.0)
        assert pcb_wizard._registration_problems() != []  # one is not enough
        verify_point(pcb_wizard, 50.0, 30.0)
        assert pcb_wizard._registration_problems() == []
        pcb_wizard.ui.wizardStack.setCurrentIndex(PAGE_REGISTER)
        pcb_wizard._update_nav()
        assert pcb_wizard.ui.btnNext.isEnabled() is True
        assert "VERIFIED" in pcb_wizard.ui.profileStatus.text()

    def test_one_point_is_enough_only_when_a_fixture_holds_the_orientation(
        self, pcb_wizard, profile_home
    ):
        # One point pins translation and says nothing about rotation about it
        prepared_profile(pcb_wizard, orientation_locked=True)
        pcb_wizard.ui.chkOrientationLocked.setChecked(True)
        verify_point(pcb_wizard, 0.0, 0.0)
        assert pcb_wizard._registration_problems() == []

    def test_two_confirmed_points_close_together_do_not_count(
        self, pcb_wizard, profile_home
    ):
        prepared_profile(pcb_wizard)
        pcb_wizard.ui.chkSnapLandmark.setChecked(False)
        verify_point(pcb_wizard, 5.0, 5.0)
        verify_point(pcb_wizard, 8.0, 5.0)
        assert len(pcb_wizard._verified_points) == 2
        assert any(
            "too close together" in problem
            for problem in pcb_wizard._registration_problems()
        )

    def test_confirming_needs_the_board_seated(self, pcb_wizard, profile_home):
        prepared_profile(pcb_wizard)
        pcb_wizard._select_landmark(0.0, 0.0)
        pcb_wizard._move_now()
        pcb_wizard.ui.chkBoardSeated.setChecked(False)
        pcb_wizard._confirm_alignment()
        assert pcb_wizard._verified_points == []
        assert any("seated" in text for text in _FakeMessageBox.warnings)

    def test_a_manual_registration_needs_no_confirmation(self, pcb_wizard):
        # Jogging onto every landmark is the verification
        register(pcb_wizard)
        assert pcb_wizard._profile is None
        assert pcb_wizard._registration_problems() == []
        assert "Manual landmarks" in pcb_wizard.ui.profileStatus.text()


class TestStaleConfirmation:
    """Moving to a point and then jogging away must not leave it confirmable."""

    def test_jogging_after_the_move_invalidates_the_candidate(
        self, pcb_wizard, profile_home
    ):
        prepared_profile(pcb_wizard)
        pcb_wizard._select_landmark(0.0, 0.0)
        pcb_wizard._move_now()
        assert pcb_wizard._candidate_is_current() is True

        pcb_wizard._jog(1, 0)
        assert pcb_wizard._candidate_is_current() is False
        pcb_wizard._confirm_alignment()
        assert pcb_wizard._verified_points == []
        assert any("no longer where" in text for text in _FakeMessageBox.warnings)

    def test_the_confirm_button_goes_dead_after_a_jog(self, pcb_wizard, profile_home):
        prepared_profile(pcb_wizard)
        pcb_wizard._select_landmark(0.0, 0.0)
        pcb_wizard._move_now()
        assert pcb_wizard.ui.btnConfirmAlignment.isEnabled() is True
        pcb_wizard._jog(0, -1)
        assert pcb_wizard.ui.btnConfirmAlignment.isEnabled() is False

    def test_selecting_a_different_point_invalidates_the_candidate(
        self, pcb_wizard, profile_home
    ):
        prepared_profile(pcb_wizard)
        pcb_wizard._select_landmark(0.0, 0.0)
        pcb_wizard._move_now()
        pcb_wizard._select_landmark(50.0, 30.0)
        assert pcb_wizard._candidate_is_current() is False

    def test_a_refit_invalidates_confirmed_points(self, pcb_wizard, profile_home):
        prepared_profile(pcb_wizard)
        verify_point(pcb_wizard, 0.0, 0.0)
        verify_point(pcb_wizard, 50.0, 30.0)
        assert pcb_wizard._registration_problems() == []
        pcb_wizard._refit_registration()
        assert pcb_wizard._verified_points == []

    def test_a_failed_move_does_not_leave_a_candidate_behind(
        self, pcb_wizard, profile_home
    ):
        prepared_profile(pcb_wizard)
        pcb_wizard._select_landmark(0.0, 0.0)
        pcb_wizard.printer.xy_error = engine_printer.PrinterTimeout("no ok")
        pcb_wizard._move_now()
        assert pcb_wizard._candidate_is_current() is False

    def test_re_homing_drops_the_whole_profile(self, pcb_wizard, profile_home):
        prepared_profile(pcb_wizard)
        verify_point(pcb_wizard, 0.0, 0.0)
        pcb_wizard._home_xy()
        assert pcb_wizard._profile is None
        assert pcb_wizard._verified_points == []
        assert pcb_wizard._registration is None

    def test_flipping_the_side_drops_the_whole_profile(self, pcb_wizard, profile_home):
        prepared_profile(pcb_wizard)
        pcb_wizard.ui.boardSide.setCurrentText("bottom")
        pcb_wizard._apply_board_side()
        assert pcb_wizard._profile is None
        assert pcb_wizard._verified_points == []


class TestOffsetCorrection:
    def test_a_small_correction_shifts_translation_and_keeps_rotation(
        self, pcb_wizard, profile_home
    ):
        prepared_profile(pcb_wizard)
        before = pcb_wizard._registration.transform
        verify_point(pcb_wizard, 0.0, 0.0)  # candidate consumed by the confirm
        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._move_now()
        pcb_wizard.printer.position = (110.18, 69.92)  # fine jogged onto the pad
        pcb_wizard._update_offset()

        after = pcb_wizard._registration.transform
        assert after.theta_deg == pytest.approx(before.theta_deg, abs=1e-9)
        assert after.x0_mm == pytest.approx(before.x0_mm + 0.18, abs=1e-6)
        assert after.y0_mm == pytest.approx(before.y0_mm - 0.08, abs=1e-6)
        assert pcb_wizard._offset_correction_mm[0] == pytest.approx(0.18)

    def test_a_correction_voids_what_was_already_confirmed(
        self, pcb_wizard, profile_home
    ):
        prepared_profile(pcb_wizard)
        verify_point(pcb_wizard, 0.0, 0.0)
        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._move_now()
        pcb_wizard.printer.position = (110.1, 70.0)
        pcb_wizard._update_offset()
        assert pcb_wizard._verified_points == []
        assert pcb_wizard._registration_problems() != []

    def test_the_corrected_point_is_immediately_confirmable(
        self, pcb_wizard, profile_home
    ):
        # The probe is standing on the true feature and the corrected transform
        # now maps that board point to exactly there, so this is not stale
        prepared_profile(pcb_wizard)
        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._move_now()
        pcb_wizard.printer.position = (110.1, 70.0)
        pcb_wizard._update_offset()
        assert pcb_wizard._candidate_is_current() is True
        pcb_wizard._confirm_alignment()
        assert len(pcb_wizard._verified_points) == 1

    def test_a_correction_beyond_the_bound_is_refused_outright(
        self, pcb_wizard, profile_home
    ):
        prepared_profile(pcb_wizard)
        before = pcb_wizard._registration.transform
        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._move_now()
        pcb_wizard.printer.position = (115.0, 70.0)  # 5 mm out: not a correction
        pcb_wizard._update_offset()
        assert pcb_wizard._registration.transform == before
        assert any("Register manually" in text for text in _FakeMessageBox.warnings)

    def test_a_correction_that_pushes_the_board_off_the_bed_is_refused(
        self, pcb_wizard, profile_home
    ):
        declare_identity(pcb_wizard)
        # Parked so the far corner sits a hair inside the 220 mm limit
        register(pcb_wizard, machine_origin=(169.5, 40.0))
        save_current(pcb_wizard)
        pcb_wizard._clear_landmarks_and_alignment()
        load_saved(pcb_wizard)
        before = pcb_wizard._registration.transform

        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._move_now()
        pcb_wizard.printer.position = (pcb_wizard.printer.position[0] + 1.0, 70.0)
        pcb_wizard._update_offset()
        assert pcb_wizard._registration.transform == before
        assert any("out of reach" in text for text in _FakeMessageBox.warnings)

    def test_no_correction_emits_z_or_g92(self, pcb_wizard, profile_home):
        prepared_profile(pcb_wizard)
        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._move_now()
        pcb_wizard.printer.position = (110.1, 70.0)
        pcb_wizard._update_offset()
        assert not any("Z" in c for c in pcb_wizard.printer.sent if c.startswith("G0"))
        assert not any(c.startswith("G92") for c in pcb_wizard.printer.sent)


class TestSaveGating:
    def test_a_manual_registration_can_be_saved(self, pcb_wizard, profile_home):
        register(pcb_wizard)
        pcb_wizard._update_nav()
        assert pcb_wizard.ui.btnSaveAsProfile.isEnabled() is True

    def test_an_unverified_profile_cannot_be_saved_back(self, pcb_wizard, profile_home):
        prepared_profile(pcb_wizard)
        pcb_wizard._update_nav()
        assert pcb_wizard._save_allowed() is False
        assert pcb_wizard.ui.btnSaveAsProfile.isEnabled() is False
        assert pcb_wizard.ui.btnUpdateProfile.isEnabled() is False

    def test_an_unconfirmed_offset_correction_cannot_be_persisted(
        self, pcb_wizard, profile_home
    ):
        prepared_profile(pcb_wizard)
        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._move_now()
        pcb_wizard.printer.position = (110.1, 70.0)
        pcb_wizard._update_offset()
        assert pcb_wizard._save_allowed() is False

    def test_a_verified_profile_can_be_saved_back(self, pcb_wizard, profile_home):
        prepared_profile(pcb_wizard)
        verify_point(pcb_wizard, 0.0, 0.0)
        verify_point(pcb_wizard, 50.0, 30.0)
        pcb_wizard._update_nav()
        assert pcb_wizard._save_allowed() is True
        assert pcb_wizard.ui.btnUpdateProfile.isEnabled() is True

    def test_a_bad_registration_cannot_be_saved(self, pcb_wizard, profile_home):
        register_points(pcb_wizard, [(0.0, 0.0), (2.0, 0.0)])
        assert pcb_wizard._save_allowed() is False

    def _selectable(self, wizard, name="board a"):
        wizard.ui.fixtureProfile = _FakeCombo()
        wizard.ui.fixtureProfile.addItem(name, name)
        wizard.ui.fixtureProfile.setCurrentIndex(0)

    def test_declining_the_overwrite_leaves_the_saved_profile_alone(
        self, pcb_wizard, profile_home
    ):
        declare_identity(pcb_wizard)
        register(pcb_wizard)
        path, _document, _digest = save_current(pcb_wizard)
        before = path.read_bytes()
        self._selectable(pcb_wizard)

        # register() again at a different origin, so a write would be visible
        register(pcb_wizard, machine_origin=(80.0, 50.0))
        pcb_wizard._update_profile()
        assert any("Overwrite" in text for text in _FakeMessageBox.questions)
        assert path.read_bytes() == before

    def test_accepting_the_overwrite_writes_the_new_alignment(
        self, pcb_wizard, profile_home
    ):
        declare_identity(pcb_wizard)
        register(pcb_wizard)
        path, _document, _digest = save_current(pcb_wizard)
        self._selectable(pcb_wizard)

        register(pcb_wizard, machine_origin=(80.0, 50.0))
        _FakeMessageBox.answer = _FakeMessageBox.Yes
        pcb_wizard._update_profile()
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["points"][0]["machine_x_mm"] == pytest.approx(80.0)
        assert "Saved" in pcb_wizard.ui.profileStatus.text()

    def test_saving_as_new_asks_before_replacing_an_existing_name(
        self, pcb_wizard, profile_home
    ):
        declare_identity(pcb_wizard)
        register(pcb_wizard)
        path, _document, _digest = save_current(pcb_wizard)
        before = path.read_bytes()

        register(pcb_wizard, machine_origin=(80.0, 50.0))
        pcb_wizard._write_profile("board a", confirm_existing=True)
        assert any("already exists" in text for text in _FakeMessageBox.questions)
        assert path.read_bytes() == before

    def test_a_profile_name_that_escapes_the_directory_is_refused(
        self, pcb_wizard, profile_home
    ):
        register(pcb_wizard)
        pcb_wizard._write_profile("../escape", confirm_existing=True)
        assert not (profile_home.parent / "escape.json").exists()
        assert any("profile name" in text for text in _FakeMessageBox.warnings)


class TestIdentityAcknowledgement:
    """The snapshot claims these were acknowledged, so the claim must be true."""

    def _stub_modal(self, monkeypatch, accept):
        class _Box:
            instances = []

            def __init__(self, _parent=None):
                self.buttons = {}
                self.informative = ""
                self._clicked = None
                _Box.instances.append(self)

            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

            def setInformativeText(self, text):
                self.informative = text

            def addButton(self, *args):
                button = object()
                self.buttons[args[0] if args else None] = button
                return button

            def exec(self):
                keys = list(self.buttons)
                self._clicked = self.buttons[keys[0 if accept else -1]]

            def clickedButton(self):
                return self._clicked

        _Box.instances = []
        _Box.Warning = "warning"
        _Box.Cancel = "cancel"
        _Box.AcceptRole = "accept"
        _Box.warning = classmethod(
            lambda cls, _p, _t, message: _FakeMessageBox.warnings.append(message)
        )
        monkeypatch.setattr(emi_map, "QMessageBox", _Box)
        return _Box

    def _mismatched(self, wizard, profile_home):
        declare_identity(wizard)
        register(wizard)
        save_current(wizard)
        wizard._clear_landmarks_and_alignment()
        declare_identity(wizard, probeSetupId="loop-6mm-holder-v3")

    def test_a_probe_change_must_be_acknowledged_explicitly(
        self, pcb_wizard, profile_home, monkeypatch
    ):
        self._mismatched(pcb_wizard, profile_home)
        box = self._stub_modal(monkeypatch, accept=True)
        load_saved(pcb_wizard)
        assert box.instances  # the operator was actually asked
        assert "probe_setup_id" in box.instances[0].informative
        assert pcb_wizard._profile is not None
        recorded = pcb_wizard._acknowledged_warnings
        assert [entry["field"] for entry in recorded] == ["probe_setup_id"]
        assert recorded[0]["saved"] == "loop-6mm-holder-v2"
        assert recorded[0]["current"] == "loop-6mm-holder-v3"
        assert recorded[0]["acknowledged_at"]

    def test_cancelling_records_nothing_and_changes_nothing(
        self, pcb_wizard, profile_home, monkeypatch
    ):
        self._mismatched(pcb_wizard, profile_home)
        self._stub_modal(monkeypatch, accept=False)
        load_saved(pcb_wizard)
        assert pcb_wizard._profile is None
        assert pcb_wizard._registration is None
        assert pcb_wizard._acknowledged_warnings == []
        assert "cancelled" in pcb_wizard.ui.profileStatus.text()

    def test_a_matching_setup_asks_nothing(self, pcb_wizard, profile_home, monkeypatch):
        declare_identity(pcb_wizard)
        register(pcb_wizard)
        save_current(pcb_wizard)
        pcb_wizard._clear_landmarks_and_alignment()
        box = self._stub_modal(monkeypatch, accept=False)
        load_saved(pcb_wizard)
        assert box.instances == []
        assert pcb_wizard._profile is not None

    def test_a_re_export_is_noted_without_a_modal(
        self, pcb_wizard, profile_home, monkeypatch
    ):
        declare_identity(pcb_wizard)
        pcb_wizard._board_model = board_50x30(sha="a" * 64)
        pcb_wizard._apply_board_side()
        register(pcb_wizard)
        save_current(pcb_wizard)
        # Same outline, a different archive: the transform is still correct, and
        # refusing here would cost a full manual registration for nothing
        pcb_wizard._board_model = board_50x30(sha="b" * 64)
        pcb_wizard._apply_board_side()
        register(pcb_wizard)
        pcb_wizard._clear_landmarks_and_alignment()

        box = self._stub_modal(monkeypatch, accept=False)
        load_saved(pcb_wizard)
        assert box.instances == []
        assert pcb_wizard._profile is not None
        assert pcb_wizard._noted_warnings
        assert pcb_wizard._acknowledged_warnings == []


class TestScanProvenance:
    def test_a_manual_registration_says_so(self, pcb_wizard):
        register(pcb_wizard)
        assert pcb_wizard._provenance() == {"registration_source": "manual_landmarks"}

    def test_a_rectangle_scan_records_no_registration_at_all(self, wizard):
        assert wizard._provenance() is None

    def test_a_profile_scan_carries_the_whole_audit_trail(
        self, pcb_wizard, profile_home
    ):
        prepared_profile(pcb_wizard)
        verify_point(pcb_wizard, 0.0, 0.0)
        verify_point(pcb_wizard, 50.0, 30.0)
        provenance = pcb_wizard._provenance()

        assert provenance["registration_source"] == "fixture_profile"
        assert provenance["fixture_profile"] == "board a"
        assert len(provenance["profile_sha256"]) == 64
        assert provenance["orientation_locked"] is False
        assert [p["board_xy"] for p in provenance["verified_points"]] == [
            (0.0, 0.0),
            (50.0, 30.0),
        ]
        assert all(point["verified_at"] for point in provenance["verified_points"])
        assert provenance["offset_correction_mm"] == [0.0, 0.0]
        assert provenance["fixture_id"] == IDENTITY["fixtureId"]
        assert provenance["profile_document"]["profile_name"] == "board a"

    def test_an_offset_correction_is_reported(self, pcb_wizard, profile_home):
        prepared_profile(pcb_wizard)
        pcb_wizard._select_landmark(50.0, 30.0)
        pcb_wizard._move_now()
        pcb_wizard.printer.position = (110.08, 69.96)
        pcb_wizard._update_offset()
        assert pcb_wizard._provenance()["offset_correction_mm"] == [0.08, -0.04]

    def test_the_worker_is_handed_the_provenance(self, pcb_wizard, monkeypatch):
        register(pcb_wizard)
        seen = {}

        def capture(config, callbacks=None, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here")

        monkeypatch.setattr(emi_map.engine_scanner, "run_scan", capture)
        worker = emi_map.ScanWorker(
            engine_config.ScanConfig(), None, None, None, pcb_wizard._provenance()
        )
        worker.run()
        assert seen["provenance"] == {"registration_source": "manual_landmarks"}

    def test_the_profile_copy_is_written_from_memory_not_from_disk(
        self, pcb_wizard, profile_home, tmp_path
    ):
        prepared_profile(pcb_wizard)
        loaded_name = pcb_wizard._profile.document["profile_name"]
        # Someone saves over the profile between loading it and scanning
        path = emi_map.engine_profiles.profile_path("board a")
        path.write_text(json.dumps({"profile_name": "replaced"}), encoding="utf-8")

        output = tmp_path / "scan"
        output.mkdir()
        pcb_wizard._write_profile_copy(output)
        written = json.loads((output / "fixture_profile.json").read_text(encoding="utf-8"))
        assert written["profile_name"] == loaded_name

    def test_no_copy_is_written_for_a_manual_registration(self, pcb_wizard, tmp_path):
        register(pcb_wizard)
        pcb_wizard._write_profile_copy(tmp_path)
        assert not (tmp_path / "fixture_profile.json").exists()


def _scan_result(**overrides):
    from types import SimpleNamespace

    result = SimpleNamespace(
        output_dir=Path("EMI_Scans/20260903_130354"),
        freqs=np.array([19e6, 21e6]),
        power={"dut": np.array([[-40.0, -32.1], [np.nan, -38.0]])},
        delta=None,
        files={
            "heatmap_png": "heatmap.png",
            "board_html": "board_overlay.html",
            "spectra": "spectra.npz",
        },
        snapshot={
            "started": "2026-09-03T13:03:54",
            "finished": "2026-09-03T13:08:10",
            "instrument": {
                "model": "tinySA Ultra",
                "firmware": "v1.4-143",
                "start_hz": 19e6,
                "stop_hz": 21e6,
                "points": 101,
                "samples_per_xy": 3,
                "rbw_khz": 0,
                "serial_baud": 115200,
                "scanraw_option": 1,
            },
            "config": {"area": {"nx": 6, "ny": 4, "step_mm": 10.0}},
            "registration_provenance": {
                "registration_source": "fixture_profile",
                "fixture_profile": "qs127m_glass_module_01 Glassboard top",
                "verified_points": [{"board_xy": (11.5, -20.4)}],
            },
        },
        plan=SimpleNamespace(
            point_count=24,
            cell_count=24,
            nx=6,
            ny=4,
            step_mm=10.0,
            side="top",
        ),
    )
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


class TestResultsOverview:
    def test_the_results_page_is_a_summary_not_a_file_dump(self, emi_widgets):
        assert "resultsBanner" in emi_widgets
        assert "resultsSummary" in emi_widgets
        assert "resultsText" not in emi_widgets

    def test_overview_names_what_was_measured(self):
        text = emi_map.results_overview(_scan_result())
        assert "19 MHz" in text and "21 MHz" in text
        assert "101 points" in text
        assert "3 samples/cell" in text
        assert "24 measured cells" in text
        assert "10 mm step" in text
        assert "tinySA Ultra" in text
        assert "PCB aligned (top side)" in text
        assert "Peak: -32.1 dBm" in text
        assert "qs127m_glass_module_01 Glassboard top" in text
        assert "1 point(s) verified" in text
        assert "EMI_Scans" in text
        assert "heatmap.png" in text

    def test_overview_does_not_dump_low_level_instrument_keys(self):
        text = emi_map.results_overview(_scan_result())
        assert "serial_baud" not in text
        assert "scanraw_option" not in text
        assert "spectra: spectra.npz" not in text

    def test_a_successful_scan_fills_the_results_page(self, wizard):
        wizard._on_succeeded(_scan_result(files={}))
        assert wizard.ui.wizardStack.currentIndex() == PAGE_RESULTS
        assert "Scan complete" in wizard.ui.resultsBanner.text()
        assert "tinySA Ultra" in wizard.ui.resultsSummary.text()
        assert "24 measured cells" in wizard.ui.resultsSummary.text()
