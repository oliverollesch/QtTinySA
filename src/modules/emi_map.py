"""EMI Near-Field Map wizard for QtTinySA.

Drives an Ender 3 over a PCB while this analyser measures, and builds a 2D map
of the emissions.  The measurement engine itself lives outside QtTinySA in the
EMI_Mapper package; this module is only Qt glue plus serial-ownership handling.

Two modes share the wizard.  Rectangle maps a plain area from an origin the
operator sets by hand with G92, and never homes.  PCB aligned homes X and Y,
imports an ODB++ board, registers it to the machine from landmarks the operator
jogs to, and clips the grid to the board outline.  Neither mode commands Z.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QFileDialog, QMessageBox
from serial.tools import list_ports

# Matches the interval Tiny.setCmdQ uses for its command FIFO
FIFO_INTERVAL_MS = 500

# Bounded read timeout while the engine owns the port. QtTinySA opens the
# analyser without a timeout, and a blocking read cannot be cancelled.
ENGINE_READ_TIMEOUT_S = 0.05

# Windows advertises every paired Bluetooth serial profile as a COM port even
# when nothing is on the other end. Opening one stalls for tens of seconds and
# then answers nothing, which is indistinguishable from a hung printer, so they
# are never offered as a printer port.
PORT_EXCLUDE_HINTS = ("bluetooth",)

# USB-serial bridges the Ender's mainboard is actually built around
PORT_PRINTER_HINTS = ("ch340", "ch341", "cp210", "ft232", "usb-serial", "wch")

# Opening the port pulses DTR and reboots the mainboard, so Marlin cannot answer
# for the first couple of seconds. M115 then proves the port really is a
# printer; without it, the wrong port costs one 30 s ok timeout per command.
PRINTER_BOOT_S = 2.0
PRINTER_GREET_S = 8.0

DIALOG_TITLE = "EMI Near-Field Map"

STEP_TITLES = ["Setup", "Board", "Register", "Rectangle origin", "Scan", "Results"]

PAGE_SETUP = 0
PAGE_BOARD = 1
PAGE_REGISTER = 2
PAGE_ORIGIN = 3
PAGE_SCAN = 4
PAGE_RESULTS = 5

MODE_RECTANGLE = "Rectangle"
MODE_PCB = "PCB aligned"

# One page sequence per mode. Navigation, the step header and Next-gating all
# read this table, so adding a mode or a page never adds another per-page
# if/elif ladder of the kind this file already shares with modules/fcc_test.py.
MODE_PAGES = {
    MODE_RECTANGLE: (PAGE_SETUP, PAGE_ORIGIN, PAGE_SCAN, PAGE_RESULTS),
    MODE_PCB: (PAGE_SETUP, PAGE_BOARD, PAGE_REGISTER, PAGE_SCAN, PAGE_RESULTS),
}

# Next names the following page so Setup cannot silently send you down the
# other mode's path. PAGE_SETUP depends on the combo; the rest do not.
NEXT_CAPTIONS = {
    PAGE_SETUP: {
        MODE_PCB: "Next: import board",
        MODE_RECTANGLE: "Next: set origin",
    },
    PAGE_BOARD: "Next: register",
    PAGE_REGISTER: "Next: scan",
    PAGE_ORIGIN: "Next: scan",
    PAGE_SCAN: "Next: results",
}

# How close a click must land to a suggested corner before it snaps there
# instead of to the nearest outline vertex, as a fraction of the diagonal.
LANDMARK_SNAP_FRACTION = 0.1

# Two landmarks closer than this in either frame are the same point: well
# under a step, and far above what M114 rounding can produce.
COINCIDENT_MM = 0.05

# A click on the drawn edge of the board is a click on the board. Outline.contains
# inflates every ring by this much rather than deciding on floating-point luck.
CLICK_EDGE_TOLERANCE_MM = 0.05

# Coarse to fine, matching the four preset buttons on the Register page
JOG_STEP_PRESETS = (10.0, 1.0, 0.1, 0.05)

# Stand-in levels for the empty live map; replaced by the first reading.
PLACEHOLDER_LEVELS = (0.0, 1.0)

TRAVEL_WARNING = (
    "The probe will move +X by {width:g} mm and +Y by {height:g} mm from the "
    "current location. Verify the full travel area is clear before setting the origin."
)

THIRD_LANDMARK_HINT = (
    "Recommended: record a third landmark. A rigid 2D fit has three degrees of "
    "freedom, so two landmarks always fit almost exactly and their residual "
    "cannot reveal a mistake. A third point is the first independent check. "
    "Scanning is allowed without it."
)


def _load_engine():
    """Import EMI_Mapper, adding the checkout root to sys.path when running from source.

    A packaged build bundles the package instead (see the nuitka-project
    include in QtTinySA.py), so the path shim is only a development fallback.
    """
    try:
        import EMI_Mapper  # noqa: F401
    except ImportError:
        root = Path(__file__).resolve().parents[3]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from EMI_Mapper import board as engine_board
    from EMI_Mapper import config as engine_config
    from EMI_Mapper import fixture_profiles as engine_profiles
    from EMI_Mapper import odbpp as engine_odbpp
    from EMI_Mapper import overlay as engine_overlay
    from EMI_Mapper import printer as engine_printer
    from EMI_Mapper import registration as engine_registration
    from EMI_Mapper import scanner as engine_scanner

    return (
        engine_board,
        engine_config,
        engine_odbpp,
        engine_overlay,
        engine_printer,
        engine_profiles,
        engine_registration,
        engine_scanner,
    )


try:
    (
        engine_board,
        engine_config,
        engine_odbpp,
        engine_overlay,
        engine_printer,
        engine_profiles,
        engine_registration,
        engine_scanner,
    ) = _load_engine()
    ENGINE_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on deployment layout
    engine_board = engine_config = engine_odbpp = engine_overlay = None
    engine_printer = engine_profiles = engine_registration = engine_scanner = None
    ENGINE_ERROR = str(exc)
    logging.info(f"EMI map engine unavailable: {exc}")


def _board_bounds(view, step_mm):
    """The board's bounding box snapped outwards onto the step grid.

    Snapping outwards keeps the requested span an exact multiple of the step and
    covers the whole profile.  Matches MapperSession's default bounds so the GUI
    and the MCP tools clip the same board to the same cells.
    """
    x0, y0, x1, y1 = view.bbox_mm
    return (
        math.floor(x0 / step_mm) * step_mm,
        math.ceil(x1 / step_mm) * step_mm,
        math.floor(y0 / step_mm) * step_mm,
        math.ceil(y1 / step_mm) * step_mm,
    )


def _spread_mm(points):
    """The largest distance between any two of the points."""
    if len(points) < 2:
        return 0.0
    coords = np.asarray(points, dtype=float)
    deltas = coords[:, None, :] - coords[None, :, :]
    return float(np.hypot(deltas[..., 0], deltas[..., 1]).max())


def _polyline_batch(arrays, close=False):
    """Flatten many polylines into one x/y/connect triple for a single curve item.

    The reference board carries 2947 strokes; one plot item each makes the
    canvas unusable, while one item with a connect mask draws in a single pass.
    """
    xs, ys, connect = [], [], []
    for points in arrays:
        polyline = np.asarray(points, dtype=float)
        if polyline.ndim != 2 or len(polyline) < 2:
            continue
        if close and not np.array_equal(polyline[0], polyline[-1]):
            polyline = np.vstack([polyline, polyline[:1]])
        xs.append(polyline[:, 0])
        ys.append(polyline[:, 1])
        flags = np.ones(len(polyline), dtype=np.uint8)
        flags[-1] = 0  # lift the pen between polylines
        connect.append(flags)
    if not xs:
        return None
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(connect)


class _MapSerial:
    """Owns the analyser for the duration of a scan.

    Tracks the device by serial number rather than index: QtTinySA renumbers
    devices whenever one connects or disconnects, so an index would silently
    retarget a different analyser.
    """

    def __init__(self, usb_instr):
        self.usb_instr = usb_instr
        self.sn = None
        self._held = False
        self._saved_timeout = None

    def _connected(self):
        return [d for d in (self.usb_instr.devices or []) if d is not None]

    def device(self):
        if self.sn is None:
            return None
        for dev in self._connected():
            if dev.sn == self.sn:
                return dev
        return None

    def adopt(self):
        devices = self._connected()
        if not devices:
            self.sn = None
            return None
        chosen = next((d for d in devices if getattr(d, "enabled", False)), devices[0])
        self.sn = chosen.sn
        logging.info(f"EMI map bound to analyser serial {self.sn} on {chosen.usbPort}")
        return chosen

    def is_open(self):
        dev = self.device()
        return dev is not None and dev.usb is not None and dev.usb.is_open

    def take(self, wait_s=4.0):
        """Stop the sweep and the command FIFO, then confirm the port is idle.

        Concurrent scanraw and serialWrite on Windows raises
        PermissionError(13) 'The device does not recognize the command.', so
        the running measurement worker has to have actually stopped reading
        before the engine may touch the same serial object.
        """
        dev = self.device()
        if dev is None:
            raise RuntimeError("Analyser is not connected")
        self.usb_instr.stop(restart=False)
        try:
            if dev.fifoTimer.isActive():
                dev.fifoTimer.stop()
        except Exception:
            pass
        try:
            if dev.usb and dev.usb.is_open:
                dev.abort()
        except Exception:
            pass

        deadline = time.time() + wait_s
        while dev.threadRunning and time.time() < deadline:
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.05)
        if dev.threadRunning:
            self.restart_fifo()
            raise RuntimeError("The analyser sweep did not stop; not starting the scan")

        try:
            if dev.usb and dev.usb.is_open:
                dev.clearBuffer()
        except Exception:
            pass
        try:
            while dev.fifo.qsize() > 0:
                dev.fifo.get_nowait()
        except Exception:
            pass

        self._saved_timeout = getattr(dev.usb, "timeout", None)
        dev.usb.timeout = ENGINE_READ_TIMEOUT_S
        self._held = True
        return dev

    def release(self):
        if not self._held:
            return
        dev = self.device()
        if dev is not None and dev.usb is not None:
            try:
                dev.usb.timeout = self._saved_timeout
            except Exception:
                pass
        self.restart_fifo()

    def restart_fifo(self):
        dev = self.device()
        if dev is not None:
            try:
                if not dev.fifoTimer.isActive():
                    dev.fifoTimer.start(FIFO_INTERVAL_MS)
            except Exception:
                pass
        self._held = False


class ScanWorker(QtCore.QObject):
    """Runs one scan off the GUI thread."""

    status = QtCore.Signal(str)
    point = QtCore.Signal(dict)
    row = QtCore.Signal(dict)
    prompt = QtCore.Signal()
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    # A Marlin reboot is reported separately from the failure text: the GUI has
    # to know the machine frame is gone, and a formatted message loses the type.
    printer_reset = QtCore.Signal()
    ended = QtCore.Signal()

    def __init__(
        self, config, tinysa_transport, printer_transport, plan=None, provenance=None
    ):
        super().__init__()
        self.config = config
        self.tinysa_transport = tinysa_transport
        self.printer_transport = printer_transport
        self.plan = plan
        self.provenance = provenance
        self._abort = threading.Event()
        self._continue = threading.Event()

    def cancel(self):
        self._abort.set()
        self._continue.set()  # unblock a worker waiting at the DUT-on prompt

    def resume(self):
        self._continue.set()

    def _prompt_dut_on(self):
        self._continue.clear()
        self.prompt.emit()
        while not self._continue.wait(0.1):
            if self._abort.is_set():
                raise engine_scanner.ScanAborted("cancelled at the DUT prompt")

    @QtCore.Slot()
    def run(self):
        try:
            callbacks = engine_scanner.Callbacks(
                on_point=self.point.emit,
                on_row=self.row.emit,
                on_status=self.status.emit,
                should_abort=self._abort.is_set,
                prompt_dut_on=self._prompt_dut_on,
            )
            result = engine_scanner.run_scan(
                self.config,
                callbacks,
                tinysa_transport=self.tinysa_transport,
                printer_transport=self.printer_transport,
                plan=self.plan,
                provenance=self.provenance,
            )
            self.succeeded.emit(result)
        except engine_scanner.ScanAborted:
            self.failed.emit("Scan cancelled. Points already measured were saved.")
        except engine_printer.PrinterReset as exc:
            self.printer_reset.emit()
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            logging.exception("EMI scan failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.ended.emit()


def _mhz(hz) -> str:
    try:
        return f"{float(hz) / 1e6:g} MHz"
    except (TypeError, ValueError):
        return "?"


def results_overview(result) -> str:
    """A human scan summary for the Results page, not a file dump.

    The snapshot already has the instrument and config; this only names the
    quantities an operator uses to judge whether the run is the one they meant.
    """
    snapshot = getattr(result, "snapshot", None) or {}
    instrument = snapshot.get("instrument") or {}
    config = snapshot.get("config") or {}
    area = config.get("area") or {}
    plan = getattr(result, "plan", None)
    freqs = np.asarray(getattr(result, "freqs", []), dtype=float)
    files = getattr(result, "files", None) or {}

    start_hz = instrument.get("start_hz")
    stop_hz = instrument.get("stop_hz")
    if freqs.size:
        start_hz = freqs[0] if start_hz is None else start_hz
        stop_hz = freqs[-1] if stop_hz is None else stop_hz
    points = instrument.get("points") or (int(freqs.size) if freqs.size else None)
    samples = instrument.get("samples_per_xy")
    rbw = instrument.get("rbw_khz", "auto")
    rbw_text = "auto" if rbw in (0, 0.0, None, "auto") else f"{rbw:g} kHz"

    rf = f"{_mhz(start_hz)} – {_mhz(stop_hz)}"
    if points:
        rf += f", {points} points"
    if samples:
        rf += f", {samples} samples/cell"
    rf += f", RBW {rbw_text}"

    if plan is not None:
        grid = (
            f"{plan.point_count} measured cells of {plan.cell_count} "
            f"({plan.nx}×{plan.ny} grid, {plan.step_mm:g} mm step, off-board skipped)"
        )
        frame = f"PCB aligned ({plan.side} side)"
    else:
        nx = area.get("nx")
        ny = area.get("ny")
        step = area.get("step_mm")
        cells = (nx * ny) if nx and ny else None
        grid = (
            f"{cells} cells ({nx}×{ny} grid, {step:g} mm step)"
            if cells
            else "Rectangle scan"
        )
        frame = "Rectangle from origin"

    model = instrument.get("model") or "analyser"
    firmware = instrument.get("firmware") or ""
    instrument_line = model if not firmware else f"{model}  ({firmware})"

    provenance = snapshot.get("registration_provenance") or {}
    source = provenance.get("registration_source")
    if source == "fixture_profile":
        alignment = f"Profile {provenance.get('fixture_profile')!r}"
        if provenance.get("verified_points"):
            alignment += f", {len(provenance['verified_points'])} point(s) verified"
    elif source == "manual_landmarks":
        alignment = "Manual landmarks"
    else:
        alignment = ""

    started = snapshot.get("started")
    finished = snapshot.get("finished")
    when = ""
    if started and finished:
        when = f"{started} → {finished}"
    elif finished:
        when = str(finished)

    names = [Path(path).name for path in files.values() if path]
    folder = getattr(result, "output_dir", "")

    lines = [
        rf,
        f"Grid: {grid}",
        f"Mode: {frame}",
        f"Instrument: {instrument_line}",
    ]
    peak = _peak_line(result)
    if peak:
        lines.append(peak)
    if alignment:
        lines.append(f"Alignment: {alignment}")
    if when:
        lines.append(f"Time: {when}")
    lines.append(f"Folder: {folder}")
    if names:
        lines.append("Wrote: " + ", ".join(names))
    return "\n".join(lines)


def _peak_line(result) -> str:
    if getattr(result, "delta", None) is not None:
        array = np.asarray(result.delta, dtype=float)
        units = "dB"
    else:
        power = getattr(result, "power", None) or {}
        if "dut" not in power:
            return ""
        array = np.asarray(power["dut"], dtype=float)
        units = "dBm"
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return "Peak: no finite readings"
    return f"Peak: {float(finite.max()):+.1f} {units}   min {float(finite.min()):+.1f} {units}"


class _BlockingPrinterJob(QtCore.QObject):
    """Marlin's ok-wait is 30s per command. Never run that on the GUI thread."""

    finished = QtCore.Signal(object)

    def __init__(self, work):
        super().__init__()
        self.work = work

    @QtCore.Slot()
    def run(self):
        try:
            self.finished.emit(self.work())
        except Exception as exc:
            self.finished.emit(exc)


class EMIMapWizard(QtCore.QObject):
    """Controller for the EMI Near-Field Map dialog loaded from emi_map.ui."""

    def __init__(self, ui, usb_instr, main_window):
        super().__init__()
        self.ui = ui
        self.serial = _MapSerial(usb_instr)
        self.main = main_window

        self.printer_serial = None
        self.origin_set = False
        self.thread = None
        self.worker = None
        self.result = None
        self._grid = None
        self._pending_updates = 0
        self._printer_job = None
        self._printer_job_ok = None
        self._printer_job_fail = None

        # PCB state in strict dependency order: the machine frame, then the
        # board, then the fit, then the frozen plan. Invalidation only ever
        # cascades downwards; see _invalidate_board_alignment.
        self._pcb_xy_homed = False
        self._board_model = None
        self._board_view = None
        self._landmarks = []
        self._selected_landmark = None
        self._registration = None
        self._fit_error = ""
        self._plan = None
        self._machine_xy = None

        # Verification of a loaded profile, tracked against two revisions
        # rather than a flag.  A bare boolean lets the operator move to a
        # landmark, jog somewhere else, and then confirm an alignment they are
        # no longer looking at; a candidate that expires when either revision
        # moves on cannot be confirmed stale.
        self._registration_revision = 0
        self._motion_revision = 0
        self._profile = None
        self._verification_candidate = None
        self._verified_points = []
        self._offset_correction_mm = (0.0, 0.0)
        self._acknowledged_warnings = []
        self._noted_warnings = []

        # Plot items, created by the pyqtgraph-touching setup methods only
        self._landmark_marks = None
        self._selected_mark = None
        self._board_trace = None
        self._results_pixmap = None
        self._scan_gate = None

        self._wire_ui()
        self._setup_plot()
        self._setup_board_plots()
        self._update_travel_warning()

    # ------------------------------------------------------------------ UI
    def _wire_ui(self):
        u = self.ui
        u.btnBack.clicked.connect(self._on_back)
        u.btnNext.clicked.connect(self._on_next)
        u.btnCancel.clicked.connect(self._on_close)
        u.btnBrowse.clicked.connect(self._browse_folder)
        u.btnSetOrigin.clicked.connect(self._set_origin_clicked)
        u.btnStartScan.clicked.connect(self._start_scan)
        u.btnAbortScan.clicked.connect(self._abort_scan)
        u.btnContinueDut.clicked.connect(self._continue_dut)
        u.btnOpenFolder.clicked.connect(self._open_folder)
        u.btnOpenOverlay.clicked.connect(self._open_overlay)
        if hasattr(u, "boardImage") and hasattr(u.boardImage, "installEventFilter"):
            u.boardImage.installEventFilter(self)

        u.chkTravelClear.stateChanged.connect(self._update_nav)
        for spin in (u.widthMm, u.heightMm, u.stepMm):
            spin.valueChanged.connect(self._on_area_changed)
        u.wizardStack.currentChanged.connect(self._on_page_changed)
        u.scanMode.currentTextChanged.connect(self._on_mode_changed)
        u.printerPort.currentTextChanged.connect(self._on_printer_port_changed)

        u.btnImportBoard.clicked.connect(self._import_board)
        u.boardSide.currentTextChanged.connect(self._apply_board_side)
        u.chkHomeClear.stateChanged.connect(self._update_nav)
        u.chkTravelClearBoard.stateChanged.connect(self._update_nav)
        u.btnHomeXy.clicked.connect(self._home_xy_clicked)
        u.btnRecordLandmark.clicked.connect(self._record_landmark)
        u.btnRemoveLandmark.clicked.connect(self._remove_landmark)
        u.btnClearLandmarks.clicked.connect(self._clear_landmarks)
        for button, dx, dy in (
            (u.btnJogXPlus, 1, 0),
            (u.btnJogXMinus, -1, 0),
            (u.btnJogYPlus, 0, 1),
            (u.btnJogYMinus, 0, -1),
        ):
            button.clicked.connect(partial(self._jog, dx, dy))
        for button, step_mm in zip(
            (u.btnStep10, u.btnStep1, u.btnStep01, u.btnStep005), JOG_STEP_PRESETS
        ):
            button.clicked.connect(partial(u.jogStep.setValue, step_mm))

        u.btnMoveHere.clicked.connect(self._move_to_selected)
        u.chkSnapLandmark.stateChanged.connect(self._on_snap_changed)
        u.btnUpdateOffset.clicked.connect(self._update_offset)
        u.btnConfirmAlignment.clicked.connect(self._confirm_alignment)
        u.btnLoadProfile.clicked.connect(self._load_profile)
        u.btnSaveAsProfile.clicked.connect(self._save_as_profile)
        u.btnUpdateProfile.clicked.connect(self._update_profile)
        u.chkBoardSeated.stateChanged.connect(self._update_nav)
        u.chkOrientationLocked.stateChanged.connect(self._on_orientation_lock_changed)

        # The dialog can be dismissed with the window chrome as well as Close
        u.rejected.connect(self._cleanup)
        self._layout_register_page()

    def _layout_register_page(self):
        """Keep the control column readable and stop spare height pooling at the top.

        Word-wrapped labels default to vertically centred. In a tall stacked
        page that looks like a blank band above the intro, and it steals height
        from the splitter so the jog pad is crushed.
        """
        intro = getattr(self.ui, "registerIntro", None)
        if intro is not None:
            intro.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            intro.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
            )
        page = getattr(self.ui, "pageRegister", None)
        page_layout = page.layout() if page is not None else None
        if page_layout is not None:
            page_layout.setAlignment(QtCore.Qt.AlignTop)
            page_layout.setStretch(2, 1)
        inner = getattr(self.ui, "registerInner", None)
        if inner is not None:
            layout = inner.layout() if callable(getattr(inner, "layout", None)) else None
            if layout is not None:
                layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        split = getattr(self.ui, "registerSplit", None)
        if split is None or not hasattr(split, "setSizes"):
            return
        split.setChildrenCollapsible(False)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        width = max(int(getattr(self.ui, "width", lambda: 1100)()), 800)
        left = min(480, max(440, width // 3))
        split.setSizes([left, max(360, width - left)])

    def _setup_plot(self):
        # Imported here rather than at module scope so the headless tests can
        # exercise the config and serial-ownership logic without pyqtgraph.
        import pyqtgraph

        pw = self.ui.emiPlot
        pw.setBackground("w")
        pw.setLabel("bottom", "X (mm from origin)")
        pw.setLabel("left", "Y (mm from origin)")
        for axis_name in ("bottom", "left"):
            axis = pw.getAxis(axis_name)
            axis.enableAutoSIPrefix(False)
            axis.setPen(pyqtgraph.mkPen("k"))
            axis.setTextPen(pyqtgraph.mkPen("k"))
        self.image = pyqtgraph.ImageItem()
        self.image.setColorMap(pyqtgraph.colormap.get("inferno"))
        pw.addItem(self.image)
        pw.setAspectLocked(True)

    def _setup_board_plots(self):
        # pyqtgraph stays inside the plot methods, as in _setup_plot, so the
        # headless tests can drive the interlocks without it installed.
        import pyqtgraph

        for plot in (self.ui.boardPlot, self.ui.regPlot):
            plot.setBackground("w")
            plot.setLabel("bottom", "Board X (mm)")
            plot.setLabel("left", "Board Y (mm)")
            for axis_name in ("bottom", "left"):
                axis = plot.getAxis(axis_name)
                axis.enableAutoSIPrefix(False)
                axis.setPen(pyqtgraph.mkPen("k"))
                axis.setTextPen(pyqtgraph.mkPen("k"))
            plot.setAspectLocked(True)

        self._landmark_marks = pyqtgraph.ScatterPlotItem(
            size=13, symbol="o", pen=pyqtgraph.mkPen("k"), brush=pyqtgraph.mkBrush("#00b070")
        )
        self._selected_mark = pyqtgraph.ScatterPlotItem(
            size=20, symbol="+", pen=pyqtgraph.mkPen("#b30000", width=2)
        )
        self.ui.regPlot.addItem(self._landmark_marks)
        self.ui.regPlot.addItem(self._selected_mark)
        self.ui.regPlot.scene().sigMouseClicked.connect(self._on_reg_plot_clicked)

    def _draw_board(self, plot, view):
        """Draw one board view: artwork, outline, then component markers."""
        import pyqtgraph

        plot.clear()
        if plot is self.ui.regPlot:
            plot.addItem(self._landmark_marks)
            plot.addItem(self._selected_mark)
        if view is None:
            return

        artwork = _polyline_batch([stroke.points for stroke in view.strokes])
        if artwork is not None:
            xs, ys, connect = artwork
            plot.addItem(
                pyqtgraph.PlotCurveItem(
                    xs, ys, connect=connect, pen=pyqtgraph.mkPen("#8899aa", width=1)
                )
            )

        outline = _polyline_batch([ring.points for ring in view.outline.rings], close=True)
        if outline is not None:
            xs, ys, connect = outline
            plot.addItem(
                pyqtgraph.PlotCurveItem(
                    xs, ys, connect=connect, pen=pyqtgraph.mkPen("k", width=2)
                )
            )

        if view.components:
            plot.addItem(
                pyqtgraph.ScatterPlotItem(
                    [part.x_mm for part in view.components],
                    [part.y_mm for part in view.components],
                    size=5,
                    symbol="s",
                    pen=None,
                    brush=pyqtgraph.mkBrush("#c04000"),
                )
            )
        # Same cap the PNG/HTML overlays use: past it the labels are unreadable
        # anyway and every one is a separate text item.
        if len(view.components) <= engine_overlay.MAX_LABELLED_COMPONENTS:
            for part in view.components:
                label = pyqtgraph.TextItem(part.refdes, color="#603000", anchor=(0.5, 1.2))
                label.setPos(part.x_mm, part.y_mm)
                plot.addItem(label)

    # ------------------------------------------------------------------ public
    def start(self):
        if engine_scanner is None:
            QMessageBox.critical(
                self.ui,
                "EMI Near-Field Map",
                "The EMI_Mapper engine could not be imported, so mapping is unavailable.\n\n"
                f"{ENGINE_ERROR}",
            )
            return
        self._refresh_ports()
        self._refresh_profiles()
        if not self.ui.outputRoot.text():
            self.ui.outputRoot.setText(str(Path.cwd() / "EMI_Scans"))
        self.ui.wizardStack.setCurrentIndex(PAGE_SETUP)
        self._update_mode_ui()
        self._on_area_changed()
        self._update_header()
        self._update_nav()
        self._fit_dialog_to_screen()
        self._layout_register_page()
        self.ui.show()
        self.ui.raise_()
        self.ui.activateWindow()

    def _fit_dialog_to_screen(self):
        """Keep Back/Next on screen. The Setup form is taller than a laptop
        viewport, so it scrolls inside setupScroll instead of stretching the
        dialog off the bottom of the display."""
        screen = self.ui.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        width = min(max(self.ui.width(), 720), available.width() - 32)
        height = min(max(560, self.ui.height()), available.height() - 32)
        self.ui.setMaximumSize(available.width(), available.height())
        self.ui.resize(width, height)
        frame = self.ui.frameGeometry()
        self.ui.move(
            available.x() + max(0, (available.width() - frame.width()) // 2),
            available.y() + max(0, (available.height() - frame.height()) // 2),
        )

    # ------------------------------------------------------------------ setup
    def _refresh_ports(self):
        """Offer the ports that could be the printer, likeliest first."""
        combo = self.ui.printerPort
        current = self._printer_port()
        combo.clear()
        for device, label in self._candidate_ports():
            combo.addItem(label, device)
        index = combo.findData(current)
        if index is not None and index >= 0:
            combo.setCurrentIndex(index)
        elif current:
            combo.setCurrentText(current)

    def _candidate_ports(self):
        """(device, label) pairs, with the recognised USB-serial bridges first."""
        # Every connected analyser, not just an adopted one: QtTinySA is
        # sweeping on those ports whether or not this dialog has claimed one.
        analysers = {
            getattr(device, "usbPort", None)
            for device in (self.serial.usb_instr.devices or [])
            if device is not None
        }
        found = []
        for port in list_ports.comports():
            description = (port.description or "").strip()
            haystack = f"{description} {port.manufacturer or ''}".lower()
            if any(hint in haystack for hint in PORT_EXCLUDE_HINTS):
                continue
            if port.device in analysers:
                continue  # QtTinySA is sweeping on that one
            label = f"{port.device} - {description}" if description else port.device
            likely = any(hint in haystack for hint in PORT_PRINTER_HINTS)
            found.append((port.device, label, likely))
        found.sort(key=lambda entry: not entry[2])
        return [(device, label) for device, label, _ in found]

    def _printer_port(self):
        """The bare device name behind whatever the combo is displaying.

        The combo is editable and its items carry a description, so the text
        can be either "COM11" or "COM11 - USB-SERIAL CH340".
        """
        combo = self.ui.printerPort
        text = (combo.currentText() or "").strip()
        data = combo.currentData()
        if data and text.startswith(str(data)):
            return str(data)
        return text.split(" ", 1)[0]

    def _on_printer_port_changed(self, _port=None):
        """A different port is a different machine, so nothing about the old
        frame carries over. Also drops the open handle so the next operation
        reconnects instead of talking to the previous printer."""
        self._close_printer()
        self._clear_pcb_homing()
        self._update_nav()

    def _browse_folder(self):
        chosen = QFileDialog.getExistingDirectory(
            self.ui, "Choose the folder that receives scan results", self.ui.outputRoot.text()
        )
        if chosen:
            self.ui.outputRoot.setText(chosen)

    def _on_area_changed(self):
        # Geometry changed, so the frozen grid is stale. The landmarks and the
        # fit are not: they still describe where the board sits. Invalidation
        # runs downwards only, so they survive this.
        self._invalidate_plan()
        self.ui.gridInfo.setText(self._grid_summary())
        self._update_travel_warning()
        # The residual limit scales with the step, so the verdict can change
        self._update_registration_ui()
        self._update_nav()

    def _grid_summary(self):
        step_mm = self.ui.stepMm.value()
        if self._mode() == MODE_PCB and self._board_view is not None:
            x_start, x_end, y_start, y_end = _board_bounds(self._board_view, step_mm)
            area = engine_config.ScanArea(
                width_mm=x_end - x_start,
                height_mm=y_end - y_start,
                step_mm=step_mm,
            )
            return f"{area.nx} x {area.ny} cells over the board, clipped to its outline"
        area = engine_config.ScanArea(
            width_mm=self.ui.widthMm.value(),
            height_mm=self.ui.heightMm.value(),
            step_mm=step_mm,
        )
        return f"{area.nx} x {area.ny} = {area.nx * area.ny} points"

    def _update_travel_warning(self):
        self.ui.travelWarning.setText(
            TRAVEL_WARNING.format(width=self.ui.widthMm.value(), height=self.ui.heightMm.value())
        )

    def _printer_config(self):
        """Machine settings shared by the scan, the jog buttons and homing.

        ``set_current_xy_as_origin`` is always false: rectangle mode sends its
        own G92 from the Origin page and a second one would move the map, and
        PCB mode works in true machine coordinates after G28 X Y.
        """
        return engine_config.PrinterConfig(
            port=self._printer_port(),
            baud=self.ui.printerBaud.value(),
            settle_s=self.ui.settleMs.value() / 1000.0,
            set_current_xy_as_origin=False,
        )

    def build_config(self):
        """Translate the form into a ScanConfig. Pure, so it is testable headless."""
        u = self.ui
        rbw = u.rbwKhz.value()
        atten = u.attenDb.value()
        baud = 115200
        dev = self.serial.device()
        if dev is not None and dev.usb is not None:
            baud = getattr(dev.usb, "baudrate", baud)

        config = engine_config.ScanConfig(
            printer=self._printer_config(),
            tinysa=engine_config.TinySAConfig(
                baud=baud,
                center_hz=u.centreMhz.value() * 1e6,
                span_hz=u.spanMhz.value() * 1e6,
                points=u.points.value(),
                rbw_khz="auto" if rbw <= 0 else rbw,
                attenuation_db="auto" if atten < 0 else atten,
                lna=u.lna.isChecked(),
                spur=u.spur.currentText(),
                samples_per_xy=u.samplesPerXy.value(),
            ),
            area=engine_config.ScanArea(
                width_mm=u.widthMm.value(),
                height_mm=u.heightMm.value(),
                step_mm=u.stepMm.value(),
            ),
            metric=u.metricBox.currentText(),
            background=u.chkBackground.isChecked(),
            background_mode=u.backgroundMode.currentText(),
            output_root=u.outputRoot.text().strip() or "EMI_Scans",
            label=u.runLabel.text().strip(),
        )
        config.validate()
        return config

    # ------------------------------------------------------- invalidation
    # Three tiers, invalidated downwards only. Routing every clear through
    # these keeps the cascade in one place instead of spread across the six
    # handlers that can trigger it.
    def _invalidate_plan(self):
        """A geometry change: the frozen grid is stale, the fit above it is not."""
        self._plan = None

    def _invalidate_board_alignment(self):
        """The board-to-machine fit is no longer trustworthy, so its plan goes too."""
        self._registration = None
        self._invalidate_plan()
        self._bump_registration()
        self._update_registration_ui()
        self._update_nav()

    def _clear_landmarks_and_alignment(self):
        self._landmarks.clear()
        self._selected_landmark = None
        self._clear_profile()
        self._invalidate_board_alignment()

    def _clear_profile(self):
        """Drop the loaded profile and everything that was verified about it."""
        self._profile = None
        self._verification_candidate = None
        self._verified_points = []
        self._offset_correction_mm = (0.0, 0.0)
        self._acknowledged_warnings = []
        self._noted_warnings = []

    # -------------------------------------------------- verification revisions
    # Two counters, because verification can be invalidated by two unrelated
    # things: the transform changing under it, and the probe moving away from
    # the point that was verified.  A candidate carries both, so it expires
    # without anything having to remember to clear it.
    def _bump_registration(self):
        """The transform changed, so nothing verified against the old one holds."""
        self._registration_revision += 1
        self._verification_candidate = None
        self._verified_points = []

    def _bump_motion(self):
        """The probe moved, so any pending candidate describes where it was."""
        self._motion_revision += 1
        self._verification_candidate = None

    def _candidate_is_current(self):
        candidate = self._verification_candidate
        return candidate is not None and (
            candidate["registration_revision"] == self._registration_revision
            and candidate["motion_revision"] == self._motion_revision
        )

    def _verification_points_required(self):
        """One point only when a fixture is holding the orientation.

        A single point pins translation exactly and says nothing at all about
        rotation about itself, so it can only be enough when something
        mechanical is preventing that rotation.
        """
        return 1 if self.ui.chkOrientationLocked.isChecked() else 2

    def _verification_problems(self):
        """Why a loaded profile is not yet fit to scan with."""
        if self._profile is None:
            return []
        required = self._verification_points_required()
        confirmed = len(self._verified_points)
        if confirmed < required:
            return [
                f"profile not verified: {confirmed} of {required} points confirmed"
            ]
        if required > 1 and not self._verified_spread_ok():
            return ["the verified points are too close together to detect rotation"]
        return []

    def _verified_spread_ok(self):
        """Two verification points only prove orientation if they are far apart.

        The same 50% of the diagonal the fit itself demands: closer than that
        and the angle they constrain is no better than a guess.
        """
        if self._board_view is None or len(self._verified_points) < 2:
            return False
        board = [point["board_xy"] for point in self._verified_points]
        required_mm = (
            engine_registration.MIN_SEPARATION_FRACTION * self._board_view.diagonal_mm
        )
        return _spread_mm(board) >= required_mm

    def _clear_pcb_homing(self):
        """The machine frame moved, so every recorded machine position is a claim
        about a frame that no longer exists."""
        self._pcb_xy_homed = False
        self._machine_xy = None
        self.ui.posLabel.setText("Not homed.")
        self._clear_landmarks_and_alignment()

    # ------------------------------------------------------------------ board
    def _import_board(self):
        path, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Choose the ODB++ job for this board",
            self.ui.boardPath.text(),
            "ODB++ archives (*.tgz *.tar *.tar.gz);;All files (*)",
        )
        if not path:
            return
        try:
            model = engine_odbpp.import_odbpp(path)
        except engine_odbpp.OdbppError as exc:
            self._board_model = None
            self._board_view = None
            self._clear_landmarks_and_alignment()
            self.ui.boardInfo.setText(f"Import failed: {exc}")
            self.ui.boardWarnings.setPlainText("")
            self._draw_board(self.ui.boardPlot, None)
            self._draw_board(self.ui.regPlot, None)
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            self._update_nav()
            return
        self._board_model = model
        self.ui.boardPath.setText(path)
        self._apply_board_side()

    def _apply_board_side(self, _side=None):
        """Derive the view for the selected side and drop what it invalidates.

        A new board or a flipped side invalidates the board-to-machine
        transform, so the landmarks recorded against the old one go with it.
        Homing survives: flipping the board does not move the Ender's frame.
        """
        if self._board_model is None:
            return
        self._board_view = self._board_model.view(self.ui.boardSide.currentText())
        self._clear_landmarks_and_alignment()
        self._describe_board()
        self._draw_board(self.ui.boardPlot, self._board_view)
        self._draw_board(self.ui.regPlot, self._board_view)
        self.ui.gridInfo.setText(self._grid_summary())
        self._update_nav()

    def _describe_board(self):
        view = self._board_view
        width_mm, height_mm = view.size_mm
        x0, y0, x1, y1 = view.bbox_mm
        self.ui.boardInfo.setText(
            f"{width_mm:.2f} x {height_mm:.2f} mm, {view.outline.island_count} outline(s), "
            f"{view.outline.hole_count} cutout(s), {len(view.components)} components, "
            f"{len(view.strokes)} artwork strokes\n"
            f"Side {view.side}{' (mirrored)' if view.mirrored else ''}, "
            f"board-view extent X[{x0:.2f}, {x1:.2f}] Y[{y0:.2f}, {y1:.2f}] mm\n"
            f"{view.height_note()}"
        )
        warnings = list(view.report.warnings) if view.report is not None else []
        self.ui.boardWarnings.setPlainText("\n".join(warnings))

    # ------------------------------------------------------- homing and jog
    def _printer(self):
        """A Printer on the shared transport, carrying the form's machine limits."""
        return engine_printer.Printer(self._open_printer(), self._printer_config())

    def _home_xy_clicked(self):
        if not self.ui.chkHomeClear.isChecked():
            return
        self._start_printer_job(
            self._do_home_xy,
            self._on_home_ok,
            self._on_home_failed,
            self.ui.posLabel,
            "Homing X/Y… the window stays usable; this can take a few seconds.",
        )

    def _do_home_xy(self):
        printer = self._printer()
        printer.drain()
        printer.prepare()
        printer.home_xy()
        return printer.get_xy()

    def _on_home_ok(self, position):
        self._pcb_xy_homed = True
        self._bump_motion()
        self._clear_landmarks_and_alignment()
        self._show_position(position)

    def _on_home_failed(self, exc):
        self._clear_pcb_homing()
        self.ui.posLabel.setText(f"Homing failed: {exc}")
        QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))

    def _home_xy(self):
        """Synchronous home, used by the headless tests."""
        try:
            self._on_home_ok(self._do_home_xy())
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            self._on_home_failed(exc)
        self._update_nav()

    def _jog(self, dx, dy):
        if not self._pcb_xy_homed:
            QMessageBox.warning(self.ui, DIALOG_TITLE, "Home X and Y before jogging.")
            return
        step_mm = self.ui.jogStep.value()
        try:
            printer = self._printer()
            x_mm, y_mm = printer.get_xy()
            # move_xy refuses anything outside the configured machine limits
            printer.move_xy(x_mm + dx * step_mm, y_mm + dy * step_mm)
            self._bump_motion()
            self._show_position(printer.get_xy())
        except engine_printer.PrinterReset as exc:
            self._clear_pcb_homing()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            # The probe may have moved before the failure, so assume it did
            self._bump_motion()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
        self._update_registration_ui()
        self._update_nav()

    def _show_position(self, position):
        self._machine_xy = (float(position[0]), float(position[1]))
        self.ui.posLabel.setText(
            f"Machine X {self._machine_xy[0]:.2f}  Y {self._machine_xy[1]:.2f} mm"
        )

    # ----------------------------------------------------------- registration
    def _on_reg_plot_clicked(self, event):
        if self._board_view is None:
            return
        view_box = self.ui.regPlot.getPlotItem().getViewBox()
        point = view_box.mapSceneToView(event.scenePos())
        self._select_landmark(point.x(), point.y())

    def _nearest_landmark(self, view, distance):
        """Snap to a suggested corner, or to the nearest outline vertex.

        The suggested corners are what make a wide baseline, so they win when
        the click lands anywhere near one; the vertex fallback covers boards
        whose usable landmark is a notch or a mounting ear, not a corner.
        """
        corners = [
            (role, spot["board_x_mm"], spot["board_y_mm"])
            for role, spot in view.suggested_registration_points().items()
        ]
        nearest = min(corners, key=distance) if corners else None
        snap_mm = LANDMARK_SNAP_FRACTION * view.diagonal_mm
        if nearest is not None and distance(nearest) <= snap_mm:
            return nearest
        vertices = [
            ("outline vertex", float(px), float(py))
            for ring in view.outline.rings
            for px, py in ring.points
        ]
        return min(vertices, key=distance) if vertices else nearest

    def _on_snap_changed(self, _state=None):
        """Turning snapping on or off makes the current selection meaningless."""
        self._clear_selection()

    def _clear_selection(self):
        self._selected_landmark = None
        self._verification_candidate = None
        if self._selected_mark is not None:
            self._selected_mark.setData([], [])
        self.ui.btnRecordLandmark.setText("Record")
        self.ui.targetLabel.setText("No point selected.")
        self._update_nav()

    def _exact_point(self, view, x_mm, y_mm):
        """An arbitrary clicked point, if it is on the board at all.

        Snapping off is for driving the probe to a component to eyeball the
        alignment, which only means anything over the board.  A click in the
        surrounding whitespace is a miss, not an instruction to move there.
        """
        inside = view.outline.contains(
            (x_mm, y_mm), tolerance_mm=CLICK_EDGE_TOLERANCE_MM
        )
        if not bool(np.asarray(inside).ravel()[0]):
            return None
        return ("clicked point", float(x_mm), float(y_mm))

    def _select_landmark(self, x_mm, y_mm):
        view = self._board_view
        if view is None:
            return

        def distance(candidate):
            return math.hypot(candidate[1] - x_mm, candidate[2] - y_mm)

        if self.ui.chkSnapLandmark.isChecked():
            chosen = self._nearest_landmark(view, distance)
        else:
            chosen = self._exact_point(view, x_mm, y_mm)
            if chosen is None:
                self._selected_landmark = None
                self._verification_candidate = None
                if self._selected_mark is not None:
                    self._selected_mark.setData([], [])
                self.ui.targetLabel.setText(
                    f"({x_mm:.2f}, {y_mm:.2f}) mm is off the board outline. "
                    "Click on the board, or tick snapping to use its corners."
                )
                self._update_nav()
                return
        if chosen is None:
            return

        # A new selection is a new question, so any pending confirmation of the
        # old one goes with it.
        self._selected_landmark = chosen
        self._verification_candidate = None
        role, board_x, board_y = chosen
        if self._selected_mark is not None:
            self._selected_mark.setData([board_x], [board_y])
        self.ui.targetLabel.setText(self._describe_target(role, board_x, board_y))
        self._update_nav()

    def _describe_target(self, role, board_x, board_y):
        """Where the probe would go, before anything is asked to move."""
        base = f"{role} at board ({board_x:.2f}, {board_y:.2f}) mm"
        if self._registration is None:
            return f"{base}. No alignment yet, so the machine position is unknown."
        machine_x, machine_y = self._machine_target(board_x, board_y)
        limits = self._limit_problem(machine_x, machine_y)
        if limits:
            return f"{base} -> machine X {machine_x:.2f} Y {machine_y:.2f}. {limits}"
        return f"{base} -> machine X {machine_x:.2f} Y {machine_y:.2f} mm"

    def _machine_target(self, board_x, board_y):
        target = self._registration.transform.to_machine((board_x, board_y))
        return float(target[0][0]), float(target[0][1])

    def _limit_problem(self, machine_x, machine_y):
        """Refuse an unreachable target here, in board terms, not as a G-code error.

        ``Printer.move_xy`` checks the same limits and is still the last line of
        defence, but its ValueError talks about machine words; by this point the
        operator wants to know which board point cannot be reached.
        """
        printer = self._printer_config()
        if not printer.x_min_mm <= machine_x <= printer.x_max_mm:
            return (
                f"X {machine_x:.2f} mm is outside the machine range "
                f"[{printer.x_min_mm:g}, {printer.x_max_mm:g}]."
            )
        if not printer.y_min_mm <= machine_y <= printer.y_max_mm:
            return (
                f"Y {machine_y:.2f} mm is outside the machine range "
                f"[{printer.y_min_mm:g}, {printer.y_max_mm:g}]."
            )
        return ""

    # ------------------------------------------------------- click to move
    def _move_to_selected(self):
        """Drive the probe to the selected board point using the current fit.

        Deliberately a button and not the plot click: a stray click on a plot
        should never start the machine moving.  It runs on the printer worker
        because a board-width travel is far longer than a jog and would freeze
        the dialog.
        """
        target = self._move_target()
        if target is None:
            return
        board_x, board_y, machine_x, machine_y = target
        self._start_printer_job(
            partial(self._do_move_to, machine_x, machine_y),
            partial(self._on_move_ok, board_x, board_y),
            self._on_move_failed,
            self.ui.targetLabel,
            f"Moving to machine X {machine_x:.2f} Y {machine_y:.2f} mm…",
        )

    def _move_target(self):
        """The board point and the machine point it maps to, or None with a reason."""
        if self._selected_landmark is None or self._registration is None:
            return None
        if not self._pcb_xy_homed:
            QMessageBox.warning(
                self.ui, DIALOG_TITLE, "Home X and Y before moving the probe."
            )
            return None
        role, board_x, board_y = self._selected_landmark
        machine_x, machine_y = self._machine_target(board_x, board_y)
        problem = self._limit_problem(machine_x, machine_y)
        if problem:
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                f"Cannot move to {role} at board ({board_x:.2f}, {board_y:.2f}) mm: "
                f"{problem}",
            )
            return None
        return board_x, board_y, machine_x, machine_y

    def _move_now(self):
        """Synchronous move, used by the headless tests."""
        target = self._move_target()
        if target is None:
            return
        board_x, board_y, machine_x, machine_y = target
        try:
            self._on_move_ok(board_x, board_y, self._do_move_to(machine_x, machine_y))
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            self._on_move_failed(exc)
        self._update_nav()

    def _do_move_to(self, machine_x, machine_y):
        printer = self._printer()
        printer.move_xy(machine_x, machine_y)
        return printer.get_xy()

    def _on_move_ok(self, board_x, board_y, position):
        # One motion, one revision: any later jog expires the candidate below.
        self._bump_motion()
        self._verification_candidate = {
            "board_xy": (float(board_x), float(board_y)),
            "machine_xy": (float(position[0]), float(position[1])),
            "registration_revision": self._registration_revision,
            "motion_revision": self._motion_revision,
        }
        self._show_position(position)
        self.ui.targetLabel.setText(
            f"Probe is at board ({board_x:.2f}, {board_y:.2f}) mm as aligned. "
            "Check it against the physical feature."
        )
        self._update_registration_ui()

    def _on_move_failed(self, exc):
        self._bump_motion()
        if isinstance(exc, engine_printer.PrinterReset):
            self._clear_pcb_homing()
        self.ui.targetLabel.setText(f"Move failed: {exc}")
        QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
        self._update_registration_ui()

    def _record_landmark(self):
        if self._selected_landmark is None or not self._pcb_xy_homed:
            return
        role, board_x, board_y = self._selected_landmark
        try:
            machine_x, machine_y = self._printer().get_xy()
        except engine_printer.PrinterReset as exc:
            self._clear_pcb_homing()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return
        clash = self._duplicate_landmark(board_x, board_y, machine_x, machine_y)
        if clash:
            QMessageBox.warning(self.ui, DIALOG_TITLE, clash)
            return
        self._landmarks.append(
            engine_registration.RegistrationPoint(
                board_x_mm=board_x,
                board_y_mm=board_y,
                machine_x_mm=machine_x,
                machine_y_mm=machine_y,
                role=role,
            )
        )
        self._show_position((machine_x, machine_y))
        self._refit_registration()

    def _duplicate_landmark(self, board_x, board_y, machine_x, machine_y):
        """Reject a landmark that repeats an earlier one in either frame.

        Distinct board points have to sit at distinct machine points, so a
        repeat is always one of two slips: the probe was not jogged, or the
        same feature was clicked twice.  Neither can improve the fit, and both
        are far easier to explain here than as a failed fit later.
        """
        for index, point in enumerate(self._landmarks, start=1):
            if math.hypot(machine_x - point.machine_x_mm, machine_y - point.machine_y_mm) <= COINCIDENT_MM:
                return (
                    f"The probe is still at machine X {machine_x:.2f}, "
                    f"Y {machine_y:.2f} mm, where landmark {index} was recorded. "
                    "Jog it onto the new physical point first."
                )
            if math.hypot(board_x - point.board_x_mm, board_y - point.board_y_mm) <= COINCIDENT_MM:
                return (
                    f"Board point ({board_x:.2f}, {board_y:.2f}) mm is already "
                    f"landmark {index}. Click a different feature on the plot."
                )
        return ""

    def _remove_landmark(self):
        """Drop one landmark, keeping the rest of the set.

        A single point on the wrong feature is the usual cause of a failed
        scale check, and re-recording all of them to fix one is needless.
        """
        if not self._landmarks:
            return
        row = self.ui.landmarkTable.currentRow()
        if not isinstance(row, int) or not 0 <= row < len(self._landmarks):
            row = len(self._landmarks) - 1  # no selection: the most recent one
        self._landmarks.pop(row)
        self._refit_registration()

    def _clear_landmarks(self):
        self._clear_landmarks_and_alignment()
        self._clear_selection()

    def _refit_registration(self):
        self._invalidate_plan()
        self._registration = None
        self._fit_error = ""
        self._bump_registration()
        if len(self._landmarks) >= 2:
            try:
                self._registration = engine_registration.fit_registration(
                    tuple(self._landmarks)
                )
            except engine_registration.RegistrationError as exc:
                self._fit_error = str(exc)
                logging.info(f"EMI map registration could not be fitted: {exc}")
        self._update_registration_ui()
        self._update_nav()

    # ---------------------------------------------------- offset correction
    def _update_offset(self):
        """Shift the whole alignment by the error the operator just jogged out.

        The probe is standing on the true feature, so the difference between
        where it is and where the transform says that board point should be is
        the fixture's translation error.  Every landmark moves by it and the
        fit is re-run, which keeps the residuals describing measurements
        instead of becoming a patched transform nobody can audit.
        """
        if self._selected_landmark is None or self._registration is None:
            return
        if not self._pcb_xy_homed:
            return
        _role, board_x, board_y = self._selected_landmark
        error = self._probe_error(board_x, board_y)
        if error is None:
            return
        actual_x, actual_y, dx, dy = error

        previous = self._registration
        try:
            shifted = engine_profiles.apply_offset(previous.points, dx, dy)
            corrected = engine_registration.fit_registration(
                shifted, rotation_locked=previous.rotation_locked
            )
        except (engine_profiles.ProfileError, engine_registration.RegistrationError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return

        refusal = self._correction_refusal(previous, corrected)
        if refusal:
            QMessageBox.warning(self.ui, DIALOG_TITLE, refusal)
            return

        self._landmarks[:] = list(shifted)
        self._registration = corrected
        self._fit_error = ""
        self._invalidate_plan()
        self._offset_correction_mm = (
            self._offset_correction_mm[0] + dx,
            self._offset_correction_mm[1] + dy,
        )
        # The transform moved, so everything verified against the old one is
        # void -- but the probe is standing on this point right now under the
        # corrected transform, so this one point is legitimately verifiable.
        self._bump_registration()
        self._verification_candidate = {
            "board_xy": (float(board_x), float(board_y)),
            "machine_xy": (float(actual_x), float(actual_y)),
            "registration_revision": self._registration_revision,
            "motion_revision": self._motion_revision,
        }
        self.ui.targetLabel.setText(
            f"Alignment shifted by X {dx:+.3f} Y {dy:+.3f} mm. Confirm this "
            "point, then verify the others."
        )
        self._update_registration_ui()
        self._update_nav()

    def _probe_error(self, board_x, board_y):
        """Where the probe is, and how far that is from where the fit says it should be."""
        try:
            actual_x, actual_y = self._printer().get_xy()
        except engine_printer.PrinterReset as exc:
            self._clear_pcb_homing()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return None
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return None
        expected_x, expected_y = self._machine_target(board_x, board_y)
        return actual_x, actual_y, actual_x - expected_x, actual_y - expected_y

    def _correction_refusal(self, previous, corrected):
        """Why a correction must not be applied, or an empty string."""
        # A pure translation cannot rotate the fit, but least squares in
        # floating point is not bit-exact, so this is a tolerance not equality.
        x0, y0, x1, y1 = self._board_view.bbox_mm
        rotation_only = engine_registration.Transform(
            theta_rad=corrected.transform.theta_rad,
            x0_mm=previous.transform.x0_mm,
            y0_mm=previous.transform.y0_mm,
        )
        if not engine_registration.transforms_agree(
            previous.transform,
            rotation_only,
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            engine_profiles.TRANSFORM_AGREEMENT_MM,
        ):
            return (
                "Correcting the offset changed the board rotation, which a "
                "translation cannot do. Register manually instead."
            )
        unreachable = self._reachability_problem(corrected)
        if unreachable:
            return f"That correction puts the board out of reach: {unreachable}"
        return ""

    def _reachability_problem(self, registration):
        """Would a scan on this fit drive off the bed? Ask the plan, not the corner.

        The same check Start runs, brought forward: a correction of a fraction
        of a millimetre can only push the board out of reach if it was already
        on the edge, and finding that out at Start means losing the whole
        registration to it.
        """
        try:
            step_mm = self.ui.stepMm.value()
            x_start, x_end, y_start, y_end = _board_bounds(self._board_view, step_mm)
            plan = engine_registration.build_plan(
                self._board_view,
                registration.transform,
                x_start=x_start,
                x_end=x_end,
                y_start=y_start,
                y_end=y_end,
                step_mm=step_mm,
                clip_to_outline=True,
                registration=registration,
            )
            plan.validate_machine_limits(self._printer_config())
        except (ValueError, RuntimeError) as exc:
            return str(exc)
        return ""

    # ------------------------------------------------------- verification
    def _confirm_alignment(self):
        """Record that the probe really is standing on the selected point.

        This is the only statement in the whole flow that software cannot
        make, so it is the one thing that must come from the operator, and it
        expires the moment anything moves.
        """
        if self._profile is None or self._selected_landmark is None:
            return
        if not self.ui.chkBoardSeated.isChecked():
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                "Confirm the board is seated as it was when the profile was saved.",
            )
            return
        if not self._candidate_is_current():
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                "Nothing to confirm. Select a point, press Move probe here, and "
                "confirm without jogging afterwards; a jog or a refit means the "
                "probe is no longer where the alignment was checked.",
            )
            return
        candidate = self._verification_candidate
        self._verified_points.append(
            {
                "board_xy": candidate["board_xy"],
                "machine_xy": candidate["machine_xy"],
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        self._verification_candidate = None
        self._update_registration_ui()
        self._update_nav()

    def _on_orientation_lock_changed(self, _state=None):
        self._update_registration_ui()
        self._update_nav()

    def _registration_problems(self):
        """Every reason the current alignment may not drive a scan."""
        if self._board_view is None:
            return ["no board imported"]
        if self._registration is None:
            return ["fewer than two landmarks recorded"]
        return (
            engine_registration.registration_problems(
                self._registration,
                diagonal_mm=self._board_view.diagonal_mm,
                step_mm=self.ui.stepMm.value(),
            )
            # A loaded profile fits perfectly by construction, so the maths
            # alone would open the gate before anyone looked at the board.
            + self._verification_problems()
        )

    # ------------------------------------------------------------- profiles
    def _refresh_profiles(self):
        """Repopulate the chooser, keeping the current selection if it survives."""
        combo = self.ui.fixtureProfile
        wanted = combo.currentText()
        try:
            summaries = engine_profiles.list_profiles()
        except (OSError, engine_profiles.ProfileError) as exc:
            logging.info(f"EMI map could not list fixture profiles: {exc}")
            summaries = []
        combo.clear()
        for summary in summaries:
            label = summary.name if summary.usable else f"{summary.name} (unreadable)"
            combo.addItem(label, summary.name if summary.usable else None)
        index = combo.findText(wanted)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _selected_profile_name(self):
        combo = self.ui.fixtureProfile
        name = combo.currentData()
        return name if name else ""

    def _identity(self):
        return {
            "fixture_id": self.ui.fixtureId.text().strip(),
            "machine_id": self.ui.machineId.text().strip(),
            "probe_setup_id": self.ui.probeSetupId.text().strip(),
        }

    def _load_profile(self):
        """Restore a saved alignment, then insist it be verified.

        Loading refits the saved points, so what lands in ``_registration`` is
        an ordinary fitted registration and every existing gate keeps working
        on it unchanged.  What it is not is checked against the board in front
        of the operator, which is what the verification state is for.
        """
        name = self._selected_profile_name()
        if not name or self._board_view is None or not self._pcb_xy_homed:
            return
        if not self.ui.chkBoardSeated.isChecked():
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                "Seat the board as it was when the profile was saved, then "
                "confirm that with the checkbox before loading.",
            )
            return
        try:
            loaded = engine_profiles.load_profile(
                name,
                self._board_view,
                printer_config=self._printer_config(),
                **self._identity(),
            )
        except engine_profiles.ProfileError as exc:
            self.ui.profileStatus.setText(f"Profile refused: {exc}")
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                f"{exc}\n\nRegister the board manually instead.",
            )
            return

        acknowledged = self._acknowledge(loaded)
        if acknowledged is None:
            self.ui.profileStatus.setText(
                f"Loading {name!r} cancelled; the previous alignment is unchanged."
            )
            return

        self._landmarks[:] = list(loaded.registration.points)
        self._registration = loaded.registration
        self._fit_error = ""
        self._invalidate_plan()
        self._bump_registration()
        self._profile = loaded
        self._offset_correction_mm = (0.0, 0.0)
        self._acknowledged_warnings = acknowledged
        self._noted_warnings = list(loaded.soft_warnings)
        self._clear_selection()
        self._update_registration_ui()
        self._update_nav()

    def _acknowledge(self, loaded):
        """Obtain a real acknowledgement of identity mismatches, or None to cancel.

        Displaying a warning is not the same as it having been read, and the
        snapshot claims these were acknowledged, so the claim has to be true:
        only the explicit second button counts.
        """
        if not loaded.hard_warnings:
            return []
        details = "\n".join(f"  {m.describe()}" for m in loaded.hard_warnings)
        box = QMessageBox(self.ui)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(DIALOG_TITLE)
        box.setText("This profile was saved for a different physical setup.")
        box.setInformativeText(
            f"{details}\n\nThe saved alignment describes where the board sat in "
            "that setup. Loading it here is only sensible if you know the "
            "difference does not matter, and it must still be verified."
        )
        anyway = box.addButton("Load and verify anyway", QMessageBox.AcceptRole)
        cancel = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(cancel)
        box.exec()
        if box.clickedButton() is not anyway:
            return None
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        return [
            dict(mismatch.to_dict(), acknowledged_at=stamp)
            for mismatch in loaded.hard_warnings
        ]

    def _save_allowed(self):
        """A manual fit needs only to pass; a loaded one needs verifying too.

        ``_registration_problems`` already folds in the verification state, so
        an unconfirmed offset correction cannot be written back over the
        profile and inherited as fact by every later scan.
        """
        return self._registration is not None and not self._registration_problems()

    def _save_as_profile(self):
        name, accepted = QtWidgets.QInputDialog.getText(
            self.ui,
            DIALOG_TITLE,
            "Name for this profile:",
            text=self._suggested_profile_name(),
        )
        if not accepted:
            return
        self._write_profile(name, confirm_existing=True)

    def _update_profile(self):
        name = self._selected_profile_name()
        if not name:
            return
        answer = QMessageBox.question(
            self.ui,
            DIALOG_TITLE,
            f"Overwrite the saved profile {name!r} with the current alignment?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._write_profile(name, confirm_existing=False)

    def _write_profile(self, name, *, confirm_existing):
        if not self._save_allowed():
            return
        try:
            path = engine_profiles.profile_path(name)
        except engine_profiles.ProfileError as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return
        overwrite = not confirm_existing
        if confirm_existing and path.exists():
            answer = QMessageBox.question(
                self.ui,
                DIALOG_TITLE,
                f"A profile named {path.stem!r} already exists. Replace it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            overwrite = True
        try:
            saved_path, _document, _digest = engine_profiles.save_profile(
                name,
                self._board_view,
                self._registration,
                orientation_locked=self.ui.chkOrientationLocked.isChecked(),
                printer_config=self._printer_config(),
                overwrite=overwrite,
                **self._identity(),
            )
        except (engine_profiles.ProfileError, OSError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return
        self._refresh_profiles()
        index = self.ui.fixtureProfile.findText(saved_path.stem)
        if index >= 0:
            self.ui.fixtureProfile.setCurrentIndex(index)
        self.ui.profileStatus.setText(f"Saved {saved_path.stem!r} to {saved_path}")
        self._update_nav()

    def _suggested_profile_name(self):
        view = self._board_view
        if view is None or view.report is None:
            return ""
        job = (view.report.job_name or Path(view.report.source_name or "").stem).strip()
        fixture = self._identity()["fixture_id"]
        parts = [part for part in (job, fixture, view.side) if part]
        return " ".join(parts)[:64]

    def _profile_report(self):
        """One line on where the current alignment came from, and its standing."""
        if self._registration is None:
            return "No alignment."
        if self._profile is None:
            return "Manual landmarks."
        required = self._verification_points_required()
        confirmed = len(self._verified_points)
        lines = [f"Profile: {self._profile.name!r} saved {self._profile.saved_at}"]
        if self._verification_problems():
            lines.append(f"UNVERIFIED - {confirmed} of {required} points confirmed")
            if self._candidate_is_current():
                lines.append("Probe is on the selected point; confirm the alignment.")
            else:
                lines.append("Select a point, Move probe here, then confirm.")
        else:
            last = self._verified_points[-1]["verified_at"]
            lines.append(f"VERIFIED at {last} on {confirmed} point(s)")
        dx, dy = self._offset_correction_mm
        if dx or dy:
            lines.append(f"Offset correction applied: X {dx:+.3f} Y {dy:+.3f} mm")
        lines += [f"Note: {text}" for text in self._noted_warnings]
        lines += [
            f"Acknowledged: {entry['field']} saved {entry['saved']!r}, "
            f"current {entry['current']!r}"
            for entry in self._acknowledged_warnings
        ]
        return "\n".join(lines)

    def _update_registration_ui(self):
        self._fill_landmark_table()
        self.ui.profileStatus.setText(self._profile_report())
        self.ui.regStatus.setText(self._registration_report())
        registration = self._registration
        # Prominent, never blocking: two well-separated landmarks are a
        # supported registration, a third is only an independent check.
        third_useful = registration is not None and registration.point_count == 2
        self.ui.landmarkHint.setText(
            THIRD_LANDMARK_HINT
            if third_useful and not self._registration_problems()
            else ""
        )

    def _unfitted_reason(self):
        """Why the recorded landmarks still produced no transform.

        Recording several board points without jogging in between leaves every
        landmark on one machine position, and the generic advice to spread the
        landmarks across the board does not describe that mistake at all.
        """
        if len(self._landmarks) < 2:
            return "Record two landmarks near opposite ends of the board."
        machine = [(p.machine_x_mm, p.machine_y_mm) for p in self._landmarks]
        board = [(p.board_x_mm, p.board_y_mm) for p in self._landmarks]
        if _spread_mm(machine) <= COINCIDENT_MM:
            return (
                f"Every landmark was recorded at machine X {machine[0][0]:.2f}, "
                f"Y {machine[0][1]:.2f} mm, so the probe never moved between "
                "them. Clear the landmarks, then for each one: click the point "
                "on the plot, jog the probe onto that same physical point, and "
                "only then record it."
            )
        if _spread_mm(board) <= COINCIDENT_MM:
            return (
                "Every landmark is the same point on the board. Click a "
                "different feature, near the opposite end, before recording."
            )
        return self._fit_error or "The landmarks do not describe a rigid fit."

    def _registration_report(self):
        registration = self._registration
        if registration is None:
            return (
                "Registration: BLOCKED\n\n"
                f"Landmarks:     {len(self._landmarks)}\n\n"
                f"{self._unfitted_reason()}"
            )
        problems = self._registration_problems()
        required_mm = (
            engine_registration.MIN_SEPARATION_FRACTION * self._board_view.diagonal_mm
        )
        # Separation carries its own verdict because it, not the residual, is
        # what a two-landmark fit can actually fail: three degrees of freedom
        # absorb two points almost exactly, so RMS there is near-tautological.
        lines = [
            f"Registration: {'BLOCKED' if problems else 'PASS'}",
            "",
            f"Landmarks:    {registration.point_count:8d}",
            f"Separation:   {registration.separation_mm:8.1f} mm   "
            f"{'PASS' if registration.separation_mm >= required_mm else 'FAIL'}"
            f"  (need {required_mm:.1f} mm)",
            f"Rotation:     {registration.transform.theta_deg:8.4f} deg",
            f"Scale:        {registration.scale:8.4f}       diagnostic only, never applied",
            f"RMS residual: {registration.rms_mm:8.3f} mm",
            "",
        ]
        lines += problems or ["Registration valid."]
        lines += self._placement_advice(problems)
        return "\n".join(lines)

    def _placement_advice(self, problems):
        """Point a residual or scale failure at the thing that usually causes it.

        Both gates are sub-millimetre, so they are unreachable while the jog
        step is coarse: the probe can only be parked within half a step of the
        feature, and no amount of re-recording beats that.
        """
        if not any("residual" in text or "scale" in text for text in problems):
            return []
        step = self.ui.jogStep.value()
        limit = engine_registration.max_rms_mm(self.ui.stepMm.value())
        if step / 2.0 <= limit:
            return []
        return [
            "",
            f"The jog step is {step:g} mm, so the probe can only be parked "
            f"within {step / 2.0:g} mm of a feature, and this fit needs "
            f"{limit:.2f} mm. Approach coarsely, then set the jog step to "
            "0.1 mm for the last millimetre before recording.",
        ]

    def _fill_landmark_table(self):
        table = self.ui.landmarkTable
        table.setRowCount(len(self._landmarks))
        # Per-point residual, so a landmark placed on the wrong feature can be
        # found and dropped instead of clearing the whole set and starting over
        residuals = (
            self._registration.residuals_mm if self._registration is not None else ()
        )
        for row, point in enumerate(self._landmarks):
            columns = [
                f"{point.board_x_mm:.2f}",
                f"{point.board_y_mm:.2f}",
                f"{point.machine_x_mm:.2f}",
                f"{point.machine_y_mm:.2f}",
                f"{residuals[row]:.2f}" if row < len(residuals) else "-",
            ]
            for column, value in enumerate(columns):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        if self._landmark_marks is not None:
            self._landmark_marks.setData(
                [point.board_x_mm for point in self._landmarks],
                [point.board_y_mm for point in self._landmarks],
            )

    # ------------------------------------------------------------------ origin
    def _open_serial(self, port, baud):
        import serial

        return serial.Serial(port, baud, timeout=ENGINE_READ_TIMEOUT_S)

    def _open_printer(self):
        if self.printer_serial is not None and self.printer_serial.is_open:
            return self.printer_serial
        port = self._printer_port()
        if not port:
            raise RuntimeError("Choose the printer serial port first")
        handle = self._open_serial(port, self.ui.printerBaud.value())
        try:
            self._greet_marlin(handle, port)
        except Exception:
            handle.close()
            raise
        self.printer_serial = handle
        return handle

    def _greet_marlin(self, handle, port):
        """Refuse a port that is not a printer, in seconds rather than minutes.

        A silent port is otherwise only discovered one 30 s ok timeout at a
        time, and the operator sees a wizard that appears to have hung.
        """
        probe = engine_printer.Printer(handle, self._printer_config())
        probe.drain(quiet_s=PRINTER_BOOT_S)
        try:
            probe.send("M115", timeout_s=PRINTER_GREET_S)
        except engine_printer.PrinterTimeout:
            raise RuntimeError(
                f"{port} did not answer as a Marlin printer. Choose the port the "
                f"Ender is on, and check nothing else already has it open."
            ) from None

    def _start_printer_job(self, work, on_ok, on_fail, status_label, busy_text):
        """Run Marlin I/O off the GUI thread so the dialog cannot go 'Not Responding'."""
        if self._printer_job is not None:
            return
        status_label.setText(busy_text)
        self._printer_job_ok = on_ok
        self._printer_job_fail = on_fail
        worker = _BlockingPrinterJob(work)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_printer_job_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._printer_job = (thread, worker)
        self._update_nav()
        thread.start()

    def _on_printer_job_finished(self, payload):
        ok, fail = self._printer_job_ok, self._printer_job_fail
        self._printer_job = None
        self._printer_job_ok = None
        self._printer_job_fail = None
        if isinstance(payload, BaseException):
            fail(payload)
        else:
            ok(payload)
        self._update_nav()

    def _set_origin_clicked(self):
        self._start_printer_job(
            self._do_set_origin,
            self._on_origin_ok,
            self._on_origin_failed,
            self.ui.originStatus,
            "Talking to the printer… the window stays usable; wait for Origin set.",
        )

    def _do_set_origin(self):
        printer = self._printer()
        printer.drain()
        printer.prepare()
        printer.set_origin()
        return True

    def _on_origin_ok(self, _ok=None):
        self.origin_set = True
        # G92 shifts Marlin's workspace offset and its software endstop
        # frame, and G92.1 needs CNC_COORDINATE_SYSTEMS, which stock Ender
        # builds do not enable. So there is no undo: a PCB scan after this
        # has to home again to get back to machine coordinates.
        self._clear_pcb_homing()
        self.ui.originStatus.setText(
            "Origin set. Heaters and fans are off. This position is now X0 Y0."
        )

    def _on_origin_failed(self, exc):
        self.origin_set = False
        self.ui.originStatus.setText(f"Origin not set: {exc}")
        QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))

    def _set_origin(self):
        """Synchronous origin, used by the headless tests."""
        try:
            self._on_origin_ok(self._do_set_origin())
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            self._on_origin_failed(exc)
        self._update_nav()

    # ------------------------------------------------------------------ scan
    def _start_scan(self, _checked=False):
        """Validate everything, then hand over. Thin on purpose: the interesting
        decisions live in _preflight and _build_validated_plan."""
        reason = self._scan_block_reason()
        if reason:
            self._refuse_start(reason)
            return
        try:
            config = self.build_config()
            plan = self._preflight(config)
            self._launch(config, plan)
        except (ValueError, RuntimeError, OSError, engine_printer.PrinterError) as exc:
            self._refuse_start(str(exc))
        except Exception as exc:
            logging.exception("EMI scan failed to start")
            self._refuse_start(f"{type(exc).__name__}: {exc}")

    def _refuse_start(self, message):
        self.ui.scanStatus.setText(message)
        QMessageBox.warning(self.ui, DIALOG_TITLE, message)

    def _scan_block_reason(self):
        """Why Start scan would no-op, shown on the page instead of a silent grey button."""
        if self.thread is not None:
            return "A scan is already running."
        if self._printer_job is not None:
            return "Wait for the printer to finish moving."
        if self._mode() != MODE_PCB:
            if not self.origin_set:
                return "Set the origin first."
            return ""
        try:
            self._pcb_preflight()
        except (ValueError, RuntimeError) as exc:
            return str(exc)
        return ""

    def _apply_scan_gate(self):
        """Keep Start in step with the visible reason; do not clobber live status."""
        scanning = self.thread is not None
        reason = "" if scanning else self._scan_block_reason()
        self.ui.btnStartScan.setEnabled(not scanning and not reason)
        self.ui.btnStartScan.setToolTip(reason or "Start the scan.")
        self.ui.btnAbortScan.setEnabled(scanning)
        if scanning:
            return
        current = self.ui.scanStatus.text()
        previous = self._scan_gate
        idle = current in ("Idle.", "Ready to scan.", previous, "")
        self._scan_gate = reason
        if reason:
            self.ui.scanStatus.setText(reason)
        elif idle:
            self.ui.scanStatus.setText("Ready to scan.")

    def _preflight(self, config):
        """Everything that must hold before the analyser is borrowed.

        Raises on refusal and returns the plan to scan with, or None for a plain
        rectangle scan.
        """
        if self._mode() != MODE_PCB:
            if not self.origin_set:
                raise RuntimeError("Set the origin first.")
            return None
        self._pcb_preflight()
        return self._build_validated_plan(config)

    def _pcb_preflight(self):
        if self._board_view is None:
            raise RuntimeError("Import an ODB++ board first.")
        if not self._pcb_xy_homed:
            raise RuntimeError(
                "Home X and Y before a PCB-aligned scan: the landmarks are "
                "machine positions and only homing fixes that frame."
            )
        problems = self._registration_problems()
        if problems:
            raise RuntimeError(
                "Registration is not fit to drive a scan:\n- " + "\n- ".join(problems)
            )
        if not self.ui.chkTravelClearBoard.isChecked():
            raise RuntimeError(
                "Confirm the probe height and the full PCB travel area are clear."
            )

    def _build_validated_plan(self, config):
        """Freeze the grid from current state and prevalidate every machine point.

        Rebuilt on every Start rather than cached, so a plan can never outlive
        the settings it was validated against.  It deliberately does not import
        a board or refit the registration: the Board and Register pages stay
        authoritative, so Start cannot silently run something else.
        """
        step_mm = self.ui.stepMm.value()
        x_start, x_end, y_start, y_end = _board_bounds(self._board_view, step_mm)
        plan = engine_registration.build_plan(
            self._board_view,
            self._registration.transform,
            x_start=x_start,
            x_end=x_end,
            y_start=y_start,
            y_end=y_end,
            step_mm=step_mm,
            clip_to_outline=True,
            registration=self._registration,
        )
        if plan.point_count == 0:
            raise ValueError(
                "no grid cell falls on the board outline; reduce the grid step"
            )
        # Every point the scan will drive to, checked before anything moves
        plan.validate_machine_limits(config.printer)

        config.frame = "board"
        config.area = engine_config.ScanArea(
            width_mm=x_end - x_start,
            height_mm=y_end - y_start,
            step_mm=step_mm,
            origin_x_mm=x_start,
            origin_y_mm=y_start,
        )
        config.validate()
        self._plan = plan
        return plan

    def _provenance(self):
        """How this scan's registration was obtained, for the snapshot.

        The plan already carries the transform; what it cannot say is whether
        anyone checked it against the physical board, which is the one thing a
        reader of an old scan will want to know.  Rectangle scans have no
        registration at all and get no block.
        """
        if self._mode() != MODE_PCB or self._registration is None:
            return None
        if self._profile is None:
            return {"registration_source": "manual_landmarks"}
        dx, dy = self._offset_correction_mm
        return {
            "registration_source": "fixture_profile",
            "fixture_profile": self._profile.name,
            "profile_sha256": self._profile.sha256,
            "profile_loaded_at": self._profile.saved_at,
            "orientation_locked": self.ui.chkOrientationLocked.isChecked(),
            "verified_points": [dict(point) for point in self._verified_points],
            "offset_correction_mm": [round(dx, 4), round(dy, 4)],
            "warnings_acknowledged": [dict(e) for e in self._acknowledged_warnings],
            "warnings_noted": list(self._noted_warnings),
            **self._identity(),
            # Embedded, so the snapshot is self-contained even if the profile
            # on disk is later overwritten or deleted.
            "profile_document": self._profile.document,
        }

    def _write_profile_copy(self, output_dir):
        """Drop the active profile beside the scan, from memory not from disk.

        Copying the file would attach whatever is on disk now, which is not
        necessarily what was loaded: a save between loading and scanning would
        silently document the wrong alignment.
        """
        if self._profile is None:
            return
        try:
            target = Path(output_dir) / "fixture_profile.json"
            target.write_text(
                json.dumps(self._profile.document, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            logging.info(f"EMI map could not write the profile copy: {exc}")

    def _launch(self, config, plan):
        """Borrow the analyser and start the worker.

        The analyser is taken last, after the board, the registration, the
        machine limits, the clearance gate and the printer port have all
        succeeded, so a refusal never costs QtTinySA its sweep.
        """
        try:
            transport = self._open_printer()
        except (OSError, RuntimeError, engine_printer.PrinterError) as exc:
            self._refuse_start(str(exc))
            return
        if self.serial.adopt() is None or not self.serial.is_open():
            self._refuse_start("No analyser is connected.")
            return
        try:
            dev = self.serial.take()
        except RuntimeError as exc:
            self._refuse_start(str(exc))
            return

        self.ui.scanStatus.setText("Starting scan…")
        try:
            self._prepare_live_plot(config, plan)
            # Only measurable cells count, so progress is not diluted by the
            # off-board cells a clipped plan will skip.
            cells = plan.point_count if plan is not None else config.area.nx * config.area.ny
            self._total = cells * (2 if config.background else 1)
            self._done = 0
            self.ui.scanProgress.setMaximum(self._total)
            self.ui.scanProgress.setValue(0)

            self.worker = ScanWorker(config, dev.usb, transport, plan, self._provenance())
            self.thread = QtCore.QThread(self)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.status.connect(self.ui.scanStatus.setText)
            self.worker.point.connect(self._on_point)
            self.worker.row.connect(self._on_row)
            self.worker.prompt.connect(self._on_prompt)
            self.worker.succeeded.connect(self._on_succeeded)
            self.worker.failed.connect(self._on_failed)
            self.worker.printer_reset.connect(self._on_printer_reset)
            # Runs for success, cancel, timeout and any worker exception alike
            self.worker.ended.connect(self._on_scan_ended)

            self._set_scanning(True)
            self.thread.start()
        except Exception:
            self.serial.release()
            if self.thread is not None:
                self.thread.quit()
                self.thread.wait(1000)
            self.thread = None
            self.worker = None
            raise

    def _prepare_live_plot(self, config, plan=None):
        area = config.area
        self._grid = np.full((area.ny, area.nx), np.nan)
        # Levels are mandatory for a float image: without them pyqtgraph raises
        # inside Qt's paint callback, and a raising paint repeats until the
        # process dies. The empty grid renders nothing, so any range will do
        # until the first reading replaces it.
        self.image.setImage(self._grid.T, autoLevels=False, levels=PLACEHOLDER_LEVELS)
        # Board-view millimetres do not start at zero, so the image rect and the
        # ranges follow the area's origin rather than assuming it.
        origin_x, origin_y = area.origin_x_mm, area.origin_y_mm
        self.image.setRect(
            QtCore.QRectF(origin_x, origin_y, area.width_mm, area.height_mm)
        )
        self.ui.emiPlot.setXRange(origin_x, origin_x + max(area.width_mm, area.step_mm))
        self.ui.emiPlot.setYRange(origin_y, origin_y + max(area.height_mm, area.step_mm))
        board = plan is not None
        self.ui.emiPlot.setLabel("bottom", "Board X (mm)" if board else "X (mm from origin)")
        self.ui.emiPlot.setLabel("left", "Board Y (mm)" if board else "Y (mm from origin)")
        if board:
            self._draw_board_under_map(plan.board_view)

    def _draw_board_under_map(self, view):
        """Trace the outline under the live map so a hotspot is locatable at a glance."""
        import pyqtgraph

        if self._board_trace is not None:
            self.ui.emiPlot.removeItem(self._board_trace)
            self._board_trace = None
        outline = _polyline_batch([ring.points for ring in view.outline.rings], close=True)
        if outline is None:
            return
        xs, ys, connect = outline
        self._board_trace = pyqtgraph.PlotCurveItem(
            xs, ys, connect=connect, pen=pyqtgraph.mkPen("k", width=2)
        )
        self.ui.emiPlot.addItem(self._board_trace)

    def _set_scanning(self, scanning):
        # The board and the alignment must not change under a running scan
        for page in (self.ui.pageSetup, self.ui.pageBoard, self.ui.pageRegister):
            page.setEnabled(not scanning)
        self._update_nav()

    def _on_point(self, info):
        self._done += 1
        self.ui.scanProgress.setValue(self._done)
        if self._grid is not None:
            self._grid[info["iy"], info["ix"]] = info["power_dbm"]
        self._pending_updates += 1
        if self._pending_updates >= 8:
            self._refresh_image()

    def _on_row(self, info):
        self._refresh_image()
        self.ui.scanStatus.setText(
            f"{info['pass']} pass: row {info['iy'] + 1} of {info['rows']} complete"
        )

    def _refresh_image(self):
        self._pending_updates = 0
        if self._grid is None:
            return
        finite = self._grid[np.isfinite(self._grid)]
        if not finite.size:
            return
        low, high = float(finite.min()), float(finite.max())
        if high - low < 1e-9:  # one distinct reading so far: give it a width
            low, high = low - 0.5, high + 0.5
        self.image.setImage(self._grid.T, autoLevels=False, levels=(low, high))

    def _on_prompt(self):
        self.ui.btnContinueDut.setEnabled(True)
        self.ui.scanStatus.setText(
            "Background pass complete. Turn the DUT ON without moving it, then continue."
        )
        QtWidgets.QApplication.beep()

    def _continue_dut(self):
        self.ui.btnContinueDut.setEnabled(False)
        if self.worker is not None:
            self.worker.resume()

    def _abort_scan(self):
        if self.worker is not None:
            self.ui.scanStatus.setText("Cancelling after the current sweep…")
            self.worker.cancel()

    def _on_succeeded(self, result):
        self.result = result
        self._write_profile_copy(result.output_dir)
        banner = getattr(self.ui, "resultsBanner", None)
        summary = getattr(self.ui, "resultsSummary", None)
        if banner is not None:
            banner.setText("Scan complete.")
        if summary is not None:
            summary.setText(results_overview(result))
        self._show_board_artifacts(result)
        self.ui.wizardStack.setCurrentIndex(PAGE_RESULTS)
        self._fit_results_image()

    def _show_board_artifacts(self, result):
        overlay_html = result.files.get("board_html")
        self.ui.btnOpenOverlay.setEnabled(bool(overlay_html))
        image_path = result.files.get("board_png") or result.files.get("heatmap_png")
        if not image_path:
            self._results_pixmap = None
            self.ui.boardImage.clear()
            return
        self._results_pixmap = QtGui.QPixmap(str(image_path))
        self._fit_results_image()

    def _fit_results_image(self):
        """Scale the heatmap to the label as it is now, not as it was at write time.

        setPixmap(scaled(label.size())) during _on_succeeded runs before the
        Results page is shown, when the label is still a few dozen pixels. That
        is why the overview used to show a stamp-sized plot in a sea of grey.
        """
        pixmap = self._results_pixmap
        label = getattr(self.ui, "boardImage", None)
        if pixmap is None or label is None or not hasattr(label, "setPixmap"):
            return
        size = label.size() if hasattr(label, "size") else None
        if size is None or not hasattr(size, "width") or size.width() < 32 or size.height() < 32:
            label.setPixmap(pixmap)
            return
        label.setPixmap(
            pixmap.scaled(
                size,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def eventFilter(self, obj, event):
        board = getattr(self.ui, "boardImage", None)
        if obj is board and event.type() == QtCore.QEvent.Type.Resize:
            self._fit_results_image()
        return super().eventFilter(obj, event)

    def _on_failed(self, message):
        self.ui.scanStatus.setText(message)
        QMessageBox.warning(self.ui, DIALOG_TITLE, message)

    def _on_printer_reset(self):
        """Marlin rebooted mid-scan, so the frame the landmarks were recorded in
        is gone. Without this the scan fails while the GUI still claims the
        machine is homed, which is the worst of both states."""
        self._clear_pcb_homing()

    def _on_scan_ended(self):
        """Unconditional teardown: success, cancel, timeout or crash."""
        self.serial.release()
        self.ui.btnContinueDut.setEnabled(False)
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(5000)
            self.thread = None
        self.worker = None
        self._set_scanning(False)

    # ------------------------------------------------------------------ nav
    def _mode(self):
        return self.ui.scanMode.currentText()

    def _sequence(self):
        """The pages this mode uses, in order. Pages outside it are skipped."""
        return MODE_PAGES.get(self._mode(), MODE_PAGES[MODE_RECTANGLE])

    def _position(self):
        sequence = self._sequence()
        index = self.ui.wizardStack.currentIndex()
        return sequence.index(index) if index in sequence else 0

    def _on_mode_changed(self, _mode=None):
        self._invalidate_plan()
        self._update_mode_ui()
        self.ui.wizardStack.setCurrentIndex(PAGE_SETUP)
        self._update_header()
        self._update_nav()

    def _update_mode_ui(self):
        # A PCB scan takes its extent from the board outline, so the rectangle
        # width and height would be ignored and must not look live.
        rectangle = self._mode() != MODE_PCB
        self.ui.widthMm.setEnabled(rectangle)
        self.ui.heightMm.setEnabled(rectangle)
        self.ui.widthMm.setVisible(rectangle)
        self.ui.heightMm.setVisible(rectangle)
        self.ui.lblWidth.setVisible(rectangle)
        self.ui.lblHeight.setVisible(rectangle)
        self.ui.grpArea.setTitle(
            "Scan area (mm from the origin you set on the next page)"
            if rectangle
            else "Scan grid (extent comes from the board outline)"
        )
        self.ui.modeHint.setText(
            "Next opens the Origin page: mark the lower-left corner as X0 Y0, "
            "then scan a rectangle. This path does not use a PCB file."
            if rectangle
            else "Next opens the Board page: import ODB++, then home X/Y and "
            "register two landmarks by jogging the probe onto them."
        )
        self.ui.gridInfo.setText(self._grid_summary())
        travel = getattr(self.ui, "chkTravelClearBoard", None)
        if travel is not None and hasattr(travel, "setVisible"):
            travel.setVisible(not rectangle)
        intro = getattr(self.ui, "scanIntro", None)
        if intro is not None:
            intro.setText(
                "Start scan maps the rectangle from the origin you set."
                if rectangle
                else "Confirm the probe height and PCB travel are clear, then Start scan."
            )

    def _update_header(self):
        sequence = self._sequence()
        position = self._position()
        self.ui.stepHeader.setText(
            f"Step {position + 1} of {len(sequence)} — {STEP_TITLES[sequence[position]]}"
        )

    def _next_allowed(self, page):
        """One gate per page, so another page never means another if/elif rung."""
        gates = {
            PAGE_BOARD: lambda: self._board_view is not None,
            PAGE_REGISTER: lambda: not self._registration_problems(),
            PAGE_ORIGIN: lambda: self.origin_set,
            PAGE_SCAN: lambda: self.result is not None,
        }
        return gates.get(page, lambda: True)()

    def _update_nav(self):
        position = self._position()
        aimed = self._selected_landmark is not None and self._pcb_xy_homed
        aligned = aimed and self._registration is not None
        seated = self.ui.chkBoardSeated.isChecked()
        allowed = {
            self.ui.btnBack: position > 0,
            self.ui.btnSetOrigin: self.ui.chkTravelClear.isChecked(),
            self.ui.btnHomeXy: self.ui.chkHomeClear.isChecked(),
            self.ui.btnRecordLandmark: aimed,
            self.ui.btnMoveHere: aligned,
            self.ui.btnUpdateOffset: aligned and len(self._landmarks) >= 2,
            self.ui.btnConfirmAlignment: (
                self._profile is not None and seated and self._candidate_is_current()
            ),
            self.ui.btnLoadProfile: (
                self._board_view is not None
                and self._pcb_xy_homed
                and seated
                and bool(self._selected_profile_name())
            ),
            self.ui.btnSaveAsProfile: self._save_allowed(),
            self.ui.btnUpdateProfile: (
                self._save_allowed() and bool(self._selected_profile_name())
            ),
            self.ui.btnNext: (
                position + 1 < len(self._sequence())
                and self._next_allowed(self.ui.wizardStack.currentIndex())
            ),
        }
        # Nothing that moves the machine or changes the plan stays live while a
        # scan is running, so idleness gates every one of them.
        idle = self.thread is None and self._printer_job is None
        for button, ready in allowed.items():
            button.setEnabled(idle and ready)
        self.ui.btnNext.setText(self._next_caption())
        self.ui.btnNext.setToolTip(self._next_block_reason())
        self._apply_scan_gate()

    def _next_caption(self):
        page = self.ui.wizardStack.currentIndex()
        caption = NEXT_CAPTIONS.get(page, "Next")
        if isinstance(caption, dict):
            caption = caption.get(self._mode(), "Next")
        return caption

    def _next_block_reason(self):
        """Why Next is disabled, for a tooltip rather than a silent grey button."""
        if self.thread is not None or self._printer_job is not None:
            return "Wait for the current printer or scan job to finish."
        page = self.ui.wizardStack.currentIndex()
        if self._next_allowed(page):
            return ""
        if page == PAGE_REGISTER:
            problems = self._registration_problems()
            if problems:
                return "Cannot scan yet:\n- " + "\n- ".join(problems)
        if page == PAGE_SCAN:
            return "Start the scan first. Results open when it finishes."
        if page == PAGE_BOARD:
            return "Import an ODB++ board first."
        if page == PAGE_ORIGIN:
            return "Set the origin first."
        return "Cannot continue yet."

    def _on_page_changed(self, index):
        self._update_header()
        self._update_nav()
        if index == PAGE_REGISTER:
            self._layout_register_page()
        if index == PAGE_RESULTS:
            self._fit_results_image()

    def _on_back(self):
        position = self._position()
        if position > 0:
            self.ui.wizardStack.setCurrentIndex(self._sequence()[position - 1])

    def _on_next(self):
        if self.ui.wizardStack.currentIndex() == PAGE_SETUP:
            try:
                self.build_config()
            except ValueError as exc:
                QMessageBox.warning(self.ui, DIALOG_TITLE, f"Invalid settings: {exc}")
                return
        sequence = self._sequence()
        position = self._position()
        if position + 1 < len(sequence):
            self.ui.wizardStack.setCurrentIndex(sequence[position + 1])

    def _open_folder(self):
        if self.result is None:
            return
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(self.result.output_dir))
        )

    def _open_overlay(self):
        overlay_html = None if self.result is None else self.result.files.get("board_html")
        if not overlay_html:
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(overlay_html)))

    def _on_close(self):
        self._cleanup()
        self.ui.hide()

    def _cleanup(self):
        if self.worker is not None:
            self.worker.cancel()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(5000)
            self.thread = None
        self.serial.release()
        self._close_printer()
        self.origin_set = False
        self.ui.originStatus.setText("Origin not set.")
        # Closing the dialog releases the port, so the homed frame cannot be
        # assumed to survive until it is reopened.
        self._clear_pcb_homing()

    def _close_printer(self):
        if self.printer_serial is None:
            return
        try:
            self.printer_serial.close()
        except OSError as exc:
            logging.info(f"EMI map could not close the printer port: {exc}")
        self.printer_serial = None
