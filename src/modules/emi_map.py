"""EMI Near-Field Map wizard for QtTinySA.

Drives an Ender 3 over a PCB while this analyser measures, and builds a 2D map
of the emissions.  The measurement engine itself lives outside QtTinySA in the
EMI_Mapper package; this module is only Qt glue plus serial-ownership handling.

Two modes share the wizard.  Rectangle maps a plain area from an origin the
operator sets by hand with G92, and never homes.  PCB aligned homes X and Y,
imports an ODB++ board, registers it to the machine from landmarks the operator
jogs to, and clips the grid to the board outline.  Z is setup-only: the scan
loop never commands it.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
from dataclasses import replace
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

# Opening a Creality CH340 port can pulse DTR and reboot Marlin. Identify from
# raw bytes (Marlin/ok), not only a newline-terminated 'ok' line: Windows CH340
# often delivers FIRMWARE_NAME before a lone ok, and a 0.05 s engine timeout
# can miss the line split. Use the full greet window; do not hold DTR down
# (that can look like a silent port on this board).
PRINTER_BOOT_S = 5.0
PRINTER_GREET_S = 8.0
PRINTER_BAUD_SETTLE_S = 1.5
PRINTER_SERIAL_TIMEOUT_S = 0.25
# After FIRMWARE_NAME, the rest of M115 (Cap: lines and ok) is still in flight.
# Returning before that idle gap lets Printer.send steal the leftover ok and
# the next real command times out with an empty buffer.
PRINTER_SYNC_QUIET_S = 0.25

DIALOG_TITLE = "EMI Near-Field Map"
PRINTER_HALTED_TEXT = "Printer halted — reset the printer, then reconnect."
_NO_BYTES_TEXT = "timeout (no bytes received)"
STEP2_AFTER_RECONNECT = (
    "Printer connected. Saved P1 XY is kept. Session homing, board-zero, and "
    "scan height were cleared. Finish P1 and Z setup on Step 2."
)


class _FixtureMotionRefused(RuntimeError):
    """A fixture XY/Z gate failed; session homing stays valid."""


def _marlin_identity_text(buf: bytes) -> bool:
    """True when the buffer is a Marlin identify reply, even without a lone ok line."""
    text = buf.decode("latin1", errors="replace").lower()
    if "firmware_name:marlin" in text:
        return True
    if "ok" in text and " t:" in text:
        return True
    if text.lstrip().startswith("ok t:"):
        return True
    return False


def _controller_text_is_halt(text: str) -> bool:
    """True only for firmware halt replies. Never applied to this app's own advice."""
    lowered = str(text).lower()
    return "printer halted" in lowered or "kill() called" in lowered


def _halt_error(message):
    """A halt is carried by type, so wizard prose can never be mistaken for one."""
    if engine_printer is not None:
        return engine_printer.PrinterHalted(message)
    return RuntimeError(message)


def _controller_preview(buf: bytes) -> str:
    text = buf.decode("latin1", errors="replace").replace("\r", "\n")
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not compact:
        return _NO_BYTES_TEXT
    if len(compact) > 240:
        return compact[:240] + "…"
    return compact


def _drain_serial(handle, quiet_s: float) -> None:
    """Read until the port is idle so a later Printer.send is not desynced."""
    idle_s = max(float(quiet_s), 0.0)
    deadline = time.monotonic() + idle_s
    while True:
        chunk = handle.read(256)
        if chunk:
            deadline = time.monotonic() + idle_s
            continue
        if time.monotonic() >= deadline:
            return
        time.sleep(0.02)


def greet_marlin_handle(handle, port, *, boot_s: float, greet_s: float) -> None:
    """Identify Marlin on an open serial handle without requiring Printer.send.

    Printer.send waits for a newline-terminated ``ok``. This board answers M115
    with FIRMWARE_NAME first; treating that as identity avoids a false
    timeout when the ok line is delayed, coalesced, or uses CR only.
    After identity, drain the rest of that reply or the next G-code times out
    with an empty buffer (``no reply from the printer: b''``). Halted
    firmware is reported from the bytes received, not guessed as power-off.
    """
    baud = getattr(handle, "baudrate", None)
    baud_text = f" at {baud} baud" if baud not in (None, "") else ""
    deadline = time.monotonic() + boot_s
    leftover = bytearray()
    while time.monotonic() < deadline:
        chunk = handle.read(256)
        if chunk:
            leftover += chunk
            logging.info("printer RX %s boot: %s", port, _controller_preview(chunk))
            if _controller_text_is_halt(leftover.decode("latin1", errors="replace")):
                raise _halt_error(
                    f"{PRINTER_HALTED_TEXT} Controller response on {port}{baud_text}: "
                    f"{_controller_preview(bytes(leftover))}"
                )
            if _marlin_identity_text(bytes(leftover)):
                _drain_serial(handle, PRINTER_SYNC_QUIET_S)
                return
        else:
            time.sleep(0.02)
    probes = []
    for command in (b"M115\n", b"M105\n"):
        logging.info("printer TX %s: %s", port, command.decode("ascii", errors="replace").strip())
        handle.write(command)
        flush = getattr(handle, "flush", None)
        if callable(flush):
            flush()
        buf = bytearray()
        end = time.monotonic() + greet_s
        while time.monotonic() < end:
            chunk = handle.read(256)
            if chunk:
                buf += chunk
                logging.info("printer RX %s: %s", port, _controller_preview(chunk))
                if _controller_text_is_halt(buf.decode("latin1", errors="replace")):
                    raise _halt_error(
                        f"{PRINTER_HALTED_TEXT} Controller response on {port}{baud_text} "
                        f"after {command.decode('ascii', errors='replace').strip()}: "
                        f"{_controller_preview(bytes(buf))}"
                    )
                if _marlin_identity_text(bytes(buf)):
                    _drain_serial(handle, PRINTER_SYNC_QUIET_S)
                    return
            else:
                time.sleep(0.02)
        probes.append(
            f"after {command.decode('ascii', errors='replace').strip()}: "
            f"{_controller_preview(bytes(buf))}"
        )
    description = _serial_port_description(port)
    boot_preview = _controller_preview(bytes(leftover))
    silent = not leftover and all(_NO_BYTES_TEXT in probe for probe in probes)
    advice = (
        # Zero bytes is the USB bridge answering alone: the CH340 enumerates
        # from USB 5 V even when the controller is not running or is held in
        # reset, so baud is not the first thing to suspect.
        "The USB bridge answered but the controller sent nothing at all. "
        "Press the printer's reset button and wait for the boot screen, "
        "check that the printer is switched on at the PSU, and close any "
        "other program holding this port, then click RETRY PRINTER "
        "CONNECTION."
        if silent
        else "Press the printer's reset button and wait for the boot screen, "
        "then click RETRY PRINTER CONNECTION. If bytes arrive but do not "
        "identify, confirm the firmware baud (normally 115200 or 250000) "
        "and the USB cable."
    )
    raise RuntimeError(
        f"{port}{f' ({description})' if description else ''} opened{baud_text}, "
        f"but Marlin did not identify. Boot-window response: {boot_preview}. "
        + " ".join(probes)
        + ". "
        + advice
    )


def _serial_port_description(device):
    for info in list_ports.comports():
        if str(getattr(info, "device", "")).upper() == str(device).upper():
            return str(getattr(info, "description", "") or "").strip()
    return ""

# TinySA Ultra has no firmware overload USB command. Cube preflight will
# refuse unless the operator acknowledges the RF chain on the Scan page.
RF_CHAIN_CONFIRM_HINT = (
    "This TinySA firmware has no overload flag. Confirm the RF chain "
    "(probe, TBWA2, attenuation) is set correctly, then tick the box. "
    "This does not claim overload was ruled out."
)

STEP_TITLES = ["Setup", "Board", "Register", "Rectangle origin", "Scan", "Results"]

PAGE_SETUP = 0
PAGE_BOARD = 1
PAGE_REGISTER = 2
PAGE_ORIGIN = 3
PAGE_SCAN = 4
PAGE_RESULTS = 5
# Runtime-only Easy page. Keeping the existing .ui indices stable avoids
# disturbing Rectangle and Advanced workflows.
PAGE_EASY_HEIGHT = 6
PAGE_FIXTURE_TEACH = 7
PAGE_SCAN_SETUP = 8

MODE_RECTANGLE = "Rectangle"
MODE_PCB = "PCB aligned"

# One page sequence per mode. Navigation, the step header and Next-gating all
# read this table, so adding a mode or a page never adds another per-page
# if/elif ladder of the kind this file already shares with modules/fcc_test.py.
MODE_PAGES = {
    MODE_RECTANGLE: (PAGE_SETUP, PAGE_ORIGIN, PAGE_SCAN, PAGE_RESULTS),
    MODE_PCB: (PAGE_SETUP, PAGE_BOARD, PAGE_REGISTER, PAGE_SCAN, PAGE_RESULTS),
}
EASY_PCB_PAGES = (
    PAGE_SETUP,
    PAGE_FIXTURE_TEACH,
    PAGE_BOARD,
    PAGE_SCAN_SETUP,
    PAGE_SCAN,
    PAGE_RESULTS,
)

OPERATOR_MODE_EASY = "Easy"
OPERATOR_MODE_ADVANCED = "Advanced"
EASY_JOG_STEPS_MM = (0.05, 0.10, 0.25, 0.50, 1.00)
LOGICAL_Z_MAX_MM = 250.0
BOARD_ZERO_COMPLETE_TEXT = "Board zero set at P1. Step 2 complete."
EASY_HEIGHT_MISSING_THICKNESS = "Step 2 incomplete: PCB thickness is missing."
EASY_HEIGHT_MISSING_VERTICAL = (
    "Step 2 incomplete: vertical E-probe calibration is missing."
)
EASY_VERTICAL_UNCALIBRATED = "E-probe vertical offset not calibrated."
EASY_HEIGHT_NOT_REACHED = "Scan height was not reached."
EASY_HEIGHT_CLEARANCE = (
    "Automatic scan-height Z needs clearance at P1 and the A/B teaching route."
)
EASY_HEIGHT_RESET = "Height reference invalidated by printer reset. Repeat Step 2."
EASY_HEIGHT_MISSING_P1 = "Step 2 incomplete: P1 contact datum is missing."
EASY_HEIGHT_NEED_BOARD_ZERO = (
    "Step 2 incomplete: Home & Set Board Zero at the saved P1."
)
EASY_HEIGHT_NEED_MANUAL_PLANE = (
    "Set PCB surface manually, then set probe height."
)
EASY_MANUAL_HEIGHT_BUTTONS = (
    "btnEasySetHeight",
    "btnEasySetPcbSurface",
    "btnEasyResetPcbSurface",
)
# Step 2 teaches P1 only: it locates the fixture origin and gives board zero.
# The STL supplies the rest of the holder geometry, with the fixture taken as
# square to X/Y. Teaching all four corners is what would measure that angle.
TEACHING_POINT_NAMES = ("P1",)
# Step 4 locates the inner pocket by teaching its two diagonal corners, once
# per board face. v2 pairs are pocket CAD corners; v1 PCB-extrema pairs are
# ignored and must be recaptured.
TAUGHT_CORNER_TEMPLATE_ID = "taught_pocket_corners_v2"
TAUGHT_CORNER_LABELS = ("Corner A", "Corner B")
SCAN_SETUP_SIDES = ("top", "bottom")
ORIENTATION_DRIFT_HINT = (
    "Orientation may have changed.\n"
    "Use Advanced → Recalibrate Fixture\n"
    "to perform a 2-point registration."
)

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
    from EMI_Mapper import fixture_teaching as engine_fixture_teaching
    from EMI_Mapper import glassboard_fixture as engine_glassboard_fixture
    from EMI_Mapper import geometry as engine_geometry
    from EMI_Mapper import height as engine_height
    from EMI_Mapper import machine_fixtures as engine_machine_fixtures
    from EMI_Mapper import odbpp as engine_odbpp
    from EMI_Mapper import overlay as engine_overlay
    from EMI_Mapper import printer as engine_printer
    from EMI_Mapper import registration as engine_registration
    from EMI_Mapper import scanner as engine_scanner
    from EMI_Mapper import selection as engine_selection
    from EMI_Mapper import scan_preview as engine_scan_preview

    return (
        engine_board,
        engine_config,
        engine_height,
        engine_odbpp,
        engine_overlay,
        engine_printer,
        engine_profiles,
        engine_fixture_teaching,
        engine_glassboard_fixture,
        engine_machine_fixtures,
        engine_registration,
        engine_scanner,
        engine_geometry,
        engine_selection,
        engine_scan_preview,
    )


try:
    (
        engine_board,
        engine_config,
        engine_height,
        engine_odbpp,
        engine_overlay,
        engine_printer,
        engine_profiles,
        engine_fixture_teaching,
        engine_glassboard_fixture,
        engine_machine_fixtures,
        engine_registration,
        engine_scanner,
        engine_geometry,
        engine_selection,
        engine_scan_preview,
    ) = _load_engine()
    ENGINE_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on deployment layout
    engine_board = engine_config = engine_height = engine_odbpp = engine_overlay = None
    engine_printer = engine_profiles = engine_registration = engine_scanner = None
    engine_geometry = engine_selection = engine_fixture_teaching = None
    engine_glassboard_fixture = engine_machine_fixtures = None
    engine_scan_preview = None
    ENGINE_ERROR = str(exc)
    logging.info(f"EMI map engine unavailable: {exc}")

from modules.board_selection_widget import (  # noqa: E402
    BoardSelectorDialog,
    ScanSelectionEditor,
    board_bounds as _board_bounds,
    polyline_batch as _polyline_batch,
)


def _spread_mm(points):
    """The largest distance between any two of the points."""
    if len(points) < 2:
        return 0.0
    coords = np.asarray(points, dtype=float)
    deltas = coords[:, None, :] - coords[None, :, :]
    return float(np.hypot(deltas[..., 0], deltas[..., 1]).max())


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
    prompt = QtCore.Signal(str)
    spectrum = QtCore.Signal(dict)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    # A Marlin reboot is reported separately from the failure text: the GUI has
    # to know the machine frame is gone, and a formatted message loses the type.
    printer_reset = QtCore.Signal()
    ended = QtCore.Signal()
    cell_begin = QtCore.Signal(dict)

    def __init__(
        self,
        config,
        tinysa_transport,
        printer_transport,
        plan=None,
        provenance=None,
        height=None,
    ):
        super().__init__()
        self.config = config
        self.tinysa_transport = tinysa_transport
        self.printer_transport = printer_transport
        self.plan = plan
        self.provenance = provenance
        self.height = height
        self._abort = threading.Event()
        self._continue = threading.Event()

    def cancel(self):
        self._abort.set()
        self._continue.set()  # unblock a worker waiting at the DUT-on prompt

    def resume(self):
        self._continue.set()

    def _prompt_dut(self, message: str):
        self._continue.clear()
        self.prompt.emit(message)
        while not self._continue.wait(0.1):
            if self._abort.is_set():
                raise engine_scanner.ScanAborted("cancelled at the DUT prompt")

    def _prompt_dut_off(self):
        self._prompt_dut(
            "Turn the DUT OFF for the ambient pass, then continue. "
            "Leave it off until this pass finishes."
        )

    def _prompt_dut_on(self):
        self._prompt_dut(
            "Ambient pass complete. Turn the DUT ON without moving it, then continue."
        )

    @QtCore.Slot()
    def run(self):
        try:
            callbacks = engine_scanner.Callbacks(
                on_point=self.point.emit,
                on_row=self.row.emit,
                on_status=self.status.emit,
                on_spectrum=self.spectrum.emit,
                on_cell_begin=self.cell_begin.emit,
                should_abort=self._abort.is_set,
                prompt_dut_off=self._prompt_dut_off,
                prompt_dut_on=self._prompt_dut_on,
            )
            result = engine_scanner.run_scan(
                self.config,
                callbacks,
                tinysa_transport=self.tinysa_transport,
                printer_transport=self.printer_transport,
                plan=self.plan,
                provenance=self.provenance,
                height=self.height,
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
    characterization_line = _characterization_overview_line(result)
    if characterization_line:
        lines.append(characterization_line)
    if alignment:
        lines.append(f"Alignment: {alignment}")
    height_line = _height_overview_line(snapshot.get("height") or {})
    if height_line:
        lines.append(height_line)
    if when:
        lines.append(f"Time: {when}")
    lines.append(f"Folder: {folder}")
    if names:
        lines.append("Wrote: " + ", ".join(names))
    return "\n".join(lines)


def _height_overview_line(height: dict) -> str:
    """Agreed scan plane from the frozen snapshot, if one was recorded."""
    requested = height.get("requested_height_above_pcb_mm")
    if requested is None:
        return ""
    line = f"Probe height: {requested:g} mm above PCB"
    source = height.get("source") or height.get("height_source")
    if source == "operator_override":
        line += " (operator override)"
    elif source == "cad+clearance":
        line += " (CAD + clearance)"
    reported = height.get("reported_height_above_pcb_mm")
    if reported is not None:
        line += f", reported {reported:g} mm"
    return line


def _characterization_overview_line(result) -> str:
    characterized = getattr(result, "characterization", None) or {}
    quantity = characterized.get("output_quantity")
    if not quantity:
        return ""
    peak_key = f"characterized_peak_{quantity}"
    values = np.asarray(characterized.get(peak_key, []), dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        peak = "no valid characterized bins"
    else:
        peak = f"peak {float(finite.max()):g} {characterized.get('output_units', '')}"
    mode = characterized.get("receiver_processing_mode", "unknown")
    return (
        f"Manufacturer characterization: {peak}, receiver mode {mode}. "
        "Diagnostic only; not EMC compliance."
    )


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
        self._printer_halted = False
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
        self._selector = ScanSelectionEditor()
        self._machine_xy = None
        self._pcb_surface_z = None
        self._height_override_mm = None
        self._last_commanded_scan_z = None
        self._easy_verified_scan_z = None
        self._easy_height_identity = None
        self._easy_height_lost_to_reset = False
        self._fixture_z_calibrations_cleared = False
        self._fixture_config_error = ""
        self._bltouch_board_zero_z = None
        self._bltouch_board_zero_frame = None
        self._z_logical_frame = 0

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
        self._easy_adjust_open = False
        self._easy_adjust_saved_xy = None
        self._easy_z_unlocked = False
        self._easy_touch_logical_z = None
        self._fixture_xyz_homed = False
        self._fixture_teaching_points = {}
        self._fixture_verified_session = False
        self._fixture_config_cache = None
        self._fixture_config_error = ""
        self._syncing_glassboard_side = False
        self._stl_board_placement = None
        # Step 4 taught inner-pocket corners, per board face: {side: {slot: (x, y)}}.
        self._taught_pcb_corner_points = {side: {} for side in SCAN_SETUP_SIDES}
        self._active_taught_side = None
        self._pocket_pair_state = {side: "missing" for side in SCAN_SETUP_SIDES}
        self._pocket_block_reason = ""
        self._corner_teach_labels = {}
        self._taught_pocket_placement = None
        self._active_machine_fixture_name = ""
        self._approved_scan_fingerprint = None
        self._scan_preview_widget = None
        self._pcb_surface_from_fixture = False
        self._machine_xy_at = 0.0
        self._pending_preview_cell = None
        self._refreshing_ports = False
        self._last_page_by_mode = {
            MODE_RECTANGLE: PAGE_SETUP,
            MODE_PCB: PAGE_SETUP,
        }

        # Plot items, created by the pyqtgraph-touching setup methods only
        self._landmark_marks = None
        self._selected_mark = None
        self._board_trace = None
        self._results_pixmap = None
        self._scan_gate = None
        self._cube_data = None
        self._cube_picks = []
        self._cube_animate = None
        self._saved_single_span = None
        self._live_spectrum = None
        self._live_spectrum_curve = None

        # Live-map presentation state.  Keep the raw OFF and ON maps separate so
        # the UI can switch to the requested DUT ON - DUT OFF view as soon as
        # the matching ON cell is measured.  The scanner remains the source of
        # truth; this is presentation-only and never changes acquisition data.
        self._live_background_grid = None
        self._live_dut_grid = None
        self._live_delta_grid = None
        self._live_background_spectra = {}
        self._live_background_kind = "xy_grid"
        self._live_stationary_background_dbm = np.nan
        self._live_label_items = {}
        self._live_colorbar = None
        self._live_units = "dBm"
        self._live_map_caption = "Measured level"
        self._live_last_pass = ""

        # Results are rendered directly at the QLabel's current size rather than
        # scaling a 31x8 RGB bitmap.  This keeps cell edges/text crisp and also
        # gives eventFilter an exact rectangle for hover/click picking.
        self._results_values = None
        self._results_units = "dBm"
        self._results_title = "EMI near-field map"
        self._results_map_rect = None

        self._wire_ui()
        self._install_cube_controls()
        self._install_easy_height_page()
        self._install_fixture_teaching_page()
        self._install_scan_setup_page()
        self._install_easy_mode_controls()
        self._install_workflow_progress()
        self._apply_simple_survey_defaults()
        self._setup_plot()
        self._setup_board_plots()
        self._update_travel_warning()
        self._refresh_height_ui()
        self._update_characterization_ui()

    @property
    def _scan_selection(self):
        return self._selector.selection

    @_scan_selection.setter
    def _scan_selection(self, value):
        self._selector.selection = value

    # ------------------------------------------------------------------ UI
    def _wire_ui(self):
        u = self.ui
        u.btnBack.clicked.connect(self._on_back)
        u.btnNext.clicked.connect(self._on_next)
        u.btnCancel.clicked.connect(self._on_close)
        self._install_next_reason_label()
        u.btnBrowse.clicked.connect(self._browse_folder)
        u.btnSetOrigin.clicked.connect(self._set_origin_clicked)
        u.btnStartScan.clicked.connect(self._start_scan)
        u.btnAbortScan.clicked.connect(self._abort_scan)
        u.btnContinueDut.clicked.connect(self._continue_dut)
        u.btnOpenFolder.clicked.connect(self._open_folder)
        u.btnOpenOverlay.clicked.connect(self._open_analyzer)
        if hasattr(u, "boardImage") and hasattr(u.boardImage, "installEventFilter"):
            u.boardImage.installEventFilter(self)
            if hasattr(u.boardImage, "setMouseTracking"):
                u.boardImage.setMouseTracking(True)
            if hasattr(u.boardImage, "setCursor"):
                u.boardImage.setCursor(QtCore.Qt.CrossCursor)

        u.chkTravelClear.stateChanged.connect(self._update_nav)
        for spin in (u.widthMm, u.heightMm, u.stepMm):
            spin.valueChanged.connect(self._on_area_changed)
        u.wizardStack.currentChanged.connect(self._on_page_changed)
        u.scanMode.currentTextChanged.connect(self._on_mode_changed)
        u.printerPort.currentTextChanged.connect(self._on_printer_port_changed)
        u.chkCharacterization.stateChanged.connect(self._update_characterization_ui)
        u.probeModel.currentTextChanged.connect(self._update_characterization_ui)
        u.amplifierModel.currentTextChanged.connect(self._update_characterization_ui)
        u.chkBackground.stateChanged.connect(self._update_characterization_ui)
        u.backgroundMode.currentTextChanged.connect(self._update_characterization_ui)

        u.btnImportBoard.clicked.connect(self._import_board)
        u.boardSide.currentTextChanged.connect(self._apply_board_view)
        u.btnRotateCcw.clicked.connect(partial(self._rotate_board, 90))
        u.btnRotateCw.clicked.connect(partial(self._rotate_board, -90))
        u.btnScanEntireBoard.clicked.connect(self._scan_entire_board)
        u.radioSelectPoint.toggled.connect(self._on_point_tool_toggled)
        u.radioSelectRect.toggled.connect(self._on_rect_tool_toggled)
        u.btnUndoSelection.clicked.connect(self._undo_scan_selection)
        u.btnClearSelection.clicked.connect(self._clear_scan_selection)
        u.btnOpenBoardSelector.clicked.connect(self._open_board_selector)
        for radio in (u.radioSelectPoint, u.radioSelectRect):
            setter = getattr(radio, "setAutoExclusive", None)
            if callable(setter):
                setter(False)
        u.chkHomeClear.stateChanged.connect(self._update_nav)
        u.chkTravelClearBoard.stateChanged.connect(self._update_nav)
        rf_confirmed = getattr(u, "chkRfConfirmed", None)
        if rf_confirmed is not None:
            rf_confirmed.stateChanged.connect(self._update_nav)
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
        u.chkAllowSetupZ.stateChanged.connect(self._refresh_height_ui)
        u.chkHeightOverride.stateChanged.connect(self._refresh_height_ui)
        u.btnApplyHeightOverride.clicked.connect(self._apply_height_override)
        u.btnSetPcbSurface.clicked.connect(self._set_pcb_surface)
        u.btnJogZUp.clicked.connect(partial(self._jog_z, 1))
        u.btnJogZDown.clicked.connect(partial(self._jog_z, -1))
        u.btnMoveToScanHeight.clicked.connect(self._move_to_scan_height)
        u.btnBoardReseated.clicked.connect(self._confirm_board_reseated)

        # The dialog can be dismissed with the window chrome as well as Close
        u.rejected.connect(self._cleanup)
        self._layout_register_page()

    def _install_cube_controls(self):
        """Add cube preset and Results sliders without a new wizard page."""
        u = self.ui

        def _attach_check(name, label, group_name, on_toggled=None):
            if getattr(u, name, None) is not None:
                return
            group = getattr(u, group_name, None)
            layout = group.layout() if group is not None and callable(getattr(group, "layout", None)) else None
            if isinstance(layout, QtWidgets.QFormLayout):
                box = QtWidgets.QCheckBox(label)
                box.setObjectName(name)
                layout.addRow(box)
                if on_toggled is not None:
                    box.toggled.connect(on_toggled)
                setattr(u, name, box)
                return
            box = type("CubeCheck", (), {})()
            box._checked = False
            box.isChecked = lambda: box._checked
            box.setChecked = lambda checked, b=box: setattr(b, "_checked", bool(checked))
            if on_toggled is not None:
                original = box.setChecked

                def _set(checked, b=box, cb=on_toggled):
                    original(checked)
                    cb(b._checked)

                box.setChecked = _set
            setattr(u, name, box)

        _attach_check(
            "chkSpectrumCube",
            "Spectrum cube (E5, 1–50 MHz)",
            "grpMeasurement",
            self._on_cube_preset_toggled,
        )
        _attach_check(
            "chkAdvanced",
            "Advanced measurement settings",
            "grpMeasurement",
            self._on_advanced_toggled,
        )
        _attach_check(
            "chkPhysicalTbwa2",
            "TBWA2 is physically connected",
            "grpCharacterization",
        )
        _attach_check(
            "chkRfConfirmed",
            "RF chain reviewed (no firmware overload flag)",
            "grpMeasurement",
        )
        amp_label = getattr(u, "lblAmplifierSurvey", None)
        if amp_label is None:
            group = getattr(u, "grpMeasurement", None)
            layout = group.layout() if group is not None else None
            if isinstance(layout, QtWidgets.QFormLayout):
                amp_label = QtWidgets.QLabel("Amplifier: TBWA2-40 (40 dB)")
                amp_label.setObjectName("lblAmplifierSurvey")
                layout.addRow(amp_label)
                u.lblAmplifierSurvey = amp_label
            else:
                class _AmpLabel:
                    def __init__(self):
                        self._text = "Amplifier: TBWA2-40 (40 dB)"

                    def text(self):
                        return self._text

                    def setText(self, text):
                        self._text = text

                u.lblAmplifierSurvey = _AmpLabel()
        amp = getattr(u, "amplifierModel", None)
        changed = getattr(amp, "currentTextChanged", None) if amp is not None else None
        if changed is not None and callable(getattr(changed, "connect", None)):
            changed.connect(self._refresh_amplifier_label)
        self._refresh_amplifier_label()
        hold = getattr(u, "backgroundHoldSweeps", None)
        if hold is None:
            group = getattr(u, "grpMeasurement", None)
            layout = group.layout() if group is not None else None
            if isinstance(layout, QtWidgets.QFormLayout):
                spin = QtWidgets.QSpinBox()
                spin.setObjectName("backgroundHoldSweeps")
                spin.setRange(1, 200)
                spin.setValue(8)
                layout.addRow("Stationary background sweeps", spin)
                u.backgroundHoldSweeps = spin
            else:
                class _HoldSweeps:
                    def value(self):
                        return 8

                    def setValue(self, *_a, **_k):
                        return None

                u.backgroundHoldSweeps = _HoldSweeps()
        results = getattr(u, "resultsLayout", None)
        if results is None or getattr(u, "cubeFreqSlider", None) is not None:
            self._set_advanced_visible(False)
            return
        if not isinstance(results, QtWidgets.QVBoxLayout):
            self._set_advanced_visible(False)
            return
        controls = QtWidgets.QHBoxLayout()
        u.cubeMapKind = QtWidgets.QComboBox()
        # Human-facing names stay short; stable machine keys live in itemData so
        # processing code and saved tests do not have to parse UI wording.
        u.cubeMapKind.addItem("DUT ON − DUT OFF", "dut_on_minus_dut_off")
        u.cubeMapKind.addItem("Peak in band", "band_max")
        u.cubeMapKind.addItem("Single frequency", "single_frequency")
        u.cubeMapKind.addItem("Integrated band power", "sum_of_measured_bin_powers")
        u.cubeMapKind.addItem("Above stationary reference", "dB_above_stationary_reference")
        u.cubeQuantity = QtWidgets.QComboBox()
        u.cubeQuantity.addItem("Raw", "raw_dbm")
        u.cubeQuantity.addItem("Characterized", "characterized")
        u.cubeFreqSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        u.cubeBandwidthMhz = QtWidgets.QDoubleSpinBox()
        u.cubeBandwidthMhz.setRange(0.0, 100.0)
        u.cubeBandwidthMhz.setValue(0.0)
        u.cubeFreqLabel = QtWidgets.QLabel("Frequency")
        controls.addWidget(QtWidgets.QLabel("Map"))
        controls.addWidget(u.cubeMapKind)
        controls.addWidget(u.cubeQuantity)
        controls.addWidget(u.cubeFreqLabel)
        controls.addWidget(u.cubeFreqSlider, 1)
        controls.addWidget(QtWidgets.QLabel("BW MHz"))
        controls.addWidget(u.cubeBandwidthMhz)
        u.cubeAnimate = QtWidgets.QCheckBox("Animate")
        u.cubeSpectrumLabel = QtWidgets.QLabel(
            "Hover for a value; click up to two completed cells to compare spectra"
        )
        u.cubeSpectrumLabel.setWordWrap(True)
        controls.addWidget(u.cubeAnimate)
        results.insertLayout(2, controls)
        results.insertWidget(3, u.cubeSpectrumLabel)
        u.cubeAnimate.toggled.connect(self._on_cube_animate_toggled)
        for widget in (u.cubeMapKind, u.cubeQuantity, u.cubeFreqSlider, u.cubeBandwidthMhz):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._refresh_cube_view)
            if hasattr(widget, "currentTextChanged"):
                widget.currentTextChanged.connect(self._refresh_cube_view)
        self._set_advanced_visible(False)

    _EASY_HEIGHT_SETUP_WIDGETS = (
        "heightCadLabel",
        "chkHeightOverride",
        "heightOverrideMm",
        "heightGapMm",
        "chkHeightReference",
        "chkAllowSetupZ",
        "btnJogZUp",
        "btnJogZDown",
        "chkHeightClearance",
        "btnMoveToScanHeight",
    )
    _EASY_HIDDEN_WIDGETS = (
        "btnRecordLandmark",
        "btnRemoveLandmark",
        "btnClearLandmarks",
        "chkSnapLandmark",
        "landmarkTable",
        "btnConfirmAlignment",
        "btnUpdateOffset",
        "btnMoveHere",
        "chkOrientationLocked",
        "fixtureId",
        "machineId",
        "probeSetupId",
        *_EASY_HEIGHT_SETUP_WIDGETS,
        "btnSaveAsProfile",
        "btnUpdateProfile",
        "chkAdvanced",
    )
    _EASY_SHOWN_WIDGETS = (
        "operatorMode",
        "easyStatusLabel",
        "btnEasyMoveToRef",
        "btnEasyFineAdjust",
        "btnEasyResetXy",
        "easyHeightLabel",
    )

    def _install_easy_height_page(self):
        """Add a dedicated Easy Z page without changing existing .ui indices."""
        self._easy_height_layout = None
        stack = getattr(self.ui, "wizardStack", None)
        if not isinstance(stack, QtWidgets.QStackedWidget):
            return
        page = QtWidgets.QWidget()
        page.setObjectName("pageEasyHeight")
        layout = QtWidgets.QVBoxLayout(page)
        intro = QtWidgets.QLabel(
            "Scan height is set in Step 2. This page only reports that status. "
            "Start and scan never move Z."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        group = QtWidgets.QGroupBox("Probe Height (Z)")
        QtWidgets.QVBoxLayout(group)
        layout.addWidget(group)
        layout.addStretch(1)
        index = stack.addWidget(page)
        if index != PAGE_EASY_HEIGHT:
            raise RuntimeError(
                f"Easy height page index changed: expected {PAGE_EASY_HEIGHT}, got {index}"
            )
        self.ui.pageEasyHeight = page
        self.ui.grpEasyHeight = group

    def _install_fixture_teaching_page(self):
        """Create the Glassboard four-point teaching page at runtime."""
        stack = getattr(self.ui, "wizardStack", None)
        if not isinstance(stack, QtWidgets.QStackedWidget):
            return
        page = QtWidgets.QWidget()
        page.setObjectName("pageFixtureTeach")
        page.setStyleSheet(
            "QGroupBox{font-weight:600;border:1px solid #667085;border-radius:8px;"
            "margin-top:10px;padding:12px} QGroupBox::title{subcontrol-origin:margin;"
            "left:12px;padding:0 5px} QPushButton{min-height:32px;padding:4px 12px}"
            "QTableWidget{border:1px solid #98A2B3;border-radius:6px;gridline-color:#D0D5DD}"
        )
        page_layout = QtWidgets.QVBoxLayout(page)
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("fixtureTeachScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        scroll.setWidget(content)
        page_layout.addWidget(scroll)
        title = QtWidgets.QLabel("Home & Set Board Zero")
        title.setStyleSheet("font-size:18px;font-weight:700")
        layout.addWidget(title)
        intro = QtWidgets.QLabel(
            "Teach P1 so the machine knows the fixture origin. Home XY for "
            "Teaching, jog the BLTouch onto the mark, then CAPTURE P1. Capture "
            "steps the pin down under M119 until it touches — that contact is "
            "the fixture Z reference. Until P1 exists, use Home XY for Teaching "
            "so you can jog. CLEAR P1 forgets it so you can recapture. Home & "
            "Set Board Zero re-homes and recaptures a saved P1; that session "
            "board-zero is enough for Next. Automatic scan-height Z at P1 runs "
            "only when vertical E-probe calibration is present and clearance is "
            "verified. Fast Verify is optional; it is not required for Next."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        selector_group = QtWidgets.QGroupBox("Choose the physical fixture")
        selector_layout = QtWidgets.QHBoxLayout(selector_group)
        machine_fixture = QtWidgets.QComboBox()
        machine_fixture.setObjectName("machineFixtureProfile")
        new_fixture = QtWidgets.QPushButton("NEW FIXTURE")
        new_fixture.setObjectName("btnNewMachineFixture")
        refresh_fixture = QtWidgets.QPushButton("REFRESH")
        refresh_fixture.setObjectName("btnRefreshMachineFixtures")
        selector_layout.addWidget(machine_fixture, 1)
        selector_layout.addWidget(new_fixture)
        selector_layout.addWidget(refresh_fixture)
        layout.addWidget(selector_group)
        summary = QtWidgets.QLabel("Fixture: not selected  •  Calibration: not taught")
        summary.setObjectName("fixtureBoardSummary")
        summary.setStyleSheet("padding:8px;background:#344054;color:white;border-radius:6px")
        layout.addWidget(summary)
        calibration = QtWidgets.QGroupBox("Teach P1 — fixture origin")
        calibration_layout = QtWidgets.QVBoxLayout(calibration)
        layout.addWidget(calibration)
        clear = QtWidgets.QCheckBox(
            "Bed, DUT, fixture, BLTouch, clamps and cables are clear for the Z "
            "lift, XY motion, the scan-height move at P1, and jogging from P1 "
            "toward A/B at that height"
        )
        clear.setObjectName("chkFixtureProbeClear")
        calibration_layout.addWidget(clear)
        connection_row = QtWidgets.QHBoxLayout()
        retry_printer = QtWidgets.QPushButton("RETRY PRINTER CONNECTION")
        retry_printer.setObjectName("btnFixtureRetryPrinter")
        retry_printer.setToolTip(
            "Reconnect without moving any axis. Tries the selected baud, then "
            "the common Marlin rates 115200 and 250000."
        )
        connection_status = QtWidgets.QLabel("Printer connection not checked.")
        connection_status.setObjectName("fixturePrinterConnectionStatus")
        connection_status.setWordWrap(True)
        connection_status.setStyleSheet(
            "padding:7px;background:#F2F4F7;color:#344054;border-radius:6px"
        )
        connection_row.addWidget(retry_printer)
        connection_row.addWidget(connection_status, 1)
        calibration_layout.addLayout(connection_row)
        home = QtWidgets.QPushButton("HOME XY FOR TEACHING")
        home.setObjectName("btnFixtureHomeXyz")
        home.setToolTip(
            "When P1 is taught: home X/Y, step the BLTouch pin to P1 contact "
            "under M119, retract, then move to the calculated scan height at "
            "P1 if thickness, vertical calibration, and this clearance tick "
            "are present. When P1 is not taught: lift Z and home X/Y so you "
            "can jog to capture P1. It never homes Z downward, never sends "
            "G30, and never uses CAD boxes or the E-probe XY offset to "
            "authorize a downward move."
        )
        calibration_layout.addWidget(home)
        reference_map = QtWidgets.QLabel(
            "<b>Central-holder origin (viewed from above)</b><br>"
            "<span style='font-family:monospace'>"
            "P1 upper-left of the 106.162 × 34.000 mm holder"
            "</span><br>"
            "Jog the BLTouch to P1, then capture it. The STL supplies the rest of "
            "the holder with the fixture taken as square to X/Y. CLEAR P1 forgets "
            "the taught point so you can recapture it."
        )
        reference_map.setObjectName("fixtureReferenceMap")
        reference_map.setWordWrap(True)
        reference_map.setStyleSheet(
            "padding:10px;border:1px solid #84ADFF;border-radius:7px;"
            "background:#EFF8FF;color:#1849A9"
        )
        calibration_layout.addWidget(reference_map)
        jog_group = QtWidgets.QGroupBox("Jog BLTouch to the selected physical mark")
        jog_layout = QtWidgets.QGridLayout(jog_group)
        jog_step = QtWidgets.QDoubleSpinBox()
        jog_step.setObjectName("fixtureJogStep")
        jog_step.setRange(0.05, 10.0)
        jog_step.setDecimals(2)
        jog_step.setValue(1.0)
        jog_layout.addWidget(QtWidgets.QLabel("Step (mm)"), 0, 0)
        jog_layout.addWidget(jog_step, 0, 1, 1, 2)
        jog_buttons = {}
        for name, label, row, column, dx, dy in (
            ("btnFixtureJogYPlus", "Y+", 1, 1, 0, 1),
            ("btnFixtureJogXMinus", "X−", 2, 0, -1, 0),
            ("btnFixtureJogXPlus", "X+", 2, 2, 1, 0),
            ("btnFixtureJogYMinus", "Y−", 3, 1, 0, -1),
        ):
            button = QtWidgets.QPushButton(label)
            button.setObjectName(name)
            button.clicked.connect(partial(self._fixture_jog, dx, dy))
            jog_layout.addWidget(button, row, column)
            jog_buttons[name] = button
        calibration_layout.addWidget(jog_group)
        table = QtWidgets.QTableWidget(len(TEACHING_POINT_NAMES), 5)
        table.setObjectName("fixtureTeachTable")
        table.setHorizontalHeaderLabels(
            ["Point", "Machine X", "Machine Y", "Taught Z", "Latest check / Δ"]
        )
        for row, name in enumerate(TEACHING_POINT_NAMES):
            item = QtWidgets.QTableWidgetItem(name)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            table.setItem(row, 0, item)
            for column in (1, 2, 3, 4):
                value = QtWidgets.QTableWidgetItem("Pending")
                value.setFlags(value.flags() & ~QtCore.Qt.ItemIsEditable)
                table.setItem(row, column, value)
        table.horizontalHeader().setStretchLastSection(True)
        calibration_layout.addWidget(table)
        table.selectRow(0)
        probe = QtWidgets.QPushButton("CAPTURE P1 — PROBE AND SAVE XYZ")
        probe.setObjectName("btnFixtureProbePoint")
        calibration_layout.addWidget(probe)
        clear_point = QtWidgets.QPushButton("CLEAR P1 — FORGET THE TAUGHT POINT")
        clear_point.setObjectName("btnFixtureClearPoint")
        clear_point.setToolTip(
            "Discard taught P1 so it can be re-jogged and captured again. "
            "Board zero and the calculated height depend on P1, so both are "
            "invalidated too. No motion."
        )
        calibration_layout.addWidget(clear_point)
        tolerance_row = QtWidgets.QHBoxLayout()
        z_limit = QtWidgets.QDoubleSpinBox()
        z_limit.setRange(0.05, 5.0)
        z_limit.setDecimals(3)
        z_limit.setValue(0.20)
        residual_limit = QtWidgets.QDoubleSpinBox()
        residual_limit.setRange(0.01, 5.0)
        residual_limit.setDecimals(3)
        residual_limit.setValue(0.10)
        z_limit.setToolTip(
            "Maximum point-to-point shape or tilt change after removing the common "
            "Marlin Z-reference shift. Recommended: 0.200 mm."
        )
        residual_limit.setToolTip(
            "Maximum deviation of a measured point from the newly fitted fixture plane."
        )
        tolerance_row.addWidget(QtWidgets.QLabel("Max point / tilt change (mm)"))
        tolerance_row.addWidget(z_limit)
        tolerance_row.addWidget(QtWidgets.QLabel("Max plane residual (mm)"))
        tolerance_row.addWidget(residual_limit)
        calibration_layout.addLayout(tolerance_row)
        save = QtWidgets.QPushButton("SAVE TAUGHT P1")
        save.setToolTip(
            "CAPTURE P1 already writes the fixture file. Use this only if that "
            "write failed and you want to retry."
        )
        verify = QtWidgets.QPushButton("FAST VERIFY — RE-PROBE P1")
        verify.setToolTip(
            "Re-probe saved P1 with the same M119 pin-contact descent used by "
            "CAPTURE P1. Does not send G30. Does not repeat Home XY when that "
            "already ran this session."
        )
        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(save)
        action_row.addWidget(verify)
        calibration_layout.addLayout(action_row)
        firmware_note = QtWidgets.QLabel(
            "CAPTURE P1 deploys the BLTouch pin and steps down 1 mm at a time "
            "until M119 z_probe triggers. That contact is the new board zero. "
            "G30 is never sent — this firmware's G30 stroke cannot reach the "
            "board from the post-home height and does not stop on the pin. "
            "FAST VERIFY re-probes saved P1 the same way. Carriage XY offset, "
            "PCB thickness, yaw, and E-probe Z live in EMI_Mapper/config.yaml, "
            "not on this page."
        )
        firmware_note.setWordWrap(True)
        firmware_note.setStyleSheet(
            "padding:8px;border-radius:6px;background:#EFF8FF;color:#175CD3"
        )
        calibration_layout.addWidget(firmware_note)
        config_note = QtWidgets.QLabel()
        config_note.setObjectName("fixtureConfigNote")
        config_note.setWordWrap(True)
        config_note.setStyleSheet(
            "padding:8px;border-radius:6px;background:#F2F4F7;color:#344054"
        )
        calibration_layout.addWidget(config_note)
        manual_height = QtWidgets.QGroupBox("E-probe scan height")
        manual_height.setObjectName("grpEasyManualHeight")
        manual_height_layout = QtWidgets.QVBoxLayout(manual_height)
        calibration_layout.addWidget(manual_height)
        self.ui.grpEasyManualHeight = manual_height
        self._easy_height_layout = manual_height_layout
        status = QtWidgets.QLabel("Fixture not taught.")
        status.setWordWrap(True)
        status.setStyleSheet("padding:8px;border-radius:6px;background:#F2F4F7")
        calibration_layout.addWidget(status)
        ready = QtWidgets.QLabel(
            "Ready check: CAPTURE P1 (or Home & Set Board Zero at a saved P1). "
            "Home XY for Teaching is available until P1 exists."
        )
        ready.setObjectName("fixtureReadySummary")
        ready.setWordWrap(True)
        layout.addWidget(ready)
        layout.addStretch(1)
        index = stack.addWidget(page)
        if index != PAGE_FIXTURE_TEACH:
            raise RuntimeError(
                f"Fixture teaching page index changed: expected {PAGE_FIXTURE_TEACH}, got {index}"
            )
        self.ui.pageFixtureTeach = page
        self.ui.fixtureTeachScroll = scroll
        self.ui.fixtureReferenceMap = reference_map
        self.ui.chkFixtureProbeClear = clear
        self.ui.btnFixtureRetryPrinter = retry_printer
        self.ui.fixturePrinterConnectionStatus = connection_status
        self.ui.btnFixtureHomeXyz = home
        self.ui.fixtureTeachTable = table
        self.ui.btnFixtureProbePoint = probe
        self.ui.fixtureJogStep = jog_step
        for name, button in jog_buttons.items():
            setattr(self.ui, name, button)
        self.ui.fixtureMaxZChange = z_limit
        self.ui.fixtureMaxResidual = residual_limit
        self.ui.btnFixtureSavePlane = save
        self.ui.btnFixtureVerify = verify
        self.ui.btnFixtureClearPoint = clear_point
        self.ui.fixtureConfigNote = config_note
        self.ui.fixtureTeachStatus = status
        self.ui.machineFixtureProfile = machine_fixture
        self.ui.btnNewMachineFixture = new_fixture
        self.ui.btnRefreshMachineFixtures = refresh_fixture
        self.ui.fixtureBoardSummary = summary
        self.ui.fixtureReadySummary = ready
        home.clicked.connect(self._fixture_home_xyz_clicked)
        retry_printer.clicked.connect(self._fixture_retry_printer_clicked)
        probe.clicked.connect(self._fixture_probe_selected_clicked)
        table.currentCellChanged.connect(self._on_fixture_point_selected)
        save.clicked.connect(self._fixture_save_plane)
        verify.clicked.connect(self._fixture_verify_clicked)
        clear_point.clicked.connect(self._fixture_clear_point_clicked)
        clear.toggled.connect(self._update_nav)
        machine_fixture.currentTextChanged.connect(self._on_machine_fixture_changed)
        new_fixture.clicked.connect(self._create_machine_fixture)
        refresh_fixture.clicked.connect(self._refresh_machine_fixtures)
        self._refresh_machine_fixtures()

    def _install_scan_setup_page(self):
        """Easy Step 4: locate the PCB insert from two taught corners per face.

        Step 3 already decided *what* is measured, so this page only answers
        *where* the PCB sits. Everything the operator can change here is a
        taught point: two diagonal corners for the top face and two for the
        bottom face, captured once and reused. The probe height is shown, not
        re-derived, because Step 2 established it from BLTouch pin contact.
        """
        stack = getattr(self.ui, "wizardStack", None)
        if not isinstance(stack, QtWidgets.QStackedWidget):
            return
        page = QtWidgets.QWidget()
        page.setObjectName("pageEasyScanSetup")
        page.setStyleSheet(
            "QGroupBox{font-weight:600;border:1px solid #667085;border-radius:8px;"
            "margin-top:10px;padding:12px} QGroupBox::title{subcontrol-origin:margin;"
            "left:12px;padding:0 5px} QPushButton{min-height:32px;padding:4px 12px}"
        )
        page_layout = QtWidgets.QVBoxLayout(page)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        scroll.setWidget(content)
        page_layout.addWidget(scroll)
        title = QtWidgets.QLabel("Locate the PCB insert")
        title.setStyleSheet("font-size:18px;font-weight:700")
        layout.addWidget(title)
        intro = QtWidgets.QLabel(
            "Jog the E-field probe onto two diagonal corners of the PCB and "
            "capture them. Teach each face once; the pair is saved with the "
            "fixture and reused. Step 3 decides what is measured."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        board_summary = QtWidgets.QLabel("Board file: not selected")
        board_summary.setObjectName("scanSetupBoardSummary")
        board_summary.setWordWrap(True)
        area_summary = QtWidgets.QLabel("Scan area: entire imported board")
        area_summary.setObjectName("scanSetupAreaSummary")
        area_summary.setWordWrap(True)
        summary_card = QtWidgets.QFrame()
        summary_card.setStyleSheet(
            "QFrame{background:#F2F4F7;border-radius:6px;padding:8px}"
        )
        summary_layout = QtWidgets.QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(8, 6, 8, 6)
        summary_layout.addWidget(board_summary)
        summary_layout.addWidget(area_summary)
        layout.addWidget(summary_card)

        placement_card = QtWidgets.QFrame()
        placement_card.setObjectName("glassboardPlacementCard")
        placement_card.setStyleSheet(
            "QFrame#glassboardPlacementCard{background:#F8FAFC;border:1px solid #CBD5E1;"
            "border-radius:8px;padding:8px}"
        )
        placement_layout = QtWidgets.QVBoxLayout(placement_card)
        # The face toward the probe is no longer an operator setting: it
        # follows whichever taught pair is applied. The combo stays as the one
        # place that answers "which side", so the STL fallback and the board
        # view cannot disagree about it.
        placement_side = QtWidgets.QComboBox()
        placement_side.setObjectName("glassboardPlacementSide")
        placement_side.addItem("Top side facing probe", "top")
        placement_side.addItem("Bottom side facing probe", "bottom")
        placement_side.setVisible(False)
        placement_side.currentIndexChanged.connect(self._on_glassboard_side_changed)
        placement_layout.addWidget(placement_side)

        jog_group = QtWidgets.QGroupBox("Jog the E-field probe")
        jog_layout = QtWidgets.QGridLayout(jog_group)
        jog_step = QtWidgets.QDoubleSpinBox()
        jog_step.setObjectName("scanSetupJogStep")
        jog_step.setRange(0.05, 10.0)
        jog_step.setDecimals(2)
        jog_step.setValue(1.0)
        jog_layout.addWidget(QtWidgets.QLabel("Step (mm)"), 0, 0)
        jog_layout.addWidget(jog_step, 0, 1, 1, 2)
        for name, label, row, column, dx, dy in (
            ("btnScanSetupJogYPlus", "Y+", 1, 1, 0, 1),
            ("btnScanSetupJogXMinus", "X−", 2, 0, -1, 0),
            ("btnScanSetupJogXPlus", "X+", 2, 2, 1, 0),
            ("btnScanSetupJogYMinus", "Y−", 3, 1, 0, -1),
        ):
            button = QtWidgets.QPushButton(label)
            button.setObjectName(name)
            button.clicked.connect(partial(self._scan_setup_jog, dx, dy))
            jog_layout.addWidget(button, row, column)
            setattr(self.ui, name, button)
        placement_layout.addWidget(jog_group)

        self._corner_teach_labels = {}
        for side in SCAN_SETUP_SIDES:
            group = QtWidgets.QGroupBox(f"{side.capitalize()} side of the PCB")
            group_layout = QtWidgets.QVBoxLayout(group)
            hint = QtWidgets.QLabel(
                "Jog the E-field probe tip onto the inner-pocket lower-left "
                "corner, capture A, then the opposite corner for B. Capture "
                "reads M114 only. A/B are the E-probe scan XY. P1 is fixture "
                "height only. Which face is MEASURE THE TOP/BOTTOM; 0°/180° "
                "yaw is fixture.pcb_yaw_deg in config.yaml, not these two corners."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#475467")
            group_layout.addWidget(hint)
            capture_row = QtWidgets.QHBoxLayout()
            for slot, corner_label in enumerate(TAUGHT_CORNER_LABELS):
                name = f"btnTeach{side.capitalize()}Corner{'AB'[slot]}"
                button = QtWidgets.QPushButton(f"CAPTURE {corner_label.upper()}")
                button.setObjectName(name)
                button.setToolTip(
                    "Record the E-field probe's M114 as this inner-pocket "
                    "corner. No motion."
                )
                button.clicked.connect(
                    partial(self._capture_pcb_corner, side, slot)
                )
                capture_row.addWidget(button)
                setattr(self.ui, name, button)
            group_layout.addLayout(capture_row)
            taught = QtWidgets.QLabel()
            taught.setObjectName(f"teach{side.capitalize()}CornerSummary")
            taught.setWordWrap(True)
            taught.setStyleSheet("font-family:monospace;padding:4px")
            group_layout.addWidget(taught)
            self._corner_teach_labels[side] = taught
            action_row = QtWidgets.QHBoxLayout()
            use_name = f"btnUse{side.capitalize()}Side"
            use_button = QtWidgets.QPushButton(f"MEASURE THE {side.upper()} SIDE")
            use_button.setObjectName(use_name)
            use_button.setToolTip(
                "Fit this face's taught pocket corners and seat the board in "
                "that insert. Face and yaw are not inferred from A/B. No motion."
            )
            use_button.clicked.connect(partial(self._use_taught_pcb_side, side))
            clear_name = f"btnClear{side.capitalize()}Corners"
            clear_button = QtWidgets.QPushButton("CLEAR")
            clear_button.setObjectName(clear_name)
            clear_button.setToolTip(
                "Forget this face's taught corners so they can be recaptured."
            )
            clear_button.clicked.connect(partial(self._clear_pcb_corners, side))
            action_row.addWidget(use_button)
            action_row.addWidget(clear_button)
            group_layout.addLayout(action_row)
            setattr(self.ui, use_name, use_button)
            setattr(self.ui, clear_name, clear_button)
            placement_layout.addWidget(group)

        placement_status = QtWidgets.QLabel(
            "Capture both corners of a face to position the board."
        )
        placement_status.setObjectName("glassboardPlacementStatus")
        placement_status.setWordWrap(True)
        placement_status.setStyleSheet(
            "padding:8px;background:#EFF8FF;color:#175CD3;border-radius:6px"
        )
        placement_layout.addWidget(placement_status)
        locate_status = QtWidgets.QLabel("")
        locate_status.setObjectName("scanSetupLocateStatus")
        locate_status.setWordWrap(True)
        placement_layout.addWidget(locate_status)
        seated = QtWidgets.QCheckBox(
            "PCB is inserted in the holder and fully seated"
        )
        seated.setObjectName("chkScanSetupBoardSeated")
        seated.setToolTip(
            "Software cannot sense seating, and taught corners are reused from "
            "disk, so this physical confirmation is still required each time."
        )
        seated.setStyleSheet("font-weight:600;padding:6px")
        placement_layout.addWidget(seated)
        layout.addWidget(placement_card)

        preview_group = QtWidgets.QGroupBox("Scan plan")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_host = QtWidgets.QWidget()
        preview_host.setObjectName("scanPreviewHost")
        preview_host_layout = QtWidgets.QVBoxLayout(preview_host)
        preview_host_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(preview_host)
        layout.addWidget(preview_group)
        review = QtWidgets.QLabel("Teach a face and confirm seating to continue.")
        review.setObjectName("scanSetupReview")
        review.setWordWrap(True)
        review.setStyleSheet("padding:10px;background:#F2F4F7;border-radius:6px")
        layout.addWidget(review)
        layout.addStretch(1)
        index = stack.addWidget(page)
        if index != PAGE_SCAN_SETUP:
            raise RuntimeError(
                f"Easy scan setup page index changed: expected {PAGE_SCAN_SETUP}, got {index}"
            )
        self.ui.pageEasyScanSetup = page
        self.ui.scanSetupBoardSummary = board_summary
        self.ui.scanSetupAreaSummary = area_summary
        self.ui.glassboardPlacementCard = placement_card
        self.ui.glassboardPlacementSide = placement_side
        self.ui.glassboardPlacementStatus = placement_status
        self.ui.scanSetupLocateStatus = locate_status
        self.ui.scanSetupJogStep = jog_step
        self.ui.chkScanSetupBoardSeated = seated
        self.ui.scanSetupReview = review
        self.ui.scanPreviewHost = preview_host
        seated.toggled.connect(self._scan_setup_seated_changed)
        self.ui.chkBoardSeated.toggled.connect(self._advanced_board_seated_changed)
        self._install_scan_preview_widget(preview_host)
        self._refresh_corner_teach_ui()

    def _install_easy_mode_controls(self):
        """Easy|Advanced toggle and Fine Adjust controls; headless-safe stubs."""
        u = self.ui

        def _stub(name, **fields):
            if getattr(u, name, None) is not None:
                return getattr(u, name)
            box = type("EasyWidget", (), {})()
            box._text = fields.get("text", "")
            box._checked = False
            box._value = fields.get("value", 0.25)
            box.enabled = True
            box.visible = True
            box.isChecked = lambda: box._checked
            box.setChecked = lambda checked, b=box: setattr(b, "_checked", bool(checked))
            box.text = lambda b=box: b._text
            box.setText = lambda text, b=box: setattr(b, "_text", str(text))
            box.setEnabled = lambda enabled, b=box: setattr(b, "enabled", bool(enabled))
            box.setVisible = lambda visible, b=box: setattr(b, "visible", bool(visible))
            box.isEnabled = lambda b=box: b.enabled
            box.isVisible = lambda b=box: b.visible
            box.value = lambda b=box: b._value
            box.setValue = lambda value, b=box: setattr(b, "_value", value)
            box.currentText = lambda b=box: b._text
            box.setCurrentText = lambda text, b=box: setattr(b, "_text", str(text))
            box.addItems = lambda items, b=box: None
            box.clicked = type("Sig", (), {"connect": lambda self, *_a, **_k: None})()
            box.currentTextChanged = type("Sig", (), {"connect": lambda self, *_a, **_k: None})()
            setattr(u, name, box)
            return box

        profile = getattr(u, "grpProfile", None)
        profile_layout = (
            profile.layout()
            if profile is not None and callable(getattr(profile, "layout", None))
            else None
        )
        height_layout = self._easy_height_layout
        root_layout = u.layout() if callable(getattr(u, "layout", None)) else None
        real = isinstance(profile_layout, QtWidgets.QVBoxLayout)

        if getattr(u, "operatorMode", None) is None:
            if real and isinstance(root_layout, QtWidgets.QVBoxLayout):
                row = QtWidgets.QHBoxLayout()
                label = QtWidgets.QLabel("Mode:")
                combo = QtWidgets.QComboBox()
                combo.setObjectName("operatorMode")
                combo.addItems([OPERATOR_MODE_EASY, OPERATOR_MODE_ADVANCED])
                combo.setCurrentText(OPERATOR_MODE_EASY)
                row.addWidget(label)
                row.addWidget(combo)
                row.addStretch(1)
                # Keep the Easy/Advanced choice visible on every wizard page.
                root_layout.insertLayout(1, row)
                u.operatorMode = combo
            else:
                combo = _stub("operatorMode", text=OPERATOR_MODE_EASY)
                combo._text = OPERATOR_MODE_EASY
        else:
            combo = u.operatorMode
            setter = getattr(combo, "setCurrentText", None)
            if callable(setter) and not combo.currentText():
                setter(OPERATOR_MODE_EASY)

        specs = (
            ("easyStatusLabel", "Easy fixture status", QtWidgets.QLabel, profile_layout),
            ("btnEasyMoveToRef", "Move to reference", QtWidgets.QPushButton, profile_layout),
            ("btnEasyFineAdjust", "Fine Adjust XY", QtWidgets.QPushButton, profile_layout),
            ("btnEasyResetXy", "Reset XY Calibration", QtWidgets.QPushButton, profile_layout),
            ("btnEasySaveXy", "Save Position", QtWidgets.QPushButton, profile_layout),
            ("btnEasyCancelXy", "Cancel", QtWidgets.QPushButton, profile_layout),
            ("easyHeightLabel", "Z reference required", QtWidgets.QLabel, height_layout),
            ("btnEasySetHeight", "SET PROBE HEIGHT", QtWidgets.QPushButton, height_layout),
            (
                "btnEasySetPcbSurface",
                "SET PCB SURFACE MANUALLY",
                QtWidgets.QPushButton,
                height_layout,
            ),
            (
                "btnEasyResetPcbSurface",
                "RESET PCB SURFACE",
                QtWidgets.QPushButton,
                height_layout,
            ),
        )
        for name, label, cls, target_layout in specs:
            if getattr(u, name, None) is not None:
                continue
            if real and target_layout is not None:
                widget = cls(label)
                widget.setObjectName(name)
                if isinstance(widget, QtWidgets.QLabel):
                    widget.setWordWrap(True)
                target_layout.addWidget(widget)
                setattr(u, name, widget)
            else:
                _stub(name, text=label)

        if getattr(u, "easyJogStep", None) is None:
            if real:
                step = QtWidgets.QDoubleSpinBox()
                step.setObjectName("easyJogStep")
                step.setRange(0.05, 1.0)
                step.setDecimals(2)
                step.setValue(0.25)
                profile_layout.addWidget(step)
                u.easyJogStep = step
            else:
                _stub("easyJogStep", value=0.25)

        for name, label in (
            ("btnEasyJogXPlus", "X+"),
            ("btnEasyJogXMinus", "X-"),
            ("btnEasyJogYPlus", "Y+"),
            ("btnEasyJogYMinus", "Y-"),
        ):
            if getattr(u, name, None) is not None:
                continue
            if real:
                button = QtWidgets.QPushButton(label)
                button.setObjectName(name)
                profile_layout.addWidget(button)
                setattr(u, name, button)
            else:
                _stub(name, text=label)

        changed = getattr(u.operatorMode, "currentTextChanged", None)
        if changed is not None and callable(getattr(changed, "connect", None)):
            changed.connect(self._on_operator_mode_changed)
        for name, slot in (
            ("btnEasyMoveToRef", self._easy_move_to_reference),
            ("btnEasyFineAdjust", self._easy_open_fine_adjust),
            ("btnEasyResetXy", self._easy_reset_xy_calibration),
            ("btnEasySaveXy", self._easy_save_position),
            ("btnEasyCancelXy", self._easy_cancel_fine_adjust),
            ("btnEasySetHeight", self._easy_set_probe_height),
            ("btnEasySetPcbSurface", self._easy_set_pcb_surface_manually),
            ("btnEasyResetPcbSurface", self._easy_reset_pcb_surface),
        ):
            button = getattr(u, name, None)
            clicked = getattr(button, "clicked", None) if button is not None else None
            if clicked is not None and callable(getattr(clicked, "connect", None)):
                clicked.connect(slot)
        for name, dx, dy in (
            ("btnEasyJogXPlus", 1, 0),
            ("btnEasyJogXMinus", -1, 0),
            ("btnEasyJogYPlus", 0, 1),
            ("btnEasyJogYMinus", 0, -1),
        ):
            button = getattr(u, name, None)
            clicked = getattr(button, "clicked", None) if button is not None else None
            if clicked is not None and callable(getattr(clicked, "connect", None)):
                clicked.connect(partial(self._easy_jog, dx, dy))
        self._set_easy_adjust_visible(False)
        self._apply_operator_mode_ui()

    def _install_workflow_progress(self):
        """Persistent five-step strip for the Easy operator path."""
        self._workflow_step_labels = []
        root = self.ui.layout() if callable(getattr(self.ui, "layout", None)) else None
        if not isinstance(root, QtWidgets.QVBoxLayout):
            return
        row = QtWidgets.QHBoxLayout()
        for number, text in enumerate(
            ("Setup", "Board zero", "Board", "Scan setup", "Measure"), start=1
        ):
            label = QtWidgets.QLabel(f"{number}  {text}")
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setMinimumHeight(30)
            row.addWidget(label, 1)
            self._workflow_step_labels.append(label)
        root.insertLayout(2, row)
        self._refresh_workflow_progress()

    def _refresh_workflow_progress(self):
        labels = getattr(self, "_workflow_step_labels", ())
        if not labels:
            return
        page = self.ui.wizardStack.currentIndex()
        pages = (PAGE_SETUP, PAGE_FIXTURE_TEACH, PAGE_BOARD, PAGE_SCAN_SETUP, PAGE_SCAN)
        current = pages.index(page) if page in pages else len(pages)
        visible = self._easy_mode_active()
        for index, label in enumerate(labels):
            label.setVisible(visible)
            if index == current:
                style = "background:#175CD3;color:white;border-radius:6px;font-weight:700"
            elif index < current and self._easy_progress_step_complete(pages[index]):
                style = "background:#067647;color:white;border-radius:6px;font-weight:600"
            else:
                style = "background:#EAECF0;color:#344054;border-radius:6px"
            label.setStyleSheet(style)

    def _easy_progress_step_complete(self, page):
        """Prior-step green is readiness, not 'the operator has visited later'."""
        if page == PAGE_SETUP:
            return True
        if page == PAGE_FIXTURE_TEACH:
            return self._fixture_gate_ready()
        if page == PAGE_BOARD:
            return self._board_view is not None
        if page == PAGE_SCAN_SETUP:
            return (
                not self._registration_problems()
                and self.ui.chkBoardSeated.isChecked()
            )
        if page == PAGE_SCAN:
            return self.result is not None
        return False

    def _easy_mode_active(self):
        combo = getattr(self.ui, "operatorMode", None)
        text = combo.currentText() if combo is not None else OPERATOR_MODE_EASY
        return self._mode() == MODE_PCB and text == OPERATOR_MODE_EASY

    def _set_operator_mode(self, mode):
        combo = getattr(self.ui, "operatorMode", None)
        if combo is None:
            return
        setter = getattr(combo, "setCurrentText", None)
        if callable(setter):
            setter(mode)
        self._on_operator_mode_changed(mode)

    def _on_operator_mode_changed(self, _mode=None):
        if self._easy_mode_active():
            advanced = getattr(self.ui, "chkAdvanced", None)
            if advanced is not None:
                advanced.setChecked(False)
        self._apply_operator_mode_ui()
        self._refresh_height_ui()
        self._update_nav()

    def _set_widget_visible(self, name, visible):
        widget = getattr(self.ui, name, None)
        setter = getattr(widget, "setVisible", None)
        if callable(setter):
            setter(visible)

    def _set_easy_adjust_visible(self, visible):
        self._easy_adjust_open = bool(visible)
        for name in (
            "btnEasySaveXy",
            "btnEasyCancelXy",
            "btnEasyJogXPlus",
            "btnEasyJogXMinus",
            "btnEasyJogYPlus",
            "btnEasyJogYMinus",
            "easyJogStep",
        ):
            self._set_widget_visible(name, visible and self._easy_mode_active())

    def _apply_operator_mode_ui(self):
        pcb = self._mode() == MODE_PCB
        easy = self._easy_mode_active()
        missing_vertical = not self._easy_vertical_calibrated()
        self._set_widget_visible("operatorMode", pcb)
        # Easy Z setup lives on Step 2. Advanced CAD height stays on the Board page.
        self._set_widget_visible("grpScanHeight", pcb and not easy)
        height_setup = set(self._EASY_HEIGHT_SETUP_WIDGETS)
        for name in self._EASY_HIDDEN_WIDGETS:
            if name == "chkAdvanced":
                self._set_widget_visible(name, pcb and not easy)
                continue
            if name in height_setup:
                self._set_widget_visible(name, pcb and not easy)
                continue
            self._set_widget_visible(name, pcb and not easy)
        for name in self._EASY_SHOWN_WIDGETS:
            self._set_widget_visible(name, easy)
        for name in EASY_MANUAL_HEIGHT_BUTTONS:
            self._set_widget_visible(name, easy and missing_vertical)
        self._set_widget_visible("grpEasyManualHeight", easy and missing_vertical)
        self._set_easy_adjust_visible(self._easy_adjust_open and easy)
        home = getattr(self.ui, "btnHomeXy", None)
        if home is not None:
            self._refresh_home_button_copy()
        if easy:
            self._set_advanced_visible(False)
        self._update_registration_ui()
        self._refresh_easy_height_label()

    def _refresh_home_button_copy(self):
        """Primary Home label: recalibrate when P1 exists, else teaching home."""
        p1 = self._p1_bltouch_commanded_xy() is not None
        easy = self._easy_mode_active()
        fixture_home = getattr(self.ui, "btnFixtureHomeXyz", None)
        if fixture_home is not None:
            fixture_home.setText(
                "HOME & SET BOARD ZERO" if p1 else "HOME XY FOR TEACHING"
            )
        home = getattr(self.ui, "btnHomeXy", None)
        if home is not None:
            if easy:
                home.setText(
                    "HOME & SET BOARD ZERO" if p1 else "HOME XY FOR TEACHING"
                )
            else:
                home.setText("Home X/Y")

    _ADVANCED_WIDGETS = (
        "lblCentre",
        "centreMhz",
        "lblSpan",
        "spanMhz",
        "lblPoints",
        "points",
        "lblSamples",
        "samplesPerXy",
        "lblMetric",
        "metricBox",
        "lblRbw",
        "rbwKhz",
        "lblAtten",
        "attenDb",
        "lblSpur",
        "spur",
        "lna",
        "lblProbeSourceVariant",
        "probeSourceVariant",
        "lblCharacterizedOutput",
        "characterizedOutput",
        "lblAmplifierModel",
        "amplifierModel",
        "lblCableId",
        "cableId",
        "lblCableCurvePath",
        "cableCurvePath",
        "lblRangePolicy",
        "characterizationRangePolicy",
        "chkBackground",
        "lblBackgroundMode",
        "backgroundMode",
        "backgroundHoldSweeps",
        "chkCharacterization",
        "lblProbeModel",
        "probeModel",
    )

    def _on_advanced_toggled(self, checked: bool):
        self._set_advanced_visible(bool(checked))
        u = self.ui
        if (
            checked
            and getattr(u, "chkSpectrumCube", None) is not None
            and u.chkSpectrumCube.isChecked()
            and u.chkBackground.isChecked()
            and u.chkCharacterization.isChecked()
        ):
            u.backgroundMode.setCurrentText("linear_subtract")

    def _set_advanced_visible(self, visible: bool):
        u = self.ui
        for name in self._ADVANCED_WIDGETS:
            widget = getattr(u, name, None)
            setter = getattr(widget, "setVisible", None)
            if callable(setter):
                setter(visible)

    def _apply_simple_survey_defaults(self):
        """Simple mode: E5 cube, TBWA2-40, 2 mm. Advanced still holds the knobs."""
        u = self.ui
        cube = getattr(u, "chkSpectrumCube", None)
        if cube is None:
            return
        if not cube.isChecked():
            cube.setChecked(True)
        if u.amplifierModel.currentText() in ("", "bypass"):
            u.amplifierModel.setCurrentText("TBWA2_40")
        self._set_advanced_visible(False)
        self._refresh_scan_intro()
        self._refresh_amplifier_label()

    def _refresh_amplifier_label(self, *_args):
        label = getattr(self.ui, "lblAmplifierSurvey", None)
        combo = getattr(self.ui, "amplifierModel", None)
        setter = getattr(label, "setText", None) if label is not None else None
        if not callable(setter) or combo is None:
            return
        names = {
            "TBWA2_40": "TBWA2-40 (40 dB)",
            "TBWA2_20": "TBWA2-20 (20 dB)",
            "bypass": "bypass (no external amp)",
        }
        amp = combo.currentText()
        setter(f"Amplifier: {names.get(amp, amp)}")

    def _cube_selected(self):
        box = getattr(self.ui, "chkSpectrumCube", None)
        return box is not None and box.isChecked()

    def _rf_confirmed(self):
        box = getattr(self.ui, "chkRfConfirmed", None)
        return box is not None and box.isChecked()

    def _refresh_scan_intro(self):
        """Cube scans need the RF-chain tick on this page; single-span does not."""
        cube = self._cube_selected()
        rf = getattr(self.ui, "chkRfConfirmed", None)
        if rf is not None and hasattr(rf, "setVisible"):
            rf.setVisible(cube)
        intro = getattr(self.ui, "scanIntro", None)
        if intro is None:
            return
        if self._mode() != MODE_PCB:
            intro.setText(
                "Confirm the RF chain is reviewed, then Start scan."
                if cube
                else "Start scan maps the rectangle from the origin you set."
            )
            return
        intro.setText(
            "Confirm the probe height, PCB travel, and RF chain, then Start scan."
            if cube
            else "Confirm the probe height and PCB travel are clear, then Start scan."
        )

    def _on_cube_preset_toggled(self, checked: bool):
        u = self.ui
        if checked:
            self._saved_single_span = {
                "centre": u.centreMhz.value(),
                "span": u.spanMhz.value(),
                "points": u.points.value(),
                "rbw": u.rbwKhz.value(),
                "atten": u.attenDb.value(),
                "samples": u.samplesPerXy.value(),
                "step": u.stepMm.value(),
                "lna": u.lna.isChecked(),
                "background": u.chkBackground.isChecked(),
                "background_mode": u.backgroundMode.currentText(),
                "characterization": u.chkCharacterization.isChecked(),
                "probe": u.probeModel.currentText(),
                "amplifier": u.amplifierModel.currentText(),
            }
            u.centreMhz.setValue(25.5)
            u.spanMhz.setValue(49.0)
            u.points.setValue(450)
            u.rbwKhz.setValue(300.0)
            u.attenDb.setValue(10)
            u.samplesPerXy.setValue(2)
            u.stepMm.setValue(2.0)
            u.lna.setChecked(False)
            u.chkBackground.setChecked(True)
            u.backgroundMode.setCurrentText("delta_db")
            u.chkCharacterization.setChecked(True)
            u.probeModel.setCurrentText("E5")
            u.characterizedOutput.setCurrentText("electric_field_dbuv_per_m")
            u.amplifierModel.setCurrentText("TBWA2_40")
            height_approved = bool(
                getattr(self, "_height_override_mm", None) is not None
                or getattr(self, "_last_commanded_scan_z", None) is not None
            )
            if not height_approved:
                clearance = getattr(u, "probeClearanceMm", None)
                if clearance is not None:
                    clearance.setValue(5.0)
        elif self._saved_single_span:
            saved = self._saved_single_span
            u.centreMhz.setValue(saved["centre"])
            u.spanMhz.setValue(saved["span"])
            u.points.setValue(saved["points"])
            u.rbwKhz.setValue(saved["rbw"])
            u.attenDb.setValue(saved["atten"])
            u.samplesPerXy.setValue(saved["samples"])
            u.stepMm.setValue(saved["step"])
            u.lna.setChecked(saved["lna"])
            u.chkBackground.setChecked(saved["background"])
            u.backgroundMode.setCurrentText(saved["background_mode"])
            u.chkCharacterization.setChecked(saved["characterization"])
            u.probeModel.setCurrentText(saved["probe"])
            if "amplifier" in saved:
                u.amplifierModel.setCurrentText(saved["amplifier"])
            self._saved_single_span = None
        self._refresh_scan_intro()
        self._update_nav()

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
        cmap = pyqtgraph.colormap.get("inferno")
        self.image.setColorMap(cmap)
        # Explicitly keep the raster nearest-neighbour.  Smooth interpolation is
        # attractive for photos, but misleading for a measurement grid because
        # it invents values between probe locations.
        if callable(getattr(self.image, "setAutoDownsample", None)):
            self.image.setAutoDownsample(False)
        self.image.setZValue(1)
        pw.addItem(self.image)
        pw.setAspectLocked(True)
        pw.setTitle("Live map — waiting for first measured cell")

        # A real colour scale belongs next to the heatmap.  ColorBarItem is
        # available in current pyqtgraph; the guarded fallback keeps older/headless
        # builds functional rather than making the wizard fail at startup.
        try:
            bar = pyqtgraph.ColorBarItem(
                values=PLACEHOLDER_LEVELS,
                colorMap=cmap,
                label="Level (dBm)",
                interactive=False,
                width=16,
            )
            bar.setImageItem(self.image, insert_in=pw.getPlotItem())
            self._live_colorbar = bar
        except Exception:
            self._live_colorbar = None
        live = getattr(self.ui, "liveSpectrum", None)
        if live is None or not callable(getattr(live, "plot", None)):
            return
        get_axis = getattr(live, "getAxis", None)
        if not callable(get_axis) or get_axis("bottom") is None:
            return
        if callable(getattr(live, "setBackground", None)):
            live.setBackground("w")
        if callable(getattr(live, "setLabel", None)):
            live.setLabel("bottom", "Frequency (MHz)")
            live.setLabel("left", "Power (dBm)")
        for axis_name in ("bottom", "left"):
            axis = get_axis(axis_name)
            if axis is None:
                continue
            if callable(getattr(axis, "enableAutoSIPrefix", None)):
                axis.enableAutoSIPrefix(False)
            if callable(getattr(axis, "setPen", None)):
                axis.setPen(pyqtgraph.mkPen("k"))
            if callable(getattr(axis, "setTextPen", None)):
                axis.setTextPen(pyqtgraph.mkPen("k"))
        self._live_spectrum_curve = live.plot(pen=pyqtgraph.mkPen("#1f4e79", width=2))

    def _setup_board_plots(self):
        # pyqtgraph stays inside the plot methods, as in _setup_plot, so the
        # headless tests can drive the interlocks without it installed.
        import pyqtgraph

        plot = self.ui.regPlot
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
        plot.addItem(self._landmark_marks)
        plot.addItem(self._selected_mark)
        plot.scene().sigMouseClicked.connect(self._on_reg_plot_clicked)
        self._selector.step_mm = self.ui.stepMm.value()

    def _draw_board(self, plot, view, *, preserve_camera=True):
        """Draw one board view: artwork, outline, then component markers."""
        import pyqtgraph

        self._selector.board_view = view
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
        sequence = self._sequence()
        resume = self._last_page_by_mode.get(self._mode(), PAGE_SETUP)
        self.ui.wizardStack.setCurrentIndex(
            resume if resume in sequence else PAGE_SETUP
        )
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
        self._refreshing_ports = True
        try:
            combo.clear()
            for device, label in self._candidate_ports():
                combo.addItem(label, device)
            index = combo.findData(current)
            if index is not None and index >= 0:
                combo.setCurrentIndex(index)
            elif current:
                combo.setCurrentText(current)
        finally:
            self._refreshing_ports = False

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
        if self._refreshing_ports:
            return
        self._close_printer()
        self._clear_height_datum()
        self._clear_pcb_homing()
        self._refresh_height_ui()
        self._update_nav()

    def _browse_folder(self):
        chosen = QFileDialog.getExistingDirectory(
            self.ui, "Choose the folder that receives scan results", self.ui.outputRoot.text()
        )
        if chosen:
            self.ui.outputRoot.setText(chosen)

    def _update_characterization_ui(self, *_args):
        u = self.ui
        enabled = u.chkCharacterization.isChecked()
        h10 = enabled and u.probeModel.currentText().upper() == "H10"
        for name in (
            "probeModel",
            "characterizedOutput",
            "amplifierModel",
            "cableId",
            "cableCurvePath",
            "characterizationRangePolicy",
        ):
            getattr(u, name).setEnabled(enabled)
        u.probeSourceVariant.setEnabled(h10)

        if not enabled:
            text = (
                "Disabled: scan output remains raw receiver dBm. Manufacturer "
                "characterization is diagnostic only, not EMC compliance data."
            )
        elif u.chkBackground.isChecked() and u.backgroundMode.currentText() != "linear_subtract":
            text = (
                "Characterized background scans require linear_subtract. Raw and "
                "background spectra are retained; nonpositive residual bins are invalid."
            )
        elif h10 and not u.probeSourceVariant.currentText():
            text = (
                "H10 requires workbook_S63 or faq_S62. Both are preserved source "
                "variants; neither is independently verified."
            )
        else:
            text = (
                "Probe → cable → optional external amplifier → TinySA. The TinySA "
                "LNA is recorded separately and is never gain-corrected twice."
            )
        u.characterizationWarning.setText(text)

    def _on_area_changed(self):
        # Geometry changed, so the frozen grid is stale. The landmarks and the
        # fit are not: they still describe where the board sits. Invalidation
        # runs downwards only, so they survive this.
        self._invalidate_plan()
        self._selector.step_mm = self.ui.stepMm.value()
        self.ui.gridInfo.setText(self._grid_summary())
        self._update_selection_info()
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
            if self._scan_selection is None:
                return f"{area.nx} x {area.ny} cells over the board, clipped to its outline"
            selected = self._selected_cell_count()
            return (
                f"{selected} selected cells of {area.nx} x {area.ny} "
                "on the board grid"
            )
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
            manage_z=self.ui.chkAllowSetupZ.isChecked(),
            z_max_mm=LOGICAL_Z_MAX_MM,
            z_travel_speed_mm_min=1200.0,
            # Drop X/Y holding after each move so stepper PWM is off during
            # settle + sweeps. Z stays held so the probe cannot sag (INV-Z-007).
            disable_steppers_during_measure=True,
            disable_stepper_axes="XY",
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

        cube = (
            getattr(u, "chkSpectrumCube", None) is not None
            and u.chkSpectrumCube.isChecked()
        )
        advanced = (
            getattr(u, "chkAdvanced", None) is not None and u.chkAdvanced.isChecked()
        )
        if cube:
            from EMI_Mapper.spectrum_cube import E5_CUBE_START_HZ, E5_CUBE_STOP_HZ

            cube_start_hz = E5_CUBE_START_HZ
            cube_stop_hz = E5_CUBE_STOP_HZ
        else:
            cube_start_hz = u.centreMhz.value() * 1e6 - u.spanMhz.value() * 5e5
            cube_stop_hz = u.centreMhz.value() * 1e6 + u.spanMhz.value() * 5e5
        hold_widget = getattr(u, "backgroundHoldSweeps", None)
        hold_sweeps = int(hold_widget.value()) if hold_widget is not None else 8
        if cube and not advanced:
            background = True
            background_kind = "stationary"
            background_mode = "delta_db"
        else:
            background = u.chkBackground.isChecked()
            background_kind = "xy_grid"
            background_mode = u.backgroundMode.currentText()
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
                unsupported_setting_policy=(
                    "reject"
                    if getattr(u, "chkSpectrumCube", None) is not None
                    and u.chkSpectrumCube.isChecked()
                    else "clip"
                ),
            ),
            measurement_chain=engine_config.MeasurementChainConfig(
                enabled=u.chkCharacterization.isChecked(),
                probe=u.probeModel.currentText(),
                probe_source_variant=(
                    u.probeSourceVariant.currentText()
                    if u.probeModel.currentText().upper() == "H10"
                    else {
                        "H20": "workbook_S80",
                        "H5": "workbook_S40",
                        "E5": "workbook_112_5",
                    }.get(u.probeModel.currentText().upper(), "")
                ),
                output_quantity=u.characterizedOutput.currentText(),
                amplifier=u.amplifierModel.currentText(),
                cable_id=u.cableId.text().strip() or "none",
                cable_curve_path=u.cableCurvePath.text().strip(),
                out_of_range_policy=u.characterizationRangePolicy.currentText(),
            ),
            area=engine_config.ScanArea(
                width_mm=u.widthMm.value(),
                height_mm=u.heightMm.value(),
                step_mm=u.stepMm.value(),
            ),
            metric=u.metricBox.currentText(),
            background=background,
            background_mode=background_mode,
            background_kind=background_kind,
            background_hold_sweeps=hold_sweeps,
            output_root=u.outputRoot.text().strip() or "EMI_Scans",
            label=u.runLabel.text().strip(),
            acquisition="spectrum_cube" if cube else "single_span",
            cube_start_hz=cube_start_hz,
            cube_stop_hz=cube_stop_hz,
            physical_amplifier_present=bool(
                getattr(u, "chkPhysicalTbwa2", None) is not None
                and u.chkPhysicalTbwa2.isChecked()
            ),
            rf_configuration_confirmed=bool(
                getattr(u, "chkRfConfirmed", None) is not None
                and u.chkRfConfirmed.isChecked()
            ),
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
        self._approved_scan_fingerprint = None
        self._refresh_scan_preview()

    def _invalidate_board_alignment(self):
        """The board-to-machine fit is no longer trustworthy, so its plan goes too."""
        self._registration = None
        self._stl_board_placement = None
        placement_status = getattr(self.ui, "glassboardPlacementStatus", None)
        if placement_status is not None:
            placement_status.setText(
                "Choose which board face is up, then place it in the middle holder."
            )
            placement_status.setStyleSheet(
                "padding:8px;background:#EFF8FF;color:#175CD3;border-radius:6px"
            )
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
        if self._easy_fixture_ready():
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

    def _easy_fixture_ready(self):
        """Easy + seated + a verified fixture replaces landmark confirmation."""
        return (
            self._easy_mode_active()
            and self.ui.chkBoardSeated.isChecked()
            and self._profile is not None
            and bool(self._profile.xy_verified)
        )

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
        self._fixture_xyz_homed = False
        self._fixture_verified_session = False
        self._machine_xy = None
        self.ui.posLabel.setText("Not homed.")
        self._clear_board_zero()
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
        self._import_board_path(path)

    def _import_board_path(self, path):
        try:
            model = engine_odbpp.import_odbpp(path)
        except engine_odbpp.OdbppError as exc:
            self._board_model = None
            self._board_view = None
            self._clear_landmarks_and_alignment()
            self.ui.boardInfo.setText(f"Import failed: {exc}")
            self.ui.boardWarnings.setPlainText("")
            self._reset_scan_selection()
            self._draw_board(self.ui.regPlot, None)
            self.ui.boardRotation.setText("0°")
            self._refresh_height_ui()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            self._update_nav()
            return
        self._board_model = model
        self.ui.boardPath.setText(path)
        self._board_view = None
        self._apply_board_view(rotation_deg=0)

    def _rotate_board(self, delta):
        if self._board_view is None:
            return
        self._apply_board_view(
            rotation_deg=(self._board_view.rotation_deg + int(delta)) % 360
        )

    def _apply_board_view(self, _side=None, *, rotation_deg=None):
        """Derive the view for the selected side and rotation.

        A changed side or rotation invalidates the board-to-machine transform
        through ``_clear_landmarks_and_alignment``. Homing survives: turning
        the drawing does not move the Ender's frame.
        """
        if self._board_model is None:
            return
        side = self.ui.boardSide.currentText()
        if rotation_deg is None:
            rotation_deg = 0 if self._board_view is None else self._board_view.rotation_deg
        old_view = self._board_view
        changed = (
            old_view is None
            or old_view.side != side
            or old_view.rotation_deg != rotation_deg
        )
        self._board_view = self._board_model.view(side=side, rotation_deg=rotation_deg)
        self._selector.board_view = self._board_view
        if changed:
            self._clear_landmarks_and_alignment()
        self._reconcile_scan_selection(old_view)
        self.ui.boardRotation.setText(f"{rotation_deg}°")
        self._describe_board()
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
            f"rotated {view.rotation_deg}°, "
            f"board-view extent X[{x0:.2f}, {x1:.2f}] Y[{y0:.2f}, {y1:.2f}] mm\n"
            f"{view.height_note()}"
        )
        warnings = list(view.report.warnings) if view.report is not None else []
        self.ui.boardWarnings.setPlainText("\n".join(warnings))
        self._refresh_height_ui()
        self._update_selection_info()

    def _empty_scan_selection(self):
        return self._selector.empty()

    def _selection_tool(self):
        if self.ui.radioSelectPoint.isChecked():
            return "point"
        if self.ui.radioSelectRect.isChecked():
            return "rectangle"
        return None

    def _set_selection_radios(self, tool):
        for radio, want in (
            (self.ui.radioSelectPoint, tool == "point"),
            (self.ui.radioSelectRect, tool == "rectangle"),
        ):
            blocker = getattr(radio, "blockSignals", None)
            if callable(blocker):
                blocker(True)
            radio.setChecked(want)
            if callable(blocker):
                blocker(False)
        self._selector.tool = tool

    def _reset_scan_selection(self):
        """Default full-board scan. Not the same as Clear."""
        self._selector.scan_entire_board()
        self._set_selection_radios(None)
        self._invalidate_plan()
        self._update_selection_info()
        self.ui.gridInfo.setText(self._grid_summary())

    def _reconcile_scan_selection(self, old_view):
        new_view = self._board_view
        if old_view is None or old_view.side != new_view.side:
            self._reset_scan_selection()
            return
        if (
            old_view.rotation_deg == new_view.rotation_deg
            or self._scan_selection is None
            or engine_selection is None
        ):
            return
        self._scan_selection = engine_selection.transform_selection(
            old_view, new_view, self._scan_selection
        )
        self._invalidate_plan()
        self._update_selection_info()

    def _on_point_tool_toggled(self, checked):
        if checked:
            self._uncheck_selection_radio(self.ui.radioSelectRect)
            before = self._scan_selection
            self._selector.set_tool("point")
            if before is None:
                self._after_selection_changed()
        elif not self.ui.radioSelectRect.isChecked():
            self._selector.tool = None

    def _on_rect_tool_toggled(self, checked):
        if checked:
            self._uncheck_selection_radio(self.ui.radioSelectPoint)
            before = self._scan_selection
            self._selector.set_tool("rectangle")
            if before is None:
                self._after_selection_changed()
        elif not self.ui.radioSelectPoint.isChecked():
            self._selector.tool = None

    def _uncheck_selection_radio(self, radio):
        blocker = getattr(radio, "blockSignals", None)
        if callable(blocker):
            blocker(True)
        radio.setChecked(False)
        if callable(blocker):
            blocker(False)

    def _scan_entire_board(self, _checked=False):
        self._reset_scan_selection()
        self._update_nav()

    def _clear_scan_selection(self, _checked=False):
        self._selector.clear()
        self._after_selection_changed()

    def _undo_scan_selection(self, _checked=False):
        if self._scan_selection is None:
            return
        self._selector.undo()
        self._after_selection_changed()

    def _after_selection_changed(self):
        self._invalidate_plan()
        self._update_selection_info()
        self.ui.gridInfo.setText(self._grid_summary())
        self._update_nav()

    def _selected_cell_count(self):
        return self._selector.selected_cell_count()

    def _assert_scan_selection_ready(self):
        if self._scan_selection is None:
            return
        if not self._scan_selection.items:
            raise RuntimeError(
                "Nothing is selected. Add a point or rectangle, or click "
                "Scan entire board."
            )
        if self._selected_cell_count() == 0:
            raise RuntimeError(
                "No selected cell falls on the board. Adjust the regions "
                "or the grid step."
            )

    def _update_selection_info(self):
        label = getattr(self.ui, "selectionInfo", None)
        if label is None:
            return
        if self._selector.last_message:
            label.setText(self._selector.last_message)
            return
        label.setText(self._selector.status_text())

    def _add_selection_point(self, x, y):
        if not self._selector.add_point(x, y):
            self._update_selection_info()
            return False
        self._after_selection_changed()
        return True

    def _add_selection_roi(self, x_min, x_max, y_min, y_max):
        if not self._selector.add_roi(x_min, x_max, y_min, y_max):
            return False
        self._after_selection_changed()
        return True

    def _open_board_selector(self, _checked=False):
        if self._board_view is None:
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                "Import a board before selecting a scan area.",
            )
            return
        parent = self.ui if isinstance(self.ui, QtWidgets.QWidget) else None
        dialog = BoardSelectorDialog(
            parent,
            self._board_view,
            self._scan_selection,
            self.ui.stepMm.value(),
            tool=self._selection_tool(),
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self._scan_selection = dialog.applied_selection()
        self._set_selection_radios(dialog.applied_tool())
        self._after_selection_changed()

    def _scan_height_plan(self):
        """CAD plan using the agreed override, independent of the checkbox.

        Apply stores the plane; unchecking without Apply does not drop it.
        Uncheck-then-Apply still clears it. An invalid override is an error,
        not a silent CAD+clearance substitute.
        """
        return engine_height.height_plan_from_view(
            self._board_view,
            clearance_mm=self._printer_config().probe_clearance_mm,
            override_height_above_pcb_mm=self._height_override_mm,
        )

    def _height_snapshot(self, reported_z=None):
        return engine_height.height_status_block(
            self._scan_height_plan(),
            pcb_surface_machine_z=self._pcb_surface_z,
            reported_machine_z=reported_z,
            last_commanded_scan_z=self._last_commanded_scan_z,
        )

    def _reported_z(self):
        handle = getattr(self, "printer", None)
        if handle is not None and hasattr(handle, "get_xyz"):
            try:
                return handle.get_xyz()[2]
            except (engine_printer.PrinterError, OSError, RuntimeError, TypeError):
                return getattr(handle, "z", None)
        if self.printer_serial is None:
            return None
        try:
            return engine_printer.Printer(
                self.printer_serial, self._printer_config()
            ).get_xyz()[2]
        except (engine_printer.PrinterError, OSError, RuntimeError):
            return None

    def _refresh_height_ui(self, *_args):
        u = self.ui
        allow_z = u.chkAllowSetupZ.isChecked()
        for widget in (u.btnJogZUp, u.btnJogZDown, u.btnMoveToScanHeight):
            widget.setEnabled(allow_z)
        self._refresh_easy_height_label()
        reported_z = (
            self._easy_verified_scan_z
            if self._easy_mode_active()
            else self._reported_z()
        )
        try:
            plan = self._scan_height_plan()
            block = self._height_snapshot(reported_z)
        except engine_height.HeightError as exc:
            u.heightCadLabel.setText(str(exc))
            u.chkAtScanPlane.setChecked(False)
            u.chkAtScanPlane.setEnabled(False)
            extra = u.boardWarnings.toPlainText().strip()
            note = str(exc)
            if extra:
                existing = extra.splitlines()
                if note not in existing:
                    u.boardWarnings.setPlainText("\n".join(existing + [note]))
            else:
                u.boardWarnings.setPlainText(note)
            return
        tallest = block.get("tallest_mm")
        recommended = block.get("recommended_height_above_pcb_mm")
        if plan.coverage == "none":
            cad = "CAD: no listed component heights. Enter an operator override."
        else:
            cad = (
                f"CAD: tallest listed is {block.get('tallest_refdes')} at "
                f"{tallest:.2f} mm. Recommended plane is {recommended:.2f} mm "
                f"above the PCB ({plan.coverage} coverage, "
                f"{block['known']}/{block['total']} heights known)."
            )
        if plan.coverage == "partial":
            cad += " Incomplete CAD is not collision protection."
        u.heightCadLabel.setText(cad)
        surface = block.get("pcb_surface_machine_z")
        if surface is None:
            u.heightSurfaceLabel.setText("PCB surface: not set. Park over a component-free area.")
        else:
            u.heightSurfaceLabel.setText(f"PCB surface datum: {surface:.2f} mm (logical).")
        reported_h = block.get("reported_height_above_pcb_mm")
        reported_z = block.get("reported_machine_z")
        if reported_z is None:
            u.heightPositionLabel.setText("Reported height: unknown until the printer answers M114.")
        elif reported_h is None:
            u.heightPositionLabel.setText(f"Reported Z {reported_z:.2f} mm. Set the PCB surface to convert to height.")
        else:
            u.heightPositionLabel.setText(
                f"Reported height {reported_h:.2f} mm above PCB (logical Z {reported_z:.2f} mm)."
            )
        at_plane = bool(block.get("at_requested_plane"))
        u.chkAtScanPlane.setChecked(at_plane)
        u.chkAtScanPlane.setEnabled(False)
        warnings = list(block.get("warnings") or [])
        extra = u.boardWarnings.toPlainText().strip()
        height_notes = [note for note in warnings if note]
        if height_notes:
            existing = extra.splitlines() if extra else []
            merged = existing + [note for note in height_notes if note not in existing]
            u.boardWarnings.setPlainText("\n".join(merged))

    def _commit_override_from_spinbox(self):
        """Store the visible override as the agreed plane. False if invalid."""
        try:
            plan = engine_height.height_plan_from_view(
                self._board_view,
                clearance_mm=self._printer_config().probe_clearance_mm,
                override_height_above_pcb_mm=self.ui.heightOverrideMm.value(),
            )
        except engine_height.HeightError as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return False
        self._height_override_mm = plan.override_height_above_pcb_mm
        return True

    def _apply_height_override(self):
        if not self.ui.chkHeightOverride.isChecked():
            self._height_override_mm = None
            self._refresh_height_ui()
            return
        if self._commit_override_from_spinbox():
            self._refresh_height_ui()

    def _set_pcb_surface(self):
        if not self.ui.chkHeightReference.isChecked():
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                "Position the probe over a component-free PCB reference area, "
                "then confirm that before setting the surface.",
            )
            return
        gap = self.ui.heightGapMm.value()
        try:
            reported_z = self._printer().get_xyz()[2]
        except (engine_printer.PrinterError, OSError, RuntimeError, AttributeError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return
        self._pcb_surface_z = reported_z - gap
        self._pcb_surface_from_fixture = False
        self._last_commanded_scan_z = None
        self._invalidate_easy_verified_height()
        self._refresh_height_ui()

    def _jog_z(self, direction):
        if not self.ui.chkAllowSetupZ.isChecked() and not self._easy_mode_active():
            return
        step = self.ui.jogStep.value() * direction
        if (
            self._easy_mode_active()
            and self._pcb_surface_z is None
            and step < 0
        ):
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                "Z reference required. Set the PCB surface in Advanced before lowering.",
            )
            return
        try:
            printer = self._printer()
            _x, _y, current = printer.get_xyz()
            target = current + step
            if target < current and not self.ui.chkHeightClearance.isChecked():
                QMessageBox.warning(
                    self.ui,
                    DIALOG_TITLE,
                    "This Z move is toward the PCB. Confirm probe clearance first.",
                )
                return
            printer.move_z(target)
            self._last_commanded_scan_z = None
            self._refresh_height_ui()
        except engine_printer.PrinterReset as exc:
            self._clear_height_datum()
            self._clear_pcb_homing()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))

    def _drive_to_agreed_plane(self, printer, *, required=True):
        """Setup-only G1 Z to the stored requested plane. No XY.

        ``required`` False skips when there is no datum or requested height
        (Home XY / Start scan). Toward-PCB still needs the clearance tick.
        """
        plan = self._scan_height_plan()
        requested = plan.requested_height_above_pcb_mm
        if self._pcb_surface_z is None or requested is None:
            if not required:
                return False
            if self._pcb_surface_z is None:
                raise RuntimeError("Set the PCB surface datum first.")
            raise RuntimeError(
                "No requested scan height. Import component heights or enter an override."
            )
        target = engine_height.target_machine_z(self._pcb_surface_z, requested)
        current = printer.get_xyz()[2]
        toward = requested < engine_height.reported_height_above_pcb(
            current, self._pcb_surface_z
        )
        if toward and not self.ui.chkHeightClearance.isChecked():
            raise RuntimeError(
                "Moving to scan height lowers the probe toward the PCB. "
                "Confirm probe clearance first."
            )
        printer.move_z(target)
        reported = printer.get_xyz()[2]
        if not engine_height.logical_position_matches(reported, target):
            raise engine_printer.PrinterError(
                "Marlin-reported position did not reach the requested logical target"
            )
        self._last_commanded_scan_z = target
        return True

    def _move_to_scan_height(self):
        if not self.ui.chkAllowSetupZ.isChecked():
            return
        if self.ui.chkHeightOverride.isChecked() and not self._commit_override_from_spinbox():
            return
        try:
            self._drive_to_agreed_plane(self._printer(), required=True)
            self._refresh_height_ui()
        except engine_printer.PrinterReset as exc:
            self._clear_height_datum()
            self._clear_pcb_homing()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
        except (
            engine_height.HeightError,
            engine_printer.PrinterError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))

    def _confirm_board_reseated(self):
        if not self.ui.chkBoardReseated.isChecked():
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                "Confirm that the PCB has been physically flipped or reseated.",
            )
            return
        self._clear_height_datum()
        self.ui.chkBoardReseated.setChecked(False)
        self._refresh_height_ui()

    def _clear_board_zero(self):
        """Every Home invalidates board zero; this does not recalibrate it."""
        if self._board_zero_committed() or self._easy_verified_scan_z is not None:
            self._easy_height_lost_to_reset = True
        self._bltouch_board_zero_z = None
        self._bltouch_board_zero_frame = None
        if self._pcb_surface_from_fixture:
            self._pcb_surface_z = None
            self._pcb_surface_from_fixture = False
        self._last_commanded_scan_z = None
        self._invalidate_easy_verified_height()

    def _board_zero_committed(self):
        return self._bltouch_board_zero_z is not None

    def _invalidate_easy_verified_height(self):
        self._easy_verified_scan_z = None
        self._easy_height_identity = None

    def _bind_printer_frame(self, printer):
        """Copy this session's logical-Z generation onto a (possibly new) Printer."""
        printer.z_logical_frame = int(self._z_logical_frame)
        return printer

    def _capture_printer_frame(self, printer):
        """Persist G92/home frame changes and drop a stale board-zero millimetre."""
        self._z_logical_frame = int(getattr(printer, "z_logical_frame", self._z_logical_frame))
        if (
            self._bltouch_board_zero_z is not None
            and self._bltouch_board_zero_frame is not None
            and int(self._z_logical_frame) != int(self._bltouch_board_zero_frame)
        ):
            self._clear_board_zero()

    def _apply_emi_surface_from_board_zero(self):
        """PCB top from P1 contact, support ledge, thickness, and vertical E-probe cal."""
        if self._bltouch_board_zero_z is None:
            return
        if engine_glassboard_fixture is None:
            self._pcb_surface_z = None
            self._pcb_surface_from_fixture = False
            return
        document = self._machine_fixture_document() or {}
        cals = engine_machine_fixtures.fixture_calibrations(document)
        surface = engine_glassboard_fixture.pcb_surface_from_p1_contact_mm(
            self._bltouch_board_zero_z,
            pcb_thickness_mm=cals.get("pcb_thickness_mm"),
            e_probe_tip_z_minus_g30_contact_mm=cals.get(
                "e_probe_tip_z_minus_g30_contact_mm"
            ),
        )
        if surface is None:
            if self._pcb_surface_from_fixture:
                self._pcb_surface_z = None
                self._pcb_surface_from_fixture = False
            return
        self._pcb_surface_z = surface
        self._pcb_surface_from_fixture = True

    def _commit_board_zero(self, result):
        self._bltouch_board_zero_z = float(result.board_zero_logical_z_mm)
        self._bltouch_board_zero_frame = int(result.z_logical_frame)
        self._z_logical_frame = int(result.z_logical_frame)
        self._apply_emi_surface_from_board_zero()

    def _clear_height_datum(self):
        had_datum = (
            self._pcb_surface_z is not None
            or self._board_zero_committed()
            or self._easy_verified_scan_z is not None
        )
        self._pcb_surface_z = None
        self._pcb_surface_from_fixture = False
        self._last_commanded_scan_z = None
        self._easy_touch_logical_z = None
        self._bltouch_board_zero_z = None
        self._bltouch_board_zero_frame = None
        self._invalidate_easy_verified_height()
        if had_datum:
            self._easy_height_lost_to_reset = True

    def _easy_scan_height_set_text(self):
        gap = self._default_probe_gap_mm()
        return f"Scan height set: {gap:.2f} mm above PCB"

    def _easy_scan_height_clearance_acknowledged(self):
        """Operator tick only. CAD boxes and emi_probe_offset never authorize Z."""
        clear = getattr(self.ui, "chkFixtureProbeClear", None)
        return clear is not None and bool(clear.isChecked())

    def _easy_resolved_z_calibrations(self):
        error = getattr(self, "_fixture_config_error", "") or ""
        if error:
            return {
                "pcb_thickness_mm": None,
                "e_probe_tip_z_minus_g30_contact_mm": None,
                "error": error,
            }
        document = self._machine_fixture_document() or {}
        cals = engine_machine_fixtures.fixture_calibrations(document)
        cals["error"] = document.get("yaml_calibration_error") or ""
        return cals

    def _easy_height_identity_now(self):
        cals = self._easy_resolved_z_calibrations()
        ledge = None
        if engine_glassboard_fixture is not None:
            ledge = float(engine_glassboard_fixture.PCB_BOTTOM_ABOVE_HOLDER_TOP_MM)
        contact = self._bltouch_board_zero_z
        return (
            cals.get("pcb_thickness_mm"),
            cals.get("e_probe_tip_z_minus_g30_contact_mm"),
            ledge,
            float(self._default_probe_gap_mm()),
            self._fixture_profile_name(),
            None if contact is None else round(float(contact), 4),
            self._bltouch_board_zero_frame,
            int(self._z_logical_frame),
        )

    def _easy_verified_height_applies(self):
        if self._easy_verified_scan_z is None or self._pcb_surface_z is None:
            return False
        if self._easy_height_identity != self._easy_height_identity_now():
            return False
        target = self._easy_scan_target_z()
        if target is None:
            return False
        return engine_height.logical_position_matches(
            self._easy_verified_scan_z, target
        )

    def _easy_p1_in_current_frame(self):
        return (
            self._bltouch_board_zero_z is not None
            and self._bltouch_board_zero_frame is not None
            and int(self._z_logical_frame) == int(self._bltouch_board_zero_frame)
        )

    def _easy_invalid_yaml_reason(self):
        if not self._easy_mode_active():
            return ""
        cals = self._easy_resolved_z_calibrations()
        error = cals.get("error") or getattr(self, "_fixture_config_error", "") or ""
        if not error:
            return ""
        return (
            "Step 2 incomplete: config.yaml fixture calibration is invalid "
            f"({error})."
        )

    def _easy_vertical_calibrated(self):
        """True only when a finite vertical offset exists. Unset is not 0."""
        cals = self._easy_resolved_z_calibrations()
        return cals.get("e_probe_tip_z_minus_g30_contact_mm") is not None

    def _easy_height_ready(self):
        return not self._easy_height_block_reason()

    def _easy_height_block_reason(self):
        """Easy Step 2 Next/banner/tooltip. Never talks to the printer."""
        yaml_reason = self._easy_invalid_yaml_reason()
        if yaml_reason:
            return yaml_reason
        cals = self._easy_resolved_z_calibrations()
        if cals.get("pcb_thickness_mm") is None:
            return EASY_HEIGHT_MISSING_THICKNESS
        if self._easy_scan_plane_cached():
            return ""
        if self._easy_height_lost_to_reset:
            return EASY_HEIGHT_RESET
        if self._p1_bltouch_commanded_xy() is None:
            return EASY_HEIGHT_MISSING_P1
        if not self._easy_vertical_calibrated():
            if self._pcb_xy_homed:
                return EASY_HEIGHT_NEED_MANUAL_PLANE
            return EASY_HEIGHT_NEED_BOARD_ZERO
        if not self._easy_p1_in_current_frame():
            if (
                self._bltouch_board_zero_z is None
                or self._bltouch_board_zero_frame is None
            ):
                return EASY_HEIGHT_NEED_BOARD_ZERO
            return EASY_HEIGHT_RESET
        if not self._easy_scan_height_clearance_acknowledged():
            return EASY_HEIGHT_CLEARANCE
        return EASY_HEIGHT_NOT_REACHED

    def _easy_scan_plane_cached(self):
        """True when this session already commanded/verified a scan Z. No M114."""
        if self._easy_verified_height_applies():
            return True
        target = self._easy_scan_target_z()
        commanded = self._last_commanded_scan_z
        return (
            target is not None
            and commanded is not None
            and engine_height.logical_position_matches(commanded, target)
        )

    def _try_finish_easy_scan_height(self, printer):
        """Move to scan Z at P1 after contact+retract. Caller owns the job."""
        self._invalidate_easy_verified_height()
        if not self._easy_mode_active():
            return
        if self._easy_invalid_yaml_reason():
            return
        if not self._easy_vertical_calibrated():
            return
        if not self._easy_scan_height_clearance_acknowledged():
            return
        self._apply_emi_surface_from_board_zero()
        target = self._easy_scan_target_z()
        if target is None:
            return
        try:
            printer.move_z(target)
        except engine_printer.PrinterReset:
            raise
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError):
            return
        reported = printer.get_xyz()[2]
        if engine_height.logical_position_matches(reported, target):
            self._last_commanded_scan_z = target
            self._easy_verified_scan_z = float(reported)
            self._easy_height_identity = self._easy_height_identity_now()
            self._easy_height_lost_to_reset = False

    def _step2_status_text(self):
        if self._easy_mode_active():
            if not self._easy_height_ready():
                return self._easy_height_block_reason()
            if self._easy_scan_plane_cached():
                return self._easy_scan_height_set_text()
            return BOARD_ZERO_COMPLETE_TEXT
        if self._board_zero_committed():
            return BOARD_ZERO_COMPLETE_TEXT
        return EASY_HEIGHT_MISSING_P1

    # ------------------------------------------------ fixture XYZ teaching
    def _fixture_profile_name(self):
        combo = getattr(self.ui, "machineFixtureProfile", None)
        if combo is not None:
            text = combo.currentText().strip()
            if text:
                return text
        return str(getattr(self, "_active_machine_fixture_name", "") or "").strip()

    def _raw_machine_fixture_document(self):
        name = self._fixture_profile_name()
        if not name:
            return None
        try:
            return engine_machine_fixtures.read_fixture(name)
        except (engine_profiles.ProfileError, OSError):
            return None

    def _fixture_config(self):
        """Carriage/PCB constants from ``EMI_Mapper/config.yaml``, cached.

        Step 2 no longer types these in, so the config file is the only YAML
        source. Blank keys do not override a saved fixture document. Invalid
        values are an error and do not fall back to saved calibration. A
        missing file is treated as blank keys.
        """
        cached = getattr(self, "_fixture_config_cache", None)
        if cached is not None:
            return cached
        config = engine_config.FixtureConfig()
        self._fixture_config_error = ""
        try:
            from EMI_Mapper.emi_mapper import DEFAULT_CONFIG, load_config

            config = load_config(DEFAULT_CONFIG).fixture
        except FileNotFoundError:
            config = engine_config.FixtureConfig()
        except Exception as exc:  # unreadable yaml or invalid values
            self._fixture_config_error = str(exc)
            config = engine_config.FixtureConfig()
        self._fixture_config_cache = config
        return config

    def _machine_fixture_document(self):
        document = self._raw_machine_fixture_document()
        if document is None:
            return None
        return self._overlay_fixture_config(document)

    def _overlay_fixture_config(self, document):
        """Apply config.yaml as optional overrides onto a stored fixture.

        Blank YAML keys leave saved thickness/vertical calibration in place.
        Explicit finite YAML values (including 0) win. Invalid YAML clears Z
        calibrations rather than falling back to the saved document. The
        E-probe XY offset may be shown as a diagnostic; it does not authorize
        scan-height motion and is not A/B registration.
        """
        if not isinstance(document, dict):
            return document
        fixture = self._fixture_config()
        return engine_machine_fixtures.overlay_yaml_fixture_values(
            document,
            pcb_thickness_mm=fixture.pcb_thickness_mm,
            e_probe_tip_z_minus_g30_contact_mm=(
                fixture.e_probe_tip_z_minus_g30_contact_mm
            ),
            pcb_yaw_deg=fixture.pcb_yaw_deg,
            emi_probe_offset_mm=fixture.emi_probe_offset_mm,
            yaml_error=getattr(self, "_fixture_config_error", "") or "",
        )

    def _reset_fixture_z_calibrations(self):
        """Clear saved thickness/vertical. Blank YAML is not this action."""
        name = self._ensure_fixture_selected()
        if not name:
            raise RuntimeError("Select a fixture before clearing Z calibration.")
        fixture = self._fixture_config()
        document = engine_machine_fixtures.clear_z_calibrations(name)
        self._invalidate_easy_verified_height()
        self._fixture_z_calibrations_cleared = True
        message = engine_machine_fixtures.yaml_override_after_z_calibration_clear(
            pcb_thickness_mm=fixture.pcb_thickness_mm,
            e_probe_tip_z_minus_g30_contact_mm=(
                fixture.e_probe_tip_z_minus_g30_contact_mm
            ),
        )
        status = "Saved fixture thickness and vertical calibration were cleared."
        if message:
            status = f"{status} {message}"
            QMessageBox.warning(self.ui, DIALOG_TITLE, status)
        fixture_status = getattr(self.ui, "fixtureTeachStatus", None)
        if fixture_status is not None:
            fixture_status.setText(status)
        self._render_emi_probe_offset()
        self._refresh_easy_height_label()
        self._update_nav()
        return document

    def _refresh_machine_fixtures(self):
        combo = getattr(self.ui, "machineFixtureProfile", None)
        if combo is None:
            return
        summaries = [
            item for item in engine_machine_fixtures.list_fixtures()
            if not item.get("error")
        ]
        names = [item["name"] for item in summaries]
        last = engine_machine_fixtures.last_selected_fixture_name()
        selected = (
            combo.currentText().strip()
            or self._active_machine_fixture_name
            or last
        )
        combo.blockSignals(True)
        combo.clear()
        for name in names:
            combo.addItem(name)
        index = combo.findText(selected) if selected else -1
        if index < 0 and names:
            taught = [
                item for item in summaries
                if item.get("taught")
            ]
            taught.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
            preferred = taught[0]["name"] if taught else names[0]
            index = combo.findText(preferred)
        if index >= 0:
            combo.setCurrentIndex(index)
            self._active_machine_fixture_name = combo.itemText(index)
            engine_machine_fixtures.remember_selected_fixture(
                self._active_machine_fixture_name
            )
        combo.blockSignals(False)
        self._on_machine_fixture_changed(self._active_machine_fixture_name)

    def _ensure_fixture_selected(self):
        """Return a fixture name, selecting the only listed one if the combo is empty.

        Does not re-render teaching: CAPTURE already holds P1 in memory and
        must not have that wiped before it is written to disk.
        """
        name = self._fixture_profile_name() or self._active_machine_fixture_name
        if name:
            return name
        combo = getattr(self.ui, "machineFixtureProfile", None)
        if combo is None:
            return ""
        if combo.count() == 0:
            for item in engine_machine_fixtures.list_fixtures():
                if not item.get("error"):
                    combo.addItem(item["name"])
        if combo.count() > 0 and combo.currentIndex() < 0:
            blocker = QtCore.QSignalBlocker(combo)
            combo.setCurrentIndex(0)
            del blocker
        self._active_machine_fixture_name = (
            combo.currentText().strip() if combo.count() else ""
        )
        return self._active_machine_fixture_name

    def _create_machine_fixture(self):
        name, accepted = QtWidgets.QInputDialog.getText(
            self.ui, DIALOG_TITLE, "Name this physical fixture:"
        )
        if not accepted or not name.strip():
            return
        try:
            engine_machine_fixtures.create_fixture(
                name,
                fixture_id=name,
                machine_id=getattr(self.ui.machineId, "text", lambda: "")(),
                probe_setup_id=getattr(self.ui.probeModel, "currentText", lambda: "")(),
            )
        except (engine_profiles.ProfileError, OSError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return
        self._refresh_machine_fixtures()
        index = self.ui.machineFixtureProfile.findText(name.strip())
        if index >= 0:
            self.ui.machineFixtureProfile.setCurrentIndex(index)

    def _on_machine_fixture_changed(self, _name=None):
        selected = self._fixture_profile_name()
        if (
            self._active_machine_fixture_name
            and selected != self._active_machine_fixture_name
        ):
            self._clear_landmarks_and_alignment()
            self.ui.chkBoardSeated.setChecked(False)
            self._clear_board_zero()
        self._active_machine_fixture_name = selected
        if selected:
            engine_machine_fixtures.remember_selected_fixture(selected)
        self._fixture_teaching_points = {}
        self._fixture_verified_session = False
        document = self._machine_fixture_document()
        self._render_fixture_teaching(
            document.get("fixture_teaching") if document else None
        )
        self._sync_board_profile_to_fixture()
        self._refresh_scan_setup_summary()
        self._update_nav()

    def _sync_board_profile_to_fixture(self):
        """Select the saved board alignment with the same fixture name when present."""
        combo = getattr(self.ui, "fixtureProfile", None)
        name = self._fixture_profile_name()
        if combo is None or not name:
            return False
        linked = ""
        document = self._machine_fixture_document()
        if document is not None and self._board_view is not None:
            linked = engine_machine_fixtures.board_alignment(
                document, self._board_view.geometry_hash
            )
        index = combo.findText(linked or name)
        if index < 0:
            return False
        combo.setCurrentIndex(index)
        return True

    def _glassboard_fixture_selected(self):
        document = self._machine_fixture_document() or {}
        identity = " ".join(
            (str(document.get("profile_name", "")), str(document.get("fixture_id", "")))
        ).lower()
        return "glass" in identity

    def _fixture_gate_ready(self):
        """Step 2 Next: Easy height ready, or Advanced board-zero contact."""
        if self._easy_mode_active():
            return self._easy_height_ready()
        return self._board_zero_committed()

    def _fixture_scan_ready(self):
        """Taught P1 XY may be reused; Fast Verify is optional. Session Z is not."""
        document = self._machine_fixture_document() or {}
        if not document.get("fixture_teaching"):
            return True
        return self._p1_bltouch_commanded_xy() is not None

    def _fixture_home_xyz_clicked(self):
        if self._printer_halted:
            QMessageBox.warning(self.ui, DIALOG_TITLE, PRINTER_HALTED_TEXT)
            return
        if not self.ui.chkFixtureProbeClear.isChecked():
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                "Confirm Z-lift, X/Y travel, P1 scan-height, and P1-to-A/B "
                "clearance first.",
            )
            return
        if self._p1_bltouch_commanded_xy() is not None:
            blocked = self._easy_invalid_yaml_reason()
            if blocked:
                QMessageBox.warning(self.ui, DIALOG_TITLE, blocked)
                return
        self._clear_board_zero()
        self._update_nav()
        p1 = self._p1_bltouch_commanded_xy() is not None
        status = (
            "Homing X/Y, probing P1, then setting board zero…"
            if p1
            else "Lifting Z, then homing X/Y for teaching…"
        )
        self._start_printer_job(
            self._do_home_xy,
            self._on_home_ok,
            self._on_home_failed,
            self.ui.fixtureTeachStatus,
            status,
        )

    def _fixture_probe_selected_clicked(self):
        if not self._fixture_xyz_homed:
            QMessageBox.warning(self.ui, DIALOG_TITLE, "Home XY for Teaching on this page first.")
            return
        if not self.ui.chkFixtureProbeClear.isChecked():
            QMessageBox.warning(self.ui, DIALOG_TITLE, "Confirm probing clearance first.")
            return
        row = self.ui.fixtureTeachTable.currentRow()
        if row < 0:
            row = 0
        name = TEACHING_POINT_NAMES[row]
        work = partial(self._do_fixture_probe_point, name)
        self._start_printer_job(
            work,
            self._on_fixture_probe_ok,
            self._on_fixture_operation_failed,
            self.ui.fixtureTeachStatus,
            f"Probing {name}…",
        )

    def _on_fixture_point_selected(self, row, _column=0, *_args):
        if 0 <= row < len(TEACHING_POINT_NAMES):
            name = TEACHING_POINT_NAMES[row]
            self.ui.btnFixtureProbePoint.setText(f"CAPTURE {name} — PROBE AND SAVE XYZ")

    def _fixture_jog(self, dx, dy):
        if not self._fixture_xyz_homed:
            QMessageBox.warning(self.ui, DIALOG_TITLE, "Lift Z and home X/Y before jogging.")
            return
        try:
            printer = self._printer(manage_z=True)
            x, y = printer.get_xy()
            step = self.ui.fixtureJogStep.value()
            printer.move_xy(x + dx * step, y + dy * step)
            self._bump_motion()
            self._show_position(printer.get_xy())
        except engine_printer.PrinterReset as exc:
            self._on_fixture_operation_failed(exc)
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))

    def _do_fixture_probe_point(self, name):
        printer = self._bind_printer_frame(self._printer(manage_z=True))
        retract_mm = self._printer_config().probe_clearance_mm
        try:
            commanded_x, commanded_y, _current_z = printer.get_xyz()
            result = printer.probe_pin_until_contact(retract_mm=retract_mm)
            fixture_x, fixture_y = engine_glassboard_fixture.REFERENCE_FIXTURE_XY[name]
            point = engine_fixture_teaching.FixtureReferencePoint(
                name, fixture_x, fixture_y,
                commanded_x, commanded_y,
                result.probe_x_mm, result.probe_y_mm,
                result.touch_z_raw_mm, result.touch_z_raw_mm,
                z_datum=engine_fixture_teaching.Z_DATUM_M119_Z_PROBE,
            )
            board_zero = None
            if result.logical_surface_z_mm is not None:
                try:
                    board_zero = engine_printer.board_zero_from_pin_contact(
                        printer,
                        (commanded_x, commanded_y),
                        result,
                        retract_mm=retract_mm,
                    )
                except (engine_printer.ProbeFailed, engine_printer.PrinterError):
                    board_zero = None
            if board_zero is not None:
                self._commit_board_zero(board_zero)
                self._try_finish_easy_scan_height(printer)
            return {"point": point, "board_zero": board_zero}
        finally:
            self._capture_printer_frame(printer)

    def _on_fixture_probe_ok(self, payload):
        if isinstance(payload, engine_fixture_teaching.FixtureReferencePoint):
            point, board_zero = payload, None
        else:
            point = payload["point"]
            board_zero = payload.get("board_zero")
        self._fixture_teaching_points[point.name] = point
        row = TEACHING_POINT_NAMES.index(point.name)
        table = getattr(self.ui, "fixtureTeachTable", None)
        if table is not None:
            for column, value in enumerate(
                (point.commanded_machine_x, point.commanded_machine_y, point.touch_z_raw), start=1
            ):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(f"{value:.3f}"))
            table.setItem(row, 4, QtWidgets.QTableWidgetItem("Not checked"))
        status = (
            f"{point.name} captured at machine X {point.commanded_machine_x:.3f}, "
            f"Y {point.commanded_machine_y:.3f}, contact Z {point.touch_z_raw:.3f}."
        )
        if board_zero is not None:
            self._commit_board_zero(board_zero)
            status = f"{status}\n{self._step2_status_text()}"
        else:
            status = (
                f"{status}\nP1 XY is captured, but board zero is unset "
                "(M114 did not follow the pin)."
            )
        self.ui.fixtureTeachStatus.setText(status)
        if self._persist_taught_p1():
            self.ui.fixtureTeachStatus.setText(
                f"{self.ui.fixtureTeachStatus.text()}\n"
                "P1 saved on this fixture for the next measurement."
            )
        else:
            self.ui.fixtureTeachStatus.setText(
                f"{self.ui.fixtureTeachStatus.text()}\n"
                "P1 is not saved on the fixture yet."
            )
        self._easy_try_load_profile()
        self._update_easy_status()
        self._refresh_height_ui()
        self._update_nav()

    def _persist_taught_p1(self):
        """Write captured P1 to the fixture file. Capture already means save."""
        profile_name = self._ensure_fixture_selected()
        if not profile_name:
            QMessageBox.warning(
                self.ui, DIALOG_TITLE,
                "Select a fixture on Step 2 so P1 can be reused next time.",
            )
            return False
        try:
            points = [self._fixture_teaching_points[name] for name in TEACHING_POINT_NAMES]
        except KeyError:
            return False
        try:
            teaching = engine_fixture_teaching.build_fixture_teaching(
                points,
                max_z_change_mm=self.ui.fixtureMaxZChange.value(),
                max_plane_residual_mm=self.ui.fixtureMaxResidual.value(),
                firmware="Marlin M119 z_probe", probe="BLTouch",
            )
            engine_machine_fixtures.save_teaching(
                profile_name, teaching, overwrite=True
            )
        except (engine_profiles.ProfileError, engine_fixture_teaching.FixtureTeachingError, OSError) as exc:
            QMessageBox.warning(
                self.ui, DIALOG_TITLE,
                f"P1 is usable now, but could not be saved for next time: {exc}",
            )
            return False
        engine_machine_fixtures.remember_selected_fixture(profile_name)
        return True

    def _fixture_save_plane(self):
        if not self._persist_taught_p1():
            if "P1" not in self._fixture_teaching_points:
                QMessageBox.warning(self.ui, DIALOG_TITLE, "Capture P1 first.")
            return False
        document = self._machine_fixture_document() or {}
        self._render_fixture_teaching(
            document.get("fixture_teaching"),
            "PASS — P1 saved on this fixture",
        )
        self._update_nav()
        return True

    def _fixture_verify_clicked(self):
        if not self.ui.chkFixtureProbeClear.isChecked():
            QMessageBox.warning(self.ui, DIALOG_TITLE, "Confirm Z-lift and X/Y probing clearance first.")
            return
        document = self._machine_fixture_document() or {}
        if not document.get("fixture_teaching"):
            QMessageBox.warning(self.ui, DIALOG_TITLE, "This fixture has not been taught.")
            return
        work = partial(
            self._do_fixture_verify,
            self.ui.fixtureMaxZChange.value(),
            self.ui.fixtureMaxResidual.value(),
        )
        status = (
            "Re-probing saved P1…"
            if self._fixture_xyz_homed
            else "Lifting Z once, homing X/Y, then re-probing P1…"
        )
        self._start_printer_job(
            work,
            self._on_fixture_verify_ok,
            self._on_fixture_operation_failed,
            self.ui.fixtureTeachStatus,
            status,
        )

    def _do_fixture_verify(self, max_z_change_mm, max_plane_residual_mm):
        profile_name = self._fixture_profile_name()
        document = engine_machine_fixtures.read_fixture(profile_name)
        teaching = document["fixture_teaching"]
        printer = self._bind_printer_frame(self._printer(manage_z=True))
        printer.drain()
        printer.prepare()
        try:
            if not self._fixture_xyz_homed:
                printer.home_xyz(self._safe_home_lift_mm())
            current = []
            for data in teaching["points"]:
                saved = engine_fixture_teaching.FixtureReferencePoint.from_dict(data)
                printer.move_xy(saved.commanded_machine_x, saved.commanded_machine_y)
                result = printer.probe_pin_until_contact(
                    retract_mm=self._printer_config().probe_clearance_mm,
                )
                current.append(engine_fixture_teaching.FixtureReferencePoint(
                    saved.name, saved.fixture_x, saved.fixture_y,
                    saved.commanded_machine_x, saved.commanded_machine_y,
                    result.probe_x_mm, result.probe_y_mm,
                    result.touch_z_raw_mm, result.touch_z_raw_mm,
                    z_datum=engine_fixture_teaching.Z_DATUM_M119_Z_PROBE,
                ))
            document = engine_machine_fixtures.save_verification(
                profile_name,
                current,
                max_z_change_mm=max_z_change_mm,
                max_plane_residual_mm=max_plane_residual_mm,
            )
            return document, document["fixture_teaching"]["latest_verification"]
        finally:
            self._capture_printer_frame(printer)

    def _on_fixture_verify_ok(self, payload):
        document, result = payload
        self._fixture_xyz_homed = True
        self._pcb_xy_homed = True
        self._fixture_verified_session = bool(result["passed"])
        verdict = (
            "PASS — P1 re-probed"
            if result["passed"]
            else f"BLOCKED — {result.get('failure_reason') or 'fixture moved or tilted'}"
        )
        self._render_fixture_teaching(document["fixture_teaching"], verdict)
        self._update_nav()

    def _render_fixture_teaching(self, teaching, verdict=None):
        if not hasattr(self.ui, "fixtureTeachStatus"):
            return
        if not hasattr(self.ui, "fixtureTeachTable"):
            return
        if not teaching:
            self._fixture_teaching_points = {}
            for row in range(len(TEACHING_POINT_NAMES)):
                for column in range(1, 5):
                    item = QtWidgets.QTableWidgetItem("Pending")
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                    self.ui.fixtureTeachTable.setItem(row, column, item)
            self.ui.fixtureMaxZChange.setValue(
                engine_fixture_teaching.DEFAULT_MAX_Z_CHANGE_MM
            )
            self.ui.fixtureMaxResidual.setValue(
                engine_fixture_teaching.DEFAULT_MAX_PLANE_RESIDUAL_MM
            )
            self.ui.fixtureTeachStatus.setText("Fixture not taught.")
            self.ui.fixtureTeachStatus.setStyleSheet(
                "padding:8px;border-radius:6px;background:#F2F4F7"
            )
            self._refresh_fixture_page_summary()
            self._render_emi_probe_offset()
            return
        self._hydrate_teaching_points(teaching)
        plane = teaching["plane"]
        points = teaching["points"]
        by_name = {
            item.get("name"): item
            for item in points
            if isinstance(item, dict)
        }
        for row, name in enumerate(TEACHING_POINT_NAMES):
            data = by_name.get(name)
            if data is None:
                continue
            for column, key in (
                (1, "commanded_machine_x"), (2, "commanded_machine_y"),
                (3, "touch_z_raw"),
            ):
                self.ui.fixtureTeachTable.setItem(
                    row, column, QtWidgets.QTableWidgetItem(f"{float(data[key]):.3f}")
                )
        verification = teaching.get("latest_verification") or {}
        differences = {
            item.get("name"): item
            for item in verification.get("differences", ())
            if isinstance(item, dict)
        }
        for row, name in enumerate(TEACHING_POINT_NAMES):
            data = by_name.get(name)
            if data is None:
                continue
            difference = differences.get(name)
            if difference:
                delta = float(
                    difference.get(
                        "relative_z_change_mm", difference.get("z_change_mm", 0.0)
                    )
                )
                current = float(difference.get("current_touch_z_raw", 0.0))
                item = QtWidgets.QTableWidgetItem(f"{current:.3f}  /  {delta:+.3f}")
                item.setToolTip(
                    "Latest contact Z / point change after removing the "
                    "common session Z-reference shift"
                )
            else:
                item = QtWidgets.QTableWidgetItem("Not checked")
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.ui.fixtureTeachTable.setItem(row, 4, item)
        tolerances = teaching.get("verification_tolerances_mm", {})
        saved_z_limit = float(tolerances.get("max_z_change", 0.20))
        # An early Step 2 build could persist the spin-box floor (0.010 mm),
        # which is below useful fixture/probe repeatability. Show the documented
        # recommendation; the next Verify persists this visible value.
        if saved_z_limit < 0.05:
            saved_z_limit = engine_fixture_teaching.DEFAULT_MAX_Z_CHANGE_MM
        self.ui.fixtureMaxZChange.setValue(saved_z_limit)
        self.ui.fixtureMaxResidual.setValue(float(tolerances.get("max_plane_residual", 0.10)))
        layout = " · ".join(
            f"{p['name']} ({p['commanded_machine_x']:.2f},{p['commanded_machine_y']:.2f})"
            for p in points
        )
        verification_text = ""
        if verification:
            verification_text = (
                f"\nLatest check: common Z reference shift "
                f"{verification.get('surface_shift_mm', 0.0):+.3f} mm; "
                f"max point / tilt change "
                f"{verification.get('max_relative_z_change_mm', verification.get('max_z_change_mm', 0.0)):.3f} mm; "
                f"residual {verification.get('plane', {}).get('max_residual_mm', 0.0):.3f} mm"
            )
        self.ui.fixtureTeachStatus.setText(
            f"{verdict or 'Fixture taught'}\nReference map: {layout}\n"
            f"Z = {plane['a']:.6f}X + {plane['b']:.6f}Y + {plane['c']:.6f}; "
            f"max residual {plane['max_residual_mm']:.3f} mm"
            f"{verification_text}"
        )
        if verdict and verdict.startswith("PASS"):
            self.ui.fixtureTeachStatus.setStyleSheet(
                "padding:10px;border:1px solid #12B76A;border-radius:6px;"
                "background:#ECFDF3;color:#05603A"
            )
        elif verdict and verdict.startswith("BLOCKED"):
            self.ui.fixtureTeachStatus.setStyleSheet(
                "padding:10px;border:1px solid #F04438;border-radius:6px;"
                "background:#FEF3F2;color:#912018"
            )
        else:
            self.ui.fixtureTeachStatus.setStyleSheet(
                "padding:8px;border-radius:6px;background:#F2F4F7"
            )
        self._refresh_fixture_page_summary()
        self._render_emi_probe_offset()

    def _hydrate_teaching_points(self, teaching):
        """Reload in-memory P1 from the fixture file so the next session has it."""
        loaded = {}
        for item in (teaching or {}).get("points") or ():
            if not isinstance(item, dict):
                continue
            try:
                point = engine_fixture_teaching.FixtureReferencePoint.from_dict(item)
            except engine_fixture_teaching.FixtureTeachingError:
                continue
            if point.name in TEACHING_POINT_NAMES:
                loaded[point.name] = point
        if loaded:
            self._fixture_teaching_points = loaded

    def _p1_bltouch_commanded_xy(self):
        point = self._fixture_teaching_points.get("P1")
        if point is not None:
            return float(point.commanded_machine_x), float(point.commanded_machine_y)
        document = self._machine_fixture_document() or {}
        for item in (document.get("fixture_teaching") or {}).get("points") or ():
            if item.get("name") == "P1":
                return (
                    float(item["commanded_machine_x"]),
                    float(item["commanded_machine_y"]),
                )
        return None

    def _on_fixture_motion_failed(self, exc):
        """Refusals are operator-correctable; anything else resets fixture state."""
        if isinstance(exc, engine_printer.PrinterReset):
            self._on_fixture_operation_failed(exc)
            return
        if isinstance(exc, _FixtureMotionRefused):
            self.ui.fixtureTeachStatus.setText(str(exc))
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return
        self._on_fixture_operation_failed(exc)

    def _fixture_clear_point_clicked(self):
        """Forget taught P1 so it can be re-jogged and captured again.

        Board zero and the calculated scan height are both derived from P1, so
        clearing it must invalidate them rather than leave a datum pointing at
        a point the operator has discarded. No motion is commanded.
        """
        name = self._fixture_profile_name()
        stored = bool((self._machine_fixture_document() or {}).get("fixture_teaching"))
        if not self._fixture_teaching_points and not stored:
            QMessageBox.information(
                self.ui, DIALOG_TITLE, "There is no taught P1 to clear."
            )
            return
        if (
            QMessageBox.question(
                self.ui,
                DIALOG_TITLE,
                "Forget taught P1?\n\n"
                "Board zero and the calculated scan height are derived from it, "
                "so both are cleared and Step 2 becomes incomplete.",
            )
            != QMessageBox.Yes
        ):
            return
        self._fixture_teaching_points = {}
        self._fixture_verified_session = False
        self._clear_board_zero()
        self._clear_height_datum()
        self._stl_board_placement = None
        self._invalidate_plan()
        if stored and name:
            try:
                engine_machine_fixtures.clear_teaching(name)
            except (engine_profiles.ProfileError, OSError, ValueError) as exc:
                QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
        self._render_fixture_teaching(None)
        self.ui.fixtureTeachStatus.setText(
            "P1 cleared. Home XY for Teaching, jog the BLTouch to the mark, "
            "then CAPTURE P1."
        )
        self._update_nav()

    def _render_emi_probe_offset(self):
        """Show the config-supplied constants Step 2 no longer edits.

        Read-only on purpose: these describe the carriage and the seated board,
        so a per-fixture edit here would silently disagree with config.yaml.
        """
        note = getattr(self.ui, "fixtureConfigNote", None)
        if note is None:
            return
        fixture = self._fixture_config()
        dx, dy = fixture.emi_probe_offset_mm
        lines = [
            f"From EMI_Mapper/config.yaml — E-probe offset {dx:+.3f}, {dy:+.3f} mm "
            "(diagnostic only; not A/B registration and not Z clearance). "
            f"PCB yaw {float(fixture.pcb_yaw_deg):g}°."
        ]
        error = getattr(self, "_fixture_config_error", "")
        if error:
            lines.append(
                f"config.yaml fixture calibration is invalid ({error}); "
                "saved thickness/vertical values are not used."
            )
        else:
            lines.extend(self._effective_z_calibration_lines())
        if getattr(self, "_fixture_z_calibrations_cleared", False):
            remaining = engine_machine_fixtures.yaml_override_after_z_calibration_clear(
                pcb_thickness_mm=fixture.pcb_thickness_mm,
                e_probe_tip_z_minus_g30_contact_mm=(
                    fixture.e_probe_tip_z_minus_g30_contact_mm
                ),
            )
            if remaining:
                lines.append(remaining)
        note.setText(" ".join(lines))

    def _effective_z_calibration_lines(self):
        yaml_cfg = self._fixture_config()
        stored = engine_machine_fixtures.fixture_calibrations(
            self._raw_machine_fixture_document()
        )
        return [
            self._z_calibration_line(
                "PCB thickness",
                yaml_cfg.pcb_thickness_mm,
                stored.get("pcb_thickness_mm"),
            ),
            (
                self._z_calibration_line(
                    "Vertical E-probe calibration",
                    yaml_cfg.e_probe_tip_z_minus_g30_contact_mm,
                    stored.get("e_probe_tip_z_minus_g30_contact_mm"),
                )
                if (
                    yaml_cfg.e_probe_tip_z_minus_g30_contact_mm is not None
                    or stored.get("e_probe_tip_z_minus_g30_contact_mm") is not None
                )
                else EASY_VERTICAL_UNCALIBRATED
            ),
        ]

    def _z_calibration_line(self, label, yaml_value, stored_value):
        if yaml_value is not None:
            return f"{label} {float(yaml_value):.3f} mm (config.yaml override)."
        if stored_value is not None:
            return f"{label} {float(stored_value):.3f} mm (saved fixture)."
        return f"{label} is missing from config.yaml and the saved fixture."

    def _refresh_fixture_page_summary(self):
        if not hasattr(self.ui, "fixtureBoardSummary"):
            return
        document = self._machine_fixture_document() or {}
        profile = document.get("fixture_name") or self._fixture_profile_name() or "not selected"
        p1 = self._p1_bltouch_commanded_xy()
        persisted = bool((self._raw_machine_fixture_document() or {}).get("fixture_teaching"))
        if p1 is not None and persisted:
            p1_text = f"P1 XY saved ({p1[0]:.2f}, {p1[1]:.2f})"
        elif p1 is not None:
            p1_text = f"P1 XY captured, not saved ({p1[0]:.2f}, {p1[1]:.2f})"
        else:
            p1_text = "P1 XY not saved"
        self.ui.fixtureBoardSummary.setText(
            f"Fixture: {profile}  •  Machine: {document.get('machine_id') or 'current machine'} "
            f" •  Probe: {document.get('probe_setup_id') or 'BLTouch'}\n{p1_text}"
        )
        ready = self._fixture_gate_ready()
        if self.thread is not None or self._printer_job is not None:
            ready = False
            reason = "Wait for the current printer or scan job to finish."
        elif not ready:
            reason = self._fixture_step2_block_reason()
        else:
            reason = (
                self._easy_scan_height_set_text()
                if self._easy_mode_active() and self._easy_scan_plane_cached()
                else BOARD_ZERO_COMPLETE_TEXT
            )
        self.ui.fixtureReadySummary.setText(
            ("FIXTURE READY — " if ready else "Complete Step 2 — ") + reason
        )
        self.ui.fixtureReadySummary.setStyleSheet(
            "padding:10px;border-radius:6px;font-weight:600;"
            + (
                "background:#ECFDF3;color:#05603A;border:1px solid #12B76A"
                if ready
                else "background:#FFFAEB;color:#7A2E0E;border:1px solid #F79009"
            )
        )
        self._render_emi_probe_offset()

    def _refresh_scan_setup_summary(self):
        if not hasattr(self.ui, "scanSetupBoardSummary"):
            return
        source = Path(self.ui.boardPath.text()).name if self.ui.boardPath.text() else "not selected"
        side = self._board_view.side if self._board_view is not None else "—"
        self.ui.scanSetupBoardSummary.setText(
            f"Board file: {source}\nView: {side}  •  Fixture: {self._fixture_profile_name() or 'not selected'}"
        )
        self.ui.scanSetupAreaSummary.setText(f"Scan coverage: {self._grid_summary()}")
        problems = self._registration_problems()
        aligned = not problems
        seated = self.ui.chkBoardSeated.isChecked()
        measurement = "Spectrum cube" if self._cube_selected() else "Single-frequency map"
        if self._active_taught_side is not None:
            alignment_name = f"{self._active_taught_side} side from taught inner pocket"
        elif self._stl_board_placement is not None:
            alignment_name = "STL middle-holder position"
        else:
            alignment_name = "PCB position"
        ready = aligned and seated
        extra = ""
        if not aligned and problems:
            extra = "Blocked: " + "; ".join(problems)
        elif ready:
            extra = (
                "READY — continue to Review & Measure. Taught corners are saved "
                "on this fixture for the next measurement."
            )
        elif not seated:
            extra = "Confirm that the PCB is seated."
        self.ui.scanSetupReview.setText(
            f"{'✓' if aligned else '○'} {alignment_name}    "
            f"{'✓' if seated else '○'} PCB fully seated    "
            f"Measurement: {measurement}\n"
            + extra
        )
        self.ui.scanSetupReview.setStyleSheet(
            "padding:10px;border-radius:6px;font-weight:600;"
            + (
                "background:#ECFDF3;color:#05603A;border:1px solid #12B76A"
                if ready
                else "background:#FFFAEB;color:#7A2E0E;border:1px solid #F79009"
            )
        )

    def _refresh_scan_setup_locate_status(self):
        label = getattr(self.ui, "scanSetupLocateStatus", None)
        if label is None:
            return
        document = self._machine_fixture_document() or {}
        lines = []
        if self._stl_board_placement is not None:
            lines.append("Board located from fixture")
        if engine_machine_fixtures.emi_probe_offset_is_set(document):
            lines.append("E-probe offset calibrated")
        saved = [
            side
            for side in SCAN_SETUP_SIDES
            if self._taught_side_complete(side)
        ]
        if saved:
            lines.append(
                "Saved insert corners: " + ", ".join(saved) + " — reused next time."
            )
        label.setText("\n".join(lines))
        label.setVisible(bool(lines))

    def _board_corners_eprobe_xy(self):
        if self._board_view is None or self._registration is None:
            return ()
        x0, y0, x1, y1 = self._board_view.bbox_mm
        corners = self._scan_transform().to_machine(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        )
        return tuple((float(x), float(y)) for x, y in corners)

    def _calculated_scan_height_info(self):
        missing = {
            "available": False,
            "reason": getattr(
                engine_glassboard_fixture,
                "PROBE_HEIGHT_NOT_CALIBRATED",
                "Probe height not calibrated",
            ),
            "pcb_surface_machine_z": None,
            "target_clearance_mm": 3.0,
            "predicted_clearance_mm": None,
        }
        if engine_glassboard_fixture is None:
            return missing
        document = self._machine_fixture_document() or {}
        teaching = document.get("fixture_teaching")
        if not teaching or self._registration is None:
            return missing
        cals = engine_machine_fixtures.fixture_calibrations(document)
        corners = self._board_corners_eprobe_xy()
        if not corners:
            return missing
        return engine_glassboard_fixture.calculated_scan_height(
            teaching,
            emi_probe_offset_mm=engine_machine_fixtures.emi_probe_offset_mm(document),
            pcb_thickness_mm=cals["pcb_thickness_mm"],
            e_probe_tip_z_minus_g30_contact_mm=cals[
                "e_probe_tip_z_minus_g30_contact_mm"
            ],
            board_corners_eprobe_xy=corners,
            min_scan_clearance_mm=cals["min_scan_clearance_mm"],
            max_scan_clearance_mm=cals["max_scan_clearance_mm"],
            session_verified=self._fixture_verified_session,
        )

    def _refresh_calculated_height(self):
        info = self._calculated_scan_height_info()
        if self._board_zero_committed():
            self._apply_emi_surface_from_board_zero()
        elif not self._easy_mode_active() and info.get("available"):
            self._pcb_surface_z = info["pcb_surface_machine_z"]
            self._pcb_surface_from_fixture = True
        self._refresh_scan_setup_locate_status()
        self._refresh_scan_setup_summary()
        self._refresh_height_ui()
        self._refresh_scan_preview()
        self._update_nav()

    def _maybe_auto_place_glassboard(self):
        if self._easy_mode_active():
            return False
        if not self._glassboard_fixture_selected():
            return False
        if self._face_blocks_stl_fallback(self._scan_setup_side()):
            return False
        if self._board_model is None or not self._pcb_xy_homed:
            return False
        if not self._fixture_verified_session:
            return False
        document = self._machine_fixture_document() or {}
        if not engine_machine_fixtures.emi_probe_offset_is_set(document):
            return False
        if self._stl_board_placement is not None and self._registration is not None:
            return True
        if self._restore_glassboard_middle_placement():
            return True
        side = "top"
        combo = getattr(self.ui, "glassboardPlacementSide", None)
        if combo is not None:
            side = combo.currentData() or "top"
        return self._set_glassboard_middle_placement(
            side, persist=True, reset_seated=False, announce=False
        )

    def _preview_scan_plan(self):
        if self._board_view is None or self._registration is None:
            return None
        step_mm = self.ui.stepMm.value()
        x_start, x_end, y_start, y_end = _board_bounds(self._board_view, step_mm)
        return engine_registration.build_plan(
            self._board_view,
            self._scan_transform(),
            x_start=x_start,
            x_end=x_end,
            y_start=y_start,
            y_end=y_end,
            step_mm=step_mm,
            clip_to_outline=True,
            registration=self._registration,
            selection=self._scan_selection,
        )

    def _current_scan_fingerprint(self):
        if engine_scan_preview is None:
            return None
        widget = getattr(self, "_scan_preview_widget", None)
        if widget is not None:
            fingerprint = widget.fingerprint()
            if fingerprint:
                return fingerprint
        try:
            plan = self._plan if self._plan is not None else self._preview_scan_plan()
        except (ValueError, RuntimeError):
            return None
        if plan is None:
            return None
        info = self._calculated_scan_height_info()
        return engine_scan_preview.scan_plan_fingerprint(
            plan,
            emi_probe_offset_mm=(0.0, 0.0),
            height_target=info.get("target_machine_z"),
        )

    def _install_scan_preview_widget(self, host):
        if host is None or not callable(getattr(host, "layout", None)):
            return
        try:
            from modules.scan_preview_widget import ScanPreviewWidget
            widget = ScanPreviewWidget(host)
        except Exception:
            return
        host.layout().addWidget(widget)
        widget.approved.connect(self._on_scan_plan_approved)
        self._scan_preview_widget = widget
        # The plan lives on Step 4 alone. Repeating it on Review & Measure
        # added a second copy of a decision already approved there.
        self._refresh_scan_preview()

    def _on_scan_plan_approved(self, fingerprint):
        if self._registration_problems():
            return
        self._approved_scan_fingerprint = fingerprint
        self._update_nav()

    def _refresh_scan_preview(self):
        widget = getattr(self, "_scan_preview_widget", None)
        if widget is None or engine_scan_preview is None:
            return
        taught_xy = []
        for slot in range(len(TAUGHT_CORNER_LABELS)):
            point = self._taught_pcb_corners(self._scan_setup_side()).get(slot)
            if point is not None:
                taught_xy.append(point)
        plan = None
        travel_error = ""
        problems = self._registration_problems() if self._board_view is not None else ["no board"]
        if self._registration is not None and self._board_view is not None and not problems:
            try:
                plan = self._preview_scan_plan()
                if plan is not None:
                    try:
                        plan.validate_machine_limits(self._printer_config())
                    except (ValueError, RuntimeError) as exc:
                        travel_error = str(exc)
            except (ValueError, RuntimeError) as exc:
                widget.set_scene(None)
                widget.status.setText(str(exc))
                return
        taught_frame = None
        placement = getattr(self, "_taught_pocket_placement", None)
        if placement and placement.get("placement") == "taught_pocket" and placement.get("scan_frame"):
            frame = placement["scan_frame"]
            taught_frame = engine_registration.Transform(
                theta_rad=float(frame["theta_rad"]),
                x0_mm=float(frame["x0_mm"]),
                y0_mm=float(frame["y0_mm"]),
            )
        if self._easy_mode_active() and self._board_zero_committed():
            calibrated = self._pcb_surface_z is not None
            height_info = {
                "available": calibrated,
                "reason": (
                    ""
                    if calibrated
                    else getattr(
                        engine_glassboard_fixture,
                        "PROBE_HEIGHT_NOT_CALIBRATED",
                        "Probe height not calibrated",
                    )
                ),
                "pcb_surface_machine_z": self._pcb_surface_z,
                "target_clearance_mm": self._default_probe_gap_mm(),
                "target_machine_z": self._easy_scan_target_z() if calibrated else None,
                "predicted_clearance_mm": None,
            }
        else:
            height_info = self._calculated_scan_height_info()
            if self._board_zero_committed() and not height_info.get("available"):
                height_info = dict(height_info)
                height_info["available"] = True
                height_info["reason"] = ""
                height_info.setdefault(
                    "target_clearance_mm", self._default_probe_gap_mm()
                )
        scene = engine_scan_preview.build_scan_preview_scene(
            plan,
            emi_probe_offset_mm=(0.0, 0.0),
            teaching=None,
            height_info=height_info if plan is not None else None,
            travel_error=travel_error,
            taught_corners_machine_xy=taught_xy,
            taught_scan_frame=taught_frame,
        )
        widget.set_scene(scene)
        stale = (
            self.thread is None
            and self._machine_xy is not None
            and (time.monotonic() - self._machine_xy_at) > 5.0
        )
        widget.set_reported_xy(self._machine_xy, stale=stale)
        if self._pending_preview_cell is not None:
            widget.set_active_cell(self._pending_preview_cell)

    def _attach_scan_preview_to_current_page(self, index):
        widget = getattr(self, "_scan_preview_widget", None)
        if widget is None or index != PAGE_SCAN_SETUP:
            return
        host = getattr(self.ui, "scanPreviewHost", None)
        if host is None or not callable(getattr(host, "layout", None)):
            return
        layout = host.layout()
        if layout is not None:
            layout.addWidget(widget)

    def _on_cell_begin(self, info):
        xy = (float(info.get("machine_x_mm", 0.0)), float(info.get("machine_y_mm", 0.0)))
        self._pending_preview_cell = xy
        widget = getattr(self, "_scan_preview_widget", None)
        if widget is not None:
            widget.set_active_cell(xy)

    def _scan_setup_seated_changed(self, checked):
        """Keep the visible Easy confirmation and the shared safety gate in sync."""
        checked = bool(checked)
        if self.ui.chkBoardSeated.isChecked() != checked:
            blocker = QtCore.QSignalBlocker(self.ui.chkBoardSeated)
            self.ui.chkBoardSeated.setChecked(checked)
            del blocker
        self._refresh_scan_setup_summary()
        self._update_nav()

    def _advanced_board_seated_changed(self, checked):
        """Reflect Advanced's shared state on the Easy Scan Setup page."""
        control = getattr(self.ui, "chkScanSetupBoardSeated", None)
        if control is None or control.isChecked() == bool(checked):
            return
        blocker = QtCore.QSignalBlocker(control)
        control.setChecked(bool(checked))
        del blocker
        self._refresh_scan_setup_summary()

    # ------------------------------------------------ taught PCB insert edge
    def _board_view_for_side(self, side):
        """The un-rotated view of one board face, or None before import."""
        if self._board_model is None:
            return None
        return self._board_model.view(side=str(side), rotation_deg=0)

    def _pcb_corner_board_points(self, view):
        """Fixture-local inner-pocket corners Step 4 teaches (not PCB outline)."""
        del view
        if engine_glassboard_fixture is None:
            return ((0.0, 0.0), (1.0, 1.0))
        return engine_glassboard_fixture.glassboard_support_corners()

    def _taught_pcb_corners(self, side):
        return dict(self._taught_pcb_corner_points.get(str(side)) or {})

    def _taught_side_complete(self, side):
        return len(self._taught_pcb_corners(side)) == len(TAUGHT_CORNER_LABELS)

    def _scan_setup_side(self):
        """The face Step 4 should position, preferring a completely taught one."""
        preferred = "top"
        combo = getattr(self.ui, "glassboardPlacementSide", None)
        if combo is not None:
            preferred = combo.currentData() or "top"
        for side in (preferred, *SCAN_SETUP_SIDES):
            if self._taught_side_complete(side):
                return side
        return preferred

    def _sync_side_combo(self, side):
        """Record which face is toward the probe without re-placing the board."""
        combo = getattr(self.ui, "glassboardPlacementSide", None)
        if combo is None:
            return
        index = combo.findData(str(side))
        if index < 0:
            return
        self._syncing_glassboard_side = True
        try:
            combo.setCurrentIndex(index)
        finally:
            self._syncing_glassboard_side = False

    def _scan_setup_jog(self, dx, dy):
        """Relative XY jog for corner teaching; Z is never commanded."""
        if not self._pcb_xy_homed:
            QMessageBox.warning(
                self.ui, DIALOG_TITLE, "Home X and Y in Step 2 before jogging."
            )
            return
        step_widget = getattr(self.ui, "scanSetupJogStep", None)
        step_mm = step_widget.value() if step_widget is not None else 1.0
        try:
            printer = self._printer()
            x_mm, y_mm = printer.get_xy()
            printer.move_xy(x_mm + dx * step_mm, y_mm + dy * step_mm)
            self._show_position(printer.get_xy())
            self._bump_motion()
        except engine_printer.PrinterReset as exc:
            self._clear_height_datum()
            self._clear_pcb_homing()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))

    def _capture_pcb_corner(self, side, slot, _checked=False):
        """Store the probe's current machine XY as one PCB corner."""
        view = self._board_view_for_side(side)
        if view is None:
            QMessageBox.warning(
                self.ui, DIALOG_TITLE, "Import the ODB++ board in Step 3 first."
            )
            return False
        if not self._pcb_xy_homed:
            QMessageBox.warning(
                self.ui, DIALOG_TITLE,
                "Home X and Y in Step 2 first: a machine coordinate taught "
                "before homing belongs to a frame that no longer exists.",
            )
            return False
        try:
            machine_x, machine_y = self._printer().get_xy()
        except engine_printer.PrinterReset as exc:
            self._clear_pcb_homing()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return False
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return False
        points = self._taught_pcb_corners(side)
        points[int(slot)] = (float(machine_x), float(machine_y))
        self._taught_pcb_corner_points[str(side)] = points
        self._show_position((machine_x, machine_y))
        self._save_taught_pcb_corners(side, view)
        if self._taught_side_complete(side):
            self._apply_taught_pcb_corners(side, announce=True)
        self._refresh_corner_teach_ui()
        self._update_nav()
        return True

    def _use_taught_pcb_side(self, side, _checked=False):
        if not self._taught_side_complete(side):
            QMessageBox.warning(
                self.ui, DIALOG_TITLE,
                f"Capture both corners of the {side} side first.",
            )
            return False
        return self._apply_taught_pcb_corners(side, announce=True)

    def _clear_pcb_corners(self, side, _checked=False):
        """Forget one face's taught corners, in memory and on disk."""
        side = str(side)
        self._taught_pcb_corner_points[side] = {}
        self._pocket_pair_state[side] = "missing"
        if self._pocket_block_reason and self._active_taught_side == side:
            self._pocket_block_reason = ""
        name = self._fixture_profile_name() or self._active_machine_fixture_name
        if name:
            try:
                engine_machine_fixtures.clear_taught_pcb_corners(name, side)
            except (engine_profiles.ProfileError, OSError) as exc:
                QMessageBox.warning(
                    self.ui, DIALOG_TITLE,
                    f"The taught corners were cleared here but not on disk: {exc}",
                )
        if self._active_taught_side == side:
            self._drop_pocket_scan_xy()
            self._update_nav()
        self._refresh_corner_teach_ui()
        self._refresh_scan_setup_summary()
        self._update_nav()
        return True

    def _set_glassboard_placement_status(self, text, stylesheet=None):
        status = getattr(self.ui, "glassboardPlacementStatus", None)
        if status is None:
            return
        status.setText(text)
        if stylesheet is not None:
            status.setStyleSheet(stylesheet)

    def _drop_pocket_scan_xy(self):
        """Forget taught-pocket XY, the scan path, and approval. Does not move Z."""
        self._registration = None
        self._landmarks.clear()
        self._active_taught_side = None
        self._taught_pocket_placement = None
        self._stl_board_placement = None
        self._invalidate_plan()

    def _apply_taught_pcb_corners(self, side, *, announce=False):
        """Seat the board from two taught E-probe inner-pocket corners."""
        side = str(side)
        view = self._board_view_for_side(side)
        points = self._taught_pcb_corners(side)
        if view is None or len(points) < len(TAUGHT_CORNER_LABELS):
            return False
        if engine_glassboard_fixture is None:
            self._blocked_corner_status("EMI_Mapper engine is unavailable", announce=announce)
            return False
        ordered = tuple(points[slot] for slot in sorted(points))
        document = self._machine_fixture_document() or {}
        yaw = 0.0
        try:
            yaw = float(
                engine_machine_fixtures.fixture_calibrations(document)["pcb_yaw_deg"]
            )
        except (engine_profiles.ProfileError, KeyError, TypeError, ValueError):
            yaw = 0.0
        blocker = QtCore.QSignalBlocker(self.ui.boardSide)
        self.ui.boardSide.setCurrentText(side)
        del blocker
        self._apply_board_view(rotation_deg=0)
        self._sync_side_combo(side)
        view = self._board_view
        try:
            registration, placement = (
                engine_glassboard_fixture.board_registration_from_taught_pocket(
                    view,
                    ordered,
                    pcb_yaw_deg=yaw,
                    step_mm=float(self.ui.stepMm.value()),
                )
            )
            unreachable = self._reachability_problem(registration)
            if unreachable:
                raise engine_glassboard_fixture.GlassboardPlacementError(
                    f"the taught pocket is outside machine travel: {unreachable}"
                )
        except (
            engine_glassboard_fixture.GlassboardPlacementError,
            engine_registration.RegistrationError,
            ValueError,
        ) as exc:
            self._pocket_pair_state[side] = "rejected"
            self._pocket_block_reason = str(exc)
            self._drop_pocket_scan_xy()
            self._blocked_corner_status(str(exc), announce=announce)
            return False
        self._landmarks[:] = list(registration.points)
        self._registration = registration
        self._active_taught_side = side
        self._taught_pocket_placement = placement
        self._pocket_pair_state[side] = "ok"
        self._pocket_block_reason = ""
        self._fit_error = ""
        self._profile = None
        self._stl_board_placement = None
        self._offset_correction_mm = (0.0, 0.0)
        self._verified_points = []
        self._invalidate_plan()
        self._bump_registration()
        self._clear_selection()
        width_mm, height_mm = view.size_mm
        theta_deg = registration.transform.to_dict()["theta_deg"]
        self.ui.glassboardPlacementStatus.setText(
            f"{side.upper()} side seated in the taught inner pocket: "
            f"{width_mm:.2f} × {height_mm:.2f} mm board at "
            f"{theta_deg:.3f}° (yaw {yaw:g}° from config). "
            "Confirm seating, then continue."
        )
        self.ui.glassboardPlacementStatus.setStyleSheet(
            "padding:8px;background:#ECFDF3;color:#05603A;"
            "border:1px solid #12B76A;border-radius:6px"
        )
        self._update_registration_ui()
        self._refresh_corner_teach_ui()
        self._refresh_calculated_height()
        self._refresh_scan_setup_summary()
        self._update_nav()
        return True

    def _blocked_corner_status(self, reason, *, announce):
        self._set_glassboard_placement_status(
            f"POSITION BLOCKED — {reason}",
            "padding:8px;background:#FEF3F2;color:#912018;"
            "border:1px solid #F04438;border-radius:6px",
        )
        if announce:
            QMessageBox.warning(self.ui, DIALOG_TITLE, reason)
        self._update_nav()

    def _save_taught_pcb_corners(self, side, view):
        """Persist one face's insert corners on the fixture, not the board file."""
        name = self._ensure_fixture_selected()
        points = self._taught_pcb_corners(side)
        if not points:
            return
        if not name:
            QMessageBox.warning(
                self.ui, DIALOG_TITLE,
                "Select a fixture on Step 2 so these corners can be reused "
                "on the next measurement.",
            )
            return
        payload = [
            {
                "slot": int(slot),
                "machine_x_mm": points[slot][0],
                "machine_y_mm": points[slot][1],
            }
            for slot in sorted(points)
        ]
        try:
            engine_machine_fixtures.save_taught_pcb_corners(name, side, payload)
        except (engine_profiles.ProfileError, OSError) as exc:
            QMessageBox.warning(
                self.ui, DIALOG_TITLE,
                f"The corner is usable now, but could not be saved: {exc}",
            )

    def _load_taught_pcb_corners(self):
        """Restore v2 pocket corners. v1 pairs are stale and must be recaptured."""
        document = self._machine_fixture_document()
        if document is None:
            return
        for side in SCAN_SETUP_SIDES:
            try:
                status = engine_machine_fixtures.taught_pocket_status(document, side)
            except engine_profiles.ProfileError:
                status = "missing"
            if status == "stale":
                self._pocket_pair_state[side] = "stale"
                self._taught_pcb_corner_points[side] = {}
                continue
            if status != "ok":
                if not self._taught_pcb_corners(side):
                    self._pocket_pair_state[side] = "missing"
                continue
            points = {}
            try:
                saved = engine_machine_fixtures.taught_pcb_corners(document, side)
            except engine_profiles.ProfileError:
                saved = []
            for entry in saved:
                points[int(entry["slot"])] = (
                    float(entry["machine_x_mm"]),
                    float(entry["machine_y_mm"]),
                )
            self._taught_pcb_corner_points[side] = points
            if self._pocket_pair_state.get(side) != "rejected":
                self._pocket_pair_state[side] = (
                    "ok" if len(points) >= len(TAUGHT_CORNER_LABELS) else "missing"
                )

    def _face_blocks_stl_fallback(self, side):
        """True when a pocket pair exists but must not be replaced by STL."""
        side = str(side)
        state = self._pocket_pair_state.get(side)
        if state in ("stale", "rejected"):
            return True
        return self._taught_side_complete(side)

    def _refresh_corner_teach_ui(self):
        """Show what each face has taught, and which one is being measured."""
        for side, label in (self._corner_teach_labels or {}).items():
            points = self._taught_pcb_corners(side)
            lines = []
            for slot, corner_label in enumerate(TAUGHT_CORNER_LABELS):
                position = points.get(slot)
                lines.append(
                    f"{corner_label}: "
                    + (
                        f"X {position[0]:.2f}  Y {position[1]:.2f} mm"
                        if position is not None
                        else "not taught"
                    )
                )
            if self._active_taught_side == side:
                lines.append("Measuring this side.")
            state = self._pocket_pair_state.get(side)
            if state == "stale":
                lines.append("Saved corners are outdated — recapture A and B.")
            elif state == "rejected" and self._pocket_block_reason:
                lines.append(self._pocket_block_reason)
            label.setText("\n".join(lines))

    def _set_glassboard_middle_placement(
        self, side, *, persist=True, reset_seated=True, announce=True
    ):
        """Place the board in the STL middle support using taught P1 and the STL."""
        if self._board_model is None:
            if announce:
                QMessageBox.warning(self.ui, DIALOG_TITLE, "Import the ODB++ board first.")
            return False
        if not self._glassboard_fixture_selected():
            if announce:
                QMessageBox.warning(
                    self.ui, DIALOG_TITLE,
                    "The STL middle-holder placement is only available for a "
                    "Glassboard fixture.",
                )
            return False
        document = self._machine_fixture_document()
        teaching = (document or {}).get("fixture_teaching")
        if teaching is None:
            if announce:
                QMessageBox.warning(
                    self.ui, DIALOG_TITLE,
                    "Teach P1 for this physical fixture before placing the PCB.",
                )
            return False
        if not self._fixture_verified_session or not self._pcb_xy_homed:
            if announce:
                QMessageBox.warning(
                    self.ui, DIALOG_TITLE,
                    "Verify the fixture in Step 2 first. Its current machine position "
                    "must be confirmed before the STL placement can be used.",
                )
            return False
        offset = engine_machine_fixtures.emi_probe_offset_mm(document)
        if not engine_machine_fixtures.emi_probe_offset_is_set(document):
            if announce:
                QMessageBox.warning(
                    self.ui, DIALOG_TITLE,
                    "Set fixture.emi_probe_offset_x_mm and emi_probe_offset_y_mm "
                    "in EMI_Mapper/config.yaml. Those are carriage constants, "
                    "not per-fixture fields.",
                )
            status = getattr(self.ui, "glassboardPlacementStatus", None)
            if status is not None:
                status.setText(
                    "E-probe offset is required in EMI_Mapper/config.yaml "
                    "before locating the board."
                )
            return False
        side = str(side or "top").lower()
        if side not in ("top", "bottom"):
            side = "top"
        blocker = QtCore.QSignalBlocker(self.ui.boardSide)
        self.ui.boardSide.setCurrentText(side)
        del blocker
        self._apply_board_view(rotation_deg=0)
        cals = engine_machine_fixtures.fixture_calibrations(document)
        try:
            registration, placement = (
                engine_glassboard_fixture.board_registration_in_middle_holder(
                    self._board_view,
                    teaching,
                    emi_probe_offset_mm=offset,
                    pcb_yaw_deg=cals["pcb_yaw_deg"],
                )
            )
            unreachable = self._reachability_problem(registration)
            if unreachable:
                raise engine_glassboard_fixture.GlassboardPlacementError(
                    f"middle-holder placement is outside machine travel: {unreachable}"
                )
        except (
            engine_glassboard_fixture.GlassboardPlacementError,
            engine_registration.RegistrationError,
            ValueError,
        ) as exc:
            self.ui.glassboardPlacementStatus.setText(f"PLACEMENT BLOCKED — {exc}")
            self.ui.glassboardPlacementStatus.setStyleSheet(
                "padding:8px;background:#FEF3F2;color:#912018;"
                "border:1px solid #F04438;border-radius:6px"
            )
            if announce:
                QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            self._update_nav()
            return False

        self._landmarks[:] = list(registration.points)
        self._registration = registration
        self._fit_error = ""
        self._profile = None
        self._stl_board_placement = placement
        self._offset_correction_mm = (0.0, 0.0)
        self._verified_points = []
        self._invalidate_plan()
        self._bump_registration()
        self._clear_selection()
        if reset_seated:
            self.ui.chkScanSetupBoardSeated.setChecked(False)
        clearance_x, clearance_y = placement["clearance_total_mm"]
        frame = placement["fixture_frame"]
        center_x, center_y = placement["machine_support_center_mm"]
        geometry_note = ""
        placement_style = (
            "padding:8px;background:#ECFDF3;color:#05603A;"
            "border:1px solid #12B76A;border-radius:6px"
        )
        if not frame.get("span_measured", True):
            # P1 alone fixes the origin but cannot observe rotation, so the
            # operator needs to know the squareness is assumed, not measured.
            geometry_note = (
                " Located from P1 and the STL with the fixture assumed square "
                "to X/Y; angle was not measured. Teach P1-P4 to check it."
            )
        elif frame["max_corner_residual_mm"] > 1.0:
            geometry_note = (
                f" Reference marks span {frame['observed_width_mm']:.3f} × "
                f"{frame['observed_height_mm']:.3f} mm versus STL "
                "106.162 × 34.000 mm; center and angle are used without scaling the PCB."
            )
            placement_style = (
                "padding:8px;background:#FFFAEB;color:#7A2E0E;"
                "border:1px solid #F79009;border-radius:6px"
            )
        saved_word = "Saved" if persist else "Restored"
        offset_note = (
            f" E-probe offset {offset[0]:+.3f}, {offset[1]:+.3f} mm "
            "(scan XY follows the E-field probe)."
        )
        self.ui.glassboardPlacementStatus.setText(
            f"{saved_word}: {side.upper()} faces the probe; PCB centered in the "
            f"58.164 × 12.940 mm middle support. Total clearance "
            f"X {clearance_x:.3f} mm, Y {clearance_y:.3f} mm. "
            f"Machine center X {center_x:.3f}, Y {center_y:.3f} mm; "
            f"fixture angle {frame['transform']['theta_deg']:.3f}°."
            + geometry_note
            + offset_note
        )
        self.ui.glassboardPlacementStatus.setStyleSheet(placement_style)
        self._refresh_scan_setup_locate_status()
        self._refresh_calculated_height()
        if persist:
            try:
                engine_machine_fixtures.save_board_placement(
                    self._fixture_profile_name(),
                    self._board_view.geometry_hash,
                    placement,
                )
            except (engine_profiles.ProfileError, OSError) as exc:
                QMessageBox.warning(
                    self.ui, DIALOG_TITLE,
                    f"The board position is usable now, but could not be saved: {exc}",
                )
        self._update_registration_ui()
        self._refresh_scan_setup_summary()
        self._update_nav()
        return True

    def _on_glassboard_side_changed(self, _index=0):
        """Re-apply a taught pocket pair when the recorded face changes.

        Easy scan XY comes from A/B only. Missing corners drop the previous
        placement rather than falling back to the STL/P1 holder.
        """
        if self._syncing_glassboard_side:
            return
        if not self._glassboard_fixture_selected() or self._board_model is None:
            return
        side = self.ui.glassboardPlacementSide.currentData() or "top"
        if self._taught_side_complete(side):
            self._apply_taught_pcb_corners(side, announce=False)
            return
        self._drop_pocket_scan_xy()

    def _restore_glassboard_middle_placement(self):
        """Restore this board/fixture association without trusting physical seating."""
        if self._board_model is None:
            return False
        if not self._glassboard_fixture_selected():
            return False
        document = self._machine_fixture_document()
        if document is None:
            return False
        preferred = self.ui.glassboardPlacementSide.currentData() or "top"
        for side in (preferred, "bottom" if preferred == "top" else "top"):
            view = self._board_model.view(side=side, rotation_deg=0)
            try:
                saved = engine_machine_fixtures.board_placement(
                    document, view.geometry_hash
                )
            except engine_profiles.ProfileError as exc:
                self.ui.glassboardPlacementStatus.setText(
                    f"Saved middle-holder position is invalid: {exc}"
                )
                return False
            if saved is None or saved.get("template_id") != engine_glassboard_fixture.TEMPLATE_ID:
                continue
            combo_index = self.ui.glassboardPlacementSide.findData(side)
            if combo_index >= 0:
                # Restoring the saved side must not look like an operator
                # choice, or it would re-place and clear the seated flag.
                self._syncing_glassboard_side = True
                try:
                    self.ui.glassboardPlacementSide.setCurrentIndex(combo_index)
                finally:
                    self._syncing_glassboard_side = False
            return self._set_glassboard_middle_placement(
                side, persist=False, reset_seated=False, announce=False
            )
        return False

    def _on_fixture_operation_failed(self, exc):
        if self._is_printer_halt(exc):
            self._handle_printer_halt(exc)
            return
        self._close_printer()
        self._clear_board_zero()
        self._fixture_xyz_homed = False
        self._fixture_verified_session = False
        self._clear_pcb_homing()
        self.ui.fixtureTeachStatus.setText(f"BLOCKED — {exc}")
        connection = getattr(self.ui, "fixturePrinterConnectionStatus", None)
        if connection is not None:
            connection.setText(f"NOT CONNECTED — {exc}")
            connection.setStyleSheet(
                "padding:7px;background:#FEF3F2;color:#912018;"
                "border:1px solid #F04438;border-radius:6px"
            )
        QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))

    # ------------------------------------------------------- homing and jog
    def _printer(self, *, manage_z=None):
        """A Printer on the shared transport, carrying the form's machine limits."""
        if self._printer_halted:
            raise engine_printer.PrinterHalted(PRINTER_HALTED_TEXT)
        config = self._printer_config()
        if manage_z is not None:
            config = replace(config, manage_z=manage_z)
        printer = engine_printer.Printer(self._open_printer(), config)
        return self._bind_printer_frame(printer)

    def _is_printer_halt(self, exc):
        """Type only. Matching halt words in text also matched this app's own advice."""
        return engine_printer is not None and isinstance(
            exc, engine_printer.PrinterHalted
        )

    def _handle_printer_halt(self, exc):
        """Stop talking to a killed controller. Do not retry homing."""
        self._printer_halted = True
        self._close_printer()
        self._clear_height_datum()
        self._clear_pcb_homing()
        message = str(exc).strip() or PRINTER_HALTED_TEXT
        if PRINTER_HALTED_TEXT not in message:
            message = f"{PRINTER_HALTED_TEXT} {message}"
        self.ui.posLabel.setText(PRINTER_HALTED_TEXT)
        fixture_status = getattr(self.ui, "fixtureTeachStatus", None)
        if fixture_status is not None:
            fixture_status.setText(message)
        connection = getattr(self.ui, "fixturePrinterConnectionStatus", None)
        if connection is not None:
            connection.setText(f"NOT CONNECTED — {PRINTER_HALTED_TEXT}")
            connection.setStyleSheet(
                "padding:7px;background:#FEF3F2;color:#912018;"
                "border:1px solid #F04438;border-radius:6px"
            )
        QMessageBox.warning(self.ui, DIALOG_TITLE, message)

    def _fixture_retry_printer_clicked(self):
        """Reconnect and identify Marlin without moving any machine axis."""
        port = self._printer_port()
        configured = int(self.ui.printerBaud.value())
        self._close_printer()
        self._clear_height_datum()
        self._clear_pcb_homing()
        self._start_printer_job(
            lambda: self._do_fixture_retry_printer(port, configured),
            self._on_fixture_retry_printer_ok,
            self._on_fixture_operation_failed,
            self.ui.fixturePrinterConnectionStatus,
            "Connecting to the selected printer port…",
        )

    def _do_fixture_retry_printer(self, port=None, configured=None):
        port = port or self._printer_port()
        configured = self.ui.printerBaud.value() if configured is None else configured
        if not port:
            raise RuntimeError("Choose the printer serial port first")
        self._close_printer()
        if PRINTER_BOOT_S:
            time.sleep(PRINTER_BOOT_S)
        attempts = tuple(dict.fromkeys((configured, 115200, 250000)))
        failures = []
        for index, baud in enumerate(attempts):
            if index:
                time.sleep(PRINTER_BAUD_SETTLE_S)
            try:
                handle = self._connect_printer(port, baud)
            except engine_printer.PrinterHalted:
                # A halt reply is an answer, not a failed baud guess. Trying
                # further rates would only reboot a controller that needs a
                # manual reset.
                raise
            except (RuntimeError, engine_printer.PrinterError) as exc:
                failures.append(str(exc))
                continue
            self.printer_serial = handle
            self._clear_height_datum()
            return port, baud, baud != configured
        tried = ", ".join(str(value) for value in attempts)
        if not failures:
            detail = "timeout (no bytes received)."
        elif len(failures) == 1:
            detail = failures[0]
        else:
            detail = (
                failures[0]
                + " Additional baud rates were tried after that and also did not "
                "identify Marlin."
            )
        raise RuntimeError(
            f"Could not auto-detect Marlin on {port}; tried {tried} baud. {detail}"
        )

    def _on_fixture_retry_printer_ok(self, payload):
        port, baud, changed = payload
        self._printer_halted = False
        if changed:
            self.ui.printerBaud.setValue(baud)
        self.ui.fixturePrinterConnectionStatus.setText(
            f"CONNECTED — Marlin answered on {port} at {baud} baud"
            + (" (auto-detected and selected)." if changed else ".")
        )
        self.ui.fixturePrinterConnectionStatus.setStyleSheet(
            "padding:7px;background:#ECFDF3;color:#05603A;"
            "border:1px solid #12B76A;border-radius:6px"
        )
        self.ui.fixtureTeachStatus.setText(STEP2_AFTER_RECONNECT)

    def _logical_z_max_mm(self):
        return LOGICAL_Z_MAX_MM

    def _safe_home_lift_mm(self):
        if self._profile is not None:
            return self._profile.safe_home_lift_mm
        document = self._selected_profile_document()
        if document is not None:
            try:
                return engine_profiles._optional_finite(
                    document.get("safe_home_lift_mm"),
                    engine_profiles.SAFE_HOME_LIFT_MM,
                )
            except engine_profiles.ProfileError:
                return engine_profiles.SAFE_HOME_LIFT_MM
        return engine_profiles.SAFE_HOME_LIFT_MM

    def _default_probe_gap_mm(self):
        if self._profile is not None:
            return self._profile.default_probe_gap_mm
        return engine_profiles.DEFAULT_PROBE_GAP_MM

    def _easy_scan_target_z(self):
        if self._pcb_surface_z is None:
            return None
        return engine_height.easy_scan_target_z(
            self._pcb_surface_z, self._default_probe_gap_mm()
        )

    def _easy_at_scan_plane(self, reported_z=None):
        if self._easy_mode_active() and reported_z is None:
            return self._easy_scan_plane_cached()
        target = self._easy_scan_target_z()
        if target is None:
            return False
        if reported_z is None:
            reported_z = self._reported_z()
        if reported_z is None:
            return False
        return engine_height.logical_position_matches(reported_z, target)

    def _easy_reenable_z_if_unlocked(self, printer=None):
        if not self._easy_z_unlocked:
            return
        handle = printer if printer is not None else self._printer(manage_z=True)
        handle.enable_z_holding()
        self._easy_z_unlocked = False

    def _selected_profile_document(self):
        name = self._selected_profile_name()
        if not name or engine_profiles is None:
            return None
        try:
            return engine_profiles.read_profile_document(name)
        except engine_profiles.ProfileError:
            return None

    def _easy_xy_offset(self):
        if self._profile is None:
            return (0.0, 0.0)
        return self._profile.easy_xy_offset_mm

    def _scan_transform(self):
        if self._registration is None:
            return None
        dx, dy = self._easy_xy_offset()
        if dx == 0.0 and dy == 0.0:
            return self._registration.transform
        return engine_profiles.with_easy_xy_offset(self._registration.transform, dx, dy)

    def _easy_home_landmark(self):
        document = None
        if self._profile is not None:
            document = self._profile.document
        else:
            document = self._selected_profile_document()
        if not document:
            return None
        try:
            return engine_profiles.effective_reference_machine(document)
        except engine_profiles.ProfileError:
            return None

    def _update_easy_status(self):
        label = getattr(self.ui, "easyStatusLabel", None)
        if label is None or not callable(getattr(label, "setText", None)):
            return
        if not self._easy_mode_active():
            return
        document = (
            self._profile.document
            if self._profile is not None
            else self._selected_profile_document()
        )
        if document is None:
            label.setText("Select a verified Glassboard fixture, then HOME & MOVE.")
            return
        try:
            base = engine_profiles.primary_reference_machine(document)
            effective = engine_profiles.effective_reference_machine(document)
            dx, dy = engine_profiles.easy_xy_offset_mm(document)
        except engine_profiles.ProfileError as exc:
            label.setText(str(exc))
            return
        verified = "✓" if (self._profile is not None and self._profile.xy_verified) or document.get("xy_verified") else "—"
        datum = "—"
        if self._registration is not None:
            point = self._registration.points[0]
            datum = f"({point.board_x_mm:.2f}, {point.board_y_mm:.2f})"
        label.setText(
            f"Fixture {verified}   board datum {datum} mm\n"
            f"Base machine XY ({base[0]:.2f}, {base[1]:.2f})\n"
            f"Effective ({effective[0]:.2f}, {effective[1]:.2f})   "
            f"ΔX {dx:+.2f}  ΔY {dy:+.2f} mm"
        )

    def _refresh_easy_height_label(self):
        u = self.ui
        missing_vertical = not self._easy_vertical_calibrated()
        for name in EASY_MANUAL_HEIGHT_BUTTONS:
            self._set_widget_visible(
                name, self._easy_mode_active() and missing_vertical
            )
        self._set_widget_visible(
            "grpEasyManualHeight", self._easy_mode_active() and missing_vertical
        )
        easy_height = getattr(u, "easyHeightLabel", None)
        if (
            self._easy_mode_active()
            and easy_height is not None
            and callable(getattr(easy_height, "setText", None))
        ):
            easy_height.setText(self._step2_status_text())
        self._refresh_fixture_page_summary()
        self._refresh_scan_setup_summary()

    def _easy_move_to_reference(self):
        target = self._easy_home_landmark()
        if target is None or not self._pcb_xy_homed:
            QMessageBox.warning(self.ui, DIALOG_TITLE, "Home XY and load a fixture first.")
            return
        try:
            printer = self._printer()
            printer.move_xy(*target)
            self._show_position(printer.get_xy())
            self._bump_motion()
        except engine_printer.PrinterReset as exc:
            self._clear_height_datum()
            self._clear_pcb_homing()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))

    def _easy_open_fine_adjust(self):
        if not self._pcb_xy_homed or self._profile is None:
            QMessageBox.warning(self.ui, DIALOG_TITLE, "Load a verified fixture first.")
            return
        try:
            self._easy_adjust_saved_xy = self._printer().get_xy()
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return
        self._set_easy_adjust_visible(True)
        self._update_easy_status()

    def _easy_jog(self, dx, dy):
        if not self._pcb_xy_homed:
            QMessageBox.warning(self.ui, DIALOG_TITLE, "Home X and Y before jogging.")
            return
        step_widget = getattr(self.ui, "easyJogStep", None)
        step_mm = step_widget.value() if step_widget is not None else 0.25
        try:
            printer = self._printer()
            x_mm, y_mm = printer.get_xy()
            printer.move_xy(x_mm + dx * step_mm, y_mm + dy * step_mm)
            self._show_position(printer.get_xy())
            self._bump_motion()
            self._update_easy_status()
        except engine_printer.PrinterReset as exc:
            self._clear_height_datum()
            self._clear_pcb_homing()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))

    def _easy_save_position(self):
        try:
            x_mm, y_mm = self._printer().get_xy()
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return False
        return self._commit_easy_xy_from_machine(x_mm, y_mm)

    def _commit_easy_xy_from_machine(self, x_mm, y_mm):
        """Persist translation-only offset. Never calls fit_registration."""
        if self._profile is None:
            QMessageBox.warning(self.ui, DIALOG_TITLE, "Load a verified fixture first.")
            return False
        try:
            base_x, base_y = engine_profiles.primary_reference_machine(
                self._profile.document
            )
        except engine_profiles.ProfileError as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return False
        dx = float(x_mm) - base_x
        dy = float(y_mm) - base_y
        try:
            document = engine_profiles.save_easy_xy_offset(self._profile.path, dx, dy)
        except engine_profiles.ProfileError as exc:
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                f"{exc}\n\n{ORIENTATION_DRIFT_HINT}",
            )
            return False
        self._profile = replace(self._profile, document=document)
        self._invalidate_plan()
        self._easy_adjust_open = False
        self._easy_adjust_saved_xy = None
        self._set_easy_adjust_visible(False)
        self._update_easy_status()
        self._update_registration_ui()
        self._update_nav()
        return True

    def _easy_cancel_fine_adjust(self):
        saved = self._easy_adjust_saved_xy
        self._easy_adjust_open = False
        self._set_easy_adjust_visible(False)
        if saved is not None and self._pcb_xy_homed:
            try:
                self._printer().move_xy(*saved)
                self._show_position(saved)
            except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
                QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
        self._easy_adjust_saved_xy = None
        self._update_easy_status()
        return True

    def _easy_reset_xy_calibration(self):
        if self._profile is None:
            return False
        try:
            document = engine_profiles.reset_easy_xy_offset(self._profile.path)
        except engine_profiles.ProfileError as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return False
        self._profile = replace(self._profile, document=document)
        self._invalidate_plan()
        landmark = self._easy_home_landmark()
        if landmark is not None and self._pcb_xy_homed:
            try:
                self._printer().move_xy(*landmark)
                self._show_position(self._printer().get_xy())
            except (engine_printer.PrinterError, OSError, RuntimeError, ValueError):
                pass
        self._set_easy_adjust_visible(False)
        self._update_easy_status()
        self._update_registration_ui()
        self._update_nav()
        return True

    def _easy_set_probe_height(self):
        if self._easy_z_unlocked:
            try:
                self._easy_reenable_z_if_unlocked()
            except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
                QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
                return False
        if self._pcb_surface_z is None:
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                "Z reference required. Use SET PCB SURFACE MANUALLY first.",
            )
            return False
        try:
            printer = self._printer(manage_z=True)
            # A newly opened serial connection pulses DTR and clears the
            # session datum. Re-read it after acquiring the printer so a
            # target calculated before reconnect can never be commanded.
            target = self._easy_scan_target_z()
            if target is None:
                raise RuntimeError(
                    "Printer reconnected, so the session Z reference was cleared. "
                    "Use SET PCB SURFACE MANUALLY again."
                )
            printer.move_z(target)
            reported = printer.get_xyz()[2]
            if not engine_height.logical_position_matches(reported, target):
                raise engine_printer.PrinterError(
                    "Marlin-reported position did not reach the requested logical target"
                )
            self._last_commanded_scan_z = target
            self._easy_verified_scan_z = float(reported)
            self._easy_height_identity = self._easy_height_identity_now()
            self._easy_height_lost_to_reset = False
            self._refresh_height_ui()
            self._update_nav()
            return True
        except engine_printer.PrinterReset as exc:
            self._clear_height_datum()
            self._clear_pcb_homing()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return False
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return False

    def _clear_easy_session_scan_plane(self):
        """Drop the E-probe plane without forgetting taught P1 XY or XY homing."""
        self._pcb_surface_z = None
        self._pcb_surface_from_fixture = False
        self._last_commanded_scan_z = None
        self._easy_touch_logical_z = None
        self._bltouch_board_zero_z = None
        self._bltouch_board_zero_frame = None
        self._invalidate_easy_verified_height()

    def _easy_begin_pcb_surface_touch(self):
        if not self._pcb_xy_homed:
            QMessageBox.warning(self.ui, DIALOG_TITLE, "Home X and Y before setting the PCB surface.")
            return False
        printer = self._printer(manage_z=True)
        # Releasing Z holding invalidates BLTouch millimetres, not taught P1 XY.
        self._clear_easy_session_scan_plane()
        landmark = self._easy_home_landmark()
        if landmark is not None:
            printer.move_xy(*landmark)
        self._easy_touch_logical_z = printer.get_xyz()[2]
        printer.release_z_holding()
        self._easy_z_unlocked = True
        return True

    def _easy_confirm_pcb_touch(self):
        if not self._easy_z_unlocked:
            return False
        printer = self._printer(manage_z=True)
        printer.enable_z_holding()
        self._easy_z_unlocked = False
        surface = self._easy_touch_logical_z
        if surface is None:
            surface = printer.get_xyz()[2]
        self._pcb_surface_z = float(surface)
        self._pcb_surface_from_fixture = False
        self._last_commanded_scan_z = None
        self._invalidate_easy_verified_height()
        self._refresh_height_ui()
        self._update_nav()
        return True

    def _easy_cancel_pcb_touch(self):
        self._easy_reenable_z_if_unlocked()
        self._clear_easy_session_scan_plane()
        return True

    def _easy_set_pcb_surface_manually(self):
        try:
            if not self._easy_begin_pcb_surface_touch():
                return False
        except engine_printer.PrinterReset as exc:
            self._easy_z_unlocked = False
            self._clear_height_datum()
            self._clear_pcb_homing()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return False
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            try:
                self._easy_cancel_pcb_touch()
            except (engine_printer.PrinterError, OSError, RuntimeError, ValueError):
                pass
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return False
        answer = QMessageBox.question(
            self.ui,
            DIALOG_TITLE,
            "Z holding is released. Turn the Z screw until the tip just touches the PCB.\n\n"
            "Yes = tip is touching the PCB.\n"
            "No = cancel (Z holding is restored; no surface is stored).",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        try:
            if answer != QMessageBox.Yes:
                self._easy_cancel_pcb_touch()
                self._refresh_easy_height_label()
                return False
            return self._easy_confirm_pcb_touch()
        except engine_printer.PrinterReset as exc:
            self._easy_z_unlocked = False
            self._clear_height_datum()
            self._clear_pcb_homing()
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return False
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
            return False

    def _easy_reset_pcb_surface(self):
        try:
            self._easy_cancel_pcb_touch()
        except (engine_printer.PrinterError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.ui, DIALOG_TITLE, str(exc))
        self._clear_easy_session_scan_plane()
        self._refresh_height_ui()
        self._update_nav()
        return True

    def _easy_try_load_profile(self):
        if not self._easy_mode_active():
            return
        if self._board_view is None or not self._pcb_xy_homed:
            return
        if not self.ui.chkBoardSeated.isChecked():
            return
        if not self._selected_profile_name():
            return
        document = self._selected_profile_document()
        if document is None or not document.get("xy_verified"):
            return
        self._load_profile()

    def _easy_auto_import_board(self):
        if self._board_model is not None:
            return
        document = self._selected_profile_document()
        source = str((document or {}).get("source_name") or "").strip()
        path = self._resolve_odbpp_path(source)
        if path is None:
            if not source:
                return
            path, _ = QFileDialog.getOpenFileName(
                self.ui,
                "Choose the ODB++ job for this board",
                self.ui.boardPath.text(),
                "ODB++ archives (*.tgz *.tar *.tar.gz);;All files (*)",
            )
        if path:
            self._import_board_path(path)

    def _resolve_odbpp_path(self, source_name):
        if not source_name:
            return None
        candidate = Path(source_name)
        if candidate.is_file():
            return str(candidate)
        cwd = Path.cwd() / source_name
        if cwd.is_file():
            return str(cwd)
        downloads = Path.home() / "Downloads" / Path(source_name).name
        if downloads.is_file():
            return str(downloads)
        return None

    def _home_xy_clicked(self):
        if self._printer_halted:
            QMessageBox.warning(self.ui, DIALOG_TITLE, PRINTER_HALTED_TEXT)
            return
        if not self.ui.chkHomeClear.isChecked():
            return
        if self._home_kind() == "board_zero":
            blocked = self._easy_invalid_yaml_reason()
            if blocked:
                QMessageBox.warning(self.ui, DIALOG_TITLE, blocked)
                return
        self._clear_board_zero()
        self._update_nav()
        kind = self._home_kind()
        status = {
            "board_zero": "Homing X/Y, probing P1, then setting board zero…",
            "teaching": "Lifting Z, then homing X/Y for teaching…",
        }.get(kind, "Homing X/Y… the window stays usable; this can take a few seconds.")
        self._start_printer_job(
            self._do_home_xy,
            self._on_home_ok,
            self._on_home_failed,
            self.ui.posLabel,
            status,
        )

    def _home_kind(self):
        """rectangle, teaching (no P1), or board_zero (taught P1)."""
        if self._mode() != MODE_PCB:
            return "rectangle"
        if self._p1_bltouch_commanded_xy() is not None:
            return "board_zero"
        return "teaching"

    def _do_home_xy(self):
        if self._printer_halted:
            raise engine_printer.PrinterHalted(PRINTER_HALTED_TEXT)
        kind = self._home_kind()
        if kind == "board_zero":
            blocked = self._easy_invalid_yaml_reason()
            if blocked:
                raise RuntimeError(blocked)
        self._clear_board_zero()
        printer = self._printer(manage_z=self._mode() == MODE_PCB)
        printer.drain()
        printer.prepare()
        note = None
        result = None
        try:
            if kind == "board_zero":
                if self._easy_mode_active():
                    self._easy_reenable_z_if_unlocked(printer)
                result = engine_printer.home_xy_and_set_board_zero(
                    printer,
                    self._p1_bltouch_commanded_xy(),
                    lift_mm=self._safe_home_lift_mm(),
                    z_max_mm=self._logical_z_max_mm(),
                    retract_mm=self._printer_config().probe_clearance_mm,
                )
                if result is not None:
                    self._commit_board_zero(result)
                    self._try_finish_easy_scan_height(printer)
            elif kind == "teaching":
                self._easy_reenable_z_if_unlocked(printer)
                engine_printer.lift_then_home_xy(
                    printer,
                    lift_mm=self._safe_home_lift_mm(),
                    z_max_mm=self._logical_z_max_mm(),
                )
            else:
                printer.home_xy()
        finally:
            self._capture_printer_frame(printer)
        return {
            "kind": kind,
            "position": printer.get_xy(),
            "result": result,
            "note": note,
        }

    def _on_home_ok(self, payload):
        if isinstance(payload, tuple):
            position, note = payload
            kind = "rectangle"
            result = None
        else:
            position = payload["position"]
            note = payload.get("note")
            kind = payload.get("kind") or self._home_kind()
            result = payload.get("result")
        self._pcb_xy_homed = True
        self._easy_height_lost_to_reset = False
        if kind in ("teaching", "board_zero"):
            self._fixture_xyz_homed = True
        self._bump_motion()
        self._clear_landmarks_and_alignment()
        self._show_position(position)
        if kind == "board_zero" and result is not None:
            self._commit_board_zero(result)
            status = self._step2_status_text()
            self.ui.posLabel.setText(status)
            fixture_status = getattr(self.ui, "fixtureTeachStatus", None)
            if fixture_status is not None:
                fixture_status.setText(status)
        elif kind == "teaching":
            status = (
                "Z lifted and X/Y homed. Jog the BLTouch over a reference mark, "
                "select its row, then probe. Step 2 is not complete until board zero is set."
            )
            fixture_status = getattr(self.ui, "fixtureTeachStatus", None)
            if fixture_status is not None:
                fixture_status.setText(status)
        self._easy_try_load_profile()
        self._update_easy_status()
        self._refresh_height_ui()
        self._update_nav()
        if note:
            QMessageBox.warning(self.ui, DIALOG_TITLE, note)

    def _on_home_failed(self, exc):
        if self._is_printer_halt(exc):
            self._handle_printer_halt(exc)
            return
        if isinstance(exc, engine_printer.PrinterReset):
            self._easy_z_unlocked = False
            self._clear_height_datum()
        self._clear_board_zero()
        self._clear_pcb_homing()
        message = f"Homing failed: {exc}"
        self.ui.posLabel.setText(message)
        fixture_status = getattr(self.ui, "fixtureTeachStatus", None)
        if fixture_status is not None:
            fixture_status.setText(message)
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
            self._clear_height_datum()
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
        self._machine_xy_at = time.monotonic()
        self.ui.posLabel.setText(
            f"Machine X {self._machine_xy[0]:.2f}  Y {self._machine_xy[1]:.2f} mm"
        )
        widget = getattr(self, "_scan_preview_widget", None)
        if widget is not None:
            widget.set_reported_xy(self._machine_xy, stale=False)

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
        transform = self._scan_transform()
        target = transform.to_machine((board_x, board_y))
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
        self._stl_board_placement = None
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
        extra = []
        side = self._scan_setup_side()
        state = self._pocket_pair_state.get(side)
        if state == "stale":
            extra.append(
                "saved pocket corners are from an older meaning; recapture A and B"
            )
        if self._pocket_block_reason:
            extra.append(self._pocket_block_reason)
        if self._board_view is None:
            return extra + ["no board imported"]
        if self._registration is None:
            if extra:
                return extra
            return extra + ["fewer than two landmarks recorded"]
        problems = engine_registration.registration_problems(
            self._registration,
            diagonal_mm=self._board_view.diagonal_mm,
            step_mm=self.ui.stepMm.value(),
        )
        return extra + problems + self._verification_problems()

    # ------------------------------------------------------------- profiles
    def _refresh_profiles(self):
        """Repopulate the chooser, keeping the current selection if it survives."""
        combo = self.ui.fixtureProfile
        wanted = combo.currentText()
        try:
            engine_profiles.seed_glassboard_v1_profile()
        except (OSError, engine_profiles.ProfileError, TypeError) as exc:
            logging.info(f"EMI map could not seed Glassboard V1 profile: {exc}")
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
        self._stl_board_placement = None
        self._fit_error = ""
        self._invalidate_plan()
        self._bump_registration()
        self._profile = loaded
        self._offset_correction_mm = (0.0, 0.0)
        self._acknowledged_warnings = acknowledged
        self._noted_warnings = list(loaded.soft_warnings)
        self._clear_selection()
        self._update_registration_ui()
        self._update_easy_status()
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
        elif self._easy_fixture_ready():
            dx, dy = self._easy_xy_offset()
            lines.append("VERIFIED fixture (Easy). Landmark confirms not required.")
            if dx or dy:
                lines.append(f"Easy XY offset: X {dx:+.3f} Y {dy:+.3f} mm")
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
        self._update_easy_status()

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

        handle = serial.Serial(
            port,
            baud,
            timeout=PRINTER_SERIAL_TIMEOUT_S,
            write_timeout=2.0,
            # Holding DTR/RTS asserted can keep an auto-reset mainboard in
            # reset, which reads back as a silent port on a live CH340.
            dsrdtr=False,
            rtscts=False,
        )
        self._pulse_controller_reset(handle)
        return handle

    @staticmethod
    def _pulse_controller_reset(handle):
        """Release the reset lines and offer one boot pulse. No G-code is sent.

        A CH340 enumerates from USB 5 V while the controller is halted or held
        in reset, so the boot window can otherwise see nothing at all. Boards
        without DTR/RTS auto-reset simply ignore this.
        """
        def _set(name, value):
            try:
                setattr(handle, name, value)
            except (AttributeError, OSError, TypeError, ValueError):
                pass

        _set("dtr", True)
        _set("rts", True)
        time.sleep(0.12)
        _set("dtr", False)
        _set("rts", False)
        reset_input = getattr(handle, "reset_input_buffer", None)
        if callable(reset_input):
            try:
                reset_input()
            except (OSError, ValueError):
                pass

    def _open_printer(self):
        if self._printer_halted:
            raise engine_printer.PrinterHalted(PRINTER_HALTED_TEXT)
        if self.printer_serial is not None and self.printer_serial.is_open:
            return self.printer_serial
        port = self._printer_port()
        if not port:
            raise RuntimeError("Choose the printer serial port first")
        baud = self.ui.printerBaud.value()
        handle = self._connect_printer(port, baud)
        self.printer_serial = handle
        # Reconnect is not a verified physical origin.
        self._clear_height_datum()
        return handle

    def _connect_printer(self, port, baud):
        """Open and identify one exact port/baud pair, closing it on refusal."""
        try:
            handle = self._open_serial(port, baud)
        except OSError as exc:
            raise RuntimeError(
                f"Could not open {port} at {baud} baud. Close Cura, Pronterface, "
                "Arduino Serial Monitor, or any other program using the printer, "
                f"then click RETRY PRINTER CONNECTION. Details: {exc}"
            ) from None
        try:
            self._greet_marlin(handle, port)
        except Exception:
            self._release_serial_handle(handle)
            raise
        return handle

    def _greet_marlin(self, handle, port):
        """Refuse a port that is not a printer, in seconds rather than minutes.

        A silent port is otherwise only discovered one 30 s ok timeout at a
        time, and the operator sees a wizard that appears to have hung.
        """
        greet_marlin_handle(
            handle, port, boot_s=PRINTER_BOOT_S, greet_s=PRINTER_GREET_S
        )

    @staticmethod
    def _release_serial_handle(handle):
        """Close without hanging the CH340 driver on a DTR edge."""
        if handle is None:
            return
        for name in ("cancel_read", "cancel_write"):
            cancel = getattr(handle, name, None)
            if callable(cancel):
                try:
                    cancel()
                except (AttributeError, OSError, TypeError, ValueError):
                    pass
        try:
            handle.timeout = 0.1
            handle.write_timeout = 0.1
        except (AttributeError, OSError, TypeError, ValueError):
            pass
        closer = getattr(handle, "close", None)
        if callable(closer):
            closer()

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
            self._maybe_resume_cube(config, plan)
            self._warn_long_cube_scan(config, plan)
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
            if self._cube_selected() and not self._rf_confirmed():
                return RF_CHAIN_CONFIRM_HINT
            return ""
        try:
            self._pcb_preflight(live_height_check=False)
        except (ValueError, RuntimeError) as exc:
            return str(exc)
        if self._cube_selected() and not self._rf_confirmed():
            return RF_CHAIN_CONFIRM_HINT
        return ""

    def _apply_scan_gate(self):
        """Keep Start in step with the visible reason; do not clobber live status."""
        scanning = self.thread is not None
        reason = "" if scanning else self._scan_block_reason()
        self.ui.btnStartScan.setEnabled(not scanning and not reason)
        self.ui.btnStartScan.setToolTip(reason or "Start the scan.")
        start_reason = getattr(self.ui, "scanStartReason", None)
        if start_reason is not None and callable(getattr(start_reason, "setText", None)):
            start_reason.setText("" if scanning else reason)
            if callable(getattr(start_reason, "setVisible", None)):
                start_reason.setVisible(not scanning and bool(reason))
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
            self._assert_cube_rf_confirmed(config)
            return None
        self._pcb_preflight()
        self._assert_cube_rf_confirmed(config)
        return self._build_validated_plan(config)

    def _assert_cube_rf_confirmed(self, config):
        if (
            getattr(config, "acquisition", "") == "spectrum_cube"
            and not config.rf_configuration_confirmed
        ):
            raise RuntimeError(RF_CHAIN_CONFIRM_HINT)

    def _pcb_preflight(self, *, live_height_check=True):
        if self._board_view is None:
            raise RuntimeError("Import an ODB++ board first.")
        if not self._fixture_scan_ready():
            raise RuntimeError(
                "Glassboard fixture teaching/verification is BLOCKED. Teach or verify "
                "P1 in this machine session before scanning."
            )
        self._assert_scan_selection_ready()
        if not self._pcb_xy_homed:
            raise RuntimeError(
                "Home X and Y before a PCB-aligned scan: the landmarks are "
                "machine positions and only homing fixes that frame."
            )
        if (
            (self._stl_board_placement is not None or self._taught_pocket_placement is not None)
            and not self.ui.chkBoardSeated.isChecked()
        ):
            raise RuntimeError(
                "Confirm that the PCB is inserted in the selected fixture holder "
                "and fully seated before scanning."
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
        if self._easy_mode_active():
            reason = self._easy_height_block_reason()
            if reason:
                raise RuntimeError(reason)
        if live_height_check:
            self._ensure_at_agreed_scan_plane()
        if self._easy_mode_active() and self._glassboard_fixture_selected():
            fingerprint = self._current_scan_fingerprint()
            if not fingerprint or fingerprint != self._approved_scan_fingerprint:
                raise RuntimeError("Approve the scan plan on Step 4 before START.")
        if live_height_check and not self._easy_mode_active():
            try:
                engine_height.apply_scan_height_preflight(
                    self._height_snapshot(self._reported_z()),
                    self._scan_height_plan(),
                )
            except engine_height.HeightError as exc:
                raise RuntimeError(str(exc)) from exc

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
            self._scan_transform(),
            x_start=x_start,
            x_end=x_end,
            y_start=y_start,
            y_end=y_end,
            step_mm=step_mm,
            clip_to_outline=True,
            registration=self._registration,
            selection=self._scan_selection,
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
        if self._stl_board_placement is not None:
            machine_fixture = self._machine_fixture_document()
            return {
                "registration_source": "glassboard_stl_middle_holder",
                "placement": dict(self._stl_board_placement),
                "board_fully_seated_confirmed": self.ui.chkBoardSeated.isChecked(),
                "machine_fixture_name": self._fixture_profile_name(),
                "machine_fixture_verified_this_session": self._fixture_verified_session,
                "machine_fixture_document": machine_fixture,
            }
        if self._profile is None:
            return {"registration_source": "manual_landmarks"}
        dx, dy = self._offset_correction_mm
        machine_fixture = self._machine_fixture_document()
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
            "machine_fixture_name": self._fixture_profile_name(),
            "machine_fixture_verified_this_session": self._fixture_verified_session,
            "machine_fixture_document": machine_fixture,
        }

    def _write_profile_copy(self, output_dir):
        """Drop the active profile beside the scan, from memory not from disk.

        Copying the file would attach whatever is on disk now, which is not
        necessarily what was loaded: a save between loading and scanning would
        silently document the wrong alignment.
        """
        try:
            if self._profile is not None:
                target = Path(output_dir) / "fixture_profile.json"
                target.write_text(
                    json.dumps(self._profile.document, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            machine_fixture = self._machine_fixture_document()
            if machine_fixture is not None:
                (Path(output_dir) / "machine_fixture.json").write_text(
                    json.dumps(machine_fixture, indent=2, sort_keys=True),
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
            self._cube_live = getattr(config, "acquisition", "single_span") == "spectrum_cube"
            self.ui.scanProgress.setMaximum(self._total)
            self.ui.scanProgress.setValue(0)

            self.worker = ScanWorker(
                config,
                dev.usb,
                transport,
                plan,
                self._provenance(),
                height=self._height_for_scan(),
            )
            self.thread = QtCore.QThread(self)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.status.connect(self.ui.scanStatus.setText)
            self.worker.point.connect(self._on_point)
            self.worker.cell_begin.connect(self._on_cell_begin)
            self.worker.row.connect(self._on_row)
            self.worker.spectrum.connect(self._on_spectrum)
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
        shape = (area.ny, area.nx)
        self._live_background_grid = np.full(shape, np.nan)
        self._live_dut_grid = np.full(shape, np.nan)
        self._live_delta_grid = np.full(shape, np.nan)
        self._live_background_spectra = {}
        self._live_background_kind = str(getattr(config, "background_kind", "xy_grid") or "xy_grid")
        self._live_stationary_background_dbm = np.nan
        self._live_last_pass = ""
        self._clear_live_cell_labels()

        # Background runs start by showing the measured DUT-OFF map.  The first
        # ON cell that has a matching OFF cell switches the display to delta dB.
        self._grid = self._live_background_grid if config.background else self._live_dut_grid
        self._live_units = "dBm"
        if config.background and self._live_background_kind == "stationary":
            self._live_map_caption = "DUT OFF / stationary reference"
        elif config.background:
            self._live_map_caption = "DUT OFF / ambient"
        else:
            self._live_map_caption = "DUT ON"

        # Levels are mandatory for a float image: without them pyqtgraph raises
        # inside Qt's paint callback, and a raising paint repeats until the
        # process dies. The empty grid renders nothing, so any range will do
        # until the first reading replaces it.
        origin_x, origin_y = area.origin_x_mm, area.origin_y_mm
        image = getattr(self, "image", None)
        if image is not None and callable(getattr(image, "setImage", None)):
            image.setImage(self._grid.T, autoLevels=False, levels=PLACEHOLDER_LEVELS)
            # Board-view millimetres do not start at zero, so the image rect and the
            # ranges follow the area's origin rather than assuming it.
            image.setRect(
                QtCore.QRectF(origin_x, origin_y, area.width_mm, area.height_mm)
            )
        self.ui.emiPlot.setXRange(origin_x, origin_x + max(area.width_mm, area.step_mm))
        self.ui.emiPlot.setYRange(origin_y, origin_y + max(area.height_mm, area.step_mm))
        board = plan is not None
        self.ui.emiPlot.setLabel("bottom", "Board X (mm)" if board else "X (mm from origin)")
        self.ui.emiPlot.setLabel("left", "Board Y (mm)" if board else "Y (mm from origin)")
        self.ui.emiPlot.setTitle(f"Live map — {self._live_map_caption}")
        if board:
            self._draw_board_under_map(plan.board_view)

    @staticmethod
    def _pass_kind(info):
        """Normalise scanner callback names without coupling Qt to scanner internals."""
        text = str(info.get("pass", info.get("phase", ""))).strip().lower()
        if text in ("background", "ambient", "dut off", "off") or "dut off" in text:
            return "background"
        if text in ("dut", "dut on", "on") or (
            text.startswith("dut") and "off" not in text
        ):
            return "dut"
        label = str(info.get("label", "")).strip().lower()
        if label in ("hold", "preflight"):
            return "background"
        return text

    def _store_live_background_spectrum(self, power, key=None):
        """Keep a running per-bin max so live subtraction matches software max-hold."""
        held = np.asarray(power, dtype=float).copy()
        if self._live_background_kind == "stationary":
            existing = self._live_background_spectra.get("stationary")
            if existing is not None and existing.shape == held.shape:
                held = np.maximum(existing, held)
            self._live_background_spectra["stationary"] = held
            finite = held[np.isfinite(held)]
            if finite.size:
                self._live_stationary_background_dbm = float(finite.max())
            return
        if key is None:
            return
        existing = self._live_background_spectra.get(key)
        if existing is not None and existing.shape == held.shape:
            held = np.maximum(existing, held)
        self._live_background_spectra[key] = held

    def _clear_live_cell_labels(self):
        plot = getattr(self.ui, "emiPlot", None)
        for item in getattr(self, "_live_label_items", {}).values():
            try:
                plot.removeItem(item)
            except Exception:
                pass
        self._live_label_items = {}

    def _update_live_cell_label(self, iy, ix):
        """Put the actual measured/calculated number in the centre of one cell."""
        if self._grid is None or not np.isfinite(self._grid[iy, ix]):
            return
        try:
            import pyqtgraph

            # Mapping through ImageItem itself guarantees the text stays centred
            # even if the image rect is not exactly nx*step by ny*step.
            point = self.image.mapToParent(QtCore.QPointF(ix + 0.5, iy + 0.5))
            value = float(self._grid[iy, ix])
            text = f"{value:+.1f}" if self._live_units == "dB" else f"{value:.1f}"
            key = (int(iy), int(ix))
            item = self._live_label_items.get(key)
            if item is None:
                item = pyqtgraph.TextItem(
                    text=text,
                    anchor=(0.5, 0.5),
                    color="w",
                    fill=pyqtgraph.mkBrush(0, 0, 0, 125),
                    border=pyqtgraph.mkPen(255, 255, 255, 70),
                )
                item.setZValue(10)
                self.ui.emiPlot.addItem(item)
                self._live_label_items[key] = item
            else:
                item.setText(text)
            item.setPos(point.x(), point.y())
        except Exception:
            # Labels are presentation-only.  A plotting-version mismatch must
            # never stop or invalidate a measurement.
            return

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
        self._board_trace.setZValue(20)
        self.ui.emiPlot.addItem(self._board_trace)

    def _set_scanning(self, scanning):
        # The board and the alignment must not change under a running scan
        for page in (self.ui.pageSetup, self.ui.pageBoard, self.ui.pageRegister):
            page.setEnabled(not scanning)
        self._update_nav()

    def _on_point(self, info):
        self._done += 1
        self.ui.scanProgress.setValue(self._done)
        widget = getattr(self, "_scan_preview_widget", None)
        if widget is not None:
            widget.mark_completed(
                (float(info.get("machine_x_mm", 0.0)), float(info.get("machine_y_mm", 0.0)))
            )

        iy, ix = int(info["iy"]), int(info["ix"])
        value = float(info["power_dbm"])
        pass_kind = self._pass_kind(info)

        if pass_kind == "background" and self._live_background_grid is not None:
            self._live_background_grid[iy, ix] = value
            if self._live_background_kind == "stationary":
                # A stationary reference is one DUT-OFF level reused for the ON
                # map.  Keep it separate so the caption never implies an XY OFF scan.
                self._live_stationary_background_dbm = value
            if self._live_last_pass != "background":
                self._grid = self._live_background_grid
                self._live_units = "dBm"
                self._live_map_caption = (
                    "DUT OFF / stationary reference"
                    if self._live_background_kind == "stationary"
                    else "DUT OFF / ambient"
                )
                self._clear_live_cell_labels()
            self._live_last_pass = "background"

        elif pass_kind == "dut" and self._live_dut_grid is not None:
            self._live_dut_grid[iy, ix] = value
            if self._live_background_kind == "stationary":
                background = self._live_stationary_background_dbm
                caption = "DUT ON − stationary DUT-OFF reference"
                state = "dut_stationary_delta"
            else:
                background = (
                    self._live_background_grid[iy, ix]
                    if self._live_background_grid is not None
                    else np.nan
                )
                caption = "DUT ON − DUT OFF"
                state = "dut_delta"
            if np.isfinite(background):
                self._live_delta_grid[iy, ix] = value - float(background)
                if self._live_last_pass != state:
                    self._grid = self._live_delta_grid
                    self._live_units = "dB"
                    self._live_map_caption = caption
                    self._clear_live_cell_labels()
                self._live_last_pass = state
            else:
                self._grid = self._live_dut_grid
                self._live_units = "dBm"
                self._live_map_caption = "DUT ON (background not available yet)"
                self._live_last_pass = "dut_raw"
        else:
            # Backwards-compatible path for scanners that do not name the pass.
            if self._live_dut_grid is not None:
                self._live_dut_grid[iy, ix] = value
                self._grid = self._live_dut_grid
            self._live_units = "dBm"
            self._live_map_caption = "Measured level"

        self._update_live_cell_label(iy, ix)

        measured = info.get("measured_cell_s")
        if measured and self.ui.scanProgress.maximum():
            remaining = max(self.ui.scanProgress.maximum() - self._done, 0) * float(measured)
            minutes = remaining / 60.0
            self.ui.scanStatus.setText(
                f"{info.get('pass', '')} {self._done}/{self.ui.scanProgress.maximum()} "
                f"ETA {minutes:.1f} min"
            )
        self._pending_updates += 1
        refresh_every = 1 if getattr(self, "_cube_live", False) else 8
        if self._pending_updates >= refresh_every:
            self._refresh_image()

    def _on_row(self, info):
        self._refresh_image()
        self.ui.scanStatus.setText(
            f"{info['pass']} pass: row {info['iy'] + 1} of {info['rows']} complete"
        )

    def _on_spectrum(self, info):
        """Draw the newest spectrum; show ON-OFF dB immediately when possible."""
        self._live_spectrum = info
        freqs = np.asarray(info.get("freqs"), dtype=float)
        power = np.asarray(info.get("power_dbm"), dtype=float)
        if freqs.size < 2 or power.shape != freqs.shape:
            return

        pass_kind = self._pass_kind(info)
        if pass_kind not in ("background", "dut"):
            label = str(info.get("label", "")).strip().lower()
            if label == "cell":
                if self._live_background_kind == "stationary":
                    pass_kind = "dut"
                elif self._live_last_pass == "background":
                    pass_kind = "background"
                elif str(self._live_last_pass).startswith("dut"):
                    pass_kind = "dut"
        iy, ix = info.get("iy"), info.get("ix")
        key = None
        if iy is not None and ix is not None:
            key = (int(iy), int(ix))

        shown = power
        units = "dBm"
        title = "Live spectrum"
        if pass_kind == "background":
            self._store_live_background_spectrum(power, key)
            title = (
                "Live spectrum — DUT OFF / stationary reference"
                if self._live_background_kind == "stationary"
                else "Live spectrum — DUT OFF"
            )
        elif pass_kind == "dut":
            if self._live_background_kind == "stationary":
                reference = self._live_background_spectra.get("stationary")
                delta_title = "Live spectrum — DUT ON − stationary DUT-OFF reference"
            else:
                reference = (
                    self._live_background_spectra.get(key) if key is not None else None
                )
                delta_title = "Live spectrum — DUT ON − DUT OFF"
            if reference is not None and reference.shape == power.shape:
                shown = power - reference
                units = "dB"
                title = delta_title
            else:
                title = "Live spectrum — DUT ON"

        curve = getattr(self, "_live_spectrum_curve", None)
        if curve is None:
            return
        curve.setData(freqs / 1e6, shown)
        live = getattr(self.ui, "liveSpectrum", None)
        if live is not None:
            if callable(getattr(live, "setLabel", None)):
                live.setLabel("left", "Difference (dB)" if units == "dB" else "Power (dBm)")
            if callable(getattr(live, "setTitle", None)):
                live.setTitle(title)
            if callable(getattr(live, "enableAutoRange", None)):
                live.enableAutoRange(axis="y", enable=True)

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
        image = getattr(self, "image", None)
        if image is not None and callable(getattr(image, "setImage", None)):
            image.setImage(self._grid.T, autoLevels=False, levels=(low, high))
        self.ui.emiPlot.setTitle(
            f"Live map — {self._live_map_caption}   "
            f"[{low:+.1f} to {high:+.1f} {self._live_units}]"
        )
        bar = getattr(self, "_live_colorbar", None)
        if bar is not None:
            try:
                bar.setLevels((low, high))
                axis = getattr(bar, "axis", None)
                if axis is not None and callable(getattr(axis, "setLabel", None)):
                    axis.setLabel(f"Level ({self._live_units})")
            except Exception:
                pass

    def _on_prompt(self, message=""):
        self.ui.btnContinueDut.setEnabled(True)
        text = message or (
            "Background pass complete. Turn the DUT ON without moving it, then continue."
        )
        self.ui.scanStatus.setText(text)
        if "OFF" in text:
            self.ui.btnContinueDut.setText("DUT is OFF — continue")
        else:
            self.ui.btnContinueDut.setText("DUT is ON — continue")
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
        self._load_cube_results(result)
        self.ui.wizardStack.setCurrentIndex(PAGE_RESULTS)
        self._fit_results_image()

    def _load_cube_results(self, result):
        spectra = Path(result.output_dir) / "spectra.npz"
        self._cube_data = None
        self._cube_picks = []
        if not spectra.exists():
            return
        from EMI_Mapper.storage import load_spectra_cube

        payload = load_spectra_cube(spectra)
        self._cube_data = payload
        slider = getattr(self.ui, "cubeFreqSlider", None)
        if slider is None:
            return
        n_freq = int(np.asarray(payload["freqs"]).size)
        slider.setMinimum(0)
        slider.setMaximum(max(n_freq - 1, 0))
        slider.setValue(min(slider.value(), max(n_freq - 1, 0)))
        self._refresh_cube_view()

    def _cube_map_kind_value(self):
        combo = getattr(self.ui, "cubeMapKind", None)
        if combo is None:
            return "band_max"
        data = combo.currentData() if callable(getattr(combo, "currentData", None)) else None
        return str(data) if data not in (None, "") else str(combo.currentText())

    def _cube_quantity_value(self):
        combo = getattr(self.ui, "cubeQuantity", None)
        if combo is None:
            return "raw_dbm"
        data = combo.currentData() if callable(getattr(combo, "currentData", None)) else None
        return str(data) if data not in (None, "") else str(combo.currentText())

    def _refresh_cube_view(self, *_args):
        if self._cube_data is None:
            return
        from EMI_Mapper.processing import (
            EmptyBand,
            band_max_map,
            single_frequency_map,
            sum_of_measured_bin_powers,
        )

        freqs = np.asarray(self._cube_data["freqs"], dtype=float)
        dut_cube = np.asarray(self._cube_data["dut"], dtype=float)
        cube = dut_cube
        kind = self._cube_map_kind_value()
        units = "dBm"
        title = "DUT ON"
        reference_note = ""

        if kind in ("dut_on_minus_dut_off", "dB_above_stationary_reference"):
            # Prefer a true XY DUT-OFF cube when the storage format provides it.
            # Falling back to a stationary reference is scientifically different,
            # so that fact is kept visible in both the map title and frequency label.
            background_cube = self._cube_data.get("background")
            if background_cube is None:
                background_cube = self._cube_data.get("background_cube")
            if background_cube is not None:
                background_cube = np.asarray(background_cube, dtype=float)
                if background_cube.shape == dut_cube.shape:
                    cube = dut_cube - background_cube
                    units = "dB"
                    title = "DUT ON − DUT OFF"
                else:
                    cube = np.full_like(dut_cube, np.nan)
                    units = "dB"
                    title = "DUT ON − DUT OFF (background shape mismatch)"
            else:
                reference = self._cube_data.get("background_reference")
                if reference is None:
                    cube = np.full_like(dut_cube, np.nan)
                    units = "dB"
                    title = "DUT ON − DUT OFF (no background saved)"
                else:
                    from EMI_Mapper.processing import db_above_stationary_reference

                    cube = db_above_stationary_reference(dut_cube, reference)
                    units = "dB"
                    title = "DUT ON − stationary DUT-OFF reference"
                    reference_note = "stationary reference, not a per-cell OFF scan"

        quantity = getattr(self.ui, "cubeQuantity", None)
        if (
            quantity is not None
            and self._cube_quantity_value() == "characterized"
            and self.result is not None
            and kind not in ("dut_on_minus_dut_off", "dB_above_stationary_reference")
        ):
            characterized = Path(self.result.output_dir) / "characterized.npz"
            if characterized.exists():
                with np.load(characterized) as data:
                    cube = np.asarray(data["characterized_spectrum"], dtype=float)
                characterized_meta = getattr(self.result, "characterization", None) or {}
                units = characterized_meta.get("output_units") or "characterized"
                title = "Characterized field"

        completed = np.asarray(
            self._cube_data.get("dut_completed", np.isfinite(dut_cube).any(axis=-1))
        )
        slider = self.ui.cubeFreqSlider
        index = int(slider.value()) if freqs.size else 0
        center = float(freqs[index]) if freqs.size else 0.0
        bandwidth = float(self.ui.cubeBandwidthMhz.value()) * 1e6
        start = center - bandwidth / 2.0
        stop = center + bandwidth / 2.0

        if bandwidth <= 0 or kind == "single_frequency":
            values, actual = single_frequency_map(cube, freqs, center)
            freq_text = f"{actual / 1e6:.3f} MHz"
        else:
            try:
                if kind == "sum_of_measured_bin_powers":
                    values = sum_of_measured_bin_powers(cube, freqs, start, stop)
                else:
                    values = band_max_map(cube, freqs, start, stop)
            except EmptyBand:
                values = np.full(cube.shape[:2], np.nan)
            freq_text = f"{center / 1e6:.3f} MHz"

        if reference_note:
            freq_text += f" — {reference_note}"
        self.ui.cubeFreqLabel.setText(freq_text)
        values = np.where(completed, values, np.nan)
        self._show_array_on_results(values, units=units, title=title)
        self._update_cube_spectrum_label()

    def cube_pick_cell(self, iy, ix):
        """Record up to two completed cells for spectrum comparison."""
        if self._cube_data is None:
            return []
        completed = np.asarray(
            self._cube_data.get("dut_completed", np.ones(self._cube_data["dut"].shape[:2]))
        )
        if iy < 0 or ix < 0 or iy >= completed.shape[0] or ix >= completed.shape[1]:
            return []
        if not completed[iy, ix]:
            return list(self._cube_picks)
        pick = (int(iy), int(ix))
        if pick in self._cube_picks:
            self._cube_picks = [item for item in self._cube_picks if item != pick]
        else:
            self._cube_picks.append(pick)
            self._cube_picks = self._cube_picks[-2:]
        self._update_cube_spectrum_label()
        self._fit_results_image()
        return list(self._cube_picks)

    def cube_cell_spectrum(self, iy, ix):
        if self._cube_data is None:
            return None
        spectrum = np.asarray(self._cube_data["dut"][iy, ix], dtype=float)
        kind = self._cube_map_kind_value()
        if kind in ("dut_on_minus_dut_off", "dB_above_stationary_reference"):
            background = self._cube_data.get("background")
            if background is None:
                background = self._cube_data.get("background_cube")
            if background is not None:
                background = np.asarray(background, dtype=float)
                if background.shape == np.asarray(self._cube_data["dut"]).shape:
                    return spectrum - background[iy, ix]
            reference = self._cube_data.get("background_reference")
            if reference is not None:
                reference = np.asarray(reference, dtype=float)
                if reference.shape == spectrum.shape:
                    return spectrum - reference
        return spectrum

    def _update_cube_spectrum_label(self):
        label = getattr(self.ui, "cubeSpectrumLabel", None)
        if label is None or self._cube_data is None:
            return
        freqs = np.asarray(self._cube_data["freqs"], dtype=float)
        if not self._cube_picks:
            label.setText("Hover for a value; click up to two completed cells to compare spectra")
            return
        delta_view = self._cube_map_kind_value() in (
            "dut_on_minus_dut_off",
            "dB_above_stationary_reference",
        )
        units = "dB" if delta_view else "dBm"
        parts = []
        for iy, ix in self._cube_picks:
            spectrum = self.cube_cell_spectrum(iy, ix)
            if spectrum is None or not np.any(np.isfinite(spectrum)):
                continue
            peak_i = int(np.nanargmax(spectrum))
            parts.append(
                f"({ix},{iy}) peak {spectrum[peak_i]:+.1f} {units} at "
                f"{freqs[peak_i] / 1e6:.3f} MHz"
            )
        if len(parts) == 2:
            parts.append("Proximity to CAD is a source candidate, not a confirmed source.")
        label.setText(" | ".join(parts) if parts else "Selected cell has no spectrum")

    def _on_cube_animate_toggled(self, checked: bool):
        if self._cube_animate is not None:
            self._cube_animate.stop()
            self._cube_animate = None
        if not checked or self._cube_data is None:
            return
        slider = getattr(self.ui, "cubeFreqSlider", None)
        if slider is None or not hasattr(QtCore, "QTimer"):
            return
        timer = QtCore.QTimer(self)
        timer.setInterval(120)

        def _step():
            if slider.maximum() <= 0:
                return
            slider.setValue((int(slider.value()) + 1) % (int(slider.maximum()) + 1))

        timer.timeout.connect(_step)
        timer.start()
        self._cube_animate = timer

    def _cube_cell_count(self, config, plan):
        if plan is not None:
            return int(plan.point_count)
        return int(config.area.nx * config.area.ny)

    def _maybe_resume_cube(self, config, plan):
        """Offer to continue the newest incomplete cube with the same plan."""
        if getattr(config, "acquisition", "single_span") != "spectrum_cube":
            return
        folder = self._find_resumable_cube(config, plan)
        if folder is None:
            return
        answer = QMessageBox.question(
            self.ui,
            DIALOG_TITLE,
            (
                f"Resume incomplete cube scan in {folder.name}?\n"
                "Yes keeps existing cells and RF settings. No starts a new folder."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            config.resume_output_dir = str(folder)

    def _find_resumable_cube(self, config, plan):
        root = Path(config.output_root)
        if not root.is_dir():
            return None
        plan_id = "" if plan is None else plan.plan_id
        for child in sorted(root.iterdir(), key=lambda path: path.name, reverse=True):
            snap_path = child / "snapshot.json"
            state_path = child / "scan_state.npz"
            if not snap_path.exists() or not state_path.exists():
                continue
            if (child / "spectra.npz").exists():
                continue
            try:
                snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if snapshot.get("acquisition") != "spectrum_cube":
                continue
            saved_plan = (snapshot.get("plan") or {}).get("plan_id", "")
            if saved_plan != plan_id:
                continue
            return child
        return None

    def _warn_long_cube_scan(self, config, plan):
        if getattr(config, "resume_output_dir", None):
            return
        if getattr(config, "acquisition", "single_span") != "spectrum_cube":
            return
        from EMI_Mapper.spectrum_cube import cube_workload

        work = cube_workload(config, self._cube_cell_count(config, plan))
        has_roi = self._scan_selection is not None
        if has_roi or work["sweeps"] < 200:
            return
        QMessageBox.warning(
            self.ui,
            DIALOG_TITLE,
            (
                f"Long cube scan without an ROI: {work['cells']} cells, "
                f"{work['segments']} segments, {work['repeats']} repeats, "
                f"{work['sweeps']} sweeps. This is a duration warning only."
            ),
        )

    def _show_cube_scan_estimate(self):
        try:
            config = self.build_config()
        except (ValueError, RuntimeError):
            return
        if getattr(config, "acquisition", "single_span") != "spectrum_cube":
            return
        from EMI_Mapper.spectrum_cube import cube_workload

        work = cube_workload(config, self._cube_cell_count(config, self._plan))
        self.ui.scanStatus.setText(
            f"Cube {work['cells']} cells, {work['segments']} segments, "
            f"{work['repeats']} repeats, {work['sweeps']} sweeps. "
            "Remaining time is measured after the first complete cell."
        )

    @staticmethod
    def _inferno_rgb(t):
        """Small dependency-free inferno approximation shared by Qt cell drawing."""
        stops = (
            (0, 0, 4),
            (40, 11, 84),
            (101, 21, 110),
            (159, 42, 99),
            (212, 72, 66),
            (245, 125, 21),
            (250, 193, 39),
            (252, 255, 164),
        )
        t = min(1.0, max(0.0, float(t)))
        p = t * (len(stops) - 1)
        i = min(len(stops) - 2, int(p))
        f = p - i
        a, b = stops[i], stops[i + 1]
        return tuple(int(round(a[k] + (b[k] - a[k]) * f)) for k in range(3))

    def _show_array_on_results(self, values, *, units=None, title=None):
        """Show a crisp, annotated measurement grid instead of a blurred bitmap."""
        values = np.asarray(values, dtype=float)
        self._results_values = values.copy()
        if units:
            self._results_units = str(units)
        if title:
            self._results_title = str(title)
        self._fit_results_image()

    def _analyzer_html_path(self, result=None):
        """Absolute EMI_Analyzer.html for this result, from the files map or disk."""
        result = self.result if result is None else result
        if result is None:
            return None
        files = getattr(result, "files", None) or {}
        recorded = files.get("analyzer_html")
        if recorded:
            recorded_path = Path(str(recorded))
            if recorded_path.is_file():
                return recorded_path
        output_dir = getattr(result, "output_dir", None)
        if output_dir:
            candidate = Path(output_dir) / "EMI_Analyzer.html"
            if candidate.is_file():
                return candidate
        return None

    def _sync_results_open_buttons(self, result=None):
        """Enable Open scan folder / Open EMI Analyzer whenever a scan folder exists."""
        result = self.result if result is None else result
        output_dir = getattr(result, "output_dir", None) if result is not None else None
        has_folder = bool(output_dir)
        html = self._analyzer_html_path(result)
        folder = getattr(self.ui, "btnOpenFolder", None)
        overlay = getattr(self.ui, "btnOpenOverlay", None)
        if folder is not None:
            folder.setEnabled(has_folder)
        if overlay is not None:
            overlay.setEnabled(has_folder)
            overlay.setToolTip(
                ""
                if html
                else "Not built during the scan. Click to build it from this scan folder."
            )

    def _show_board_artifacts(self, result):
        self._sync_results_open_buttons(result)
        image_path = (
            result.files.get("board_png")
            or result.files.get("heatmap_png")
            or result.files.get("characterized_heatmap_png")
        )
        if not image_path:
            self._results_pixmap = None
            self.ui.boardImage.clear()
            return
        self._results_pixmap = QtGui.QPixmap(str(image_path))
        self._results_values = None
        self._fit_results_image()

    def _fit_results_image(self):
        """Render results at widget resolution with hard cell edges and dB labels."""
        label = getattr(self.ui, "boardImage", None)
        if label is None or not hasattr(label, "setPixmap"):
            return
        size = label.size() if hasattr(label, "size") else None
        if size is None or not hasattr(size, "width") or size.width() < 64 or size.height() < 64:
            if self._results_values is None and self._results_pixmap is not None:
                label.setPixmap(self._results_pixmap)
            return

        # Non-cube/static artifact fallback: if there is no numerical grid yet,
        # at least preserve hard pixels rather than Qt's photo-style smoothing.
        if self._results_values is None:
            pixmap = self._results_pixmap
            if pixmap is None:
                return
            label.setPixmap(
                pixmap.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            )
            return

        values = np.asarray(self._results_values, dtype=float)
        if values.ndim != 2:
            return
        ny, nx = values.shape
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            label.clear()
            return
        lo, hi = float(finite.min()), float(finite.max())
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5

        width, height = int(size.width()), int(size.height())
        pixmap = QtGui.QPixmap(width, height)
        pixmap.fill(QtGui.QColor("#f7f8fa"))
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

        left_margin, top_margin, bottom_margin, right_margin = 58, 30, 42, 112
        available = QtCore.QRectF(
            left_margin,
            top_margin,
            max(1, width - left_margin - right_margin),
            max(1, height - top_margin - bottom_margin),
        )
        grid_aspect = nx / max(ny, 1)
        if available.width() / max(available.height(), 1.0) > grid_aspect:
            map_h = available.height()
            map_w = map_h * grid_aspect
            map_x = available.x() + (available.width() - map_w) / 2.0
            map_y = available.y()
        else:
            map_w = available.width()
            map_h = map_w / max(grid_aspect, 1e-9)
            map_x = available.x()
            map_y = available.y() + (available.height() - map_h) / 2.0
        map_rect = QtCore.QRectF(map_x, map_y, map_w, map_h)
        self._results_map_rect = map_rect
        cw, ch = map_rect.width() / nx, map_rect.height() / ny

        # Draw every measured cell exactly once.  No interpolation: each rectangle
        # is one physical probe location.  A subtle grid makes the sampling pitch visible.
        grid_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 105))
        grid_pen.setWidthF(0.8)
        painter.setPen(grid_pen)
        draw_numbers = cw >= 23 and ch >= 17
        font = QtGui.QFont(label.font())
        font.setPointSizeF(max(6.0, min(9.0, min(cw, ch) * 0.28)))
        font.setBold(True)
        painter.setFont(font)

        for iy in range(ny):
            display_row = ny - 1 - iy
            for ix in range(nx):
                value = values[iy, ix]
                rect = QtCore.QRectF(
                    map_rect.left() + ix * cw,
                    map_rect.top() + display_row * ch,
                    cw,
                    ch,
                )
                if not np.isfinite(value):
                    painter.fillRect(rect, QtGui.QColor(238, 240, 243))
                    painter.drawRect(rect)
                    continue
                t = (float(value) - lo) / (hi - lo)
                rgb = self._inferno_rgb(t)
                colour = QtGui.QColor(*rgb)
                painter.fillRect(rect, colour)
                painter.drawRect(rect)
                if draw_numbers:
                    # Luminance-based contrast keeps the small cell value readable
                    # across the complete inferno scale.
                    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                    painter.setPen(QtGui.QColor("#111111") if lum > 155 else QtGui.QColor("#ffffff"))
                    text = (
                        f"{float(value):+.1f}"
                        if self._results_units == "dB"
                        else f"{float(value):.1f}"
                    )
                    painter.drawText(rect, QtCore.Qt.AlignCenter, text)
                    painter.setPen(grid_pen)

        # Highlight up to two clicked cells; their spectra are compared below.
        pick_pen = QtGui.QPen(QtGui.QColor("#00d4ff"))
        pick_pen.setWidth(3)
        painter.setPen(pick_pen)
        for iy, ix in getattr(self, "_cube_picks", []):
            if 0 <= iy < ny and 0 <= ix < nx:
                display_row = ny - 1 - iy
                rect = QtCore.QRectF(
                    map_rect.left() + ix * cw,
                    map_rect.top() + display_row * ch,
                    cw,
                    ch,
                )
                painter.drawRect(rect.adjusted(1.5, 1.5, -1.5, -1.5))

        # Outer border + titles.
        painter.setPen(QtGui.QPen(QtGui.QColor("#20242a"), 1.2))
        painter.drawRect(map_rect)
        title_font = QtGui.QFont(label.font())
        title_font.setBold(True)
        title_font.setPointSizeF(9.5)
        painter.setFont(title_font)
        painter.drawText(
            QtCore.QRectF(map_rect.left(), 2, map_rect.width(), 24),
            QtCore.Qt.AlignCenter,
            self._results_title,
        )

        axis_font = QtGui.QFont(label.font())
        axis_font.setPointSizeF(8.0)
        painter.setFont(axis_font)
        painter.drawText(
            QtCore.QRectF(map_rect.left(), map_rect.bottom() + 7, map_rect.width(), 22),
            QtCore.Qt.AlignCenter,
            "Board X (grid cells)",
        )
        painter.save()
        painter.translate(16, map_rect.center().y())
        painter.rotate(-90)
        painter.drawText(
            QtCore.QRectF(-map_rect.height() / 2, -10, map_rect.height(), 20),
            QtCore.Qt.AlignCenter,
            "Board Y (grid cells)",
        )
        painter.restore()

        # Vertical inferno colourbar with numeric dB/dBm ticks.  This is the
        # requested second visual scale: colour is never left unexplained.
        bar_x = map_rect.right() + 30
        bar_w = 18
        bar_rect = QtCore.QRectF(bar_x, map_rect.top(), bar_w, map_rect.height())
        gradient = QtGui.QLinearGradient(0, bar_rect.bottom(), 0, bar_rect.top())
        stops = (
            (0.0, (0, 0, 4)),
            (1 / 7, (40, 11, 84)),
            (2 / 7, (101, 21, 110)),
            (3 / 7, (159, 42, 99)),
            (4 / 7, (212, 72, 66)),
            (5 / 7, (245, 125, 21)),
            (6 / 7, (250, 193, 39)),
            (1.0, (252, 255, 164)),
        )
        for stop, rgb in stops:
            gradient.setColorAt(stop, QtGui.QColor(*rgb))
        painter.fillRect(bar_rect, gradient)
        painter.setPen(QtGui.QPen(QtGui.QColor("#20242a"), 1.0))
        painter.drawRect(bar_rect)
        painter.setFont(axis_font)
        for j in range(5):
            frac = j / 4.0
            value = lo + frac * (hi - lo)
            y = bar_rect.bottom() - frac * bar_rect.height()
            painter.drawLine(QtCore.QPointF(bar_rect.right(), y), QtCore.QPointF(bar_rect.right() + 5, y))
            tick = f"{value:+.1f}" if self._results_units == "dB" else f"{value:.1f}"
            painter.drawText(
                QtCore.QRectF(bar_rect.right() + 8, y - 9, 52, 18),
                QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
                tick,
            )
        painter.save()
        painter.translate(bar_rect.right() + 70, bar_rect.center().y())
        painter.rotate(-90)
        painter.drawText(
            QtCore.QRectF(-bar_rect.height() / 2, -10, bar_rect.height(), 20),
            QtCore.Qt.AlignCenter,
            self._results_units,
        )
        painter.restore()

        painter.end()
        self._results_pixmap = pixmap
        label.setPixmap(pixmap)

    def eventFilter(self, obj, event):
        board = getattr(self.ui, "boardImage", None)
        if obj is board and event.type() == QtCore.QEvent.Type.Resize:
            self._fit_results_image()
            return super().eventFilter(obj, event)
        if obj is board and self._cube_data is not None:
            if event.type() == QtCore.QEvent.Type.MouseMove:
                pos = event.position() if hasattr(event, "position") else event.pos()
                cell = self._results_cell_from_pos(pos)
                if cell is None:
                    board.setToolTip("")
                else:
                    iy, ix = cell
                    value = np.nan
                    if self._results_values is not None:
                        value = self._results_values[iy, ix]
                    if np.isfinite(value):
                        shown = (
                            f"{float(value):+.2f}"
                            if self._results_units == "dB"
                            else f"{float(value):.2f}"
                        )
                        board.setToolTip(
                            f"Cell ({ix}, {iy})  •  {shown} {self._results_units}\n"
                            "Click to inspect/compare its spectrum."
                        )
                    else:
                        board.setToolTip(f"Cell ({ix}, {iy}) — not measured")
                return False
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                pos = event.position() if hasattr(event, "position") else event.pos()
                cell = self._results_cell_from_pos(pos)
                if cell is not None:
                    self.cube_pick_cell(*cell)
                return True
        return super().eventFilter(obj, event)

    def _results_cell_from_pos(self, pos):
        if self._cube_data is None or self._results_map_rect is None:
            return None
        values = self._results_values
        if values is None or np.asarray(values).ndim != 2:
            return None
        ny, nx = np.asarray(values).shape
        if nx < 1 or ny < 1:
            return None
        x = pos.x() if hasattr(pos, "x") else pos[0]
        y = pos.y() if hasattr(pos, "y") else pos[1]
        rect = self._results_map_rect
        if not rect.contains(QtCore.QPointF(float(x), float(y))):
            return None
        ix = int(np.clip((float(x) - rect.left()) / rect.width() * nx, 0, nx - 1))
        display_row = int(
            np.clip((float(y) - rect.top()) / rect.height() * ny, 0, ny - 1)
        )
        iy = ny - 1 - display_row
        return iy, ix

    def _on_failed(self, message):
        self.ui.scanStatus.setText(message)
        QMessageBox.warning(self.ui, DIALOG_TITLE, message)

    def _height_for_scan(self):
        """Freeze the current height block into scan provenance. No Z motion."""
        block = self._height_snapshot(self._reported_z())
        if self._easy_mode_active():
            return block
        return engine_height.apply_scan_height_preflight(block, self._scan_height_plan())

    def _ensure_at_agreed_scan_plane(self):
        """Refuse Easy Start unless M114 already matches the Easy target. Never move Z."""
        if self._easy_mode_active():
            if self._printer_job is not None:
                raise RuntimeError("Wait for the printer to finish moving.")
            if self._easy_z_unlocked:
                self._easy_reenable_z_if_unlocked()
                raise RuntimeError(
                    "Z holding was released. Repeat Step 2 to restore scan height."
                )
            reason = self._easy_height_block_reason()
            if reason:
                raise RuntimeError(reason)
            target = self._easy_scan_target_z()
            reported = self._reported_z()
            if (
                target is None
                or reported is None
                or not engine_height.logical_position_matches(reported, target)
            ):
                self._invalidate_easy_verified_height()
                raise RuntimeError(EASY_HEIGHT_NOT_REACHED)
            self._easy_verified_scan_z = float(reported)
            self._easy_height_identity = self._easy_height_identity_now()
            return
        try:
            plan = self._scan_height_plan()
        except engine_height.HeightError as exc:
            raise RuntimeError(str(exc)) from exc
        requested = plan.requested_height_above_pcb_mm
        if self._pcb_surface_z is None or requested is None:
            return
        target = engine_height.target_machine_z(self._pcb_surface_z, requested)
        reported = self._reported_z()
        if reported is not None and engine_height.logical_position_matches(
            reported, target
        ):
            return
        if not self.ui.chkAllowSetupZ.isChecked():
            raise RuntimeError(
                f"probe is not at the {requested:g} mm plane you set"
            )
        try:
            self._drive_to_agreed_plane(self._printer(), required=True)
        except engine_height.HeightError as exc:
            raise RuntimeError(str(exc)) from exc
        reported = self._reported_z()
        if reported is None or not engine_height.logical_position_matches(
            reported, target
        ):
            raise RuntimeError(
                f"probe is not at the {requested:g} mm plane you set"
            )

    def _on_printer_reset(self):
        """Marlin rebooted mid-scan, so the frame the landmarks were recorded in
        is gone. Without this the scan fails while the GUI still claims the
        machine is homed, which is the worst of both states."""
        self._easy_z_unlocked = False
        self._clear_height_datum()
        self._easy_height_lost_to_reset = True
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
        if self._easy_mode_active():
            return EASY_PCB_PAGES
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
        if rectangle:
            hint = (
                "Next opens the Origin page: mark the lower-left corner as X0 Y0, "
                "then scan a rectangle. This path does not use a PCB file."
            )
        elif self._easy_mode_active():
            hint = (
                "Next selects the physical fixture first. Teach its four BLTouch "
                "points only once; the ODB++ board is imported afterward."
            )
        else:
            hint = (
                "Next opens the Board page: import ODB++, then home X/Y and "
                "register board landmarks."
            )
        self.ui.modeHint.setText(hint)
        self.ui.gridInfo.setText(self._grid_summary())
        travel = getattr(self.ui, "chkTravelClearBoard", None)
        if travel is not None and hasattr(travel, "setVisible"):
            travel.setVisible(not rectangle)
        self._refresh_scan_intro()
        self._apply_operator_mode_ui()

    def _update_header(self):
        self._refresh_workflow_progress()
        sequence = self._sequence()
        position = self._position()
        if self._easy_mode_active():
            titles = {
                PAGE_SETUP: "Measurement Setup",
                PAGE_FIXTURE_TEACH: "Home & Set Board Zero",
                PAGE_BOARD: "Insert Board & Import ODB++",
                PAGE_REGISTER: "Advanced Board Alignment",
                PAGE_SCAN_SETUP: "Scan Area & Probe Height",
                PAGE_EASY_HEIGHT: "Probe Height (Z)",
                PAGE_SCAN: "Review & Measure",
                PAGE_RESULTS: "Results",
            }
            title = titles[sequence[position]]
            if sequence[position] == PAGE_RESULTS:
                self.ui.stepHeader.setText(f"Complete — {title}")
                return
            self.ui.stepHeader.setText(f"Step {position + 1} of 5 — {title}")
            return
        else:
            title = STEP_TITLES[sequence[position]]
        self.ui.stepHeader.setText(
            f"Step {position + 1} of {len(sequence)} — {title}"
        )

    def _next_allowed(self, page):
        """One gate per page, so another page never means another if/elif rung."""
        gates = {
            PAGE_BOARD: lambda: self._board_view is not None,
            PAGE_REGISTER: lambda: not self._registration_problems(),
            PAGE_FIXTURE_TEACH: self._fixture_gate_ready,
            PAGE_SCAN_SETUP: lambda: (
                not self._registration_problems()
                and self.ui.chkBoardSeated.isChecked()
            ),
            PAGE_EASY_HEIGHT: lambda: self._easy_at_scan_plane(),
            PAGE_ORIGIN: lambda: self.origin_set,
            PAGE_SCAN: lambda: self.result is not None,
        }
        return gates.get(page, lambda: True)()

    def _update_nav(self):
        self._refresh_home_button_copy()
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
            self.ui.btnRotateCcw: self._board_view is not None,
            self.ui.btnRotateCw: self._board_view is not None,
            self.ui.btnNext: (
                position + 1 < len(self._sequence())
                and self._next_allowed(self.ui.wizardStack.currentIndex())
            ),
        }
        easy_move = getattr(self.ui, "btnEasyMoveToRef", None)
        if easy_move is not None:
            allowed[easy_move] = (
                self._easy_mode_active()
                and self._pcb_xy_homed
                and self._easy_home_landmark() is not None
            )
        easy_fine = getattr(self.ui, "btnEasyFineAdjust", None)
        if easy_fine is not None:
            allowed[easy_fine] = (
                self._easy_mode_active() and self._pcb_xy_homed and self._profile is not None
            )
        easy_reset = getattr(self.ui, "btnEasyResetXy", None)
        if easy_reset is not None:
            allowed[easy_reset] = self._easy_mode_active() and self._profile is not None
        easy_set = getattr(self.ui, "btnEasySetHeight", None)
        manual_height = (
            self._easy_mode_active() and not self._easy_vertical_calibrated()
        )
        if easy_set is not None:
            allowed[easy_set] = manual_height and self._pcb_surface_z is not None
        for name in ("btnEasySetPcbSurface", "btnEasyResetPcbSurface"):
            button = getattr(self.ui, name, None)
            if button is not None:
                allowed[button] = manual_height
        fixture_home = getattr(self.ui, "btnFixtureHomeXyz", None)
        if fixture_home is not None:
            allowed[fixture_home] = (
                self._easy_mode_active()
                and bool(self._fixture_profile_name())
                and self.ui.chkFixtureProbeClear.isChecked()
            )
        retry_printer = getattr(self.ui, "btnFixtureRetryPrinter", None)
        if retry_printer is not None:
            allowed[retry_printer] = bool(self._printer_port())
        fixture_probe = getattr(self.ui, "btnFixtureProbePoint", None)
        if fixture_probe is not None:
            allowed[fixture_probe] = (
                self._easy_mode_active()
                and self._fixture_xyz_homed
                and self.ui.chkFixtureProbeClear.isChecked()
            )
        for name in (
            "btnFixtureJogXPlus", "btnFixtureJogXMinus",
            "btnFixtureJogYPlus", "btnFixtureJogYMinus",
        ):
            button = getattr(self.ui, name, None)
            if button is not None:
                allowed[button] = self._easy_mode_active() and self._fixture_xyz_homed
        fixture_save = getattr(self.ui, "btnFixtureSavePlane", None)
        if fixture_save is not None:
            allowed[fixture_save] = all(
                name in self._fixture_teaching_points
                for name in TEACHING_POINT_NAMES
            )
        fixture_verify = getattr(self.ui, "btnFixtureVerify", None)
        if fixture_verify is not None:
            document = self._machine_fixture_document() or {}
            allowed[fixture_verify] = (
                bool(document.get("fixture_teaching"))
                and self.ui.chkFixtureProbeClear.isChecked()
            )
        clear_point = getattr(self.ui, "btnFixtureClearPoint", None)
        if clear_point is not None:
            allowed[clear_point] = bool(
                self._fixture_teaching_points
                or (self._machine_fixture_document() or {}).get("fixture_teaching")
            )
        # Step 4 corner teaching: capturing needs a homed frame and an
        # imported board, because it pairs a board millimetre with a machine
        # one. Applying and clearing are pure software.
        teachable = self._board_model is not None and self._pcb_xy_homed
        for name in (
            "btnScanSetupJogXPlus", "btnScanSetupJogXMinus",
            "btnScanSetupJogYPlus", "btnScanSetupJogYMinus",
        ):
            button = getattr(self.ui, name, None)
            if button is not None:
                allowed[button] = self._pcb_xy_homed
        for side in SCAN_SETUP_SIDES:
            for slot in range(len(TAUGHT_CORNER_LABELS)):
                button = getattr(
                    self.ui, f"btnTeach{side.capitalize()}Corner{'AB'[slot]}", None
                )
                if button is not None:
                    allowed[button] = teachable
            use_button = getattr(self.ui, f"btnUse{side.capitalize()}Side", None)
            if use_button is not None:
                allowed[use_button] = (
                    self._board_model is not None and self._taught_side_complete(side)
                )
            clear_button = getattr(self.ui, f"btnClear{side.capitalize()}Corners", None)
            if clear_button is not None:
                allowed[clear_button] = bool(self._taught_pcb_corners(side))
        # Nothing that moves the machine or changes the plan stays live while a
        # scan is running, so idleness gates every one of them.
        idle = self.thread is None and self._printer_job is None
        for button, ready in allowed.items():
            button.setEnabled(idle and ready)
        # Results open-actions are not motion gates. Keep them live even while
        # the scan thread is still winding down, otherwise Open EMI Analyzer
        # stays at the .ui default (disabled) on the Complete page.
        self._sync_results_open_buttons()
        self.ui.btnNext.setText(self._next_caption())
        self._show_next_block_reason()
        self._refresh_fixture_page_summary()
        self._apply_scan_gate()

    def _install_next_reason_label(self):
        if getattr(self.ui, "nextBlockReason", None) is not None:
            return
        button = getattr(self.ui, "btnNext", None)
        nav = getattr(self.ui, "navLayout", None)
        find_child = getattr(self.ui, "findChild", None)
        if nav is None and callable(find_child):
            nav = find_child(QtWidgets.QHBoxLayout, "navLayout")
        layout_fn = getattr(self.ui, "layout", None)
        if nav is None and callable(layout_fn) and button is not None:
            try:
                nav = self._layout_containing_widget(layout_fn(), button)
            except Exception:
                nav = None
        if (
            button is None
            or nav is None
            or not hasattr(nav, "indexOf")
            or not hasattr(nav, "insertWidget")
        ):
            return
        index = nav.indexOf(button)
        if index < 0:
            return
        label = QtWidgets.QLabel()
        label.setObjectName("nextBlockReason")
        label.setWordWrap(True)
        label.setStyleSheet("color:#B42318;")
        nav.insertWidget(index, label, 1)
        self.ui.nextBlockReason = label

    def _layout_containing_widget(self, layout, widget):
        if layout is None:
            return None
        if hasattr(layout, "indexOf") and layout.indexOf(widget) >= 0:
            return layout
        count = layout.count() if hasattr(layout, "count") else 0
        for index in range(count):
            item = layout.itemAt(index)
            child = item.layout() if item is not None else None
            found = self._layout_containing_widget(child, widget)
            if found is not None:
                return found
        return None

    def _show_next_block_reason(self):
        reason = self._next_block_reason()
        self.ui.btnNext.setToolTip(reason)
        label = getattr(self.ui, "nextBlockReason", None)
        if label is None or not callable(getattr(label, "setText", None)):
            return
        enabled = bool(self.ui.btnNext.isEnabled())
        label.setText("" if enabled else reason)
        if callable(getattr(label, "setVisible", None)):
            label.setVisible(not enabled and bool(reason))

    def _next_caption(self):
        page = self.ui.wizardStack.currentIndex()
        if self._easy_mode_active():
            return {
                PAGE_SETUP: "Next: select fixture",
                PAGE_FIXTURE_TEACH: "Next: insert board",
                PAGE_BOARD: "Next: scan setup",
                PAGE_REGISTER: "Next: teach fixture",
                PAGE_SCAN_SETUP: "Next: review & measure",
                PAGE_EASY_HEIGHT: "Next: scan",
                PAGE_SCAN: "Next: results",
            }.get(page, "Next")
        caption = NEXT_CAPTIONS.get(page, "Next")
        if isinstance(caption, dict):
            caption = caption.get(self._mode(), "Next")
        return caption

    def _fixture_step2_block_reason(self):
        """Same Next refusal used by the Step 2 banner."""
        if self._easy_mode_active():
            return self._easy_height_block_reason()
        if self._p1_bltouch_commanded_xy() is None:
            return (
                "Teach P1 (Home XY for Teaching if the machine is not homed), "
                "then CAPTURE P1 to set board zero."
            )
        if not self._board_zero_committed():
            return (
                "CAPTURE P1 (or Home & Set Board Zero at the saved P1). "
                "Pin contact and retract must both succeed, and M114 must "
                "follow the pin."
            )
        return ""

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
        if page == PAGE_FIXTURE_TEACH:
            return self._fixture_step2_block_reason()
        if page == PAGE_SCAN_SETUP:
            problems = self._registration_problems()
            if problems:
                side = self._scan_setup_side()
                if not self._taught_side_complete(side):
                    return (
                        f"Capture both inner-pocket corners of the {side} side: "
                        "jog the E-field probe onto the pocket, then CAPTURE A and B. "
                        "P1 is height only; A/B are scan XY."
                    )
                return "Cannot position the board yet:\n- " + "\n- ".join(problems)
            if not self.ui.chkBoardSeated.isChecked():
                return "Confirm that the PCB is inserted in the holder and fully seated."
            return ""
        if page == PAGE_EASY_HEIGHT:
            return "Repeat Step 2 to set scan height before continuing."
        if page == PAGE_SCAN:
            return "Start the scan first. Results open when it finishes."
        if page == PAGE_BOARD:
            return "Import an ODB++ board first."
        if page == PAGE_ORIGIN:
            return "Set the origin first."
        return "Cannot continue yet."

    def _on_page_changed(self, index):
        self._last_page_by_mode[self._mode()] = index
        self._update_header()
        self._apply_operator_mode_ui()
        self._refresh_height_ui()
        self._update_nav()
        if index == PAGE_REGISTER:
            self._layout_register_page()
        if index == PAGE_FIXTURE_TEACH:
            document = self._machine_fixture_document() or {}
            teaching = document.get("fixture_teaching")
            if teaching or not self._fixture_teaching_points:
                self._render_fixture_teaching(teaching)
            self._refresh_fixture_page_summary()
        if index == PAGE_SCAN_SETUP:
            self._sync_board_profile_to_fixture()
            self._advanced_board_seated_changed(self.ui.chkBoardSeated.isChecked())
            # Easy scan XY is the taught A/B pocket. Do not STL-place or
            # restore a P1/middle-holder pose on this page.
            self._load_taught_pcb_corners()
            side = self._scan_setup_side()
            applied = (
                self._active_taught_side == side and self._registration is not None
            )
            # Re-fitting an unchanged face would invalidate the plan, and with
            # it the approval, every time the operator stepped back here.
            if not applied and not self._apply_taught_pcb_corners(side):
                self._drop_pocket_scan_xy()
                if self._pocket_pair_state.get(side) == "stale":
                    self._pocket_block_reason = (
                        self._pocket_block_reason
                        or "saved pocket corners are from an older meaning; recapture A and B"
                    )
                    self._set_glassboard_placement_status(
                        "POSITION BLOCKED — recapture the inner-pocket corners.",
                        "padding:8px;background:#FEF3F2;color:#912018;"
                        "border:1px solid #F04438;border-radius:6px",
                    )
                elif not self._taught_side_complete(side):
                    self._set_glassboard_placement_status(
                        "Capture inner-pocket corners A and B. P1 is height only; "
                        "A/B are the E-probe scan XY.",
                        "padding:8px;background:#EFF8FF;color:#175CD3;border-radius:6px",
                    )
            self._refresh_corner_teach_ui()
            self._refresh_calculated_height()
            self._refresh_scan_setup_summary()
            self._attach_scan_preview_to_current_page(index)
            self._refresh_scan_preview()
        if index == PAGE_SCAN:
            self._show_cube_scan_estimate()
        if index == PAGE_RESULTS:
            self._sync_results_open_buttons()
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
        if self.ui.wizardStack.currentIndex() == PAGE_REGISTER:
            self._easy_try_load_profile()
            self._update_easy_status()

    def _open_local_path(self, target):
        """Open a local folder or HTML file, including OneDrive paths with spaces."""
        path = Path(target).expanduser()
        try:
            path = path.resolve()
        except OSError:
            path = path.absolute()
        url = QtCore.QUrl.fromLocalFile(str(path))
        if QtGui.QDesktopServices.openUrl(url):
            return True
        import webbrowser

        return bool(webbrowser.open(path.as_uri()))

    def _open_folder(self):
        if self.result is None or not getattr(self.result, "output_dir", None):
            return
        self._open_local_path(self.result.output_dir)

    def _open_analyzer(self):
        analyzer_html = self._analyzer_html_path()
        if analyzer_html is None:
            analyzer_html = self._build_analyzer()
        if not analyzer_html:
            return
        if not self._open_local_path(analyzer_html):
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                f"Could not open the EMI Analyzer at {analyzer_html}",
            )

    def _build_analyzer(self):
        """Build the dashboard from this scan's saved artifacts. Reads only.

        The scan writes it too, but swallows any failure into a status line, so
        the operator otherwise faces a dead button. Failures are named here.
        """
        output_dir = None if self.result is None else getattr(self.result, "output_dir", None)
        if not output_dir:
            QMessageBox.warning(
                self.ui, DIALOG_TITLE, "There is no scan folder to build the EMI Analyzer from."
            )
            return None
        try:
            from EMI_Mapper.analyzer import write_analyzer

            path = write_analyzer(output_dir)
        except ImportError as exc:
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                "The EMI Analyzer needs the plotting dependency: "
                f"{exc}. Install requirements.txt (plotly) into this "
                "environment, then click OPEN EMI ANALYZER again.",
            )
            return None
        except Exception as exc:
            QMessageBox.warning(
                self.ui,
                DIALOG_TITLE,
                f"Could not build the EMI Analyzer from {output_dir}: {exc}",
            )
            return None
        files = getattr(self.result, "files", None)
        if files is None:
            self.result.files = {}
            files = self.result.files
        files["analyzer_html"] = path
        self._sync_results_open_buttons()
        return path

    def _on_close(self):
        self._last_page_by_mode[self._mode()] = self.ui.wizardStack.currentIndex()
        self._cleanup()
        self.ui.hide()

    def _cleanup(self):
        if self.worker is not None:
            self.worker.cancel()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(5000)
            self.thread = None
        if self._easy_z_unlocked:
            try:
                self._easy_cancel_pcb_touch()
            except (engine_printer.PrinterError, OSError, RuntimeError, ValueError):
                # The port may already have disappeared. Never retain a datum
                # from an incomplete touch sequence.
                self._easy_z_unlocked = False
                self._clear_height_datum()
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
            self._release_serial_handle(self.printer_serial)
        except OSError as exc:
            logging.info(f"EMI map could not close the printer port: {exc}")
        self.printer_serial = None
