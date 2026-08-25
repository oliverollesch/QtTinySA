"""EMI Near-Field Map wizard for QtTinySA.

Drives an Ender 3 over a PCB while this analyser measures, and builds a 2D map
of the emissions.  The measurement engine itself lives outside QtTinySA in the
EMI_Mapper package; this module is only Qt glue plus serial-ownership handling.

Guides the operator through: setup -> set the XY origin by hand -> optional
DUT-off background pass -> DUT pass -> results.  No Z motion, no homing.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
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

STEP_TITLES = ["Setup", "Origin", "Scan", "Results"]

PAGE_SETUP = 0
PAGE_ORIGIN = 1
PAGE_SCAN = 2
PAGE_RESULTS = 3

TRAVEL_WARNING = (
    "The probe will move +X by {width:g} mm and +Y by {height:g} mm from the "
    "current location. Verify the full travel area is clear before setting the origin."
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
    from EMI_Mapper import config as engine_config
    from EMI_Mapper import printer as engine_printer
    from EMI_Mapper import scanner as engine_scanner

    return engine_config, engine_printer, engine_scanner


try:
    engine_config, engine_printer, engine_scanner = _load_engine()
    ENGINE_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on deployment layout
    engine_config = engine_printer = engine_scanner = None
    ENGINE_ERROR = str(exc)
    logging.info(f"EMI map engine unavailable: {exc}")


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
    ended = QtCore.Signal()

    def __init__(self, config, tinysa_transport, printer_transport):
        super().__init__()
        self.config = config
        self.tinysa_transport = tinysa_transport
        self.printer_transport = printer_transport
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
            )
            self.succeeded.emit(result)
        except engine_scanner.ScanAborted:
            self.failed.emit("Scan cancelled. Points already measured were saved.")
        except Exception as exc:
            logging.exception("EMI scan failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.ended.emit()


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

        self._wire_ui()
        self._setup_plot()
        self._update_travel_warning()

    # ------------------------------------------------------------------ UI
    def _wire_ui(self):
        u = self.ui
        u.btnBack.clicked.connect(self._on_back)
        u.btnNext.clicked.connect(self._on_next)
        u.btnCancel.clicked.connect(self._on_close)
        u.btnBrowse.clicked.connect(self._browse_folder)
        u.btnSetOrigin.clicked.connect(self._set_origin)
        u.btnStartScan.clicked.connect(self._start_scan)
        u.btnAbortScan.clicked.connect(self._abort_scan)
        u.btnContinueDut.clicked.connect(self._continue_dut)
        u.btnOpenFolder.clicked.connect(self._open_folder)

        u.chkTravelClear.stateChanged.connect(self._update_nav)
        for spin in (u.widthMm, u.heightMm, u.stepMm):
            spin.valueChanged.connect(self._on_area_changed)
        u.wizardStack.currentChanged.connect(self._on_page_changed)

        # The dialog can be dismissed with the window chrome as well as Close
        u.rejected.connect(self._cleanup)

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
        if not self.ui.outputRoot.text():
            self.ui.outputRoot.setText(str(Path.cwd() / "EMI_Scans"))
        self.ui.wizardStack.setCurrentIndex(PAGE_SETUP)
        self._on_area_changed()
        self._update_header()
        self._update_nav()
        self.ui.show()
        self.ui.raise_()
        self.ui.activateWindow()

    # ------------------------------------------------------------------ setup
    def _refresh_ports(self):
        combo = self.ui.printerPort
        current = combo.currentText()
        combo.clear()
        for port in list_ports.comports():
            combo.addItem(port.device)
        if current:
            combo.setCurrentText(current)

    def _browse_folder(self):
        chosen = QFileDialog.getExistingDirectory(
            self.ui, "Choose the folder that receives scan results", self.ui.outputRoot.text()
        )
        if chosen:
            self.ui.outputRoot.setText(chosen)

    def _on_area_changed(self):
        area = engine_config.ScanArea(
            width_mm=self.ui.widthMm.value(),
            height_mm=self.ui.heightMm.value(),
            step_mm=self.ui.stepMm.value(),
        )
        points = area.nx * area.ny
        self.ui.gridInfo.setText(f"{area.nx} x {area.ny} = {points} points")
        self._update_travel_warning()

    def _update_travel_warning(self):
        self.ui.travelWarning.setText(
            TRAVEL_WARNING.format(width=self.ui.widthMm.value(), height=self.ui.heightMm.value())
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
            printer=engine_config.PrinterConfig(
                port=u.printerPort.currentText().strip(),
                baud=u.printerBaud.value(),
                settle_s=u.settleMs.value() / 1000.0,
                # The wizard sets the origin itself on the Origin page, so the
                # engine must not send a second G92 when the scan starts.
                set_current_xy_as_origin=False,
            ),
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

    # ------------------------------------------------------------------ origin
    def _open_printer(self):
        import serial

        if self.printer_serial is not None and self.printer_serial.is_open:
            return self.printer_serial
        port = self.ui.printerPort.currentText().strip()
        if not port:
            raise RuntimeError("Choose the printer serial port first")
        self.printer_serial = serial.Serial(port, self.ui.printerBaud.value(), timeout=0.05)
        return self.printer_serial

    def _set_origin(self):
        try:
            transport = self._open_printer()
            printer = engine_printer.Printer(
                transport,
                engine_config.PrinterConfig(
                    port=self.ui.printerPort.currentText().strip(),
                    baud=self.ui.printerBaud.value(),
                ),
            )
            printer.drain()
            printer.prepare()
            printer.set_origin()
        except Exception as exc:
            self.origin_set = False
            self.ui.originStatus.setText(f"Origin not set: {exc}")
            QMessageBox.warning(self.ui, "EMI Near-Field Map", str(exc))
        else:
            self.origin_set = True
            self.ui.originStatus.setText(
                "Origin set. Heaters and fans are off. This position is now X0 Y0."
            )
        self._update_nav()

    # ------------------------------------------------------------------ scan
    def _start_scan(self):
        if self.thread is not None:
            return
        if not self.origin_set:
            QMessageBox.warning(self.ui, "EMI Near-Field Map", "Set the origin first.")
            return
        try:
            config = self.build_config()
        except Exception as exc:
            QMessageBox.warning(self.ui, "EMI Near-Field Map", f"Invalid settings: {exc}")
            return

        if self.serial.adopt() is None or not self.serial.is_open():
            QMessageBox.warning(self.ui, "EMI Near-Field Map", "No analyser is connected.")
            return
        try:
            dev = self.serial.take()
        except Exception as exc:
            QMessageBox.warning(self.ui, "EMI Near-Field Map", str(exc))
            return

        try:
            transport = self._open_printer()
        except Exception as exc:
            self.serial.release()
            QMessageBox.warning(self.ui, "EMI Near-Field Map", str(exc))
            return

        self._prepare_live_plot(config)
        self._total = config.area.nx * config.area.ny * (2 if config.background else 1)
        self._done = 0
        self.ui.scanProgress.setMaximum(self._total)
        self.ui.scanProgress.setValue(0)

        self.worker = ScanWorker(config, dev.usb, transport)
        self.thread = QtCore.QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.ui.scanStatus.setText)
        self.worker.point.connect(self._on_point)
        self.worker.row.connect(self._on_row)
        self.worker.prompt.connect(self._on_prompt)
        self.worker.succeeded.connect(self._on_succeeded)
        self.worker.failed.connect(self._on_failed)
        # Runs for success, cancel, timeout and any worker exception alike
        self.worker.ended.connect(self._on_scan_ended)

        self._set_scanning(True)
        self.thread.start()

    def _prepare_live_plot(self, config):
        area = config.area
        self._grid = np.full((area.ny, area.nx), np.nan)
        self.image.setImage(self._grid.T, autoLevels=False)
        self.image.setRect(QtCore.QRectF(0.0, 0.0, area.width_mm, area.height_mm))
        self.ui.emiPlot.setXRange(0, max(area.width_mm, area.step_mm))
        self.ui.emiPlot.setYRange(0, max(area.height_mm, area.step_mm))

    def _set_scanning(self, scanning):
        self.ui.btnStartScan.setEnabled(not scanning)
        self.ui.btnAbortScan.setEnabled(scanning)
        self.ui.pageSetup.setEnabled(not scanning)
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
        if finite.size:
            self.image.setImage(
                self._grid.T, autoLevels=False, levels=(float(finite.min()), float(finite.max()))
            )

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
        lines = [f"Scan folder: {result.output_dir}", ""]
        lines += [f"{name}: {Path(path).name}" for name, path in result.files.items()]
        instrument = result.snapshot.get("instrument", {})
        lines += ["", "Instrument:"]
        lines += [f"  {key}: {value}" for key, value in instrument.items()]
        self.ui.resultsText.setPlainText("\n".join(lines))
        self.ui.wizardStack.setCurrentIndex(PAGE_RESULTS)

    def _on_failed(self, message):
        self.ui.scanStatus.setText(message)
        QMessageBox.warning(self.ui, "EMI Near-Field Map", message)

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
    def _update_header(self):
        index = self.ui.wizardStack.currentIndex()
        self.ui.stepHeader.setText(f"Step {index + 1} of 4 — {STEP_TITLES[index]}")

    def _update_nav(self):
        index = self.ui.wizardStack.currentIndex()
        scanning = self.thread is not None
        self.ui.btnBack.setEnabled(index > 0 and not scanning)
        self.ui.btnSetOrigin.setEnabled(
            self.ui.chkTravelClear.isChecked() and not scanning
        )
        if index == PAGE_ORIGIN:
            self.ui.btnNext.setEnabled(self.origin_set)
        elif index == PAGE_SCAN:
            self.ui.btnNext.setEnabled(self.result is not None)
        else:
            self.ui.btnNext.setEnabled(index < PAGE_RESULTS)

    def _on_page_changed(self, index):
        self._update_header()
        self._update_nav()

    def _on_back(self):
        index = self.ui.wizardStack.currentIndex()
        if index > 0:
            self.ui.wizardStack.setCurrentIndex(index - 1)

    def _on_next(self):
        index = self.ui.wizardStack.currentIndex()
        if index == PAGE_SETUP:
            try:
                self.build_config()
            except Exception as exc:
                QMessageBox.warning(self.ui, "EMI Near-Field Map", f"Invalid settings: {exc}")
                return
        if index < PAGE_RESULTS:
            self.ui.wizardStack.setCurrentIndex(index + 1)

    def _open_folder(self):
        if self.result is None:
            return
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(self.result.output_dir))
        )

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
        if self.printer_serial is not None:
            try:
                self.printer_serial.close()
            except Exception:
                pass
            self.printer_serial = None
        self.origin_set = False
        self.ui.originStatus.setText("Origin not set.")
