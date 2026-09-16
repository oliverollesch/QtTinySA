"""Step 4 millimetre scan preview. Display only: no printer I/O.

QtTinySA constructs the EMI map wizard at process start, so this widget
exists even before the operator opens the map. A hidden ``GLViewWidget``
on Windows mixed Qt's default ANGLE/Direct3D path with native OpenGL and
aborted the process (``0xC0000409``). The preview is therefore a 2D
``PlotWidget`` on the same stack as the spectrum graphs — never
``pyqtgraph.opengl``.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from PySide6 import QtCore, QtWidgets

from EMI_Mapper.scan_preview import LAST_REPORTED_POSITION, ScanPreviewScene


class ScanPreviewWidget(QtWidgets.QWidget):
    """Draws a ``ScanPreviewScene``. Never imports Printer or sends G-code."""

    approved = QtCore.Signal(str)
    simulation_toggled = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene: ScanPreviewScene | None = None
        self._ghost_index = -1
        self._simulating = False
        self._completed = set()
        self._active = None
        self._reported_xy = None
        self._reported_stale = False
        self._plot = None
        self._path_item = None
        self._outline_item = None
        self._fixture_items = []
        self._inferred_items = []
        self._ghost_item = None
        self._tip_item = None
        self._done_item = None
        self._corner_item = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._advance_ghost)
        self._build()

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status = QtWidgets.QLabel("Scan plan preview is empty.")
        self.status.setWordWrap(True)
        self.status.setObjectName("scanPreviewStatus")
        layout.addWidget(self.status)
        self.notice = QtWidgets.QLabel(
            "Display only — not collision detection. Top view in millimetres. "
            "Mesh missing; audited boxes shown."
        )
        self.notice.setWordWrap(True)
        self.notice.setObjectName("scanPreviewNotice")
        layout.addWidget(self.notice)
        try:
            import pyqtgraph

            plot = pyqtgraph.PlotWidget(self)
            plot.setObjectName("scanPreviewPlot")
            plot.setMinimumHeight(240)
            plot.setBackground("w")
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setAspectLocked(True)
            plot.setLabel("bottom", "Machine X", units="mm")
            plot.setLabel("left", "Machine Y", units="mm")
            self._path_item = plot.plot(
                pen=pyqtgraph.mkPen((50, 115, 230), width=2), name="path"
            )
            self._outline_item = plot.plot(
                pen=pyqtgraph.mkPen((25, 180, 80), width=2), name="outline"
            )
            self._ghost_item = pyqtgraph.ScatterPlotItem(
                size=14, brush=pyqtgraph.mkBrush(240, 190, 25), pen=None
            )
            self._tip_item = pyqtgraph.ScatterPlotItem(
                size=12, brush=pyqtgraph.mkBrush(220, 40, 40), pen=None
            )
            self._done_item = pyqtgraph.ScatterPlotItem(
                size=7, brush=pyqtgraph.mkBrush(50, 180, 80), pen=None
            )
            self._corner_item = pyqtgraph.ScatterPlotItem(
                size=16,
                brush=pyqtgraph.mkBrush(30, 30, 30),
                pen=pyqtgraph.mkPen((240, 190, 25), width=2),
                symbol="x",
            )
            plot.addItem(self._ghost_item)
            plot.addItem(self._tip_item)
            plot.addItem(self._done_item)
            plot.addItem(self._corner_item)
            self._plot = plot
            layout.addWidget(plot, 1)
        except Exception:
            fallback = QtWidgets.QLabel(
                "Scan preview plot unavailable; path numbers still apply."
            )
            fallback.setObjectName("scanPreviewFallback")
            fallback.setMinimumHeight(80)
            layout.addWidget(fallback)
        cam_row = QtWidgets.QHBoxLayout()
        for name, slot in (
            ("Top", lambda: self._set_camera("top")),
            ("Side", lambda: self._set_camera("side")),
            ("Isometric", lambda: self._set_camera("iso")),
        ):
            button = QtWidgets.QPushButton(name)
            button.clicked.connect(slot)
            cam_row.addWidget(button)
        layout.addLayout(cam_row)
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("PLAY SIMULATION")
        self.btn_play.setObjectName("btnPlayScanSimulation")
        self.btn_stop = QtWidgets.QPushButton("STOP SIMULATION")
        self.btn_stop.setObjectName("btnStopScanSimulation")
        self.btn_approve = QtWidgets.QPushButton("APPROVE SCAN PLAN")
        self.btn_approve.setObjectName("btnApproveScanPlan")
        self.btn_play.clicked.connect(self.play_simulation)
        self.btn_stop.clicked.connect(self.stop_simulation)
        self.btn_approve.clicked.connect(self.approve_plan)
        self.btn_approve.setEnabled(False)
        self.btn_play.setEnabled(False)
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_approve)
        layout.addLayout(btn_row)
        self.ghost_label = QtWidgets.QLabel("")
        self.ghost_label.setObjectName("scanPreviewGhostLabel")
        layout.addWidget(self.ghost_label)

    def fingerprint(self) -> str:
        return "" if self._scene is None else self._scene.fingerprint

    def set_scene(self, scene: ScanPreviewScene | None):
        self.stop_simulation()
        self._scene = scene
        self._completed.clear()
        self._active = None
        has_path = bool(scene is not None and scene.path_machine_xy)
        if getattr(self, "btn_approve", None) is not None:
            self.btn_approve.setEnabled(has_path)
        if getattr(self, "btn_play", None) is not None:
            self.btn_play.setEnabled(has_path)
        self._redraw()
        self._refresh_status()

    def set_reported_xy(self, xy, *, stale=False):
        self._reported_xy = None if xy is None else (float(xy[0]), float(xy[1]))
        self._reported_stale = bool(stale)
        self._redraw_markers()
        self._refresh_status()

    def set_active_cell(self, machine_xy):
        self._active = None if machine_xy is None else (float(machine_xy[0]), float(machine_xy[1]))
        self._redraw_markers()

    def mark_completed(self, machine_xy):
        if machine_xy is None:
            return
        self._completed.add((round(float(machine_xy[0]), 3), round(float(machine_xy[1]), 3)))
        self._redraw_markers()

    def play_simulation(self):
        if self._scene is None or not self._scene.path_machine_xy:
            return
        self._simulating = True
        self._ghost_index = 0
        self.ghost_label.setText("Simulation")
        self._timer.start()
        self._redraw_markers()
        self.simulation_toggled.emit(True)

    def stop_simulation(self):
        self._timer.stop()
        self._simulating = False
        self._ghost_index = -1
        self.ghost_label.setText("")
        self._redraw_markers()
        self.simulation_toggled.emit(False)

    def approve_plan(self):
        if self._scene is None or not self._scene.path_machine_xy:
            return
        self.approved.emit(self._scene.fingerprint)

    def _set_camera(self, kind: str):
        # Side/iso keep the millimetre top view; OpenGL cameras were the crash.
        del kind
        if self._plot is None:
            return
        self._plot.autoRange()

    def _refresh_status(self):
        scene = self._scene
        if scene is None:
            self.status.setText("Scan plan preview is empty.")
            return
        height = (
            f"Target clearance: {scene.target_clearance_mm:.2f} mm"
            if scene.height_calibrated and scene.target_clearance_mm is not None
            else scene.height_message or "Probe height not calibrated"
        )
        predicted = ""
        if scene.predicted_clearance_mm is not None:
            low, high = scene.predicted_clearance_mm
            predicted = f" Predicted PCB clearance {low:.3f}–{high:.3f} mm."
        travel = f" {scene.travel_error}" if scene.travel_error else ""
        stale = f" {LAST_REPORTED_POSITION}." if self._reported_stale else ""
        inferred = (
            " Holder is the taught pocket placement from A/B."
            if scene.taught_placement_label or scene.inferred_holder_label
            else ""
        )
        self.status.setText(
            f"{scene.point_count} points, {scene.direction}. {height}.{predicted}{travel}{stale}{inferred}"
        )

    def _xy(self, points):
        if not points:
            return [], []
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return xs, ys

    def _redraw(self):
        if self._path_item is None or self._outline_item is None:
            return
        scene = self._scene
        if scene is None:
            self._path_item.setData([], [])
            self._outline_item.setData([], [])
            self._clear_fixture_items()
            self._redraw_markers()
            return
        self._path_item.setData(*self._xy(scene.path_machine_xy))
        self._outline_item.setData(*self._xy(scene.pcb_outline_machine_xy))
        self._clear_fixture_items()
        if self._plot is not None:
            import pyqtgraph

            for poly in scene.fixture_polylines_machine_xy:
                item = self._plot.plot(
                    *self._xy(poly),
                    pen=pyqtgraph.mkPen((120, 120, 130), width=1),
                )
                self._fixture_items.append(item)
        self._redraw_markers()
        if self._plot is not None:
            self._plot.autoRange()

    def _clear_fixture_items(self):
        if self._plot is None:
            self._fixture_items = []
            self._inferred_items = []
            return
        for item in self._fixture_items + self._inferred_items:
            self._plot.removeItem(item)
        self._fixture_items = []
        self._inferred_items = []

    def _redraw_markers(self):
        if self._ghost_item is None or self._tip_item is None or self._done_item is None:
            return
        ghost = []
        if self._simulating and self._scene is not None and 0 <= self._ghost_index < len(self._scene.path_machine_xy):
            ghost = [self._scene.path_machine_xy[self._ghost_index]]
        self._ghost_item.setData(*self._xy(ghost))
        tip = []
        if self._active is not None:
            tip = [self._active]
        elif self._reported_xy is not None:
            tip = [self._reported_xy]
        self._tip_item.setData(*self._xy(tip))
        self._done_item.setData(*self._xy(list(self._completed)))
        corners = []
        if self._scene is not None:
            corners = list(self._scene.taught_corners_machine_xy)
        if self._corner_item is not None:
            self._corner_item.setData(*self._xy(corners))

    def _advance_ghost(self):
        if self._scene is None or not self._scene.path_machine_xy:
            self.stop_simulation()
            return
        self._ghost_index += 1
        if self._ghost_index >= len(self._scene.path_machine_xy):
            self.stop_simulation()
            return
        self._redraw_markers()
