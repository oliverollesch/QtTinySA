"""Reusable BoardView scan-selection canvas, widget, and large selector dialog.

Snapping, rasterization, and view transforms stay in EMI_Mapper.selection.
This module is Qt only: the same interaction for the wizard preview and the
large selector.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets

FIT_PADDING = 0.08
SELECTOR_SIZE = (1400, 900)
SELECTOR_MIN_SIZE = (720, 480)
SELECTOR_SCREEN_MARGIN = 32

try:
    from EMI_Mapper import geometry as engine_geometry
    from EMI_Mapper import overlay as engine_overlay
    from EMI_Mapper import selection as engine_selection
except ImportError:  # pragma: no cover - packaged layout uses the wizard shim
    root = Path(__file__).resolve().parents[3]
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from EMI_Mapper import geometry as engine_geometry
    from EMI_Mapper import overlay as engine_overlay
    from EMI_Mapper import selection as engine_selection


def board_bounds(view, step_mm):
    x0, y0, x1, y1 = view.bbox_mm
    return (
        math.floor(x0 / step_mm) * step_mm,
        math.ceil(x1 / step_mm) * step_mm,
        math.floor(y0 / step_mm) * step_mm,
        math.ceil(y1 / step_mm) * step_mm,
    )


def polyline_batch(arrays, close=False):
    """Flatten many polylines into one x/y/connect triple for a single curve item."""
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
        flags[-1] = 0
        connect.append(flags)
    if not xs:
        return None
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(connect)


def cursor_text(x, y) -> str:
    return f"Board X: {x:.2f} mm    Board Y: {y:.2f} mm"


def preview_summary(selection, cell_count: int) -> str:
    if selection is None:
        return f"Current selection: entire board · {cell_count} scan cells"
    n_roi = sum(1 for item in selection.items if isinstance(item, engine_selection.BoardROI))
    n_point = sum(1 for item in selection.items if isinstance(item, engine_selection.BoardPoint))
    if not selection.items:
        return "Current selection: nothing selected"
    return (
        f"Current selection: {n_roi} ROI(s) · {n_point} point(s) · {cell_count} scan cells"
    )


def selection_status_text(selection, cell_count: int) -> str:
    summary = preview_summary(selection, cell_count)
    if selection is None:
        return "Scanning the entire board.\n" + summary
    if not selection.items:
        return (
            "Nothing selected. Add a point or rectangle, or scan the entire board.\n"
            + summary
        )
    last = selection.items[-1]
    n_items = len(selection.items)
    extra = f" in {n_items} items" if n_items > 1 else ""
    if isinstance(last, engine_selection.BoardROI):
        name = last.name or "Rectangle"
        detail = (
            f"{name}\n"
            f"X: {last.x_min:.1f} – {last.x_max:.1f} mm\n"
            f"Y: {last.y_min:.1f} – {last.y_max:.1f} mm\n"
            f"{cell_count} selected cells{extra}"
        )
    else:
        name = last.name or "Point"
        detail = (
            f"{name}\n"
            f"X: {last.x:.2f} mm    Y: {last.y:.2f} mm\n"
            f"{cell_count} selected cells{extra}"
        )
    return f"{summary}\n{detail}"


class ScanSelectionEditor:
    """None vs empty vs items. Plot-free so headless tests can drive it."""

    def __init__(self):
        self.board_view = None
        self.selection = None
        self.step_mm = 10.0
        self.tool = None
        self.last_message = ""

    def empty(self):
        return engine_selection.ScanSelection(items=())

    def scan_entire_board(self):
        self.selection = None
        self.tool = None
        self.last_message = ""

    def clear(self):
        self.selection = self.empty()
        self.last_message = ""

    def undo(self):
        if self.selection is None:
            return
        self.selection = self.selection.without_last()
        self.last_message = ""

    def enter_subset(self):
        if self.selection is None:
            self.selection = self.empty()

    def set_tool(self, tool):
        if tool not in (None, "point", "rectangle"):
            raise ValueError(f"unknown selection tool {tool!r}")
        self.tool = tool
        if tool in ("point", "rectangle"):
            self.enter_subset()

    def selection_grid(self):
        view = self.board_view
        if view is None or engine_geometry is None or self.step_mm <= 0:
            return None
        x_start, x_end, y_start, y_end = board_bounds(view, self.step_mm)
        xs = engine_geometry.grid_axis(x_start, x_end, self.step_mm)
        ys = engine_geometry.grid_axis(y_start, y_end, self.step_mm)
        mask = engine_geometry.outline_mask(view.outline, xs, ys)
        return xs, ys, mask, self.step_mm

    def selected_cell_count(self) -> int:
        if self.selection is None:
            grid = self.selection_grid()
            return 0 if grid is None else int(grid[2].sum())
        if not self.selection.items or engine_selection is None:
            return 0
        grid = self.selection_grid()
        if grid is None:
            return 0
        xs, ys, mask, _step = grid
        selected = engine_selection.rasterize(
            self.selection, xs, ys, mask, self.board_view.outline
        )
        return int((mask & selected).sum())

    def _next_name(self, kind: str) -> str:
        items = () if self.selection is None else self.selection.items
        cls = engine_selection.BoardPoint if kind == "point" else engine_selection.BoardROI
        n = 1 + sum(1 for item in items if isinstance(item, cls))
        return f"Point {n}" if kind == "point" else f"ROI {n}"

    def add_point(self, x, y) -> bool:
        if self.board_view is None or engine_selection is None:
            return False
        grid = self.selection_grid()
        if grid is None:
            return False
        xs, ys, mask, _step = grid
        if engine_selection.snap_point(xs, ys, mask, self.board_view.outline, x, y) is None:
            self.last_message = "Click is off the board or in a cutout."
            return False
        self.enter_subset()
        point = engine_selection.BoardPoint(x, y, name=self._next_name("point"))
        self.selection = self.selection.with_item(point)
        self.last_message = ""
        return True

    def add_roi(self, x_min, x_max, y_min, y_max) -> bool:
        if self.board_view is None or engine_selection is None:
            return False
        if not (x_min < x_max and y_min < y_max):
            return False
        self.enter_subset()
        roi = engine_selection.BoardROI(
            x_min, x_max, y_min, y_max, name=self._next_name("roi")
        )
        self.selection = self.selection.with_item(roi)
        self.last_message = ""
        return True

    def status_text(self) -> str:
        return selection_status_text(self.selection, self.selected_cell_count())

    def preview_summary(self) -> str:
        return preview_summary(self.selection, self.selected_cell_count())


class BoardSelectionView(QtCore.QObject):
    """PlotWidget interaction: draw, overlay, wheel zoom, pan, ROI tools."""

    cursorMoved = QtCore.Signal(str)
    selectionChanged = QtCore.Signal()

    def __init__(self, plot, editor: ScanSelectionEditor, parent=None):
        super().__init__(parent)
        self.plot = plot
        self.editor = editor
        self._overlay_items = []
        self._rubber_item = None
        self._drag_origin = None
        self._camera_ready = False
        self._prepare_plot()
        self._install_navigation()
        scene = plot.scene()
        scene.sigMouseClicked.connect(self._on_clicked)
        scene.sigMouseMoved.connect(self._on_mouse_moved)
        viewport = plot.viewport()
        if viewport is not None:
            viewport.installEventFilter(self)
        destroyed = getattr(plot, "destroyed", None)
        if destroyed is not None:
            destroyed.connect(self._on_plot_destroyed)

    def _on_plot_destroyed(self, *_args):
        self.plot = None

    def _prepare_plot(self):
        import pyqtgraph

        self.plot.setBackground("w")
        self.plot.setLabel("bottom", "Board X (mm)")
        self.plot.setLabel("left", "Board Y (mm)")
        for axis_name in ("bottom", "left"):
            axis = self.plot.getAxis(axis_name)
            axis.enableAutoSIPrefix(False)
            axis.setPen(pyqtgraph.mkPen("k"))
            axis.setTextPen(pyqtgraph.mkPen("k"))
        self.plot.setAspectLocked(True)
        hide = getattr(self.plot, "hideButtons", None)
        if callable(hide):
            hide()

    @property
    def view_box(self):
        try:
            return self.plot.getPlotItem().getViewBox()
        except (AttributeError, TypeError):
            return None

    def disable_auto_range(self):
        view_box = self.view_box
        if view_box is not None:
            view_box.disableAutoRange()
            view_box.setMouseEnabled(True, True)

    def capture_camera(self):
        if not self._camera_ready:
            return None
        view_box = self.view_box
        if view_box is None:
            return None
        (x0, x1), (y0, y1) = view_box.viewRange()
        return (float(x0), float(x1), float(y0), float(y1))

    def restore_camera(self, camera):
        view_box = self.view_box
        if view_box is None or camera is None:
            return
        x0, x1, y0, y1 = camera
        view_box.disableAutoRange()
        view_box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def fit_board(self):
        view_box = self.view_box
        view = self.editor.board_view
        if view_box is None or view is None:
            return
        x0, y0, x1, y1 = view.bbox_mm
        pad_x = max((x1 - x0) * FIT_PADDING, 1e-6)
        pad_y = max((y1 - y0) * FIT_PADDING, 1e-6)
        view_box.disableAutoRange()
        view_box.setRange(
            xRange=(x0 - pad_x, x1 + pad_x),
            yRange=(y0 - pad_y, y1 + pad_y),
            padding=0,
        )
        self._camera_ready = True

    def zoom_at(self, x, y, factor):
        view_box = self.view_box
        if view_box is None or factor <= 0:
            return
        view_box.disableAutoRange()
        view_box.scaleBy((factor, factor), center=(x, y))
        self._camera_ready = True

    def pan(self, dx, dy):
        view_box = self.view_box
        if view_box is None:
            return
        view_box.disableAutoRange()
        view_box.translateBy(x=dx, y=dy)
        self._camera_ready = True

    def _install_navigation(self):
        view_box = self.view_box
        if view_box is None:
            return
        view_box.setMenuEnabled(False)
        view_box.disableAutoRange()
        view_box.setMouseEnabled(True, True)

        def mouse_drag_event(ev, axis=None):
            button = ev.button()
            if button == QtCore.Qt.MouseButton.LeftButton:
                ev.ignore()
                return
            if button in (
                QtCore.Qt.MouseButton.MiddleButton,
                QtCore.Qt.MouseButton.RightButton,
            ):
                now = view_box.mapToView(ev.pos())
                last = view_box.mapToView(ev.lastPos())
                self.pan(last.x() - now.x(), last.y() - now.y())
                ev.accept()
                return
            ev.ignore()

        def mouse_click_event(ev):
            if ev.double():
                self.fit_board()
                ev.accept()
                return
            ev.ignore()

        def wheel_event(ev, axis=None):
            delta = ev.delta() if hasattr(ev, "delta") else 0
            if not delta and hasattr(ev, "angleDelta"):
                delta = ev.angleDelta().y()
            if not delta:
                ev.accept()
                return
            center = view_box.mapToView(ev.pos())
            self.zoom_at(center.x(), center.y(), 0.85 ** (delta / 120.0))
            ev.accept()

        view_box.mouseDragEvent = mouse_drag_event
        view_box.mouseClickEvent = mouse_click_event
        view_box.wheelEvent = wheel_event

    def draw(self, *, preserve_camera=True):
        import pyqtgraph

        camera = self.capture_camera() if preserve_camera else None
        self.plot.clear()
        self._overlay_items = []
        self._rubber_item = None
        self.disable_auto_range()
        view = self.editor.board_view
        if view is None:
            return
        artwork = polyline_batch([stroke.points for stroke in view.strokes])
        if artwork is not None:
            xs, ys, connect = artwork
            self.plot.addItem(
                pyqtgraph.PlotCurveItem(
                    xs, ys, connect=connect, pen=pyqtgraph.mkPen("#8899aa", width=1)
                )
            )
        outline = polyline_batch([ring.points for ring in view.outline.rings], close=True)
        if outline is not None:
            xs, ys, connect = outline
            self.plot.addItem(
                pyqtgraph.PlotCurveItem(
                    xs, ys, connect=connect, pen=pyqtgraph.mkPen("k", width=2)
                )
            )
        if view.components:
            self.plot.addItem(
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
                self.plot.addItem(label)
        self.refresh_overlay()
        if camera is not None:
            self.restore_camera(camera)

    def refresh_overlay(self):
        camera = self.capture_camera()
        try:
            self._paint_overlay()
        finally:
            if camera is not None:
                self.restore_camera(camera)
            else:
                self.disable_auto_range()

    def _paint_overlay(self):
        import pyqtgraph

        plot = self.plot
        for item in self._overlay_items:
            try:
                plot.removeItem(item)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        self._overlay_items = []
        selection = self.editor.selection
        if selection is None or not selection.items:
            return
        grid = self.editor.selection_grid()
        if grid is None:
            return
        xs, ys, mask, step_mm = grid
        selected = engine_selection.rasterize(
            selection, xs, ys, mask, self.editor.board_view.outline
        )
        half = step_mm / 2.0
        squares = [
            (
                (x - half, y - half),
                (x + half, y - half),
                (x + half, y + half),
                (x - half, y + half),
            )
            for iy, y in enumerate(ys)
            for ix, x in enumerate(xs)
            if selected[iy, ix]
        ]
        cells = polyline_batch(squares, close=True)
        if cells is not None:
            cx, cy, connect = cells
            item = pyqtgraph.PlotCurveItem(
                cx, cy, connect=connect, pen=pyqtgraph.mkPen("#1a73e8", width=1)
            )
            plot.addItem(item)
            self._overlay_items.append(item)
        boxes = [
            (
                (item.x_min, item.y_min),
                (item.x_max, item.y_min),
                (item.x_max, item.y_max),
                (item.x_min, item.y_max),
            )
            for item in selection.items
            if isinstance(item, engine_selection.BoardROI)
        ]
        outlines = polyline_batch(boxes, close=True)
        if outlines is not None:
            ox, oy, connect = outlines
            item = pyqtgraph.PlotCurveItem(
                ox, oy, connect=connect, pen=pyqtgraph.mkPen("#d93025", width=2)
            )
            plot.addItem(item)
            self._overlay_items.append(item)

    def xy_from_scene(self, scene_pos):
        view_box = self.view_box
        if view_box is None or not view_box.sceneBoundingRect().contains(scene_pos):
            return None
        point = view_box.mapSceneToView(scene_pos)
        return float(point.x()), float(point.y())

    def _xy_from_viewport_event(self, event):
        qt_pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        return self.xy_from_scene(self.plot.mapToScene(qt_pos))

    def _on_clicked(self, event):
        if hasattr(event, "double") and event.double():
            self.fit_board()
            return
        if self.editor.tool != "point" or self.editor.board_view is None:
            return
        button = event.button() if hasattr(event, "button") else QtCore.Qt.LeftButton
        if button not in (QtCore.Qt.LeftButton, QtCore.Qt.MouseButton.LeftButton):
            return
        pos = self.xy_from_scene(event.scenePos())
        if pos is None:
            return
        if self.editor.add_point(*pos):
            self.refresh_overlay()
            self.selectionChanged.emit()
        else:
            self.selectionChanged.emit()

    def _on_mouse_moved(self, pos):
        xy = self.xy_from_scene(pos)
        self.cursorMoved.emit("" if xy is None else cursor_text(*xy))

    def _update_rubber_band(self, origin, current):
        import pyqtgraph

        xs = [origin[0], current[0], current[0], origin[0], origin[0]]
        ys = [origin[1], origin[1], current[1], current[1], origin[1]]
        if self._rubber_item is None:
            self._rubber_item = pyqtgraph.PlotCurveItem(
                xs,
                ys,
                pen=pyqtgraph.mkPen("#d93025", width=1, style=QtCore.Qt.PenStyle.DashLine),
            )
            self.plot.addItem(self._rubber_item)
            return
        self._rubber_item.setData(xs, ys)

    def _clear_rubber_band(self):
        item = self._rubber_item
        self._rubber_item = None
        if item is None:
            return
        try:
            self.plot.removeItem(item)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    def eventFilter(self, obj, event):
        plot = self.plot
        if plot is None:
            return False
        try:
            viewport = plot.viewport() if callable(getattr(plot, "viewport", None)) else None
        except RuntimeError:
            return False
        if viewport is not None and obj is viewport:
            if self._handle_viewport_event(event):
                return True
        return super().eventFilter(obj, event)

    def _handle_viewport_event(self, event):
        if self.editor.tool != "rectangle":
            return False
        etype = event.type()
        press = QtCore.QEvent.Type.MouseButtonPress
        move = QtCore.QEvent.Type.MouseMove
        release = QtCore.QEvent.Type.MouseButtonRelease
        if etype not in (press, move, release):
            return False
        left = QtCore.Qt.LeftButton
        if etype == press:
            if hasattr(event, "button") and event.button() != left:
                return False
            pos = self._xy_from_viewport_event(event)
            if pos is None:
                return False
            self._drag_origin = pos
            return True
        if self._drag_origin is None:
            return False
        if etype == move:
            pos = self._xy_from_viewport_event(event)
            if pos is not None:
                self._update_rubber_band(self._drag_origin, pos)
            return True
        origin = self._drag_origin
        self._drag_origin = None
        self._clear_rubber_band()
        pos = self._xy_from_viewport_event(event)
        if pos is None:
            return True
        x0, x1 = sorted((origin[0], pos[0]))
        y0, y1 = sorted((origin[1], pos[1]))
        if x1 > x0 and y1 > y0 and self.editor.add_roi(x0, x1, y0, y1):
            self.refresh_overlay()
            self.selectionChanged.emit()
        return True


class BoardSelectionWidget(QtWidgets.QWidget):
    """Toolbar + large plot + status. Used by the selector dialog."""

    selectionChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        import pyqtgraph

        self.editor = ScanSelectionEditor()
        self.plot = pyqtgraph.PlotWidget(self)
        self.view = BoardSelectionView(self.plot, self.editor, parent=self)
        self.view.selectionChanged.connect(self._on_view_changed)
        self.view.cursorMoved.connect(self._set_cursor)

        self._radio_point = QtWidgets.QRadioButton("Point", self)
        self._radio_rect = QtWidgets.QRadioButton("Rectangle", self)
        self._radio_point.setAutoExclusive(False)
        self._radio_rect.setAutoExclusive(False)
        self._status = QtWidgets.QLabel(self)
        self._status.setWordWrap(True)
        self._cursor = QtWidgets.QLabel(self)

        tools = QtWidgets.QHBoxLayout()
        self._radio_point.toggled.connect(self._on_point_toggled)
        self._radio_rect.toggled.connect(self._on_rect_toggled)
        tools.addWidget(self._radio_point)
        tools.addWidget(self._radio_rect)
        for text, slot in (
            ("Undo", self._on_undo),
            ("Clear", self._on_clear),
            ("Scan entire board", self._on_entire),
            ("Fit board", self.view.fit_board),
        ):
            button = QtWidgets.QPushButton(text, self)
            button.clicked.connect(slot)
            tools.addWidget(button)
        tools.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(tools)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self._cursor)
        layout.addWidget(self._status)
        self._refresh_status()

    def set_board_view(self, view):
        self.editor.board_view = view
        self.view.draw(preserve_camera=False)
        self.view.fit_board()
        self._refresh_status()

    def set_selection(self, selection):
        self.editor.selection = selection
        self.view.refresh_overlay()
        self._refresh_status()

    def set_step_mm(self, step_mm):
        self.editor.step_mm = float(step_mm)
        self.view.refresh_overlay()
        self._refresh_status()

    def set_tool(self, tool):
        self._radio_point.blockSignals(True)
        self._radio_rect.blockSignals(True)
        self._radio_point.setChecked(tool == "point")
        self._radio_rect.setChecked(tool == "rectangle")
        self._radio_point.blockSignals(False)
        self._radio_rect.blockSignals(False)
        self.editor.set_tool(tool)
        self._refresh_status()

    def selection(self):
        return self.editor.selection

    def fit_board(self):
        self.view.fit_board()

    def add_point(self, x, y):
        ok = self.editor.add_point(x, y)
        self.view.refresh_overlay()
        self._refresh_status()
        return ok

    def add_roi(self, x_min, x_max, y_min, y_max):
        ok = self.editor.add_roi(x_min, x_max, y_min, y_max)
        self.view.refresh_overlay()
        self._refresh_status()
        return ok

    def _uncheck(self, radio):
        radio.blockSignals(True)
        radio.setChecked(False)
        radio.blockSignals(False)

    def _on_point_toggled(self, checked):
        if checked:
            self._uncheck(self._radio_rect)
            self.editor.set_tool("point")
        elif not self._radio_rect.isChecked():
            self.editor.tool = None
        self._refresh_status()

    def _on_rect_toggled(self, checked):
        if checked:
            self._uncheck(self._radio_point)
            self.editor.set_tool("rectangle")
        elif not self._radio_point.isChecked():
            self.editor.tool = None
        self._refresh_status()

    def _on_entire(self):
        self.editor.scan_entire_board()
        self.set_tool(None)
        self.view.refresh_overlay()
        self._refresh_status()
        self.selectionChanged.emit()

    def _on_clear(self):
        self.editor.clear()
        self.view.refresh_overlay()
        self._refresh_status()
        self.selectionChanged.emit()

    def _on_undo(self):
        self.editor.undo()
        self.view.refresh_overlay()
        self._refresh_status()
        self.selectionChanged.emit()

    def _on_view_changed(self):
        self._refresh_status()
        self.selectionChanged.emit()

    def _set_cursor(self, text):
        self._cursor.setText(text)

    def _refresh_status(self):
        if self.editor.last_message:
            self._status.setText(self.editor.last_message)
            return
        self._status.setText(self.editor.status_text())


class BoardSelectorDialog(QtWidgets.QDialog):
    """Working-copy editor. Apply commits; Cancel discards."""

    def __init__(self, parent, board_view, selection, step_mm, tool=None):
        super().__init__(parent)
        self.setWindowTitle("Select scan area")
        self.setModal(True)
        self.setSizeGripEnabled(True)
        self._widget = BoardSelectionWidget(self)
        self._widget.set_step_mm(step_mm)
        self._widget.set_board_view(board_view)
        working = (
            None
            if selection is None
            else engine_selection.ScanSelection(items=tuple(selection.items))
        )
        self._widget.set_selection(working)
        if tool:
            self._widget.set_tool(tool)
        self._buttons = QtWidgets.QDialogButtonBox(self)
        self._buttons.addButton("Apply selection", QtWidgets.QDialogButtonBox.AcceptRole)
        self._buttons.addButton("Cancel", QtWidgets.QDialogButtonBox.RejectRole)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._widget, 1)
        layout.addWidget(self._buttons)
        self._fit_to_screen()

    def applied_selection(self):
        return self._widget.selection()

    def applied_tool(self):
        return self._widget.editor.tool

    def _available_geometry(self):
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return QtCore.QRect(0, 0, *SELECTOR_SIZE)
        return screen.availableGeometry()

    def _fit_to_screen(self):
        """Keep Apply/Cancel on the usable desktop, not behind the taskbar."""
        available = self._available_geometry()
        max_w = max(available.width() - SELECTOR_SCREEN_MARGIN, 480)
        max_h = max(available.height() - SELECTOR_SCREEN_MARGIN, 360)
        width = min(SELECTOR_SIZE[0], max_w)
        height = min(SELECTOR_SIZE[1], max_h)
        self.setMinimumSize(min(SELECTOR_MIN_SIZE[0], width), min(SELECTOR_MIN_SIZE[1], height))
        self.setMaximumSize(available.width(), available.height())
        self.resize(width, height)
        frame = self.frameGeometry()
        extra_w = max(0, frame.width() - available.width())
        extra_h = max(0, frame.height() - available.height())
        if extra_w or extra_h:
            self.resize(max(self.width() - extra_w, 480), max(self.height() - extra_h, 360))
            frame = self.frameGeometry()
        self.move(
            available.x() + max(0, (available.width() - frame.width()) // 2),
            available.y() + max(0, (available.height() - frame.height()) // 2),
        )

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_to_screen()
        self._widget.fit_board()
