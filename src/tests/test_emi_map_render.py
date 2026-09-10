"""Renders the PCB pages for real, which needs pyqtgraph and a QApplication.

test_emi_map.py stays display-free and covers the interlocks by stubbing the
two canvases; nothing there can catch a misused pyqtgraph call.  This module
does exactly that and nothing else, and skips where no QApplication can exist.

Run from the src directory:
    python -m pytest tests -q
"""

import os
from pathlib import Path

import pytest

pyqtgraph = pytest.importorskip("pyqtgraph")

# Must be set before the first QApplication: these tests never want a window
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402
from PySide6.QtCore import QFile  # noqa: E402
from PySide6.QtUiTools import QUiLoader  # noqa: E402

from modules import emi_map  # noqa: E402
from test_emi_map import (  # noqa: E402
    _fake_ui,
    _FakeDevice,
    _FakeMessageBox,
    _FakePrinter,
    _FakeUsbInstr,
    engine_board,
    register,
)


class _RenderWizard(emi_map.EMIMapWizard):
    """The real wizard, real canvases, only the printer transport faked."""

    def _printer(self):
        return self.printer


class _Click:
    def __init__(self, scene_pos, double=False):
        self._scene_pos = scene_pos
        self._double = double

    def scenePos(self):
        return self._scene_pos

    def double(self):
        return self._double

    def button(self):
        return QtCore.Qt.LeftButton


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def wizard(app, monkeypatch):
    monkeypatch.setattr(emi_map, "QMessageBox", _FakeMessageBox)
    _FakeMessageBox.warnings = []
    ui = _fake_ui()
    for name in ("emiPlot", "regPlot", "liveSpectrum"):
        setattr(ui, name, pyqtgraph.PlotWidget())
    made = _RenderWizard(ui, _FakeUsbInstr([_FakeDevice()]), None)
    made.printer = _FakePrinter()
    cube = getattr(made.ui, "chkSpectrumCube", None)
    if cube is not None:
        cube.setChecked(False)
    made.ui.scanMode.setCurrentText(emi_map.MODE_PCB)
    made._board_model = engine_board.rectangular_board(
        50.0,
        30.0,
        components=(
            engine_board.Component(refdes="U1", x_mm=12.0, y_mm=8.0, side="top"),
            engine_board.Component(refdes="L2", x_mm=38.0, y_mm=22.0, side="top"),
        ),
    )
    return made


def items(plot):
    return plot.getPlotItem().items


class _UiLoader(QUiLoader):
    """QtTinySA's CustomLoader: pyqtgraph widgets cannot come from the loader."""

    def createWidget(self, class_name, parent=None, name=""):
        if class_name == "PlotWidget":
            return pyqtgraph.PlotWidget(parent=parent)
        return super().createWidget(class_name, parent, name)


@pytest.fixture
def real_ui(app):
    handle = QFile(str(Path(emi_map.__file__).with_name("emi_map.ui")))
    handle.open(QFile.ReadOnly)
    try:
        loaded = _UiLoader().load(handle)
    finally:
        handle.close()
    assert loaded is not None, "emi_map.ui failed to load"
    return loaded


class TestTheShippedDialog:
    """emi_map.ui and the controller are edited together and drift apart
    silently: the stub UI elsewhere answers every attribute, so only loading
    the real file proves the two still agree."""

    @pytest.fixture
    def wizard(self, real_ui, monkeypatch):
        monkeypatch.setattr(emi_map, "QMessageBox", _FakeMessageBox)
        _FakeMessageBox.warnings = []
        made = emi_map.EMIMapWizard(real_ui, _FakeUsbInstr([_FakeDevice()]), None)
        made.start()
        return made

    def test_opening_lands_on_setup_in_pcb_mode(self, wizard):
        assert wizard.ui.wizardStack.currentIndex() == emi_map.PAGE_SETUP
        assert wizard.ui.scanMode.currentText() == emi_map.MODE_PCB
        assert wizard.ui.btnNext.text() == "Next: import board"

    def test_the_rectangle_size_boxes_follow_the_mode(self, wizard):
        assert not wizard.ui.widthMm.isEnabled()
        wizard.ui.scanMode.setCurrentText(emi_map.MODE_RECTANGLE)
        assert wizard.ui.widthMm.isEnabled()
        assert wizard.ui.btnNext.text() == "Next: set origin"
        wizard.ui.scanMode.setCurrentText(emi_map.MODE_PCB)
        assert not wizard.ui.widthMm.isEnabled()

    def test_next_walks_the_pcb_pages(self, wizard):
        wizard._on_next()
        assert wizard.ui.wizardStack.currentIndex() == emi_map.PAGE_BOARD

    def test_the_setup_form_scrolls_instead_of_growing(self, wizard):
        scroll = wizard.ui.setupScroll
        assert scroll.widgetResizable()
        # Back/Next live outside the scroller, so they cannot be pushed away
        assert wizard.ui.btnNext.parent() is not scroll.widget()

    def test_rotate_buttons_exist_and_start_disabled(self, wizard):
        assert wizard.ui.btnRotateCcw.text() == "Rotate 90° CCW"
        assert wizard.ui.btnRotateCw.text() == "Rotate 90° CW"
        assert not wizard.ui.btnRotateCcw.isEnabled()
        assert not wizard.ui.btnRotateCw.isEnabled()

    def test_the_board_page_has_no_embedded_plot(self, wizard):
        wizard._on_next()
        assert wizard.ui.findChild(QtWidgets.QWidget, "boardPlot") is None
        assert wizard.ui.findChild(QtWidgets.QWidget, "btnFitBoard") is None
        assert wizard.ui.btnOpenBoardSelector.isVisible()
        assert wizard.ui.btnOpenBoardSelector.text() == "Open large board selector…"
        assert wizard.ui.btnOpenBoardSelector.parent() is not wizard.ui.pageRegister


class TestBoardRendering:
    def test_importing_a_board_draws_it_on_the_register_plot(self, wizard):
        wizard._apply_board_view()
        # Outline, component scatter, two labels, plus two landmark layers
        assert len(items(wizard.ui.regPlot)) == 6
        assert "50.00 x 30.00 mm" in wizard.ui.boardInfo.text()
        assert "2 components" in wizard.ui.boardInfo.text()

    def test_a_selected_point_does_not_draw_on_the_register_plot(self, wizard):
        wizard._apply_board_view()
        before_reg = len(items(wizard.ui.regPlot))
        wizard._add_selection_point(12.0, 8.0)
        assert len(items(wizard.ui.regPlot)) == before_reg
        assert "Point 1" in wizard.ui.selectionInfo.text()

    def test_redrawing_replaces_rather_than_stacks(self, wizard):
        wizard._apply_board_view()
        before = len(items(wizard.ui.regPlot))
        wizard._apply_board_view()
        assert len(items(wizard.ui.regPlot)) == before

    def test_rotating_90_ccw_swaps_size_and_moves_a_right_hand_part_up(self, wizard):
        wizard._board_model = engine_board.rectangular_board(
            50.0,
            30.0,
            components=(
                engine_board.Component(refdes="U1", x_mm=35.0, y_mm=15.0, side="top"),
            ),
        )
        wizard._apply_board_view(rotation_deg=0)
        before = len(items(wizard.ui.regPlot))
        wizard._rotate_board(90)
        assert len(items(wizard.ui.regPlot)) == before
        assert "30.00 x 50.00 mm" in wizard.ui.boardInfo.text()
        assert "rotated 90°" in wizard.ui.boardInfo.text()
        part = wizard._board_view.component("U1")
        assert part.x_mm == pytest.approx(25.0)
        assert part.y_mm == pytest.approx(25.0)
        scatter = next(
            item
            for item in items(wizard.ui.regPlot)
            if isinstance(item, pyqtgraph.ScatterPlotItem)
            and item not in (wizard._landmark_marks, wizard._selected_mark)
        )
        xs, ys = scatter.getData()
        assert xs[0] == pytest.approx(25.0)
        assert ys[0] == pytest.approx(25.0)

    def test_the_bottom_view_drops_parts_that_are_only_on_top(self, wizard):
        wizard.ui.boardSide.setCurrentText("bottom")
        wizard._apply_board_view()
        assert wizard._board_view.components == ()
        # Outline plus the two empty landmark layers
        assert len(items(wizard.ui.regPlot)) == 3
        assert "0 components" in wizard.ui.boardInfo.text()
        assert "mirrored" in wizard.ui.boardInfo.text()

    def test_recorded_landmarks_appear_on_the_register_plot(self, wizard):
        wizard._apply_board_view()
        register(wizard)
        assert len(wizard._landmark_marks.getData()[0]) == 2

    def test_a_click_on_the_plot_snaps_to_a_landmark(self, wizard):
        wizard._apply_board_view()
        view_box = wizard.ui.regPlot.getPlotItem().getViewBox()
        scene_pos = view_box.mapViewToScene(QtCore.QPointF(1.0, 1.5))
        wizard._on_reg_plot_clicked(_Click(scene_pos))
        assert wizard._selected_landmark[1:] == (0.0, 0.0)
        assert wizard._selected_mark.getData()[0].tolist() == [0.0]

    def test_the_live_map_is_placed_in_board_coordinates(self, wizard):
        wizard._apply_board_view()
        register(wizard)
        wizard.ui.chkTravelClearBoard.setChecked(True)
        config = wizard.build_config()
        plan = wizard._preflight(config)
        wizard._prepare_live_plot(config, plan)
        rect = wizard.image.boundingRect()
        assert (rect.width(), rect.height()) == pytest.approx((plan.nx, plan.ny))
        # The board outline is traced under the map so hotspots are locatable
        assert wizard._board_trace in items(wizard.ui.emiPlot)

    def test_the_empty_live_map_can_be_painted(self, wizard):
        """A float image with no levels raises inside Qt's paint callback, and
        a paint that raises is retried until the whole application dies."""
        wizard._apply_board_view()
        register(wizard)
        wizard.ui.chkTravelClearBoard.setChecked(True)
        config = wizard.build_config()
        wizard._prepare_live_plot(config, wizard._preflight(config))
        wizard.image.render()  # exactly what ImageItem.paint does

    def test_the_map_paints_with_only_one_reading_in_it(self, wizard):
        wizard._apply_board_view()
        register(wizard)
        wizard.ui.chkTravelClearBoard.setChecked(True)
        config = wizard.build_config()
        wizard._prepare_live_plot(config, wizard._preflight(config))
        wizard._grid[0, 0] = -42.0  # a zero-width level range
        wizard._refresh_image()
        wizard.image.render()


class _Wheel:
    def __init__(self, pos, delta=120):
        self._pos = pos
        self._delta = delta

    def pos(self):
        return self._pos

    def delta(self):
        return self._delta

    def accept(self):
        pass

    def ignore(self):
        pass


class _Drag:
    def __init__(self, button, pos, last):
        self._button = button
        self._pos = pos
        self._last = last

    def button(self):
        return self._button

    def pos(self):
        return self._pos

    def lastPos(self):
        return self._last

    def accept(self):
        pass

    def ignore(self):
        pass


class TestBoardNavigation:
    """CAD-style zoom/pan in the large selector; Register clicks stay landmarks."""

    def _open(self, wizard):
        wizard._apply_board_view()
        dialog = emi_map.BoardSelectorDialog(
            None,
            wizard._board_view,
            wizard._scan_selection,
            wizard.ui.stepMm.value(),
        )
        dialog._widget.plot.resize(800, 400)
        dialog.show()
        QtWidgets.QApplication.processEvents()
        dialog._widget.fit_board()
        return dialog

    def test_wheel_zoom_shrinks_the_view_around_the_cursor(self, wizard):
        dialog = self._open(wizard)
        try:
            view_box = dialog._widget.plot.getPlotItem().getViewBox()
            before = [list(axis) for axis in view_box.viewRange()]
            local = view_box.mapFromView(QtCore.QPointF(12.0, 8.0))
            view_box.wheelEvent(_Wheel(local, 120))
            after = view_box.viewRange()
            assert after[0][1] - after[0][0] < before[0][1] - before[0][0]
            assert after[0][0] < 12.0 < after[0][1]
            assert after[1][0] < 8.0 < after[1][1]
            before_frac = (12.0 - before[0][0]) / (before[0][1] - before[0][0])
            after_frac = (12.0 - after[0][0]) / (after[0][1] - after[0][0])
            assert after_frac == pytest.approx(before_frac, abs=0.05)
        finally:
            dialog.close()

    def test_pan_moves_the_view_range(self, wizard):
        dialog = self._open(wizard)
        try:
            view_box = dialog._widget.plot.getPlotItem().getViewBox()
            before = view_box.viewRange()
            dialog._widget.view.pan(5.0, -2.0)
            after = view_box.viewRange()
            assert after[0][0] == pytest.approx(before[0][0] + 5.0)
            assert after[1][0] == pytest.approx(before[1][0] - 2.0)
        finally:
            dialog.close()

    def test_left_drag_does_not_pan(self, wizard):
        dialog = self._open(wizard)
        try:
            view_box = dialog._widget.plot.getPlotItem().getViewBox()
            before = [list(axis) for axis in view_box.viewRange()]
            view_box.mouseDragEvent(
                _Drag(
                    QtCore.Qt.MouseButton.LeftButton,
                    QtCore.QPointF(40, 20),
                    QtCore.QPointF(10, 10),
                )
            )
            after = view_box.viewRange()
            assert after[0] == pytest.approx(before[0])
            assert after[1] == pytest.approx(before[1])
        finally:
            dialog.close()

    def test_middle_drag_pans(self, wizard):
        dialog = self._open(wizard)
        try:
            view_box = dialog._widget.plot.getPlotItem().getViewBox()
            before = view_box.viewRange()
            view_box.mouseDragEvent(
                _Drag(
                    QtCore.Qt.MouseButton.MiddleButton,
                    QtCore.QPointF(40, 20),
                    QtCore.QPointF(10, 10),
                )
            )
            after = view_box.viewRange()
            assert abs(after[0][0] - before[0][0]) + abs(after[1][0] - before[1][0]) > 0
        finally:
            dialog.close()

    def test_adding_an_roi_does_not_reset_zoom(self, wizard):
        dialog = self._open(wizard)
        try:
            view = dialog._widget.view
            view_box = view.view_box
            view.zoom_at(12.0, 8.0, 0.5)
            zoomed = [list(axis) for axis in view_box.viewRange()]
            dialog._widget.add_point(12.0, 8.0)
            dialog._widget.add_roi(10.0, 20.0, 5.0, 15.0)
            dialog._widget.editor.undo()
            dialog._widget.set_step_mm(5.0)
            after = view_box.viewRange()
            assert after[0] == pytest.approx(zoomed[0], abs=1e-6)
            assert after[1] == pytest.approx(zoomed[1], abs=1e-6)
            assert dialog._widget.selection().items
        finally:
            dialog.close()

    def test_roi_still_snaps_after_zoom(self, wizard):
        dialog = self._open(wizard)
        try:
            dialog._widget.view.zoom_at(12.0, 8.0, 0.5)
            assert dialog._widget.add_point(12.0, 8.0) is True
            point = dialog._widget.selection().items[0]
            assert (point.x, point.y) == (12.0, 8.0)
        finally:
            dialog.close()

    def test_fit_board_covers_the_board_bbox(self, wizard):
        dialog = self._open(wizard)
        try:
            view = dialog._widget.view
            view.zoom_at(12.0, 8.0, 0.4)
            view.fit_board()
            (x0, x1), (y0, y1) = view.view_box.viewRange()
            bx0, by0, bx1, by1 = wizard._board_view.bbox_mm
            assert x0 <= bx0 and x1 >= bx1
            assert y0 <= by0 and y1 >= by1
        finally:
            dialog.close()

    def test_cursor_readout_uses_board_millimetres(self, wizard):
        dialog = self._open(wizard)
        try:
            view = dialog._widget.view
            scene_pos = view.view_box.mapViewToScene(QtCore.QPointF(23.42, 15.81))
            view._on_mouse_moved(scene_pos)
            text = dialog._widget._cursor.text()
            assert "23.42" in text
            assert "15.81" in text
        finally:
            dialog.close()

    def test_register_clicks_are_unchanged(self, wizard):
        wizard._apply_board_view()
        view_box = wizard.ui.regPlot.getPlotItem().getViewBox()
        scene_pos = view_box.mapViewToScene(QtCore.QPointF(1.0, 1.5))
        wizard._on_reg_plot_clicked(_Click(scene_pos))
        assert wizard._selected_landmark[1:] == (0.0, 0.0)
        assert view_box.state["mouseEnabled"] == [True, True]


class TestLargeBoardSelector:
    """Precision ROI editing happens in a working-copy dialog, not a second model."""

    def _dialog(self, wizard, selection=None, tool=None):
        wizard._apply_board_view()
        dialog = emi_map.BoardSelectorDialog(
            None,
            wizard._board_view,
            selection if selection is not None else wizard._scan_selection,
            wizard.ui.stepMm.value(),
            tool=tool,
        )
        dialog._widget.plot.resize(1200, 700)
        dialog.show()
        QtWidgets.QApplication.processEvents()
        return dialog

    def test_the_window_fits_the_available_screen(self, wizard):
        dialog = self._dialog(wizard)
        try:
            screen = dialog.screen() or QtWidgets.QApplication.primaryScreen()
            available = screen.availableGeometry()
            frame = dialog.frameGeometry()
            assert frame.left() >= available.left() - 8
            assert frame.top() >= available.top() - 8
            assert frame.right() <= available.right() + 8
            assert frame.bottom() <= available.bottom() + 8
            box = dialog.findChild(QtWidgets.QDialogButtonBox)
            assert box is not None
            assert box.isVisible()
            assert "Apply selection" in [button.text() for button in box.buttons()]
            assert box.geometry().bottom() <= dialog.rect().bottom()
        finally:
            dialog.close()

    def test_fit_on_open_covers_the_board(self, wizard):
        dialog = self._dialog(wizard)
        try:
            view_box = dialog._widget.plot.getPlotItem().getViewBox()
            (x0, x1), (y0, y1) = view_box.viewRange()
            bx0, by0, bx1, by1 = wizard._board_view.bbox_mm
            assert x0 <= bx0 and x1 >= bx1
            assert y0 <= by0 and y1 >= by1
        finally:
            dialog.close()

    def test_cancel_leaves_the_wizard_selection_alone(self, wizard, monkeypatch):
        wizard._apply_board_view()
        wizard._add_selection_point(12.0, 8.0)
        before = wizard._scan_selection

        def fake_exec(dialog):
            dialog._widget.editor.clear()
            dialog._widget.add_roi(10.0, 20.0, 5.0, 15.0)
            return QtWidgets.QDialog.Rejected

        monkeypatch.setattr(emi_map.BoardSelectorDialog, "exec", fake_exec)
        wizard._open_board_selector()
        assert wizard._scan_selection is before
        assert isinstance(
            wizard._scan_selection.items[0], emi_map.engine_selection.BoardPoint
        )

    def test_apply_replaces_the_wizard_selection(self, wizard, monkeypatch):
        wizard._apply_board_view()
        wizard._add_selection_point(12.0, 8.0)

        def fake_exec(dialog):
            dialog._widget.editor.clear()
            dialog._widget.add_roi(10.0, 20.0, 5.0, 15.0)
            return QtWidgets.QDialog.Accepted

        monkeypatch.setattr(emi_map.BoardSelectorDialog, "exec", fake_exec)
        wizard._open_board_selector()
        assert len(wizard._scan_selection.items) == 1
        assert isinstance(
            wizard._scan_selection.items[0], emi_map.engine_selection.BoardROI
        )

    def test_apply_entire_board_clears_the_subset(self, wizard, monkeypatch):
        wizard._apply_board_view()
        wizard._add_selection_point(12.0, 8.0)

        def fake_exec(dialog):
            dialog._widget.editor.scan_entire_board()
            dialog._widget.set_tool(None)
            return QtWidgets.QDialog.Accepted

        monkeypatch.setattr(emi_map.BoardSelectorDialog, "exec", fake_exec)
        wizard._open_board_selector()
        assert wizard._scan_selection is None
        assert not wizard.ui.radioSelectPoint.isChecked()

    def test_the_dialog_uses_the_same_snap_as_the_embedded_plot(self, wizard):
        wizard._apply_board_view()
        dialog = self._dialog(wizard, selection=None)
        try:
            assert dialog._widget.add_point(-0.1, 15.0) is False
            assert wizard._add_selection_point(-0.1, 15.0) is False
            assert dialog._widget.add_point(12.0, 8.0) is True
            assert wizard._add_selection_point(12.0, 8.0) is True
            point = dialog._widget.selection().items[0]
            other = wizard._scan_selection.items[0]
            assert (point.x, point.y) == (other.x, other.y) == (12.0, 8.0)
        finally:
            dialog.close()

    def test_overlay_inside_the_dialog_keeps_zoom(self, wizard):
        dialog = self._dialog(wizard)
        try:
            view = dialog._widget.view
            view.zoom_at(12.0, 8.0, 0.5)
            zoomed = [list(axis) for axis in view.view_box.viewRange()]
            dialog._widget.add_point(12.0, 8.0)
            after = view.view_box.viewRange()
            assert after[0] == pytest.approx(zoomed[0], abs=1e-6)
            assert after[1] == pytest.approx(zoomed[1], abs=1e-6)
        finally:
            dialog.close()


