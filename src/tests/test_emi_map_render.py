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
    def __init__(self, scene_pos):
        self._scene_pos = scene_pos

    def scenePos(self):
        return self._scene_pos


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def wizard(app, monkeypatch):
    monkeypatch.setattr(emi_map, "QMessageBox", _FakeMessageBox)
    _FakeMessageBox.warnings = []
    ui = _fake_ui()
    for name in ("emiPlot", "boardPlot", "regPlot"):
        setattr(ui, name, pyqtgraph.PlotWidget())
    made = _RenderWizard(ui, _FakeUsbInstr([_FakeDevice()]), None)
    made.printer = _FakePrinter()
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


class TestBoardRendering:
    def test_importing_a_board_draws_it_on_both_pages(self, wizard):
        wizard._apply_board_side()
        # Outline curve, one component scatter, one label each
        assert len(items(wizard.ui.boardPlot)) == 4
        # The register page also keeps its two landmark layers
        assert len(items(wizard.ui.regPlot)) == 6
        assert "50.00 x 30.00 mm" in wizard.ui.boardInfo.text()
        assert "2 components" in wizard.ui.boardInfo.text()

    def test_redrawing_replaces_rather_than_stacks(self, wizard):
        wizard._apply_board_side()
        before = len(items(wizard.ui.boardPlot))
        wizard._apply_board_side()
        assert len(items(wizard.ui.boardPlot)) == before

    def test_the_bottom_view_drops_parts_that_are_only_on_top(self, wizard):
        wizard.ui.boardSide.setCurrentText("bottom")
        wizard._apply_board_side()
        assert wizard._board_view.components == ()
        # Just the outline: nothing on this side to mark or label
        assert len(items(wizard.ui.boardPlot)) == 1
        assert "0 components" in wizard.ui.boardInfo.text()
        assert "mirrored" in wizard.ui.boardInfo.text()

    def test_recorded_landmarks_appear_on_the_register_plot(self, wizard):
        wizard._apply_board_side()
        register(wizard)
        assert len(wizard._landmark_marks.getData()[0]) == 2

    def test_a_click_on_the_plot_snaps_to_a_landmark(self, wizard):
        wizard._apply_board_side()
        view_box = wizard.ui.regPlot.getPlotItem().getViewBox()
        scene_pos = view_box.mapViewToScene(QtCore.QPointF(1.0, 1.5))
        wizard._on_reg_plot_clicked(_Click(scene_pos))
        assert wizard._selected_landmark[1:] == (0.0, 0.0)
        assert wizard._selected_mark.getData()[0].tolist() == [0.0]

    def test_the_live_map_is_placed_in_board_coordinates(self, wizard):
        wizard._apply_board_side()
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
        wizard._apply_board_side()
        register(wizard)
        wizard.ui.chkTravelClearBoard.setChecked(True)
        config = wizard.build_config()
        wizard._prepare_live_plot(config, wizard._preflight(config))
        wizard.image.render()  # exactly what ImageItem.paint does

    def test_the_map_paints_with_only_one_reading_in_it(self, wizard):
        wizard._apply_board_side()
        register(wizard)
        wizard.ui.chkTravelClearBoard.setChecked(True)
        config = wizard.build_config()
        wizard._prepare_live_plot(config, wizard._preflight(config))
        wizard._grid[0, 0] = -42.0  # a zero-width level range
        wizard._refresh_image()
        wizard.image.render()
