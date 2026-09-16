"""Renders the PCB pages for real, which needs pyqtgraph and a QApplication.

test_emi_map.py stays display-free and covers the interlocks by stubbing the
two canvases; nothing there can catch a misused pyqtgraph call.  This module
does exactly that and nothing else, and skips where no QApplication can exist.

Run from the src directory:
    python -m pytest tests -q
"""

import os
from pathlib import Path

os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
# Must be set before the first QApplication: these tests never want a window
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pyqtgraph = pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtWidgets  # noqa: E402
from PySide6.QtCore import QFile  # noqa: E402
from PySide6.QtUiTools import QUiLoader  # noqa: E402

from modules import emi_map  # noqa: E402
from test_emi_map import (  # noqa: E402
    _agree_easy_scan_plane,
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

    def _printer(self, *, manage_z=None):
        return self._bind_printer_frame(self.printer)


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
        assert wizard.ui.btnNext.text() == "Next: select fixture"

    def test_easy_pcb_has_visible_fixture_and_scan_setup_workspaces(self, wizard):
        assert wizard.ui.wizardStack.indexOf(wizard.ui.pageFixtureTeach) == emi_map.PAGE_FIXTURE_TEACH
        assert wizard.ui.wizardStack.indexOf(wizard.ui.pageEasyScanSetup) == emi_map.PAGE_SCAN_SETUP
        assert wizard.ui.fixtureTeachScroll.widgetResizable() is True
        assert wizard.ui.fixtureTeachTable.rowCount() == 1
        assert wizard.ui.fixtureTeachTable.columnCount() == 5
        assert len(wizard._workflow_step_labels) == 5
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_FIXTURE_TEACH)
        assert "Step 2 of 5" in wizard.ui.stepHeader.text()
        expected_home = (
            "HOME & SET BOARD ZERO"
            if wizard._p1_bltouch_commanded_xy() is not None
            else "HOME XY FOR TEACHING"
        )
        assert wizard.ui.btnFixtureHomeXyz.text() == expected_home
        assert wizard.ui.btnFixtureRetryPrinter.text() == "RETRY PRINTER CONNECTION"
        assert "without moving" in wizard.ui.btnFixtureRetryPrinter.toolTip()
        assert "250000" in wizard.ui.btnFixtureRetryPrinter.toolTip()
        assert wizard.ui.fixtureMaxZChange.value() == pytest.approx(0.20)
        assert wizard.ui.fixtureMaxZChange.minimum() == pytest.approx(0.05)
        assert "M119" in wizard.ui.btnFixtureVerify.toolTip()
        assert "Does not send G30" in wizard.ui.btnFixtureVerify.toolTip()
        assert "P1 upper-left" in wizard.ui.fixtureReferenceMap.text()
        assert wizard.ui.btnFixtureClearPoint.objectName() == "btnFixtureClearPoint"
        assert "config.yaml" in wizard.ui.fixtureConfigNote.text()
        assert not hasattr(wizard.ui, "emiProbeOffsetX")
        assert not hasattr(wizard.ui, "grpFixtureCalibrations")
        assert "A/B" in wizard.ui.chkFixtureProbeClear.text()
        wizard._fixture_config_cache = emi_map.engine_config.FixtureConfig(
            emi_probe_offset_x_mm=-104.0,
            emi_probe_offset_y_mm=1.0,
            pcb_thickness_mm=0.746,
            pcb_yaw_deg=0.0,
            e_probe_tip_z_minus_g30_contact_mm=None,
        )
        wizard._machine_fixture_document = lambda: wizard._overlay_fixture_config(
            {"pcb_thickness_mm": 0.746, "e_probe_tip_z_minus_g30_contact_mm": None}
        )
        wizard._apply_operator_mode_ui()
        wizard._refresh_easy_height_label()
        assert wizard.ui.btnEasySetHeight.isHidden() is False
        assert wizard.ui.btnEasySetPcbSurface.isHidden() is False
        assert wizard.ui.btnEasyResetPcbSurface.isHidden() is False
        assert wizard.ui.grpEasyManualHeight.isHidden() is False
        assert wizard.ui.grpScanHeight.isHidden() is True
        page = wizard.ui.pageFixtureTeach
        assert page.findChild(QtWidgets.QPushButton, "btnEasySetPcbSurface") is not None
        assert page.findChild(QtWidgets.QPushButton, "btnEasySetHeight") is not None

        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_SCAN_SETUP)
        assert wizard.ui.glassboardPlacementSide.count() == 2
        assert wizard.ui.glassboardPlacementSide.itemData(0) == "top"
        assert wizard.ui.glassboardPlacementSide.itemData(1) == "bottom"
        assert wizard.ui.chkScanSetupBoardSeated.isVisible()

    def test_progress_strip_follows_height_readiness(self, wizard):
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_SCAN)
        wizard._refresh_workflow_progress()
        step2 = wizard._workflow_step_labels[1]
        assert "#067647" not in step2.styleSheet()
        _agree_easy_scan_plane(wizard)
        wizard._refresh_workflow_progress()
        assert "#067647" in step2.styleSheet()

    def test_scan_setup_offers_only_corner_teaching_and_seating(self, wizard):
        """Step 4 answers 'where is the PCB' with taught points and nothing else."""
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_SCAN_SETUP)
        for side in emi_map.SCAN_SETUP_SIDES:
            for letter in "AB":
                button = getattr(wizard.ui, f"btnTeach{side.capitalize()}Corner{letter}")
                assert button.isVisible()
            assert getattr(wizard.ui, f"btnUse{side.capitalize()}Side").isVisible()
            assert getattr(wizard.ui, f"btnClear{side.capitalize()}Corners").isVisible()
        assert wizard.ui.chkScanSetupBoardSeated.isVisible()
        # The face toward the probe follows the taught pair, so it is no longer
        # an operator choice, and the STL/legacy position controls are gone.
        assert wizard.ui.glassboardPlacementSide.isVisible() is False
        for name in (
            "btnApplyGlassboardPlacement",
            "btnPreviewPcbCenter",
            "scanSetupAlignment",
            "btnScanSetupLoadAlignment",
            "scanSetupHeightStatus",
        ):
            assert not hasattr(wizard.ui, name)
        assert wizard.ui.pageEasyScanSetup.findChild(
            QtWidgets.QPushButton, "btnEasySetHeight"
        ) is None
        hints = [
            widget.text()
            for widget in wizard.ui.pageEasyScanSetup.findChildren(QtWidgets.QLabel)
            if widget.text()
        ]
        assert any(
            "inner-pocket" in text
            and "pcb_yaw_deg" in text
            and "height only" in text
            for text in hints
        )

    def test_review_and_measure_no_longer_repeats_the_scan_plan(self, wizard):
        assert not hasattr(wizard.ui, "scanLivePreviewHost")
        widget = wizard._scan_preview_widget
        assert widget is not None
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_SCAN)
        assert widget.parent() is wizard.ui.scanPreviewHost


    # -------------------------------------------- Step 4 taught PCB corners
    def _ready_to_teach(self, wizard, monkeypatch):
        """A homed machine with an imported board and a recording fixture file."""
        saved = []
        wizard._board_model = emi_map.engine_board.rectangular_board(57.7621, 12.54)
        wizard._apply_board_view(rotation_deg=0)
        monkeypatch.setattr(wizard, "_fixture_profile_name", lambda: "Glassboard")
        monkeypatch.setattr(
            wizard,
            "_machine_fixture_document",
            lambda: {
                "fixture_id": "Glassboard",
                "fixture_teaching": None,
                "pcb_yaw_deg": 0.0,
            },
        )
        monkeypatch.setattr(
            emi_map.engine_machine_fixtures,
            "save_taught_pcb_corners",
            lambda name, side, corners: saved.append(
                (name, side, [dict(entry) for entry in corners])
            ),
        )
        monkeypatch.setattr(
            emi_map.engine_machine_fixtures,
            "clear_taught_pcb_corners",
            lambda name, side: saved.append((name, side, [])),
        )
        wizard.printer = _FakePrinter()
        wizard._printer = lambda **_kwargs: wizard.printer
        wizard._pcb_xy_homed = True
        return saved

    def _teach(self, wizard, side, origin=(20.0, 40.0)):
        """Capture both corners of one face at a known translation."""
        view = wizard._board_view_for_side(side)
        corners = wizard._pcb_corner_board_points(view)
        for slot, (board_x, board_y) in enumerate(corners):
            wizard.printer.position = (origin[0] + board_x, origin[1] + board_y)
            assert wizard._capture_pcb_corner(side, slot)
        return corners

    def test_two_captured_corners_position_the_board_without_motion(
        self, wizard, monkeypatch
    ):
        taught = self._ready_to_teach(wizard, monkeypatch)
        wizard.printer.sent.clear()
        self._teach(wizard, "top")

        assert wizard._active_taught_side == "top"
        assert wizard._board_view.side == "top"
        assert wizard._registration is not None
        assert len(wizard._landmarks) == 2
        assert wizard._registration.rms_mm == pytest.approx(0.0, abs=1e-6)
        assert wizard._registration_problems() == []
        # Teaching reads M114; it never drives the carriage itself.
        assert [code for code in wizard.printer.sent if code.startswith("G0")] == []
        assert [entry[0] for entry in taught] == ["Glassboard", "Glassboard"]
        assert taught[-1][1] == "top"
        assert len(taught[-1][2]) == 2

    def test_each_face_is_taught_and_stored_separately(self, wizard, monkeypatch):
        taught = self._ready_to_teach(wizard, monkeypatch)
        self._teach(wizard, "top", origin=(20.0, 40.0))
        self._teach(wizard, "bottom", origin=(25.0, 45.0))

        assert wizard._active_taught_side == "bottom"
        assert wizard._board_view.side == "bottom"
        assert wizard._taught_side_complete("top")
        assert wizard._taught_side_complete("bottom")
        assert {entry[1] for entry in taught} == {"top", "bottom"}
        # Switching back reuses the stored pair instead of re-teaching.
        assert wizard._use_taught_pcb_side("top")
        assert wizard._board_view.side == "top"

    def test_saved_corners_are_restored_for_the_imported_board(
        self, wizard, monkeypatch
    ):
        taught = self._ready_to_teach(wizard, monkeypatch)
        self._teach(wizard, "top")
        payload = {entry[1]: entry[2] for entry in taught if entry[2]}
        store = {
            side: {
                "template_id": emi_map.engine_glassboard_fixture.TAUGHT_POCKET_TEMPLATE_ID,
                "corners": corners,
            }
            for side, corners in payload.items()
        }
        monkeypatch.setattr(
            wizard,
            "_machine_fixture_document",
            lambda: {"fixture_id": "Glassboard", "taught_pcb_corners": store},
        )
        wizard._taught_pcb_corner_points = {
            side: {} for side in emi_map.SCAN_SETUP_SIDES
        }
        wizard._load_taught_pcb_corners()

        assert wizard._taught_side_complete("top")
        assert wizard._apply_taught_pcb_corners("top")

    def test_clearing_the_measured_face_drops_the_position_it_produced(
        self, wizard, monkeypatch
    ):
        self._ready_to_teach(wizard, monkeypatch)
        self._teach(wizard, "top")
        assert wizard._registration is not None

        wizard._clear_pcb_corners("top")

        assert wizard._taught_pcb_corners("top") == {}
        assert wizard._active_taught_side is None
        assert wizard._registration is None
        assert wizard._approved_scan_fingerprint is None
        assert wizard._registration_problems() != []

    def test_pocket_diagonal_mismatch_blocks_next(self, wizard, monkeypatch):
        self._ready_to_teach(wizard, monkeypatch)
        view = wizard._board_view_for_side("top")
        a, b = wizard._pcb_corner_board_points(view)
        wizard.printer.position = (20.0, 40.0)
        assert wizard._capture_pcb_corner("top", 0)
        wizard.printer.position = (
            20.0 + 0.5 * (b[0] - a[0]),
            40.0 + 0.5 * (b[1] - a[1]),
        )
        assert wizard._capture_pcb_corner("top", 1)
        assert wizard._registration is None
        assert wizard._pocket_pair_state["top"] == "rejected"
        assert wizard._face_blocks_stl_fallback("top") is True
        assert wizard._easy_mode_active()
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_SCAN_SETUP)
        wizard.ui.chkScanSetupBoardSeated.setChecked(True)
        wizard._update_nav()
        problems = wizard._registration_problems()
        assert problems
        assert any("scale" in item for item in problems)
        assert wizard.ui.btnNext.isEnabled() is False

    def test_v1_saved_pair_requires_recapture_and_does_not_stl_place(
        self, wizard, monkeypatch
    ):
        self._ready_to_teach(wizard, monkeypatch)
        monkeypatch.setattr(
            wizard,
            "_machine_fixture_document",
            lambda: {
                "fixture_id": "Glassboard",
                "fixture_teaching": {"points": []},
                "taught_pcb_corners": {
                    "top": [
                        {"slot": 0, "machine_x_mm": 75.0, "machine_y_mm": 130.0},
                        {"slot": 1, "machine_x_mm": 131.0, "machine_y_mm": 142.0},
                    ]
                },
            },
        )
        wizard._load_taught_pcb_corners()
        side = wizard._scan_setup_side()
        applied = wizard._apply_taught_pcb_corners(side)
        if not applied:
            wizard._drop_pocket_scan_xy()
        assert wizard._pocket_pair_state["top"] == "stale"
        assert wizard._taught_pcb_corners("top") == {}
        assert applied is False
        assert wizard._stl_board_placement is None
        assert any("older meaning" in item for item in wizard._registration_problems())
        wizard._fixture_verified_session = True
        assert wizard._maybe_auto_place_glassboard() is False
        assert wizard._stl_board_placement is None

    def test_empty_teaching_does_not_stl_fallback(self, wizard, monkeypatch):
        self._ready_to_teach(wizard, monkeypatch)
        side = wizard._scan_setup_side()
        if not wizard._apply_taught_pcb_corners(side):
            wizard._drop_pocket_scan_xy()
        assert wizard._easy_mode_active()
        assert wizard._maybe_auto_place_glassboard() is False
        assert wizard._registration is None
        assert wizard._stl_board_placement is None
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_SCAN)
        wizard.ui.chkTravelClearBoard.setChecked(True)
        with pytest.raises(
            RuntimeError,
            match="Capture both inner-pocket|Approve the scan plan|Registration is not fit",
        ):
            wizard._pcb_preflight()

    def test_cad_length_ab_installs_even_when_p1_xy_disagrees(
        self, wizard, monkeypatch
    ):
        self._ready_to_teach(wizard, monkeypatch)
        p1 = emi_map.engine_fixture_teaching.FixtureReferencePoint(
            "P1", -53.081, 17.0, 80.0, 125.0, 80.0, 125.0, 1.0, 1.0
        )
        teaching = emi_map.engine_fixture_teaching.build_fixture_teaching([p1])
        monkeypatch.setattr(
            wizard,
            "_machine_fixture_document",
            lambda: {
                "fixture_id": "Glassboard",
                "fixture_teaching": teaching,
                "emi_probe_offset_mm": {"x": -104.0, "y": 1.0},
                "emi_probe_offset_source": "typed",
                "emi_probe_offset_convention": "v2_tip_minus_bltouch",
                "pcb_yaw_deg": 0.0,
            },
        )
        self._teach(wizard, "top")
        assert wizard._registration is not None
        assert wizard._pocket_pair_state["top"] == "ok"
        assert wizard._taught_pocket_placement["placement"] == "taught_pocket"
        assert not any("disagree" in item for item in wizard._registration_problems())
        wizard._fixture_verified_session = True
        wizard._approved_scan_fingerprint = "stale"
        wizard.ui.chkScanSetupBoardSeated.setChecked(True)
        wizard.ui.chkBoardSeated.setChecked(True)
        wizard.ui.chkTravelClearBoard.setChecked(True)
        _agree_easy_scan_plane(wizard)
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_SCAN)
        with pytest.raises(RuntimeError, match="Approve the scan plan"):
            wizard._pcb_preflight()
        wizard._clear_pcb_corners("top")
        assert wizard._approved_scan_fingerprint is None
        assert wizard._registration is None

    def test_top_bottom_selection_does_not_stl_place(self, wizard, monkeypatch):
        self._ready_to_teach(wizard, monkeypatch)
        self._teach(wizard, "top")
        assert wizard._registration is not None
        wizard.ui.glassboardPlacementSide.setCurrentIndex(1)
        assert wizard._stl_board_placement is None
        assert wizard._registration is None
        assert wizard._taught_pocket_placement is None
        assert wizard._maybe_auto_place_glassboard() is False

    def test_pocket_path_approve_and_start_do_not_move_z(self, wizard, monkeypatch):
        self._ready_to_teach(wizard, monkeypatch)
        self._teach(wizard, "top")
        wizard._fixture_verified_session = True
        wizard.ui.chkScanSetupBoardSeated.setChecked(True)
        wizard.ui.chkBoardSeated.setChecked(True)
        wizard.ui.chkTravelClearBoard.setChecked(True)
        _agree_easy_scan_plane(wizard)
        wizard._refresh_scan_preview()
        printer = wizard.printer
        printer.sent.clear()
        widget = wizard._scan_preview_widget
        widget.approve_plan()
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_SCAN)
        wizard._pcb_preflight()
        assert not any(
            cmd.startswith("G1 Z") or cmd.startswith("G0 Z") for cmd in printer.sent
        )

    def test_teaching_before_homing_is_refused(self, wizard, monkeypatch):
        self._ready_to_teach(wizard, monkeypatch)
        wizard._pcb_xy_homed = False
        _FakeMessageBox.warnings = []

        assert wizard._capture_pcb_corner("top", 0) is False
        assert wizard._taught_pcb_corners("top") == {}
        assert "Home X and Y" in _FakeMessageBox.warnings[-1]

    def test_visible_scan_setup_seating_confirmation_controls_next(self, wizard):
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_SCAN_SETUP)
        wizard._registration_problems = lambda: []
        wizard._easy_at_scan_plane = lambda: True
        wizard.ui.chkScanSetupBoardSeated.setChecked(False)
        wizard._update_nav()
        assert wizard.ui.chkBoardSeated.isChecked() is False
        assert wizard.ui.btnNext.isEnabled() is False
        assert "fully seated" in wizard.ui.btnNext.toolTip()

        wizard.ui.chkScanSetupBoardSeated.setChecked(True)
        assert wizard.ui.chkBoardSeated.isChecked() is True
        assert wizard.ui.btnNext.isEnabled() is True

    def test_stl_middle_holder_places_top_or_bottom_without_manual_landmarks(
        self, wizard, monkeypatch, tmp_path
    ):
        points = [
            emi_map.engine_fixture_teaching.FixtureReferencePoint(
                name, fixture_x, fixture_y,
                machine_x, machine_y, machine_x, machine_y, 2.0, 2.0
            )
            for name, fixture_x, fixture_y, machine_x, machine_y in (
                ("P1", -53.081, 17.0, 76.919, 117.0),
                ("P2", -53.081, -17.0, 76.919, 83.0),
                ("P3", 53.081, -17.0, 183.081, 83.0),
                ("P4", 53.081, 17.0, 183.081, 117.0),
            )
        ]
        teaching = emi_map.engine_fixture_teaching.build_fixture_teaching(points)
        monkeypatch.setattr(
            wizard, "_machine_fixture_document", lambda: {
                "fixture_id": "Glassboard",
                "fixture_teaching": teaching,
                "emi_probe_offset_mm": {"x": 32.0, "y": -4.0},
                "emi_probe_offset_source": "typed",
                "emi_probe_offset_convention": "v2_tip_minus_bltouch",
            }
        )
        wizard._fixture_verified_session = True
        wizard._pcb_xy_homed = True
        wizard._board_model = emi_map.engine_board.rectangular_board(57.762, 12.54)
        wizard.ui.glassboardPlacementSide.setCurrentIndex(1)

        assert wizard._set_glassboard_middle_placement("bottom", persist=False)
        assert wizard._board_view.side == "bottom"
        assert wizard._registration is not None
        assert wizard._stl_board_placement["placement"] == "middle_glassboard_support"
        assert wizard._stl_board_placement["board_side_facing_probe"] == "bottom"
        assert len(wizard._landmarks) == 2
        assert "BOTTOM faces the probe" in wizard.ui.glassboardPlacementStatus.text()
        provenance = wizard._provenance()
        assert provenance["registration_source"] == "glassboard_stl_middle_holder"
        assert provenance["machine_fixture_document"]["fixture_teaching"] == teaching
        wizard._write_profile_copy(tmp_path)
        assert (tmp_path / "machine_fixture.json").exists()
        assert not (tmp_path / "fixture_profile.json").exists()

    def test_saved_middle_holder_association_restores_for_later_measurements(
        self, wizard, monkeypatch
    ):
        points = [
            emi_map.engine_fixture_teaching.FixtureReferencePoint(
                name, x, y, x + 130.0, y + 100.0,
                x + 130.0, y + 100.0, 2.0, 2.0
            )
            for name, (x, y) in
            emi_map.engine_glassboard_fixture.REFERENCE_FIXTURE_XY.items()
        ]
        teaching = emi_map.engine_fixture_teaching.build_fixture_teaching(points)
        wizard._board_model = emi_map.engine_board.rectangular_board(57.762, 12.54)
        top_view = wizard._board_model.view("top", 0)
        document = {
            "fixture_id": "Glassboard",
            "fixture_teaching": teaching,
            "emi_probe_offset_mm": {"x": 32.0, "y": -4.0},
            "emi_probe_offset_source": "typed",
            "emi_probe_offset_convention": "v2_tip_minus_bltouch",
            "board_placements": {
                top_view.geometry_hash: {
                    "template_id": emi_map.engine_glassboard_fixture.TEMPLATE_ID,
                    "board_side_facing_probe": "top",
                }
            },
        }
        monkeypatch.setattr(wizard, "_machine_fixture_document", lambda: document)
        wizard._fixture_verified_session = True
        wizard._pcb_xy_homed = True

        assert wizard._restore_glassboard_middle_placement()
        assert wizard._board_view.side == "top"
        assert wizard._stl_board_placement is not None
        assert "Restored" in wizard.ui.glassboardPlacementStatus.text()

    def test_emi_probe_offset_shifts_placed_machine_center(self, wizard, monkeypatch):
        points = [
            emi_map.engine_fixture_teaching.FixtureReferencePoint(
                name, fixture_x, fixture_y,
                machine_x, machine_y, machine_x, machine_y, 2.0, 2.0
            )
            for name, fixture_x, fixture_y, machine_x, machine_y in (
                ("P1", -53.081, 17.0, 76.919, 117.0),
                ("P2", -53.081, -17.0, 76.919, 83.0),
                ("P3", 53.081, -17.0, 183.081, 83.0),
                ("P4", 53.081, 17.0, 183.081, 117.0),
            )
        ]
        teaching = emi_map.engine_fixture_teaching.build_fixture_teaching(points)
        wizard._board_model = emi_map.engine_board.rectangular_board(57.762, 12.54)
        wizard._apply_board_view(rotation_deg=0)
        monkeypatch.setattr(
            wizard, "_machine_fixture_document", lambda: {
                "fixture_id": "Glassboard",
                "fixture_teaching": teaching,
                "emi_probe_offset_mm": {"x": 32.0, "y": -4.0},
            }
        )
        wizard._fixture_verified_session = True
        wizard._pcb_xy_homed = True
        assert wizard._set_glassboard_middle_placement(
            "top", persist=False, announce=False
        )
        _, expected = emi_map.engine_glassboard_fixture.board_registration_in_middle_holder(
            wizard._board_view, teaching, emi_probe_offset_mm=(32.0, -4.0)
        )
        assert wizard._stl_board_placement["machine_support_center_mm"] == pytest.approx(
            expected["machine_support_center_mm"]
        )
        assert "E-probe offset +32.000, -4.000 mm" in wizard.ui.glassboardPlacementStatus.text()

    def test_place_pcb_warns_when_e_probe_offset_unset(self, wizard, monkeypatch):
        points = [
            emi_map.engine_fixture_teaching.FixtureReferencePoint(
                name, x, y, x + 130.0, y + 100.0,
                x + 130.0, y + 100.0, 2.0, 2.0
            )
            for name, (x, y) in
            emi_map.engine_glassboard_fixture.REFERENCE_FIXTURE_XY.items()
        ]
        teaching = emi_map.engine_fixture_teaching.build_fixture_teaching(points)
        monkeypatch.setattr(
            wizard, "_machine_fixture_document", lambda: {
                "fixture_id": "Glassboard",
                "fixture_teaching": teaching,
            }
        )
        wizard._fixture_verified_session = True
        wizard._pcb_xy_homed = True
        wizard._board_model = emi_map.engine_board.rectangular_board(57.762, 12.54)
        emi_map.QMessageBox.warnings = []
        assert wizard._set_glassboard_middle_placement("top", persist=False) is False
        assert wizard._stl_board_placement is None
        assert any(
            "config.yaml" in text
            for text in emi_map.QMessageBox.warnings
        )

    def test_fixture_verify_moves_to_taught_bltouch_xy_not_e_probe_offset(
        self, wizard, monkeypatch
    ):
        points = [
            emi_map.engine_fixture_teaching.FixtureReferencePoint(
                name, fixture_x, fixture_y,
                machine_x, machine_y, machine_x, machine_y, 2.0, 2.0
            )
            for name, fixture_x, fixture_y, machine_x, machine_y in (
                ("P1", -53.081, 17.0, 79.0, 128.0),
                ("P2", -53.081, -17.0, 79.0, 97.0),
                ("P3", 53.081, -17.0, 179.0, 97.0),
                ("P4", 53.081, 17.0, 179.0, 127.0),
            )
        ]
        teaching = emi_map.engine_fixture_teaching.build_fixture_teaching(points)
        document = {
            "fixture_id": "Glassboard",
            "fixture_teaching": teaching,
            "emi_probe_offset_mm": {"x": 32.0, "y": -4.0},
        }
        printer = _FakePrinter()
        wizard._printer = lambda **_kwargs: printer
        monkeypatch.setattr(wizard, "_fixture_profile_name", lambda: "Glassboard")
        monkeypatch.setattr(
            emi_map.engine_machine_fixtures, "read_fixture", lambda _name: document
        )
        monkeypatch.setattr(
            emi_map.engine_machine_fixtures,
            "save_verification",
            lambda *_args, **_kwargs: {
                **document,
                "fixture_teaching": {
                    **teaching,
                    "latest_verification": {"passed": True},
                },
            },
        )
        wizard._do_fixture_verify(0.20, 0.10)
        assert [item for item in printer.sent if item.startswith("G0 ")] == [
            "G0 X79.0 Y128.0",
            "G0 X79.0 Y97.0",
            "G0 X179.0 Y97.0",
            "G0 X179.0 Y127.0",
        ]
        assert printer.sent.count("G28 X Y") == 1

    def test_fixture_verify_skips_home_when_already_homed(self, wizard, monkeypatch):
        points = [
            emi_map.engine_fixture_teaching.FixtureReferencePoint(
                name, fixture_x, fixture_y,
                machine_x, machine_y, machine_x, machine_y, 2.0, 2.0
            )
            for name, fixture_x, fixture_y, machine_x, machine_y in (
                ("P1", -53.081, 17.0, 79.0, 128.0),
                ("P2", -53.081, -17.0, 79.0, 97.0),
                ("P3", 53.081, -17.0, 179.0, 97.0),
                ("P4", 53.081, 17.0, 179.0, 127.0),
            )
        ]
        teaching = emi_map.engine_fixture_teaching.build_fixture_teaching(points)
        document = {"fixture_id": "Glassboard", "fixture_teaching": teaching}
        printer = _FakePrinter()
        wizard._printer = lambda **_kwargs: printer
        wizard._fixture_xyz_homed = True
        monkeypatch.setattr(wizard, "_fixture_profile_name", lambda: "Glassboard")
        monkeypatch.setattr(
            emi_map.engine_machine_fixtures, "read_fixture", lambda _name: document
        )
        monkeypatch.setattr(
            emi_map.engine_machine_fixtures,
            "save_verification",
            lambda *_args, **_kwargs: {
                **document,
                "fixture_teaching": {
                    **teaching,
                    "latest_verification": {"passed": True},
                },
            },
        )
        wizard._do_fixture_verify(0.20, 0.10)
        assert "G28 X Y" not in printer.sent
        assert [item for item in printer.sent if item.startswith("G0 ")] == [
            "G0 X79.0 Y128.0",
            "G0 X79.0 Y97.0",
            "G0 X179.0 Y97.0",
            "G0 X179.0 Y127.0",
        ]

    def test_fixture_verify_uses_the_limits_visible_to_the_operator(self, wizard):
        captured = {}
        wizard._machine_fixture_document = lambda: {"fixture_teaching": {"points": []}}
        wizard._start_printer_job = lambda work, *_args: captured.update(work=work)
        wizard.ui.chkFixtureProbeClear.setChecked(True)
        wizard.ui.fixtureMaxZChange.setValue(0.325)
        wizard.ui.fixtureMaxResidual.setValue(0.125)
        wizard._fixture_verify_clicked()
        assert captured["work"].args == pytest.approx((0.325, 0.125))

    def test_fixture_verification_pass_unlocks_insert_board(self, wizard):
        points = [
            emi_map.engine_fixture_teaching.FixtureReferencePoint(
                name, x, y, x, y, x, y, 10.0, 10.0
            )
            for name, x, y in (
                ("P1", 10.0, 10.0),
                ("P2", 90.0, 10.0),
                ("P3", 10.0, 50.0),
                ("P4", 90.0, 50.0),
            )
        ]
        teaching = emi_map.engine_fixture_teaching.build_fixture_teaching(points)
        result = emi_map.engine_fixture_teaching.compare_fixture_verification(
            teaching, points
        )
        teaching["latest_verification"] = result
        document = {"fixture_teaching": teaching}
        wizard._machine_fixture_document = lambda: document
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_FIXTURE_TEACH)
        wizard._on_fixture_verify_ok((document, result))
        assert wizard._fixture_verified_session is True
        assert wizard.ui.btnNext.isEnabled() is False
        assert "FIXTURE READY" not in wizard.ui.fixtureReadySummary.text()
        _agree_easy_scan_plane(wizard)
        wizard._update_nav()
        assert wizard.ui.btnNext.isEnabled() is True
        assert wizard.ui.btnNext.text() == "Next: insert board"
        assert wizard.ui.fixtureReadySummary.text().startswith("FIXTURE READY")
        assert wizard.ui.nextBlockReason.text() == ""

    def test_the_rectangle_size_boxes_follow_the_mode(self, wizard):
        assert not wizard.ui.widthMm.isEnabled()
        wizard.ui.scanMode.setCurrentText(emi_map.MODE_RECTANGLE)
        assert wizard.ui.widthMm.isEnabled()
        assert wizard.ui.btnNext.text() == "Next: set origin"
        wizard.ui.scanMode.setCurrentText(emi_map.MODE_PCB)
        assert not wizard.ui.widthMm.isEnabled()

    def test_next_walks_the_pcb_pages(self, wizard):
        wizard._on_next()
        assert wizard.ui.wizardStack.currentIndex() == emi_map.PAGE_FIXTURE_TEACH

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
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_BOARD)
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
        _agree_easy_scan_plane(wizard)
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
        _agree_easy_scan_plane(wizard)
        wizard.ui.chkTravelClearBoard.setChecked(True)
        config = wizard.build_config()
        wizard._prepare_live_plot(config, wizard._preflight(config))
        wizard.image.render()  # exactly what ImageItem.paint does

    def test_the_map_paints_with_only_one_reading_in_it(self, wizard):
        wizard._apply_board_view()
        register(wizard)
        _agree_easy_scan_plane(wizard)
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


class TestGlassboardAutoSetup:
    """STL point datums, P1-only origin, and the Step 4 preview gates."""

    @pytest.fixture
    def wizard(self, real_ui, monkeypatch):
        monkeypatch.setattr(emi_map, "QMessageBox", _FakeMessageBox)
        _FakeMessageBox.warnings = []
        made = emi_map.EMIMapWizard(real_ui, _FakeUsbInstr([_FakeDevice()]), None)
        made.start()
        return made

    def test_probed_point_stores_stl_xy_and_m119_datum(self, wizard, monkeypatch):
        saved = []
        monkeypatch.setattr(wizard, "_fixture_profile_name", lambda: "Glassboard")
        monkeypatch.setattr(
            emi_map.engine_machine_fixtures,
            "save_teaching",
            lambda name, teaching, overwrite=False: saved.append(
                (name, teaching, overwrite)
            ) or {"fixture_teaching": teaching},
        )
        printer = _FakePrinter(position=(10.0, 20.0))
        printer.z = 15.0
        wizard._printer = lambda **_kwargs: printer
        payload = wizard._do_fixture_probe_point("P1")
        point = payload["point"]
        assert (point.fixture_x, point.fixture_y) == (
            emi_map.engine_glassboard_fixture.REFERENCE_FIXTURE_XY["P1"]
        )
        assert (point.commanded_machine_x, point.commanded_machine_y) == pytest.approx(
            (10.0, 20.0)
        )
        assert point.z_datum == "m119_z_probe"
        assert point.touch_z_raw == pytest.approx(10.0)
        assert payload["board_zero"] is not None
        assert payload["board_zero"].board_zero_logical_z_mm == pytest.approx(10.0)
        assert "G30" not in printer.sent
        assert "M119" in printer.sent
        wizard._on_fixture_probe_ok(payload)
        assert wizard._bltouch_board_zero_z == pytest.approx(10.0)
        assert "Scan height set:" not in wizard.ui.fixtureTeachStatus.text()
        assert emi_map.BOARD_ZERO_COMPLETE_TEXT not in wizard.ui.fixtureTeachStatus.text()
        assert emi_map.EASY_HEIGHT_NEED_BOARD_ZERO in wizard.ui.fixtureTeachStatus.text()
        assert saved and saved[0][0] == "Glassboard" and saved[0][2] is True
        assert saved[0][1]["points"][0]["name"] == "P1"
        assert "next measurement" in wizard.ui.fixtureTeachStatus.text()

    def test_saved_p1_is_restored_when_the_fixture_is_selected(self, wizard):
        point = emi_map.engine_fixture_teaching.FixtureReferencePoint(
            "P1", -53.081, 17.0, 79.0, 128.0, 79.0, 128.0, 2.0, 2.0
        )
        teaching = emi_map.engine_fixture_teaching.build_fixture_teaching([point])
        wizard._fixture_teaching_points = {}
        wizard._render_fixture_teaching(teaching)
        assert wizard._p1_bltouch_commanded_xy() == pytest.approx((79.0, 128.0))
        assert wizard.ui.fixtureTeachTable.item(0, 1).text() == "79.000"

    def test_refresh_selects_a_fixture_when_the_combo_has_no_index(self, wizard, monkeypatch):
        point = emi_map.engine_fixture_teaching.FixtureReferencePoint(
            "P1", -53.081, 17.0, 79.0, 128.0, 79.0, 128.0, 2.0, 2.0
        )
        teaching = emi_map.engine_fixture_teaching.build_fixture_teaching([point])
        monkeypatch.setattr(
            emi_map.engine_machine_fixtures,
            "list_fixtures",
            lambda: [{"name": "Glassboard", "taught": True, "error": ""}],
        )
        monkeypatch.setattr(
            wizard,
            "_machine_fixture_document",
            lambda: {"fixture_name": "Glassboard", "fixture_teaching": teaching},
        )
        combo = wizard.ui.machineFixtureProfile
        combo.blockSignals(True)
        combo.clear()
        combo.blockSignals(False)
        wizard._active_machine_fixture_name = ""
        wizard._refresh_machine_fixtures()
        assert combo.currentText() == "Glassboard"
        assert wizard._p1_bltouch_commanded_xy() == pytest.approx((79.0, 128.0))

    def test_config_overlay_outranks_stored_fixture_calibrations(self, wizard):
        stored = {
            "fixture_id": "Glassboard",
            "pcb_thickness_mm": 9.9,
            "pcb_yaw_deg": 180.0,
            "e_probe_tip_z_minus_g30_contact_mm": 7.0,
            "emi_probe_offset_mm": {"x": 1.0, "y": 2.0},
        }
        wizard._fixture_config_cache = emi_map.engine_config.FixtureConfig(
            emi_probe_offset_x_mm=-104.0,
            emi_probe_offset_y_mm=1.0,
            pcb_thickness_mm=1.6,
            pcb_yaw_deg=0.0,
            e_probe_tip_z_minus_g30_contact_mm=-2.0,
        )
        overlaid = wizard._overlay_fixture_config(stored)
        assert stored["emi_probe_offset_mm"] == {"x": 1.0, "y": 2.0}
        assert overlaid["emi_probe_offset_mm"] == {"x": -104.0, "y": 1.0}
        assert overlaid["pcb_thickness_mm"] == pytest.approx(1.6)
        assert overlaid["pcb_yaw_deg"] == pytest.approx(0.0)
        assert overlaid["e_probe_tip_z_minus_g30_contact_mm"] == pytest.approx(-2.0)

    def test_blank_yaml_does_not_erase_stored_fixture_calibrations(self, wizard):
        stored = {
            "fixture_id": "Glassboard",
            "pcb_thickness_mm": 1.6,
            "e_probe_tip_z_minus_g30_contact_mm": 0.0,
        }
        wizard._fixture_config_cache = emi_map.engine_config.FixtureConfig(
            emi_probe_offset_x_mm=-104.0,
            emi_probe_offset_y_mm=1.0,
            pcb_thickness_mm=None,
            pcb_yaw_deg=0.0,
            e_probe_tip_z_minus_g30_contact_mm=None,
        )
        wizard._fixture_config_error = ""
        overlaid = wizard._overlay_fixture_config(stored)
        assert overlaid["pcb_thickness_mm"] == pytest.approx(1.6)
        assert overlaid["e_probe_tip_z_minus_g30_contact_mm"] == pytest.approx(0.0)

    def test_invalid_yaml_does_not_fall_back_to_stored_cal(self, wizard):
        stored = {
            "fixture_id": "Glassboard",
            "pcb_thickness_mm": 1.6,
            "e_probe_tip_z_minus_g30_contact_mm": 0.0,
        }
        wizard._fixture_config_cache = emi_map.engine_config.FixtureConfig()
        wizard._fixture_config_error = "fixture pcb_thickness_mm must be finite and >= 0"
        overlaid = wizard._overlay_fixture_config(stored)
        assert overlaid["pcb_thickness_mm"] is None
        assert overlaid["e_probe_tip_z_minus_g30_contact_mm"] is None
        assert "invalid" in wizard._easy_height_block_reason()

    def test_clear_p1_forgets_in_memory_point_and_board_zero(self, wizard):
        wizard._fixture_teaching_points["P1"] = (
            emi_map.engine_fixture_teaching.FixtureReferencePoint(
                "P1", -53.081, 17.0, 79.0, 128.0, 79.0, 128.0, 2.0, 2.0
            )
        )
        wizard._bltouch_board_zero_z = 12.0
        wizard._bltouch_board_zero_frame = 1
        wizard._z_logical_frame = 1
        wizard._pcb_surface_z = 10.0
        emi_map.QMessageBox.answer = emi_map.QMessageBox.Yes
        try:
            wizard._fixture_clear_point_clicked()
            assert wizard._fixture_teaching_points == {}
            assert wizard._bltouch_board_zero_z is None
            assert wizard._pcb_surface_z is None
            assert wizard._p1_bltouch_commanded_xy() is None
            assert "P1 cleared" in wizard.ui.fixtureTeachStatus.text()
        finally:
            emi_map.QMessageBox.answer = emi_map.QMessageBox.No

    def test_approve_and_play_send_zero_printer_commands(self, wizard):
        printer = _FakePrinter()
        wizard._printer = lambda **_kwargs: printer
        wizard.printer = printer
        widget = wizard._scan_preview_widget
        assert widget is not None
        assert widget.btn_approve.isEnabled() is False
        printer.sent.clear()
        widget.play_simulation()
        widget.stop_simulation()
        widget.approve_plan()
        assert printer.sent == []
        assert "not collision detection" in widget.notice.text().lower()

    def test_scan_preview_uses_plotwidget_not_opengl(self, wizard):
        import sys

        widget = wizard._scan_preview_widget
        assert widget is not None
        assert isinstance(widget._plot, pyqtgraph.PlotWidget)
        assert "pyqtgraph.opengl" not in sys.modules

    def test_invalidating_the_plan_clears_approval(self, wizard):
        wizard._approved_scan_fingerprint = "stale-fingerprint"
        wizard._invalidate_plan()
        assert wizard._approved_scan_fingerprint is None
        assert wizard._plan is None

    def test_step_four_does_not_repeat_probe_height(self, wizard):
        wizard._refresh_calculated_height()
        assert not hasattr(wizard.ui, "scanSetupHeightStatus")
        page = wizard.ui.pageEasyScanSetup
        assert page.findChild(QtWidgets.QGroupBox, "scanSetupHeightStatus") is None
        assert "not calibrated" not in page.objectName().lower()

    def test_incomplete_step_2_never_says_fixture_ready(self, wizard):
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_FIXTURE_TEACH)
        wizard._update_nav()
        assert wizard.ui.btnNext.isEnabled() is False
        summary = wizard.ui.fixtureReadySummary.text()
        assert "FIXTURE READY" not in summary
        assert summary.startswith("Complete Step 2 — ")
        reason = wizard._fixture_step2_block_reason()
        assert reason
        assert reason in summary
        assert reason in wizard.ui.btnNext.toolTip()
        assert reason in wizard.ui.nextBlockReason.text()
        assert "calibration saved" not in summary

    def test_full_height_setup_enables_next_and_says_ready(self, wizard):
        wizard.ui.wizardStack.setCurrentIndex(emi_map.PAGE_FIXTURE_TEACH)
        _agree_easy_scan_plane(wizard)
        wizard._update_nav()
        assert wizard._easy_height_ready() is True
        assert wizard.ui.btnNext.isEnabled() is True
        assert wizard.ui.fixtureReadySummary.text().startswith("FIXTURE READY")
        assert (
            emi_map.BOARD_ZERO_COMPLETE_TEXT in wizard.ui.fixtureReadySummary.text()
            or "Scan height set:" in wizard.ui.fixtureReadySummary.text()
        )
        assert wizard.ui.nextBlockReason.text() == ""

    def test_saved_p1_is_loaded_by_a_new_session(self, real_ui, monkeypatch, tmp_path):
        root = tmp_path / "machine_fixtures"
        monkeypatch.setattr(emi_map.engine_machine_fixtures, "fixtures_dir", lambda: root)
        monkeypatch.setattr(emi_map, "QMessageBox", _FakeMessageBox)
        _FakeMessageBox.warnings = []
        emi_map.engine_machine_fixtures.create_fixture("Glassboard")
        emi_map.engine_machine_fixtures.save_fixture_calibrations(
            "Glassboard",
            pcb_thickness_mm=0.746,
        )
        first = emi_map.EMIMapWizard(real_ui, _FakeUsbInstr([_FakeDevice()]), None)
        first.start()
        first.ui.wizardStack.setCurrentIndex(emi_map.PAGE_FIXTURE_TEACH)
        printer = _FakePrinter(position=(80.0, 125.0))
        printer.z = 15.0
        first._printer = lambda **_kwargs: printer
        first._fixture_xyz_homed = True
        first.ui.chkFixtureProbeClear.setChecked(True)
        payload = first._do_fixture_probe_point("P1")
        first._on_fixture_probe_ok(payload)
        assert first._fixture_save_plane() is True
        stored = emi_map.engine_machine_fixtures.read_fixture("Glassboard")
        points = stored["fixture_teaching"]["points"]
        assert points[0]["name"] == "P1"
        assert points[0]["commanded_machine_x"] == pytest.approx(80.0)
        assert points[0]["commanded_machine_y"] == pytest.approx(125.0)
        assert stored.get("e_probe_tip_z_minus_g30_contact_mm") in (None, "")
        assert stored.get("pcb_thickness_mm") == pytest.approx(0.746)
        assert "PASS — P1 saved on this fixture" in first.ui.fixtureTeachStatus.text()
        assert "P1 XY saved (80.00, 125.00)" in first.ui.fixtureBoardSummary.text()
        assert first._bltouch_board_zero_z == pytest.approx(10.0)
        first.ui.hide()

        handle = QFile(str(Path(emi_map.__file__).with_name("emi_map.ui")))
        handle.open(QFile.ReadOnly)
        try:
            second_ui = _UiLoader().load(handle)
        finally:
            handle.close()
        second = emi_map.EMIMapWizard(second_ui, _FakeUsbInstr([_FakeDevice()]), None)
        second._fixture_config_cache = emi_map.engine_config.FixtureConfig(
            emi_probe_offset_x_mm=-104.0,
            emi_probe_offset_y_mm=1.0,
            pcb_thickness_mm=0.746,
            pcb_yaw_deg=0.0,
            e_probe_tip_z_minus_g30_contact_mm=None,
        )
        second._fixture_config_error = ""
        second._render_emi_probe_offset()
        second.start()
        second.ui.wizardStack.setCurrentIndex(emi_map.PAGE_FIXTURE_TEACH)
        second._update_nav()
        assert second._fixture_profile_name() == "Glassboard"
        assert second._p1_bltouch_commanded_xy() == pytest.approx((80.0, 125.0))
        assert second.ui.btnFixtureHomeXyz.text() == "HOME & SET BOARD ZERO"
        assert second._bltouch_board_zero_z is None
        assert second._easy_verified_scan_z is None
        assert second._easy_height_ready() is False
        assert second.ui.btnNext.isEnabled() is False
        summary = second.ui.fixtureReadySummary.text()
        assert "FIXTURE READY" not in summary
        assert emi_map.EASY_HEIGHT_NEED_BOARD_ZERO in summary
        assert emi_map.EASY_HEIGHT_NEED_BOARD_ZERO in second.ui.nextBlockReason.text()
        assert "P1 XY saved (80.00, 125.00)" in second.ui.fixtureBoardSummary.text()
        note = second.ui.fixtureConfigNote.text()
        assert "PCB thickness 0.746 mm (config.yaml override)" in note
        assert emi_map.EASY_VERTICAL_UNCALIBRATED in note
        assert "Vertical E-probe calibration 0.000 mm" not in note
        assert "is missing from config.yaml and the saved fixture" not in note
        assert "calibration saved" not in summary

        printer2 = _FakePrinter(position=(80.0, 125.0))
        printer2.z = 15.0
        second._printer = lambda **_kwargs: printer2
        second.ui.chkFixtureProbeClear.setChecked(True)
        second._home_xy()
        assert second._p1_bltouch_commanded_xy() == pytest.approx((80.0, 125.0))
        assert second._bltouch_board_zero_z is not None
        assert second._easy_verified_scan_z is None
        cals = second._easy_resolved_z_calibrations()
        assert cals["pcb_thickness_mm"] == pytest.approx(0.746)
        assert cals["e_probe_tip_z_minus_g30_contact_mm"] is None
        assert second._easy_height_ready() is False
        assert second.ui.btnNext.isEnabled() is False
        assert second._easy_height_block_reason() == emi_map.EASY_HEIGHT_NEED_MANUAL_PLANE
        assert emi_map.EASY_HEIGHT_NEED_MANUAL_PLANE in second.ui.fixtureReadySummary.text()
        assert second.ui.btnNext.toolTip() == emi_map.EASY_HEIGHT_NEED_MANUAL_PLANE
        assert second.ui.nextBlockReason.text() == emi_map.EASY_HEIGHT_NEED_MANUAL_PLANE
        assert "FIXTURE READY" not in second.ui.fixtureReadySummary.text()
        assert emi_map.EASY_VERTICAL_UNCALIBRATED in second.ui.fixtureConfigNote.text()
        _FakeMessageBox.answer = _FakeMessageBox.Yes
        assert second._easy_set_pcb_surface_manually() is True
        assert second._p1_bltouch_commanded_xy() == pytest.approx((80.0, 125.0))
        assert second._easy_scan_plane_cached() is False
        assert second._easy_set_probe_height() is True
        second._update_nav()
        assert second._easy_scan_plane_cached() is True
        assert second._easy_height_ready() is True
        assert second.ui.btnNext.isEnabled() is True
        assert second.ui.btnNext.text() == "Next: insert board"
        assert second.ui.btnNext.toolTip() == ""
        assert second.ui.fixtureReadySummary.text().startswith("FIXTURE READY")
        assert "Scan height set:" in second.ui.fixtureReadySummary.text()
        assert second.ui.nextBlockReason.text() == ""
        second.ui.hide()


