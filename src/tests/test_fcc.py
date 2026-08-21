"""
Headless checks for the hardware-independent parts of the FCC wizard.

Run from the src directory so that `modules` is importable:
    python -m pytest tests -q
"""

import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fcc_test import (  # noqa: E402
    FCC_CLASS_B,
    compute_emc_schwarzbeck,
    compute_emc_tekbox,
    parse_float_lines,
    sanitize_label,
)


class TestParseFloatLines:
    def test_extracts_one_value_per_line(self):
        assert parse_float_lines("-12.5\n-30.25\n-99\n") == [-12.5, -30.25, -99.0]

    def test_rewrites_the_firmware_underflow_sentinel(self):
        # tinySA emits '-:.0' instead of '-10.0' when the level underflows
        assert parse_float_lines("-:.0\n-20.0\n") == [-10.0, -20.0]

    def test_skips_blank_and_unparseable_lines(self):
        assert parse_float_lines("\n  \nch>\n-42.0\n") == [-42.0]

    def test_accepts_scientific_notation(self):
        assert parse_float_lines("1.5e8\n") == [1.5e8]

    def test_empty_input_gives_empty_list(self):
        assert parse_float_lines("") == []


class TestSanitizeLabel:
    def test_keeps_safe_characters(self):
        assert sanitize_label("Run-1_v2.0") == "Run-1_v2.0"

    def test_replaces_path_separators(self):
        assert sanitize_label("a/b\\c") == "a_b_c"

    def test_falls_back_when_nothing_survives(self):
        assert sanitize_label("///") == "unnamed"
        assert sanitize_label("") == "unnamed"


class TestComputeEmcTekbox:
    def test_scalar_matches_the_closed_form(self):
        # E is returned in uV/m, so dBuV/m carries the +120 dB offset:
        # dBuV/m = 120 + (dBm - gain) + 112.5 - 20*log10(f_MHz)
        actual, e_field, dbuv = compute_emc_tekbox(100e6, -40.0, amp_gain_db=40.0)
        assert actual == pytest.approx(-80.0)
        assert dbuv == pytest.approx(120.0 - 80.0 + 112.5 - 20 * np.log10(100.0))
        assert dbuv == pytest.approx(112.5)
        assert e_field == pytest.approx(10 ** (dbuv / 20.0))

    def test_scalar_input_returns_floats(self):
        result = compute_emc_tekbox(100e6, -40.0)
        assert all(isinstance(v, float) for v in result)

    def test_array_input_is_elementwise(self):
        freqs = np.array([50e6, 100e6, 200e6])
        dbm = np.array([-40.0, -40.0, -40.0])
        _, _, dbuv = compute_emc_tekbox(freqs, dbm, amp_gain_db=40.0)
        # Halving frequency raises the field by 20*log10(2) dB at constant power
        assert dbuv[0] - dbuv[1] == pytest.approx(20 * np.log10(2))
        assert dbuv[1] - dbuv[2] == pytest.approx(20 * np.log10(2))

    def test_gain_subtracts_directly(self):
        _, _, high = compute_emc_tekbox(100e6, -40.0, amp_gain_db=20.0)
        _, _, low = compute_emc_tekbox(100e6, -40.0, amp_gain_db=40.0)
        assert high - low == pytest.approx(20.0)

    def test_zero_frequency_is_nan_not_an_exception(self):
        _, e_field, dbuv = compute_emc_tekbox(np.array([0.0]), np.array([-40.0]))
        assert np.isnan(e_field[0])
        assert np.isnan(dbuv[0])


class TestComputeEmcSchwarzbeck:
    def test_scalar_matches_the_closed_form(self):
        # E(dBuV/m) = dBm + 107 + AF, independent of frequency
        actual, e_field, dbuv = compute_emc_schwarzbeck(100e6, -40.0, 20.0)
        assert actual == pytest.approx(-40.0)
        assert dbuv == pytest.approx(87.0)
        assert e_field == pytest.approx(10 ** ((87.0 - 120.0) / 20.0) * 1e6)

    def test_frequency_argument_is_ignored(self):
        _, _, at_100 = compute_emc_schwarzbeck(100e6, -40.0, 20.0)
        _, _, at_900 = compute_emc_schwarzbeck(900e6, -40.0, 20.0)
        assert at_100 == pytest.approx(at_900)

    def test_antenna_factor_adds_directly(self):
        _, _, low = compute_emc_schwarzbeck(100e6, -40.0, 20.0)
        _, _, high = compute_emc_schwarzbeck(100e6, -40.0, 47.0)
        assert high - low == pytest.approx(27.0)

    def test_array_input_is_elementwise(self):
        _, _, dbuv = compute_emc_schwarzbeck(100e6, np.array([-40.0, -30.0]), 20.0)
        assert dbuv == pytest.approx([87.0, 97.0])


class TestFccClassBLimits:
    def test_bands_are_contiguous_and_ascending(self):
        for (_, prev_stop, _), (next_start, _, _) in zip(FCC_CLASS_B, FCC_CLASS_B[1:]):
            assert prev_stop == next_start

    def test_covers_30_mhz_to_1_ghz(self):
        assert FCC_CLASS_B[0][0] == 30.0
        assert FCC_CLASS_B[-1][1] == 1000.0

    def test_limits_are_the_published_values(self):
        assert [lim for _, _, lim in FCC_CLASS_B] == [40.0, 43.5, 46.0, 54.0]


# --------------------------------------------------------------------------
# Exporters. These need a QApplication because the shared base class builds a
# Parameter tree and can raise QMessageBox, so skip cleanly if Qt is headless.

pyqtgraph = pytest.importorskip("pyqtgraph")


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        try:
            app = QtWidgets.QApplication([])
        except Exception as e:  # no display available
            pytest.skip(f"Qt unavailable: {e}")
    return app


@pytest.fixture
def plot_item(qapp):
    """
    A PlotItem holding two 5-point traces.

    Two are needed because the exporters reject a plot with no more than
    `trace * 2` columns, so a single trace fails the default trace=1 check.
    """
    item = pyqtgraph.PlotItem()
    freqs = np.arange(100_000_000, 100_000_005, dtype=float)
    item.plot(freqs, np.array([-40.0, -41.0, -42.0, -43.0, -44.0]))
    item.plot(freqs, np.array([-50.0, -51.0, -52.0, -53.0, -54.0]))
    return item


class TestExporters:
    def test_wwb_writes_mhz_and_level_columns(self, plot_item, tmp_path):
        from modules.exporters import WWBExporter

        out = tmp_path / "wwb.csv"
        WWBExporter(plot_item).export(str(out))
        rows = list(csv.reader(out.open(newline="", encoding="utf-8")))
        assert len(rows) == 5
        # x converted Hz -> MHz with 3 decimals, y is the dBm level
        assert rows[0][0] == "100.000"
        assert float(rows[0][1]) == pytest.approx(-40.0)

    def test_wsm_writes_header_then_semicolon_rows(self, plot_item, tmp_path):
        from modules.exporters import WSMExporter

        out = tmp_path / "wsm.csv"
        WSMExporter(plot_item).export(str(out))
        rows = list(csv.reader(out.open(newline="", encoding="utf-8"), delimiter=";"))
        assert rows[0][0] == "Receiver"
        assert rows[13][0] == "Frequency Range [kHz]"
        assert rows[13][3] == "5"  # point count
        assert rows[14] == ["Frequency", "RF level (%)", "RF level"]
        # x converted Hz -> kHz
        assert rows[15][0] == "100000"

    def test_both_exporters_share_the_base_class(self):
        from modules.exporters import WSMExporter, WWBExporter, _TraceCSVExporter

        assert issubclass(WWBExporter, _TraceCSVExporter)
        assert issubclass(WSMExporter, _TraceCSVExporter)

    def test_empty_plot_raises_export_exception(self, qapp):
        from modules.exporters import QtTinySAExportException, WWBExporter

        exporter = WWBExporter(pyqtgraph.PlotItem())
        with pytest.raises(QtTinySAExportException):
            exporter._collect_columns()
