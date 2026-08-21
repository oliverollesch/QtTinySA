import csv
import itertools

import numpy as np

from PySide6.QtWidgets import QMessageBox

from pyqtgraph import ErrorBarItem, PlotItem
from pyqtgraph.exporters import CSVExporter
from pyqtgraph.parametertree import Parameter


class QtTinySAExportException(Exception):
    """Custom exception for custom pyqtgraph exporters."""


class _TraceCSVExporter(CSVExporter):
    """Shared collection and error handling for per-trace CSV exporters."""

    def __init__(self, item):
        CSVExporter.__init__(self, item)
        self.params = Parameter.create(name='params', type='group', children=[
            {'name': 'trace', 'title': 'Export trace', 'type': 'list',
             'value': 1, 'limits': [1, 2, 3, 4]}
        ])
        self.index_counter = itertools.count(start=0)
        self.header = []
        self.data = []

    def _exportPlotDataItem(self, plotDataItem) -> None:
        """Export selected trace data points to class data variable."""
        if hasattr(plotDataItem, 'getOriginalDataset'):
            cd = plotDataItem.getOriginalDataset()
        else:
            cd = plotDataItem.getData()
        if cd[0] is None:
            return None
        self.data.append(cd)
        return None

    def _trace_xy_indices(self):
        t = int(self.params['trace'])
        row_x = (t - 1) * 2
        return row_x, row_x + 1

    def _collect_columns(self):
        for item in self.item.items:
            if isinstance(item, ErrorBarItem):
                self._exportErrorBarItem(item)
            elif hasattr(item, 'implements') and item.implements('plotData'):
                self._exportPlotDataItem(item)
        columns = [column for dataset in self.data for column in dataset]
        if len(columns) == 0:
            raise QtTinySAExportException(
                'No column data to export / empty graph!')
        if len(columns) <= self.params['trace'] * 2:
            raise QtTinySAExportException(
                "Missing column data for selected trace - can't export!")
        return columns

    def _report_error(self, err):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setText(str(err))
        error_dialog.setWindowTitle('Error')
        error_dialog.exec()

    def _write_rows(self, fileName, columns):
        raise NotImplementedError

    def export(self, fileName=None):
        """Export handler to prepare and export the data to a CSV file."""
        if not isinstance(self.item, PlotItem):
            raise TypeError("Must have a PlotItem selected for CSV export.")
        if fileName is None:
            self.fileSaveDialog(filter=["*.csv", "*.tsv"])
            return
        try:
            columns = self._collect_columns()
            self._write_rows(fileName, columns)
        except QtTinySAExportException as e:
            self._report_error(e)
        self.header.clear()
        self.data.clear()


class WWBExporter(_TraceCSVExporter):
    """Shure Wireless Workbench CSV exporter class."""

    Name = "CSV for WWB (Shure)"

    def _write_rows(self, fileName, columns):
        row_x, row_y = self._trace_xy_indices()
        with open(fileName, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for row in itertools.zip_longest(*columns, fillvalue=""):
                if isinstance(row[row_x], str):
                    x = row[row_x]
                else:
                    x = np.format_float_positional(int(row[row_x]) / 1000000,
                                                   precision=3,
                                                   unique=True,
                                                   min_digits=3,
                                                   fractional=True)
                if isinstance(row[row_y], str):
                    y = row[row_y]
                else:
                    y = np.format_float_positional(int(row[row_y]),
                                                   precision=4,
                                                   unique=True,
                                                   min_digits=4,
                                                   fractional=True)
                writer.writerow([x, y])


class WSMExporter(_TraceCSVExporter):
    """Sennheiser WSM CSV exporter class."""
    Name = "CSV for WSM (Sennheiser)"

    def _write_rows(self, fileName, columns):
        row_x, row_y = self._trace_xy_indices()
        self.header = [["Receiver", ''],
                       ["Date/Time", ''],
                       ["RFUnit", "dBm"],
                       ["Owner", ''],
                       ["ScanCity", ''],
                       ["ScanComment", ''],
                       ["ScanCountry", ''],
                       ["ScanDescription", ''],
                       ["ScanInteriorExterior", ''],
                       ["ScanLatitude", ''],
                       ["ScanLongitude", ''],
                       ["ScanName", ''],
                       ["ScanPostalCode", ''],
                       ["Frequency Range [kHz]",
                           f"{int(columns[row_x][0] / 1000)}",
                           f"{int(columns[row_x][-1] / 1000)}",
                           f"{len(columns[row_x])}"],
                       ["Frequency", "RF level (%)", "RF level"]]
        with open(fileName, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(self.header)
            for row in itertools.zip_longest(*columns, fillvalue=""):
                if isinstance(row[row_x], str):
                    x = row[row_x]
                else:
                    x = int(int(row[row_x]) / 1000)
                if isinstance(row[row_y], str):
                    y = row[row_y]
                else:
                    y = np.format_float_positional(float(row[row_y]), precision=5)
                writer.writerow([x, '', y])
