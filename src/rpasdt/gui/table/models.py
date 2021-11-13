import typing
from datetime import datetime

from PyQt5 import QtCore
from PyQt5.QtCore import Qt


class ListTableModel(QtCore.QAbstractTableModel):
    def __init__(
        self, data: typing.List[typing.List[typing.Any]], columns: typing.List[str]
    ):
        super(ListTableModel, self).__init__()
        self._data = data
        self._columns = columns

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # Get the raw value
            value = self._data[index.row()][index.column()]

            # Perform per-type checks and render accordingly.
            if isinstance(value, datetime):
                # Render time to YYY-MM-DD.
                return value.strftime("%Y-%m-%d")

            if isinstance(value, float):
                # Render float to 2 dp
                return "%.2f" % value

            # Default (anything not captured above: e.g. int, str)
            return value

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._columns)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = ...
    ) -> typing.Any:
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._columns[section]
        return super(ListTableModel, self).headerData(section, orientation, role)
