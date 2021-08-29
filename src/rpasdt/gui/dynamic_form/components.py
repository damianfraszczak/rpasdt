import typing

from PyQt5.QtWidgets import QColorDialog, QFileDialog, QPushButton, QWidget


class QColorField(QPushButton):
    DEFAULT_COLOR = "#000000"

    def __init__(
        self,
        color: typing.Optional[str] = DEFAULT_COLOR,
        parent: typing.Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.color = color

        self.clicked.connect(self.show_picker)

    def show_picker(self):
        self.color = QColorDialog.getColor().name()

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value or self.DEFAULT_COLOR
        self.setStyleSheet(f"background-color: {self._color}")


class QFileField(QPushButton):
    def __init__(
        self,
        file_path: typing.Optional[str] = None,
        parent: typing.Optional[QWidget] = None,
        open_file_mode: bool = True,
    ) -> None:
        super().__init__(parent)
        self.file_path = file_path
        self.clicked.connect(self.show_picker)
        self.open_file_mode = open_file_mode

    def show_picker(self):
        result = (
            QFileDialog.getOpenFileName()
            if self.open_file_mode
            else QFileDialog.getSaveFileName()
        )
        self.file_path = result[0] if result else ""

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, value):
        self._file_path = value
        self.setText(self._file_path)
