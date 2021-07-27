import typing

from PyQt5.QtWidgets import QColorDialog, QPushButton, QWidget


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
