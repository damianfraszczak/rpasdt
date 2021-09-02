import typing
from typing import Any

from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from rpasdt.common.utils import get_object_value, set_object_value
from rpasdt.gui.dynamic_form.models import DynamicFormConfig, FormFieldConfig
from rpasdt.gui.dynamic_form.utils import (
    get_component_for_field_config,
    get_component_value,
    get_field_config,
    get_type_representation_for_field_config,
    set_component_value,
)
from rpasdt.model.constants import APP_ICON_PATH


def _get_help_icon(text: str):
    label = QLabel(text)
    label.setFont(QFont("Times", 8))
    label.setStyleSheet("color: grey;font-style: italic;")
    return label


class DynamicForm(QWidget):
    def __init__(
        self,
        object: Any,
        field_config: typing.Dict[str, FormFieldConfig] = None,
        parent: typing.Optional["QWidget"] = None,
    ) -> None:
        super().__init__(parent)
        self.object: Any = object
        self.field_config: dict[str, FormFieldConfig] = {
            **get_field_config(object),
            **(field_config or {}),
        }
        self.field_row_map: dict = {}
        self.field_component_map: dict = {}
        self.setLayout(QFormLayout())
        self.init_fields()
        self.copy_data_to_fields()

    def init_fields(self):
        row_index = 0
        for field_name, field_config in self.field_config.items():
            field_config.type_representation = get_type_representation_for_field_config(
                field_config
            )
            if not field_config.type_representation:
                continue
            component = get_component_for_field_config(field_config)

            label = field_config.label or field_name
            if component and label:
                self.layout().addRow(label, component)
                self.field_component_map[field_name] = component
                self.field_row_map[field_name] = row_index
                row_index += 1
                if field_config.help_text:
                    self.layout().addRow(_get_help_icon(field_config.help_text))

    def copy_data_to_fields(self):
        for field_name, component in self.field_component_map.items():
            value = (
                get_object_value(self.object, field_name)
                or self.field_config[field_name].default_value
            )
            set_component_value(
                component=component,
                value=value,
                options=self.field_config[field_name].options,
            )

    def copy_fields_to_data(self):
        for field_name, component in self.field_component_map.items():
            value = get_component_value(
                component=component,
                type_representation=self.field_config[field_name].type_representation,
                options=self.field_config[field_name].options,
            )
            set_object_value(self.object, field_name, value)


class DynamicDialog(QDialog):
    def __init__(
        self,
        object: Any,
        config: typing.Optional[DynamicFormConfig] = None,
        parent: typing.Optional["QWidget"] = None,
    ) -> None:
        super().__init__(parent)
        self.config = config or DynamicFormConfig()
        # creating a dialog button for ok and cancel
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        # adding action when form is accepted
        self.buttonBox.accepted.connect(self.save)

        # adding action when form is rejected
        self.buttonBox.rejected.connect(self.cancel)

        # creating a vertical layout
        mainLayout = QVBoxLayout()
        self.form = DynamicForm(
            parent=parent, object=object, field_config=self.config.field_config
        )

        # adding form group box to the layout
        mainLayout.addWidget(self.form)

        # adding button box to the layout
        mainLayout.addWidget(self.buttonBox)

        # setting lay out
        self.setLayout(mainLayout)
        self.setModal(True)
        self.setWindowTitle(self.config.title)
        self.setWindowIcon(QIcon(APP_ICON_PATH))
        self.setMinimumWidth(600)

    @property
    def object(self):
        return self.form.object

    def save(self):
        self.form.copy_fields_to_data()
        self.accept()

    def cancel(self):
        self.reject()
