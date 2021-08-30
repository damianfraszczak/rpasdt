import logging
from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, get_type_hints

from PyQt5.QtCore import QItemSelectionModel, QModelIndex
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QListWidget,
    QSpinBox,
    QWidget,
)

from rpasdt.common.enums import StringChoiceEnum
from rpasdt.gui.dynamic_form.components import QColorField, QFileField
from rpasdt.gui.dynamic_form.models import FieldInputType, FormFieldConfig

logger = logging.getLogger(__name__)


def get_multi_select(options: List[Tuple]):
    if not options:
        return None
    list = QListWidget()
    list.setSelectionMode(QAbstractItemView.MultiSelection)
    for _, label in options:
        list.addItem(label)
    return list


def get_combo_box(options: List[Tuple]):
    if not options:
        return None
    combo = QComboBox()
    setattr(combo, "options", options)
    for key, label in options:
        combo.addItem(label, key)
    return combo


def get_int_field(range: Optional[Tuple]):
    box = QSpinBox()
    if range:
        box.setRange(int(range[0]), int(range[1]))
    else:
        box.setRange(0, 1000000)
    return box


def get_float_field(range: Optional[Tuple]):
    box = QDoubleSpinBox()
    if range:
        box.setRange(float(range[0]), float(range[1]))
    else:
        box.setRange(0, 1000000)
    return box


def get_component_type(cls: type):
    if issubclass(cls, Enum):
        return FieldInputType.COMBOBOX
    elif issubclass(cls, bool):
        return FieldInputType.CHECKBOX
    elif issubclass(cls, int):
        return FieldInputType.INTEGER
    elif issubclass(cls, float):
        return FieldInputType.DOUBLE
    elif issubclass(cls, str):
        return FieldInputType.SINGLE_TEXT


def set_component_value(
    component: QWidget, value: Any, options: Optional[List[Tuple]] = None
):
    if isinstance(component, QLineEdit):
        component.setText(value or "")
    elif isinstance(component, QCheckBox):
        component.setChecked(bool(value or False))
    elif isinstance(component, QSpinBox):
        component.setValue(int(value or 0))
    elif isinstance(component, QDoubleSpinBox):
        component.setValue(float(value or 0.0))
    elif isinstance(component, QComboBox) and options:
        component.setCurrentText(dict(options)[value])
    elif isinstance(component, QColorField):
        component.color = value
    elif isinstance(component, QFileField):
        component.file_path = value
    elif isinstance(component, QListWidget):
        options_index = {option[0]: index for index, option in enumerate(options)}
        for single_value in value:
            list_index = component.model().index(
                options_index[single_value], 0, QModelIndex()
            )
            component.selectionModel().setCurrentIndex(
                list_index, QItemSelectionModel.Select
            )


def get_component_value(
    component: QWidget, cls: type, options: Optional[List[Tuple]] = None
):
    if isinstance(component, QLineEdit):
        return component.text()
    elif isinstance(component, QCheckBox):
        return component.isChecked()
    elif isinstance(component, QComboBox) and options:
        return get_option_value(
            cls=cls, option_value=options[component.currentIndex()][0]
        )
    elif isinstance(component, QSpinBox):
        return component.value()
    elif isinstance(component, QDoubleSpinBox):
        return component.value()
    elif isinstance(component, QColorField):
        return component.color
    elif isinstance(component, QFileField):
        return component.file_path
    elif isinstance(component, QListWidget):
        return [options[index.row()][0] for index in component.selectedIndexes()]
    return None


def get_field_options(cls: type) -> List[Tuple]:
    if issubclass(cls, StringChoiceEnum):
        return cls.choices


def get_option_value(cls: type, option_value: Any) -> Any:
    if isinstance(cls, type) and issubclass(cls, StringChoiceEnum):
        return cls(option_value)
    return option_value


def get_component_for_field_config(field_config: FormFieldConfig) -> Optional[QWidget]:
    widget = None
    if FieldInputType.COMBOBOX == field_config.type:
        widget = get_combo_box(field_config.options)
        widget = get_multi_select(field_config.options)
    elif FieldInputType.CHECKBOX == field_config.type:
        widget = QCheckBox()
    elif FieldInputType.INTEGER == field_config.type:
        widget = get_int_field(field_config.range)
    elif FieldInputType.DOUBLE == field_config.type:
        widget = get_float_field(field_config.range)
    elif FieldInputType.SINGLE_TEXT == field_config.type:
        widget = QLineEdit()
    elif FieldInputType.COLOR == field_config.type:
        widget = QColorField()
    elif FieldInputType.FILE == field_config.type:
        widget = QFileField()
    if widget and field_config.read_only:
        read_only_method = getattr(widget, "setReadOnly", None)
        if read_only_method:
            read_only_method(field_config.read_only)
        else:
            logger.warning(
                f"Component for field {field_config.field_name} does not support read only option."
            )
    return widget


def format_field_label(val: str) -> str:
    return (val or "").replace("_", " ").title()


def clear_type_hint(type_hint):
    if hasattr(type_hint, "__origin__") and getattr(type_hint, "__origin__") is Union:
        return type_hint.__args__[0]
    return type_hint if isinstance(type_hint, type) else None


def get_field_config(object: Any) -> Dict[str, FormFieldConfig]:
    result: Dict[str, FormFieldConfig] = OrderedDict()
    cls = type(object)
    type_hints = get_type_hints(cls)
    for field_name, inner_type in type_hints.items():
        inner_type = clear_type_hint(inner_type)
        if inner_type:
            result[field_name] = FormFieldConfig(
                field_name=field_name,
                label=format_field_label(field_name),
                type=get_component_type(inner_type),
                default_value=getattr(object, field_name, None),
                options=get_field_options(inner_type),
                inner_type=inner_type,
            )

    return result
