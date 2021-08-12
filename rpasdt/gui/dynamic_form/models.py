from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from rpasdt.common.enums import StringChoiceEnum


class FieldInputType(StringChoiceEnum):
    SINGLE_TEXT = "SINGLE_TEXT"
    COMBOBOX = "COMBOBOX"
    CHECKBOX = "CHECKBOX"
    TOGGLE = "TOGGLE"
    DATE = "DATE"
    DOUBLE = "DOUBLE"
    INTEGER = "INTEGER"
    PASSWORD = "PASSWORD"
    COLOR = "COLOR"
    FILE = "FILE"


@dataclass
class FormFieldConfig:
    field_name: str
    label: str = ""
    help_text: str = ""
    type: FieldInputType = FieldInputType.SINGLE_TEXT
    default_value: Any = None
    options: Optional[List[Tuple]] = None
    visible: Union[bool, Callable] = None
    read_only: bool = False
    range: List = None
    inner_type: Type = None


@dataclass
class DynamicFormConfig:
    title: str = ""
    field_config: Optional[Dict[str, FormFieldConfig]] = None
    read_only_fields: Optional[List[str]] = None
