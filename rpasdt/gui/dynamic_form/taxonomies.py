from typing import List, Tuple, Dict, Union, Callable, Optional

from dataclasses import dataclass

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


@dataclass
class FormFieldConfig:
    field_name: str
    label: str = ''
    help_text: str = ''
    type: FieldInputType = FieldInputType.SINGLE_TEXT
    default_value: any = None
    options: List[Tuple] = None
    visible: Union[bool, Callable] = None
    read_only: bool = False
    range: List = None
    inner_type: type = None


@dataclass
class DynamicFormConfig:
    title: str = ''
    field_config: Dict[str, FormFieldConfig] = None
    read_only_fields: Optional[List[str]] = None
