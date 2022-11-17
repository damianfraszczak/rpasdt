"""Common methods shared throughout the app."""
import ast
import collections
import inspect
import re
from collections import defaultdict
from enum import Enum
from functools import wraps
from pathlib import Path
from time import process_time
from typing import Any, Dict, List, Mapping, Optional, Type, Union

from dataclasses_json import dataclass_json


def write_to_file(text: str, file_path: str) -> None:
    """Save provided text into file with given path."""
    with open(file_path, "w") as file:
        file.write(text)


def read_from_file(file_path: str) -> str:
    """Read content from the file."""
    with open(file_path, "r") as file:
        return file.read()


def get_object_type_as_json_exportable(object: Any) -> Type:
    """Get object type to be json exportable.

    It appends to_json method to an object type dynamically.
    """
    object_type = object if isinstance(object, type) else type(object)
    if not getattr(object_type, "to_json", None):
        object_type = dataclass_json(object_type, undefined=Undefined.EXCLUDE)
    return object_type


def export_dataclass_as_json(object: Any, file_path: Optional[str] = None) -> str:
    json_data = get_object_type_as_json_exportable(object).to_json(object)
    if file_path:
        with open(file_path, "w") as file:
            file.write(json_data)
    return json_data


def import_dataclass_from_json(
    type: Type, json_data: Optional[str] = None, file_path: Optional[str] = None
) -> Any:
    if not (json_data or file_path):
        raise ValueError("Either `json_data` or `file_path` is required.")
    json_data = json_data or read_from_file(file_path)
    return get_object_type_as_json_exportable(type).from_json(json_data)


def get_enum(value: Union[Enum, str], enum_type: type) -> Enum:
    if isinstance(value, str):
        try:
            return enum_type(value)
        except Exception as e:  # noqa
            return enum_type[value]
    return value


def object_to_dict(object: Any) -> dict:
    """Return dict representation of object."""
    return object.__dict__


def copy_dict_values_into_object(object: Any, dict: dict) -> None:
    """Update object properties with dict values."""
    for key, value in dict.items():
        setattr(object, key, value)


def get_object_value(object: Any, field_name: str) -> Optional[Any]:
    if isinstance(object, dict):
        return object.get(field_name, None)
    else:
        return getattr(object, field_name, None)


def set_object_value(object: Any, field_name: str, value: Any) -> None:
    if isinstance(object, dict):
        object[field_name] = value
    else:
        setattr(object, field_name, value)


def group_dict_by_values(dict: Dict):
    res = defaultdict(list)
    for key, val in sorted(dict.items()):
        res[val].append(key)
    return res


def sort_dict_by_value(data: Dict[any, any], reverse=True) -> Dict[any, any]:
    return collections.OrderedDict(
        sorted(data.items(), key=lambda x: x[1], reverse=reverse)
    )


def normalize_dict_values(data: Dict[any, float]) -> Dict[any, float]:
    maxv = max(data.values())
    if maxv == 0:
        maxv = 1.0
    return {key: value / maxv for key, value in data.items()}


def eval_if_str(val: Any):
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val


def format_label(string: str, required_words: Optional[List[str]] = None) -> str:
    string = string.replace("_", " ")
    # if required_words:
    #     for word in (for word in required_words if word not in string):
    #         string += f' {word}'
    return string.title()


def multi_sum(iterable, *attributes, **kwargs):
    sums = dict.fromkeys(attributes, kwargs.get("start", 0))
    for it in iterable:
        for attr in attributes:
            sums[attr] += getattr(it, attr)
    return sums


def camel_case_split(str):
    return re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", str)


def get_function_params(func) -> Mapping[str, inspect.Parameter]:
    return inspect.signature(func).parameters


def get_default_value(type: Any, default: Optional[Any]) -> Optional[Any]:
    if default and default != inspect.Parameter.empty:
        return default
    try:
        return type()
    except Exception:
        return None


def get_function_default_kwargs(func) -> Dict[str, Any]:
    return {
        name: get_default_value(type=param.annotation, default=param.default)
        for name, param in get_function_params(func).items()
    }


def default_on_error(def_val=0):
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                return def_val

        return inner

    return decorator


def method_time(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(process_time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(process_time() * 1000)) - start
            print(f"Total execution time {func.__name__}: {end_ if end_ > 0 else 0} ms")

    return _time_it


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent
