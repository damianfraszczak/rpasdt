import ast
from collections import defaultdict
from typing import Optional, Any, Dict, List

def object_to_dict(object: Any) -> dict:
    return object.__dict__


def copy_dict_values_into_object(object: Any, dict: dict) -> None:
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


def sort_dict_by_value(data: Dict[any, any], reverse=True):
    return sorted(data.items(), key=lambda x: x[1], reverse=reverse)


def eval_if_str(val: Any):
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val


def format_label(string: str,
                 required_words: Optional[List[str]] = None) -> str:
    string = string.replace('_', ' ')
    # if required_words:
    #     for word in (for word in required_words if word not in string):
    #         string += f' {word}'
    return string.title()


def multi_sum(iterable, *attributes, **kwargs):
    sums = dict.fromkeys(attributes, kwargs.get('start', 0))
    for it in iterable:
        for attr in attributes:
            sums[attr] += getattr(it, attr)
    return sums
