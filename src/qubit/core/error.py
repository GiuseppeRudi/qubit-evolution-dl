

from enum import Enum
# TODO check all the config loader without the default values 

def parse_enum(enum_cls: type[Enum], value: str, field: str):
    try:
        return enum_cls(value)
    except ValueError:
        allowed = [e.value for e in enum_cls]
        raise ValueError(
            f"Invalid value for {field}: '{value}'. "
            f"Allowed: {allowed}"
        ) from None
