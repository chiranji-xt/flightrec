"""Turning arbitrary call arguments and responses into JSON and back."""

import dataclasses
import hashlib
import json
import sys
from typing import Any, Optional


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of SDK objects, dataclasses and the like to plain JSON types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Box):
        return to_jsonable(obj.model_dump())
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "__dict__"):
        return to_jsonable(vars(obj))
    return repr(obj)


def type_path(obj: Any) -> Optional[str]:
    """`module:Qualname` for pydantic-style SDK objects, so replay can rebuild the real type."""
    cls = type(obj)
    if hasattr(cls, "model_validate") and hasattr(obj, "model_dump"):
        return f"{cls.__module__}:{cls.__qualname__}"
    return None


def rehydrate(path: str, data: Any) -> Any:
    """Rebuild an SDK object from its recorded dump, or return None if that is not possible.

    Only classes in modules the program has already imported are used. A recording is just a
    file, and reading one must never import code.
    """
    module_name, _, qualname = path.partition(":")
    target: Any = sys.modules.get(module_name)
    for part in qualname.split(".") if target is not None else ():
        target = getattr(target, part, None)
    validate = getattr(target, "model_validate", None)
    if validate is None:
        return None
    try:
        return validate(data)
    except Exception:
        return None


def digest(obj: Any) -> str:
    """Stable short hash of a JSON-able value, used to detect a changed request on replay."""
    blob = json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class Box:
    """Attribute-access view over recorded JSON, so replayed SDK responses look like the originals."""

    def __init__(self, data: Any):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict) and name in data:
            return _wrap(data[name])
        raise AttributeError(name)

    def __getitem__(self, key: Any) -> Any:
        return _wrap(object.__getattribute__(self, "_data")[key])

    def __iter__(self):
        data = object.__getattribute__(self, "_data")
        return iter(_wrap(v) for v in data) if isinstance(data, list) else iter(data)

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_data"))

    def __eq__(self, other: Any) -> bool:
        other = other._data if isinstance(other, Box) else other
        return object.__getattribute__(self, "_data") == other

    def __repr__(self) -> str:
        return f"Box({object.__getattribute__(self, '_data')!r})"

    def model_dump(self) -> Any:
        return object.__getattribute__(self, "_data")


def _wrap(value: Any) -> Any:
    return Box(value) if isinstance(value, (dict, list)) else value
