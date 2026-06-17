"""
Set up yaml serializer to read and write config files (based on ``pydantic.BaseModel``).
"""

from __future__ import annotations

import datetime
import pathlib
from typing import Dict, Any, Self, Callable
import types
import inspect
import textwrap
import ast

try:
    import isodate

    ISODATE = True
except ImportError:
    ISODATE = False

import pydantic
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def path_representer(dumper: yaml.Dumper, path: pathlib.Path):
    """Represent ``pathlib.Path`` as string in yaml."""
    return dumper.represent_scalar(tag="!path", value=str(path))


def path_constructor(loader: yaml.loader, node) -> pathlib.Path:
    """Convert string back to ``pathlib.Path`` object."""
    return pathlib.Path(loader.construct_scalar(node))


def timedelta_representer(dumper: yaml.Dumper, td: datetime.timedelta):
    """Represent ``datetime.timedelta`` as ISO 8601 duration string with yaml tag ``!timedelta``."""
    if not ISODATE:
        raise ImportError("Please install isodate to use timedelta_representer.")
    return dumper.represent_scalar(tag="!timedelta", value=isodate.duration_isoformat(td))


def timedelta_constructor(loader: yaml.Loader, node) -> datetime.timedelta:
    """Convert seconds back to ``timedelta`` object.
    Attention! Changing constructors might result in old files becoming unreadable!
    """
    if not ISODATE:
        raise ImportError("Please install isodate to use timedelta_representer.")
    return isodate.parse_duration(loader.construct_scalar(node))


def tuple_representer(dumper: yaml.Dumper, t: tuple):
    """Convert tuple to yaml list. Attention! Deserialisation will be list not tuple! Pydantic will fix that."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", t, flow_style=True)


def lambda_representer(dumper: yaml.Dumper, func: types.FunctionType):
    """Represent a lambda function as a string.
    Tries to extract the specific lambda source code from the definition line.
    """
    if func.__name__ != "<lambda>":
        # Fallback for named functions if you ever want to serialize those
        # Or raise an error to prevent accidental serialization of large functions
        return dumper.represent_scalar("!func", func.__name__)

    try:
        # Get the source lines of the function
        source_lines = inspect.getsource(func)
    except OSError:
        raise ValueError(f"Could not retrieve source code for lambda: {func}")

    # The source might be indented or contain "x = lambda..."
    # We parse the AST to find the exact Lambda node.
    dedented_source = textwrap.dedent(source_lines)

    try:
        tree = ast.parse(dedented_source)
    except SyntaxError:
        raise ValueError(
            f"Could not parse source for lambda extraction: {dedented_source}. "
            f"Your lambda may be too complex (-> use named functions instead), "
            "or multiple lambdas on one line (-> spread out list to multiple lines)."
        )

    lambda_node = None

    # Walk the tree to find the lambda.
    # NOTE: If there are multiple lambdas on one line, this naively takes the first one.
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            lambda_node = node
            break

    if lambda_node:
        # Extract the segment corresponding to the lambda
        # Requires Python 3.8+ for get_source_segment, fallback usually not needed for pydantic v2 envs
        lambda_text = ast.get_source_segment(dedented_source, lambda_node)
        return dumper.represent_scalar("!lambda", lambda_text)

    # Fallback: dump the whole line (might fail eval if it's an assignment)
    return dumper.represent_scalar("!lambda", dedented_source.strip())


def lambda_constructor(loader: yaml.Loader, node) -> Callable:
    """Convert a string definition of a lambda back into a callable.
    WARNING: Uses eval(). Unsafe for untrusted input.
    """
    value = loader.construct_scalar(node)
    try:
        return eval(value)
    except Exception as e:
        raise ValueError(f"Could not evaluate lambda string '{value}': {e}")


# Register path representer and constructor for Posix and Windows to Dumper
yaml.add_representer(pathlib.Path, path_representer, Dumper=Dumper)
yaml.add_representer(pathlib.PosixPath, path_representer, Dumper=Dumper)
yaml.add_representer(pathlib.WindowsPath, path_representer, Dumper=Dumper)
yaml.add_constructor("!path", path_constructor, Loader=Loader)

# Register representer and constructor to convert timedelta between Python and yaml.
yaml.add_representer(datetime.timedelta, timedelta_representer, Dumper=Dumper)
yaml.add_constructor("!timedelta", timedelta_constructor, Loader=Loader)

# Register representer and constructor to convert tuple between Python and yaml.
yaml.add_representer(tuple, tuple_representer, Dumper=Dumper)

# Register Lambda
# specific check for LambdaType (which is usually just FunctionType)
yaml.add_representer(types.LambdaType, lambda_representer, Dumper=Dumper)
yaml.add_constructor("!lambda", lambda_constructor, Loader=Loader)


def yaml_to_dict(yaml_str: str) -> Dict:
    """Convert yaml string to dict."""
    return yaml.load(yaml_str, Loader=Loader)


def dict_to_yaml(d: Dict) -> str:
    """Convert dict to yaml string."""
    return yaml.dump(d, Dumper=Dumper)


class BaseYAMLConfig(pydantic.BaseModel):
    """Mixin to add yaml dumping and loading support to pydantic ``BaseModel``.
    Following pydantic v2 paradigm.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)  # Allow, e.g., numpy arrays or custom types

    def model_dump_yaml(self, *, exclude: Dict[str, Any] = None) -> str:
        """Convert model to yaml string."""
        if exclude is None:
            exclude = {}
        return yaml.dump(self.model_dump(exclude=exclude), Dumper=Dumper)

    @classmethod
    def model_from_yaml(cls, yaml_str: str) -> Self:
        """Load model from yaml string."""
        return cls(**yaml_to_dict(yaml_str))  # noqa: unexpected arguments
