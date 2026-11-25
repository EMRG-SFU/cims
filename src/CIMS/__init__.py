"""
CIMS: the Canadian Integrated Modelling System.

The model's public names (Model, load_model, and the others below) load on
first use, not when the package is imported. So importing a subpackage that
doesn't need the model, such as CIMS.data_processing.technology_database,
doesn't load the model and its dependencies (networkx and the rest).
`from CIMS import Model` and `CIMS.Model` work as before.
"""
from importlib import import_module
from typing import TYPE_CHECKING

from .about import __version__

if TYPE_CHECKING:  # read by editors and type checkers only; never runs. Keep in step with _LAZY.
    from .download import download_models
    from .logging import log_model, search_parameter
    from .model import Model, load_model
    from .quantities import ProvidedQuantity, RequestedQuantity

# Public name -> module that defines it
_LAZY = {
    'Model': '.model',
    'load_model': '.model',
    'ProvidedQuantity': '.quantities',
    'RequestedQuantity': '.quantities',
    'log_model': '.logging',
    'search_parameter': '.logging',
    'download_models': '.download',
}

__all__ = ['__version__', *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value      # later lookups skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
