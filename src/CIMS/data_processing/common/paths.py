"""Every data layer's root, in one place (plan Phase 2, conventions §2).

The data folder follows the code: ``<repo root>/data``, beside the code that is
imported, unless the ``CIMS_DATA_ROOT`` environment variable points somewhere
else. Every stage builds its paths from the constants here, so a layer moves by
changing one line (phase 3).

Reads
-----
Nothing. The constants are paths, and nothing here checks that they exist,
except the data root itself.

Usage
-----
    from CIMS.data_processing.common import paths
    paths.EXTRACTED / 'gov-can/statcan/cpi/18100005.xlsx'

    python -m CIMS.data_processing.common.paths            # print the roots
    python -m CIMS.data_processing.common.paths --strict   # and fail on a mismatch
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_DATA_ROOT = 'CIMS_DATA_ROOT'
ENV_STRICT = 'CIMS_STRICT_PATHS'

# This file is src/CIMS/data_processing/common/paths.py, so parents[4] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / 'src' / 'CIMS'
PIPELINE_ROOT = CODE_ROOT / 'data_processing'


def _data_root() -> Path:
    override = os.environ.get(ENV_DATA_ROOT)
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'{ENV_DATA_ROOT} is set to {root}, which is not a directory.')
        return root
    root = REPO_ROOT / 'data'
    if not root.is_dir():
        raise FileNotFoundError(f'Could not locate the CIMS data directory at {root}. '
                                f'Set the {ENV_DATA_ROOT} environment variable to point at it.')
    return root


DATA_ROOT = _data_root()
DATA_ROOT_FROM_ENV = bool(os.environ.get(ENV_DATA_ROOT))

# The layers (conventions §2). Phase 3 moves the folders (tools/migrate_data_layout.py).
PRIMARY_SOURCES = DATA_ROOT / 'primary-sources'            # documents as obtained; the build never reads them
EXTRACTED = DATA_ROOT / 'extracted'                         # as-published values, by publisher (gov-can/statcan/, ...)
LEGACY = EXTRACTED / 'legacy'                               # JCIMS-era inputs
FIXED_DATA = LEGACY / 'fixed-data'                          # the JCIMS model data (phase 13 retires it)
CONFIG = DATA_ROOT / 'config'                               # hand-edited records, in git
ASSUMPTIONS = CONFIG / 'assumption'                         # hand-set growth rates and shares
DEFAULTS = CONFIG / 'model-default'                         # hand-made model defaults
CALIBRATION_CONFIG = CONFIG / 'calibration'                 # the FIC proxy map and decline list
TECHDB_RECORDS = CONFIG / 'technology-database'             # the technology database's hand-edited records
PROCESSED = DATA_ROOT / 'processed'                         # written by the source stages
MODEL = DATA_ROOT / 'model'                                 # everything CIMS reads; written by the build
MODEL_BASE = MODEL / 'base'                                 # cims-base_<region>.csv
MODEL_SECTOR = MODEL / 'sector'                             # <sector>/<sector>_<region>.csv
MODEL_COST = MODEL / 'cost'                                 # declining-capital-cost/, declining-intangible-cost/, fixed-intangible-cost/
MODEL_EXOGENOUS = MODEL / 'exogenous'                       # exogenous-price/, exogenous-demand/, market-share-limit/
POLICIES = MODEL / 'policy'                                 # policy files (hand-made until the policy stage)
CALIBRATION_TARGETS = MODEL / 'calibration' / 'target'      # calibration targets, written by the pipeline
CALIBRATION_OUTPUTS = MODEL / 'calibration' / 'fitted'      # written by calibration: fitted FICs and lifetimes
TECHDB_EXTRACTED = EXTRACTED / 'technology-database'        # its extractions and manifest
TECHDB_OUTPUT = PROCESSED / 'technology-database'           # its build output
RESULTS = REPO_ROOT / 'results'                             # model results and logs (gitignored)
LOGS = RESULTS / 'logs'                                     # run and optimizer logs


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def root_warnings(cwd: Path | None = None) -> list[str]:
    """Return a warning for each way the roots don't match the current folder.

    A notebook or script that reads ``data/...`` relative to the current folder
    would otherwise use different files from the ones the pipeline reads and writes.

    Parameters
    ----------
    cwd : Path, optional
        The folder to compare against. Default: the current working directory.

    Returns
    -------
    list of str
        Empty when the data root is inside ``cwd``.
    """
    cwd = Path.cwd() if cwd is None else Path(cwd)
    if _inside(DATA_ROOT, cwd):
        return []
    how = f'set by {ENV_DATA_ROOT}' if DATA_ROOT_FROM_ENV else 'beside the imported code'
    return [f'The data root ({DATA_ROOT}, {how}) is not inside the current folder ({cwd}). '
            f'A script or notebook reading data/... from the current folder would use other files. '
            f'Run from {REPO_ROOT}, or use the environment of the worktree you are in.']


def strict_from_env() -> bool:
    """True when ``CIMS_STRICT_PATHS`` is set to anything other than '', '0', or 'false'."""
    return os.environ.get(ENV_STRICT, '').strip().lower() not in ('', '0', 'false', 'no')


def announce(strict: bool | None = None, cwd: Path | None = None, out=sys.stdout) -> None:
    """Print the code and data roots, and warn when they don't match the current folder.

    Parameters
    ----------
    strict : bool, optional
        Raise instead of warning. Default: ``CIMS_STRICT_PATHS`` (see ``strict_from_env``).
    cwd : Path, optional
        The folder to compare against. Default: the current working directory.
    out : file, optional
        Where to print. Default: standard output.

    Raises
    ------
    RuntimeError
        In strict mode, when the data root isn't inside the current folder.
    """
    strict = strict_from_env() if strict is None else strict
    print(f'Code root: {CODE_ROOT}', file=out)
    print(f'Data root: {DATA_ROOT}' + (f'  (from {ENV_DATA_ROOT})' if DATA_ROOT_FROM_ENV else ''), file=out)
    warnings = root_warnings(cwd)
    if warnings and strict:
        raise RuntimeError(' '.join(warnings))
    for w in warnings:
        print(f'WARNING: {w}', file=out)
    print(file=out)


if __name__ == '__main__':
    announce(strict=True if '--strict' in sys.argv[1:] else None)
