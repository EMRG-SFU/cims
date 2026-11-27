"""Where the technology database's files are, for the extraction scripts.

The folders come from common/paths.py, like every other stage, so CIMS_DATA_ROOT
applies here too:

    RAW       paths.TECHDB_EXTRACTED: the extractions and the manifest (FRDR)
    PRIMARY   paths.PRIMARY_SOURCES / technology-database: the source documents (local
              only; may be absent)
    RECORDS   paths.TECHDB_RECORDS: the hand-edited records
    OUTPUT    paths.TECHDB_OUTPUT: what the build writes

The scripts written for single sources sit beside their extractions (and so in the
FRDR deposit); each reads its document from the same folder under PRIMARY.
"""
from __future__ import annotations

from CIMS.data_processing.common import paths

ROOT = paths.REPO_ROOT
DATA = paths.DATA_ROOT
RAW = paths.TECHDB_EXTRACTED
PRIMARY = paths.PRIMARY_SOURCES / "technology-database"
RECORDS = paths.TECHDB_RECORDS
OUTPUT = paths.TECHDB_OUTPUT
