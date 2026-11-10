"""Where the technology database's files are, for the extraction scripts.

The same layout in tech_data and in CIMS: the folder that holds
data/technology_database/ is the repo root (found by walking up from here).

    data/raw_data/technology_database/      the extractions and the manifest (FRDR)
    primary-sources/                        the source documents (local only; may be absent)
    data/technology_database/               the hand-edited records
    data/processed_data/technology_database/  what the build writes

The scripts written for single sources sit beside their extractions in the raw data
(and so in the FRDR deposit); each reads its document from the same folder under
primary-sources/.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent


def _repo_root() -> Path:
    for p in HERE.parents:
        if (p / "data" / "technology_database").is_dir():
            return p
    raise SystemExit("data/technology_database/ not found above " + str(HERE))


ROOT = _repo_root()
DATA = ROOT / "data"
RAW = DATA / "raw_data" / "technology_database"
PRIMARY = ROOT / "primary-sources"
RECORDS = DATA / "technology_database"
OUTPUT = DATA / "processed_data" / "technology_database"
