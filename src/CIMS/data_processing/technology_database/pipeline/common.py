"""Paths and CSV helpers shared by every stage."""
from __future__ import annotations

import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent                     # src/technology_database (src/CIMS/data_processing/technology_database in CIMS)


def _repo_root() -> Path:
    """The folder that holds data/technology_database/: the repo root, found by
    walking up from this file, so the package runs from tech_data and from CIMS."""
    for p in HERE.parents:
        if (p / "data" / "technology_database").is_dir():
            return p
    raise SystemExit("data/technology_database/ not found above " + str(HERE))


ROOT = _repo_root()
DATA = ROOT / "data"
RAW = DATA / "raw_data" / "technology_database"            # the extractions and the manifest (FRDR)
PRIMARY = ROOT / "primary-sources"                         # the source documents (local only, may be absent)
RECORDS = DATA / "technology_database"                     # hand-edited records
VOCAB = RECORDS / "vocabulary"            # controlled vocabularies
REFERENCE = RECORDS / "reference"         # conversion and ranking tables
OUTPUT = DATA / "processed_data" / "technology_database"   # everything the build writes
DB = DATA                                 # paths in messages are shown relative to data/
HIERARCHY = RECORDS / "cims-hierarchy.csv"

def find_file(rel: str) -> Path | None:
    """A manifest path: under the extractions, then the source documents, then the
    repo root (for the records' own files). None when it is on none of them."""
    for base in (RAW, PRIMARY, ROOT):
        if rel and (base / rel).exists():
            return base / rel
    return None


MONEY = re.compile(r"^(\d{4}_[A-Z]{3})")


def read(path: Path) -> list[dict]:
    """Rows of a CSV as dicts; an absent file is an empty table."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write(path: Path, cols: list[str], rows: list[dict], quiet: bool = False) -> None:
    """Write rows in the given column order; missing keys are blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:   # with a BOM, so Excel reads the accents and symbols (Brad 2026-11-06)
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c, "")) for c in cols})
    if not quiet:
        try:
            shown = path.relative_to(DB)
        except ValueError:
            shown = path
        print(f"  {str(shown):42s} {len(rows):6d} rows")


def split_list(cell: str) -> list[str]:
    """A ;-separated list cell as a list of stripped, non-empty items."""
    return [x.strip() for x in (cell or "").split(";") if x.strip()]


def fnum(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def currency_year(unit: str) -> str:
    """The currency-year a money unit is expressed in, or "" if it is not money.
    `currency` holds `2013_USD` and nothing else; what the money is per belongs
    to `unit`."""
    m = MONEY.match((unit or "").strip())
    return m.group(1) if m else ""
