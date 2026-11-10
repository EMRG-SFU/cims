"""Check the source manifest against the disk and against the database.

Run from anywhere:   python src/technology_database/extraction/check_manifest.py

The manifest is two hand-maintained CSVs -- source-details_sources.csv (one row
per source) and source-details_files.csv (one row per file held for a source).
It was a workbook until 2026-08-29; the conversion is recorded in the README.
Nothing generates these files, so this checks the things that quietly rot:

  * a file row whose file does not exist where it says it does;
  * a file row pointing at a source that is not registered;
  * a source-id used in the harmonized database that is not registered;
  * rows out of alphabetical order.

The reference manifest, source-details_reference.csv, is checked here too: the
constants, conversion data and currency series behind the conversions, kept apart
from the technology sources (design.md s5). Its ids must not repeat the main
manifest's, and a file it says is held must exist. That every factor's source is
registered is checked by the build (pipeline/checks.py).

Exit status is non-zero if anything fails, so it can gate a commit.
"""
from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import OUTPUT, PRIMARY, RAW, RECORDS, ROOT  # noqa: E402

SOURCES = RAW / "source-details_sources.csv"
FILES = RAW / "source-details_files.csv"
REFERENCE = RAW / "source-details_reference.csv"


def find(rel: str):
    """A manifest path under the extractions, the source documents or the repo root."""
    for base in (RAW, PRIMARY, ROOT):
        if (base / rel).exists():
            return base / rel
    return None


def read(p: Path) -> list[dict]:
    return list(csv.DictReader(p.open(encoding="utf-8-sig"))) if p.exists() else []


def main() -> int:
    srcs, files = read(SOURCES), read(FILES)
    if not srcs:
        print(f"manifest not found at {SOURCES}", file=sys.stderr)
        return 1
    fail = collections.defaultdict(list)

    ids = [d["source_id"] for d in srcs]
    if len(ids) != len(set(ids)):
        dup = [k for k, v in collections.Counter(ids).items() if v > 1]
        fail["duplicate source_id"] = dup
    if ids != sorted(ids):
        fail["sources not in alphabetical order"] = ["run the sort in the README"]
    fkeys = [(d["source_id"], d["file_name"]) for d in files]
    if fkeys != sorted(fkeys):
        fail["files not in alphabetical order"] = ["sort by source_id then file_name"]

    for d in files:
        if d["source_id"] not in set(ids):
            fail["file row for an unregistered source"].append(
                f"{d['source_id']} / {d['file_name']}")
        on_disk = (d.get("file_on_disk") or "").strip()
        if not on_disk:
            fail["file row with no file on disk"].append(
                f"{d['source_id']} / {d['file_name']}")
            continue
        rel = f"{d['file_path']}/{on_disk}" if d["file_path"] else on_disk
        # a source document is checked only where primary-sources/ is on this disk
        # (an FRDR download holds the extractions alone)
        # an extraction is a .csv the manifest does not describe as a source file
        extraction = on_disk.endswith(".csv") and not (d.get("description") or "").startswith("source")
        if find(rel) is None and (PRIMARY.is_dir() or extraction):
            fail["file recorded but missing"].append(rel)

    ref = read(REFERENCE)
    rids = [d["source_id"] for d in ref]
    if len(rids) != len(set(rids)):
        fail["duplicate source_id in the reference manifest"] = [
            k for k, v in collections.Counter(rids).items() if v > 1]
    if rids != sorted(rids):
        fail["reference manifest not in alphabetical order"] = ["sort by source_id"]
    for k in sorted(set(rids) & set(ids)):
        fail["registered in both the main and the reference manifest"].append(k)
    for d in ref:
        held = (d.get("file_on_disk") or "").strip()
        if held and find(held) is None and (PRIMARY.is_dir() or held.startswith("data/")):
            fail["reference file recorded but missing"].append(held)

    used = collections.Counter()
    for path in (OUTPUT / "example-parameter.csv",
                 OUTPUT / "example-flow.csv",
                 OUTPUT / "example-constraint.csv",
                 RECORDS / "source-example.csv"):
        for r in read(path):
            for col in ("source-id", "via-source-id"):
                sid = (r.get(col) or "").strip()
                if sid:
                    used[sid] += 1
    for sid in sorted(set(used) - set(ids)):
        fail["used in the database but not registered"].append(
            f"{sid} ({used[sid]} rows)")

    print(f"  {len(srcs)} sources, {len(files)} files, "
          f"{len(used)} source-ids used in the database; {len(ref)} reference sources")
    unused = sorted(set(ids) - set(used))
    print(f"  {len(set(ids) & set(used))} registered sources are used; "
          f"{len(unused)} are registered and not yet used")
    if not fail:
        print("  all checks passed")
        return 0
    for k, v in fail.items():
        print(f"\n  {k.upper()} ({len(v)}):")
        for x in v[:12]:
            print(f"      {x}")
        if len(v) > 12:
            print(f"      ... and {len(v) - 12} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
