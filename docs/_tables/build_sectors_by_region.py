"""Build the sectors-by-region table for the Model Architecture chapter.

Writes ``sectors-by-region.md`` next to this script. The chapter pulls it
in with ``{{< include _tables/sectors-by-region.md >}}``, so rendering the
book runs no Python. Rerun this script whenever the model inputs change
which sectors each region carries, then re-render.

A sector is marked for a region when at least one row in any CSV under
``data/model_inputs/`` has ``Parameter == service_provide`` and a
``Branch`` that starts with that region's branch from
``data/mappings_conversions/region_primary.csv``. Rows follow the ``cims``
column of ``data/mappings_conversions/sector_map.csv``.

When a region aggregation (``data/mappings_conversions/region_aggregation_*.csv``)
is present in the model inputs, it replaces its child regions as a single
column, with a footnote naming them. Regions with no sectors are dropped.

Uses the Python standard library only. Usage, from the repository root::

    python docs/_tables/build_sectors_by_region.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "sectors-by-region.md"
LABEL = "tbl-sectors-by-region"
CAPTION = "Sectors represented by region"

# Pandoc sets pipe-table column widths in proportion to the dashes in the
# separator row: a wide first column, and equal shares for the regions.
SECTOR_WEIGHT = 40
REGION_WEIGHT = 4


def find_root() -> Path:
    """The repository root: the first parent holding data/mappings_conversions."""
    for folder in [HERE, *HERE.parents]:
        if (folder / "data" / "mappings_conversions").exists():
            return folder
    raise SystemExit("Could not find the project root containing data/mappings_conversions")


def read_rows(path: Path, skip_lines: int = 0) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        for _ in range(skip_lines):
            f.readline()
        return list(csv.DictReader(f))


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def main() -> int:
    root = find_root()
    mc_dir = root / "data" / "mappings_conversions"
    inputs_dir = root / "data" / "model_inputs"
    if not inputs_dir.exists():
        raise SystemExit(f"No model inputs at {inputs_dir}; generate them before building the table")

    regions = read_rows(mc_dir / "region_primary.csv")
    sectors = read_rows(mc_dir / "sector_map.csv")

    # Primary regions as (lowercase branch, uppercase code), in file order.
    primary = [(clean(r["branch"]).lower(), clean(r["code"]).upper()) for r in regions]
    primary_branch_by_code = {code: branch for branch, code in primary}
    region_name_by_code = {clean(r["code"]).upper(): clean(r["name"]) for r in regions}

    # Aggregations: each file maps one aggregation code to its child regions.
    # The aggregation's branch is a sibling of its children's branches.
    aggregations: dict[str, dict] = {}
    for agg_file in sorted(mc_dir.glob("region_aggregation_*.csv")):
        rows = read_rows(agg_file)
        if not rows:
            continue
        code = clean(rows[0]["code"]).upper()
        children = [clean(r["region_primary"]).upper() for r in rows]
        sample = next((primary_branch_by_code[c] for c in children if c in primary_branch_by_code), None)
        if sample is None:
            continue
        aggregations[code] = {"branch": f"{sample.rsplit('.', 1)[0]}.{code.lower()}",
                              "children": children, "name": clean(rows[0]["name"])}

    # Most specific branch prefix wins.
    candidates = primary + [(info["branch"], code) for code, info in aggregations.items()]
    candidates.sort(key=lambda c: -len(c[0]))

    # Every model input CSV: the first line is a "navigate by typing" banner,
    # so the header is on line 2.
    found: set[tuple[str, str]] = set()
    for path in sorted(inputs_dir.rglob("*.csv")):
        try:
            rows = read_rows(path, skip_lines=1)
        except (UnicodeDecodeError, csv.Error):
            continue
        for row in rows:
            branch, parameter, sector = (clean(row.get(k)) for k in ("Branch", "Parameter", "Sector"))
            if not (branch and sector) or parameter != "service_provide":
                continue
            branch = branch.lower()
            for cand_branch, code in candidates:
                if branch == cand_branch or branch.startswith(cand_branch + "."):
                    found.add((code, sector))
                    break

    # Active aggregations replace their children, where the children were.
    codes_with_data = {code for code, _ in found}
    active = [code for code in aggregations if code in codes_with_data]
    columns = [code for _, code in primary]
    for code in active:
        children = aggregations[code]["children"]
        at = min((columns.index(ch) for ch in children if ch in columns), default=len(columns))
        columns = [c for c in columns if c not in children]
        columns.insert(min(at, len(columns)), code)
    columns = [c for c in columns if c in codes_with_data]

    def header(code: str) -> str:
        if code not in active:
            return f"**{code}**"
        names = [region_name_by_code.get(ch, ch) for ch in aggregations[code]["children"]]
        listed = ", ".join(names[:-1]) + ", and " + names[-1] if len(names) > 1 else "".join(names)
        return f"**{code}**^[{aggregations[code]['name']} includes {listed}.]"

    lines = [
        "<!-- Generated by build_sectors_by_region.py from data/model_inputs. Do not edit. -->",
        "",
        "| **Sectors** | " + " | ".join(header(c) for c in columns) + " |",
        "|:" + "-" * SECTOR_WEIGHT + "|" + "|".join(":" + "-" * REGION_WEIGHT + ":" for _ in columns) + "|",
    ]
    # In HTML, style.css shades cells holding .cell-on and hides the "+";
    # without the stylesheet (PDF), the "+" shows as the marker.
    for sector in (clean(s["cims"]) for s in sectors):
        cells = ["[+]{.cell-on}" if (code, sector) in found else "" for code in columns]
        lines.append(f"| {sector} | " + " | ".join(cells) + " |")
    lines += ["", f": {CAPTION} {{#{LABEL}}}", ""]

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT.name}: {len(columns)} regions, {len(found)} sector-region pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
