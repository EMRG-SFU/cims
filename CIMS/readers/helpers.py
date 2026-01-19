from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, List, Tuple
from tabulate import tabulate

COL_NAME = "File Name"
COL_LOADED_TOTAL = ""
COL_LOADED = "Loaded"
COL_NOT_FOUND = "Not Found"

def _build_rows(summary: dict, total_regions: int) -> List[dict]:
    """Build rows for the summary table with alphabetized region lists."""
    rows = []
    for name, info in summary.items():
        loaded = sorted(info["loaded_regions"])
        missing = sorted(info["missing_regions"])
        rows.append({
            COL_NAME: name,
            COL_LOADED_TOTAL: f"{len(loaded)}/{total_regions}",
            COL_LOADED: ", ".join(loaded) if loaded else "none",
            COL_NOT_FOUND: ", ".join(missing) if missing else "none",
        })
    return rows


def _print_combined_summary(rows):
    """Print a single table for base and update files, preserving row order and indenting output."""
    if not rows:
        print("    No files found")
        return

    table = [[r[COL_NAME], r[COL_LOADED_TOTAL], r[COL_LOADED], r[COL_NOT_FOUND]] for r in rows]
    table_str = tabulate(table, headers=[COL_NAME, COL_LOADED_TOTAL, COL_LOADED, COL_NOT_FOUND], tablefmt="plain")
    indented = "\n".join(f"    {line}" for line in table_str.splitlines())

    print(indented)


def _add_path(cur: Path, summary: dict, key: str, reg: str, found: List[str], missing: List[str]) -> None:
    """Record a path as found or missing and update summary."""
    if cur.exists():
        found.append(str(cur))
        summary[key]["loaded_regions"].add(reg)
    else:
        missing.append(str(cur))
        summary[key]["missing_regions"].add(reg)


def collect_base_paths(
    model_path: str,
    base_model: str,
    region_list: Iterable[str],
) -> Tuple[List[str], List[str], List[dict]]:
    """Collect base model CSV paths and summarize loaded/not found regions."""
    regions = list(region_list)
    found, missing = [], []
    summary = defaultdict(lambda: {"loaded_regions": set(), "missing_regions": set()})
    total_regions = len(regions)

    for reg in regions:
        cur = Path(model_path) / base_model / f"{base_model}_{reg}.csv"
        _add_path(cur, summary, base_model, reg, found, missing)

    rows = _build_rows(summary, total_regions)
    return found, missing, rows


def collect_update_paths(
    update_files: Mapping[str, Iterable[str]],
    region_list: Iterable[str],
) -> Tuple[List[str], List[str], List[dict]]:
    """Collect update CSV paths and summarize loaded/not found regions."""
    regions = list(region_list)
    found, missing = [], []
    summary = defaultdict(lambda: {"loaded_regions": set(), "missing_regions": set()})
    total_regions = len(regions)

    if update_files:
        for dir_name, files in update_files.items():
            for file in files:
                for reg in regions:
                    cur = Path(dir_name) / file / f"{file}_{reg}.csv"
                    _add_path(cur, summary, file, reg, found, missing)

    rows = _build_rows(summary, total_regions)
    return found, missing, rows


def collect_all_paths(
    model_path: str,
    base_model: str,
    region_list: Iterable[str],
    update_files: Mapping[str, Iterable[str]],
) -> Tuple[List[str], List[str]]:
    """
    Collect base and update paths, print a combined summary table, and return found lists.
    Returns (base_found, update_found).
    """
    base_found, _, base_rows = collect_base_paths(model_path, base_model, region_list)
    update_found, _, update_rows = collect_update_paths(update_files, region_list)

    _print_combined_summary(base_rows + update_rows)
    return base_found, update_found
