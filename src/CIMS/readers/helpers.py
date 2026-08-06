from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, List, Tuple
from tabulate import tabulate

import numpy as np
import pandas as pd
import polars as pl

from ..utils.model_description import column_list as COL
from ..utils.parameter import list as PARAM

COL_NAME = "File Name"
COL_FOUND = "Files Found"
COL_UNREADABLE = "Unreadable"
REGION_COMPETITION_TYPE = "Region"


def _glob_csvs(directory: Path) -> List[Path]:
    """List every CSV file directly inside directory.

    No naming convention is required — a file may contain rows for any mix of
    regions/sectors, since filtering happens later on the row data itself
    (see filter_model_data), not by which files are selected.
    """
    if not directory.exists():
        return []
    return sorted(directory.glob("*.csv"))


def _scannable(path) -> bool:
    """Check a CSV has the columns filter_model_data requires, without fully reading it."""
    try:
        cols = pl.scan_csv(path, infer_schema_length=0).collect_schema().names()
        return COL.region in cols and COL.sector in cols and COL.year in cols and COL.value in cols
    except Exception:
        return False


def _build_rows(summary: dict) -> List[dict]:
    """Build rows for the per-entry file health table."""
    rows = []
    for name, info in summary.items():
        rows.append({
            COL_NAME: name,
            COL_FOUND: info["found"],
            COL_UNREADABLE: len(info["unreadable"]),
            "unreadable_paths": info["unreadable"],
        })
    return rows


MAX_LISTED_UNREADABLE = 3


def _print_terse_health(rows: List[dict]):
    """Print one folder-level row per directory (found/unreadable counts), with a
    capped list of unreadable filenames underneath any folder that has them."""
    by_dir = {}
    for row in rows:
        d = row.get("dir") or row[COL_NAME]
        group = by_dir.setdefault(d, {"found": 0, "unreadable": 0, "paths": []})
        group["found"] += row[COL_FOUND]
        group["unreadable"] += row[COL_UNREADABLE]
        group["paths"].extend(row["unreadable_paths"])

    ordered_dirs = list(by_dir)  # base model first, then update folders in the order provided

    print("\n    File health (folder: files found / unreadable):\n")
    table = [[d, by_dir[d]["found"], by_dir[d]["unreadable"]] for d in ordered_dirs]
    lines = tabulate(table, headers=["Folder", COL_FOUND, COL_UNREADABLE], tablefmt="simple").splitlines()
    for line in lines:
        print(f"      {line}")

    for d in ordered_dirs:
        group = by_dir[d]
        if not group["unreadable"]:
            continue
        print(f"\n      Unreadable under {d}:")
        for p in group["paths"][:MAX_LISTED_UNREADABLE]:
            try:
                p = p.relative_to(d)
            except ValueError:
                pass
            print(f"        ! {p}")
        remaining = len(group["paths"]) - MAX_LISTED_UNREADABLE
        if remaining > 0:
            print(f"        ! ... and {remaining} more")
    if any(g["unreadable"] for g in by_dir.values()):
        print("\n      (call model.print_file_health(verbose=True) for the full per-entry table)")


def _print_verbose_health(rows: List[dict]):
    """Print one row per update entry (sector/policy), grouped under its source folder."""
    # Format all rows together so column widths are consistent across groups
    table = [[r[COL_NAME], r[COL_FOUND], r[COL_UNREADABLE]] for r in rows]
    lines = tabulate(table, headers=[COL_NAME, COL_FOUND, COL_UNREADABLE], tablefmt="plain").splitlines()
    header, data_lines = lines[0], lines[1:]

    print(f"      {header}")

    by_dir = {}
    for i, row in enumerate(rows):
        d = row.get("dir")
        by_dir.setdefault(d, []).append(i)

    for d, indices in by_dir.items():
        if d:
            print(f"    {d}/")
        for i in indices:
            print(f"      {data_lines[i]}")
            for p in rows[i]["unreadable_paths"]:
                print(f"          ! unreadable: {p.name}")


def print_file_health(rows: List[dict], verbose: bool = False):
    """Print file health, rolled up per folder by default, or the full per-entry table if verbose.

    Terse mode (default) reports one line per folder — found/unreadable file counts,
    with unreadable filenames listed underneath only when there are any — since that's
    the level a user actually decides at ("did my policies folder read ok?"), and the
    per-entry (per sector/policy) breakdown is rarely needed. Pass verbose=True for that
    full per-entry table instead.
    """
    if not rows:
        print("    No files found")
        return
    if verbose:
        _print_verbose_health(rows)
    else:
        _print_terse_health(rows)


def collect_base_files(model_path: str, base_model: str) -> Tuple[List[str], List[dict]]:
    """Collect base model CSV paths and report which are readable.

    Every CSV in the base model directory is collected for reading; whether a file
    can actually be parsed (has the expected columns) is checked here, separately
    from whether requested regions/sectors have any data (see print_coverage_matrix).
    """
    base_dir = Path(model_path) / base_model
    found_paths = _glob_csvs(base_dir)
    unreadable = [p for p in found_paths if not _scannable(str(p))]

    summary = {str(base_dir): {"found": len(found_paths), "unreadable": unreadable}}
    rows = _build_rows(summary)
    return [str(p) for p in found_paths], rows


def collect_update_files(update_files: Mapping[str, Iterable[str]]) -> Tuple[List[str], List[dict]]:
    """Collect update CSV paths and report which are readable.

    Every CSV in each update directory is collected for reading; whether a file
    can actually be parsed (has the expected columns) is checked here, separately
    from whether requested regions/sectors have any data (see print_coverage_matrix).
    """
    found = []
    summary = {}

    key_to_dir = {}
    if update_files:
        all_entries = [(d, f) for d, files in update_files.items() for f in files]
        duplicates = {f for f, n in Counter(f for _, f in all_entries).items() if n > 1}
        for dir_name, file in all_entries:
            summary_key = str(Path(dir_name) / file) if file in duplicates else file
            key_to_dir[summary_key] = dir_name

            sub_dir = Path(dir_name) / file
            found_paths = _glob_csvs(sub_dir)
            unreadable = [p for p in found_paths if not _scannable(str(p))]
            summary[summary_key] = {"found": len(found_paths), "unreadable": unreadable}

            found.extend(str(p) for p in found_paths)

    rows = _build_rows(summary)
    for row in rows:
        d = key_to_dir.get(row[COL_NAME])
        row["dir"] = d
        if d:
            row[COL_NAME] = Path(row[COL_NAME]).name
    return found, rows


def _coverage_data(paths: Iterable[str]) -> Tuple[set, set]:
    """Scan readable paths and return (pairs, region_only): the set of (Region, Sector)
    combinations with both fields non-null, and the set of regions that have at least
    one row with a null Sector.

    Rows with a null Sector are core/structural rows not tied to any specific sector
    (e.g. node/branch definitions) — they don't count toward a Region/Sector pair, but
    their presence for a region is itself useful: it shows whether that region has any
    data in the dataset at all, independent of sector-level content.
    """
    pairs = set()
    region_only = set()
    for path in paths:
        if not _scannable(path):
            continue
        try:
            df = pl.scan_csv(path, infer_schema_length=0).select(COL.region, COL.sector).collect()
        except Exception:
            continue
        non_null = df.drop_nulls()
        pairs.update(zip(non_null[COL.region].to_list(), non_null[COL.sector].to_list()))
        no_sector = df.filter(pl.col(COL.sector).is_null() & pl.col(COL.region).is_not_null())
        region_only.update(no_sector[COL.region].to_list())
    return pairs, region_only


def print_coverage_matrix(paths: Iterable[str], region_list: Iterable[str], sector_list: Iterable[str]):
    """Print a region x sector matrix of which requested combinations have at least one row.

    Built from the actual Region/Sector column values found across all readable
    files combined, not from file or folder names — a single file may contain rows
    for any mix of regions/sectors, so coverage can only be determined from the data.
    """
    regions = list(region_list)
    sectors = list(sector_list)
    if not regions or not sectors:
        return

    pairs, region_only = _coverage_data(paths)
    table = [["(no sector)"] + ["x" if region in region_only else "." for region in regions]]
    table.extend(
        [sector] + ["x" if (region, sector) in pairs else "." for region in regions]
        for sector in sectors
    )
    print("\n    Region x Sector coverage (x = data found, . = none found):\n")
    print("      '(no sector)' row: does the region have any data at all, ignoring sector")
    lines = tabulate(table, headers=["Sector"] + regions, tablefmt="simple").splitlines()
    for line in lines:
        print(f"      {line}")

    covered_sectors = sum(1 for sector in sectors if any((region, sector) in pairs for region in regions))
    print(f"\n      Summary: {covered_sectors} of {len(sectors)} requested sectors have data in at least one region.")


def collect_files(
    model_path: str,
    base_model: str,
    region_list: Iterable[str],
    sector_list: Iterable[str],
    update_files: Mapping[str, Iterable[str]],
) -> Tuple[List[str], List[str], List[dict]]:
    """
    Collect base and update paths, print a summary, and return found lists plus health rows.
    Returns (base_found, update_found, health_rows); health_rows can be re-printed later
    via print_file_health(health_rows, verbose=True) for the full per-entry breakdown.
    """
    base_found, base_rows = collect_base_files(model_path, base_model)
    update_found, update_rows = collect_update_files(update_files)
    health_rows = base_rows + update_rows

    print_coverage_matrix(base_found + update_found, region_list, sector_list)
    print_file_health(health_rows)
    return base_found, update_found, health_rows


def _excluded_region_branches(paths: Iterable[str], regions: List[str]) -> set:
    """Find branches that represent an excluded administrative region (e.g. a province
    not in region_list), so its structural parent's unconditional request to it can be
    dropped along with everything else about that region (see filter_model_data).

    A branch is a region node if it has a competition row with Value == "Region"; its
    own Region column value is the region it represents.
    """
    region_of_branch = {}
    for path in paths:
        if not _scannable(path):
            continue
        try:
            df = (
                pl.scan_csv(path, infer_schema_length=0)
                .filter(pl.col(COL.parameter) == PARAM.competition_type)
                .select(COL.branch, COL.region, COL.value)
                .collect()
            )
        except Exception:
            continue
        region_rows = df.filter(pl.col(COL.value) == REGION_COMPETITION_TYPE)
        region_of_branch.update(zip(region_rows[COL.branch].to_list(), region_rows[COL.region].to_list()))

    return {branch for branch, region in region_of_branch.items() if region not in regions}


def filter_model_data(
    paths: Iterable[str],
    region_list: Iterable,
    sector_list: Iterable,
    year_list: Iterable,
    col_list: Iterable[str],
) -> pd.DataFrame:
    """Read, concatenate, and filter model input CSVs by region, sector, and year in one lazy pass.

    Region, Sector, and Year are filtered independently: a row is dropped only if one of
    its own values is non-null and not in the requested list. A null value in a column
    always passes that column's filter, but doesn't exempt the row from the others. An
    empty/falsy region_list or sector_list applies no filtering for that column. Exact
    duplicate rows (e.g. national defaults inlined into every per-region file) are dropped
    after filtering. Unreadable files are silently skipped — already reported by
    print_file_health/collect_files at collection time.

    See _excluded_region_branches for the one exception to per-column independence: a
    structural parent's unconditional request to an excluded region node is also dropped.
    """
    year_strs = [str(y) for y in year_list]
    sectors = list(sector_list) if sector_list else None
    regions = list(region_list) if region_list else None

    excluded_region_branches = list(_excluded_region_branches(paths, regions)) if regions else []

    lazy_frames = []
    for path in paths:
        if not _scannable(path):
            continue
        lf = pl.scan_csv(path, infer_schema_length=0)
        if regions:
            lf = lf.filter(pl.col(COL.region).is_in(regions) | pl.col(COL.region).is_null())
        if sectors:
            lf = lf.filter(pl.col(COL.sector).is_in(sectors) | pl.col(COL.sector).is_null())
        lf = lf.filter(pl.col(COL.year).is_in(year_strs) | pl.col(COL.year).is_null())
        if excluded_region_branches:
            is_structural_request_to_excluded_region = (
                (pl.col(COL.parameter) == PARAM.service_request)
                & pl.col(COL.target).is_in(excluded_region_branches)
                & (pl.col(COL.branch) == pl.col(COL.target).str.replace(r"\.[^.]+$", ""))
            )
            lf = lf.filter(~is_structural_request_to_excluded_region)
        lazy_frames.append(lf)

    if not lazy_frames:
        return pd.DataFrame(columns=list(col_list) + [COL.year, COL.value])

    df = (
        pl.concat(lazy_frames, how="diagonal_relaxed")
        .collect()
        .unique(maintain_order=True)
        .to_pandas()
        .replace({np.nan: None, "": None})
    )

    meta_cols = [c for c in df.columns if c not in (COL.year, COL.value) and c in col_list]
    return df[meta_cols + [COL.year, COL.value]]
