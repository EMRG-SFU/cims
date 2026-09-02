from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, List, Tuple
from tabulate import tabulate

import numpy as np
import pandas as pd
import polars as pl

from ..utils.model_description import column_list as COL
from ..utils.parameter import list as PARAM

NEST_INDENT = "  "
COL_NAME = "File Name"
COL_FOUND = "Files Found"
COL_UNREADABLE = "Unreadable"
COL_MISSING = "Folder Not Found"
MISSING_MARK = "<--"
REGION_COMPETITION_TYPE = "Region"

# Every column filter_model_data touches. Branch is included even though the
# filters do not key off it, because every consumer downstream does.
REQUIRED_COLUMNS = (COL.branch, COL.region, COL.sector, COL.year, COL.value)

# Rows _scannable parses before deciding a file is usable. Enough to catch a file
# that is structurally wrong from the start (ragged rows, unbalanced quotes,
# non-CSV content) without paying for a full parse of every file at collection.
SCAN_PROBE_ROWS = 200


def _glob_csvs(directory: Path) -> List[Path]:
    """List every CSV file directly inside directory.

    No naming convention is required — a file may contain rows for any mix of
    regions/sectors, since filtering happens later on the row data itself
    (see filter_model_data), not by which files are selected.
    """
    if not directory.exists():
        return []
    return sorted(directory.glob("*.csv"))


@lru_cache(maxsize=None)
def _scannable(path) -> bool:
    """Check a CSV can be read the way filter_model_data reads it.

    Checks that every column any consumer needs is present. Also check that 
    the file actually parses: reading the first SCAN_PROBE_ROWS rows surfaces
    ragged rows and unbalanced quotes near the top of a file, which pass a 
    header check and then raise ComputeError mid-load. Damage deeper in a file 
    is caught at load time instead (see filter_model_data).

    Cached per path: one run asks the same question from collect_base_files,
    collect_update_files, _coverage_data, _excluded_region_branches and
    filter_model_data.
    """
    try:
        head = pl.read_csv(
            path,
            infer_schema_length=0,
            null_values=[""],
            n_rows=SCAN_PROBE_ROWS,
        )
    except Exception:
        return False
    return all(c in head.columns for c in REQUIRED_COLUMNS)


def clear_scannable_cache() -> None:
    """Forget cached per-file results. Call after input CSVs change on disk."""
    _scannable.cache_clear()


def _build_rows(summary: dict) -> List[dict]:
    """Build rows for the per-entry file health table."""
    rows = []
    for name, info in summary.items():
        rows.append({
            COL_NAME: name,
            COL_FOUND: info["found"],
            COL_UNREADABLE: len(info["unreadable"]),
            "unreadable_paths": info["unreadable"],
            "missing": info.get("missing", False),
            "path": info.get("path", name),
        })
    return rows


MAX_LISTED_UNREADABLE = 3


def _print_terse_health(rows: List[dict]):
    """Print one folder-level row per directory (found/unreadable counts), with a
    capped list of unreadable filenames underneath any folder that has them."""
    by_dir = {}
    for row in rows:
        d = row.get("dir") or row[COL_NAME]
        group = by_dir.setdefault(d, {"found": 0, "unreadable": 0, "missing": [], "paths": []})
        group["found"] += row[COL_FOUND]
        group["unreadable"] += row[COL_UNREADABLE]
        if row.get("missing"):
            group["missing"].append(row[COL_NAME])
        group["paths"].extend(row["unreadable_paths"])

    ordered_dirs = list(by_dir)  # base model first, then update folders in the order provided

    # A folder row carries the counts for everything collected under it; each entry that
    # was requested but does not exist gets its own indented row beneath it, so the name
    # of the offending folder is visible without switching to verbose. Counts stay on the
    # parent, since a missing folder contributed nothing to them.
    table = []
    for d in ordered_dirs:
        group = by_dir[d]
        table.append([d, group["found"], group["unreadable"], None])
        for name in group["missing"]:
            table.append([NEST_INDENT + name, None, None, MISSING_MARK])
    lines = tabulate(
        table,
        headers=["Folder", COL_FOUND, COL_UNREADABLE, COL_MISSING],
        tablefmt="simple",
        missingval="",
        preserve_whitespace=True,
        colalign=("left", "right", "right", "right"),
    ).splitlines()
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
    # Nest entries that belong to a registered folder one level under that folder's
    # header; the base model has no parent folder, so it stays flush left.
    # preserve_whitespace keeps the leading indent, which tabulate strips by default.
    indents = [NEST_INDENT if row.get("dir") else "" for row in rows]
    # Blank for folders that were found, so a missing folder is the only thing in the 
    # column and reads at a glance.
    table = [
        [
            ind + r[COL_NAME],
            r[COL_FOUND],
            r[COL_UNREADABLE],
            MISSING_MARK if r.get("missing") else "",
        ]
        for ind, r in zip(indents, rows)
    ]
    lines = tabulate(
        table,
        headers=[COL_NAME, COL_FOUND, COL_UNREADABLE, COL_MISSING],
        tablefmt="plain",
        preserve_whitespace=True,
        colalign=("left", "right", "right", "right"),
    ).splitlines()
    header, data_lines = lines[0], lines[1:]

    print(f"      {header}")

    by_dir = {}
    for i, row in enumerate(rows):
        d = row.get("dir")
        by_dir.setdefault(d, []).append(i)

    for d, indices in by_dir.items():
        if d:
            print(f"      {d}/")
        for i in indices:
            print(f"      {data_lines[i]}")
            for p in rows[i]["unreadable_paths"]:
                print(f"      {indents[i]}    ! unreadable: {p.name}")


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
    
    print("\n    File health:\n")
    if verbose:
        _print_verbose_health(rows)
    else:
        _print_terse_health(rows)

    print(f"\n")


def collect_base_files(model_path: str, base_model: str) -> Tuple[List[str], List[dict]]:
    """Collect base model CSV paths and report which are readable.

    Every CSV in the base model directory is collected for reading; whether a file
    can actually be parsed (has the expected columns) is checked here, separately
    from whether requested regions/sectors have any data (see print_coverage_matrix).
    """
    base_dir = Path(model_path) / base_model
    found_paths = _glob_csvs(base_dir)
    unreadable = [p for p in found_paths if not _scannable(str(p))]

    summary = {base_dir.as_posix(): {
        "found": len(found_paths), 
        "unreadable": unreadable,
        "missing": not base_dir.is_dir(),
        "path": base_dir.as_posix(),
    }}
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
            summary[summary_key] = {
                "found": len(found_paths),
                "unreadable": unreadable,
                "missing": not sub_dir.is_dir(),
                "path": sub_dir.as_posix(),
            }

            found.extend(str(p) for p in found_paths)

    rows = _build_rows(summary)
    for row in rows:
        d = key_to_dir.get(row[COL_NAME])
        row["dir"] = d
        if d:
            row[COL_NAME] = Path(row[COL_NAME]).name
    return found, rows


def _coverage_data(paths: Iterable[str], sector_paths: set) -> Tuple[set, set]:
    """Scan readable paths and return (pairs, region_only): the set of (Region, Sector)
    combinations with both fields non-null, and the set of regions that have data at all.

    Only files in sector_paths contribute Region/Sector pairs, so the matrix answers "was
    this sector actually defined for this region" rather than the weaker "is this sector
    named anywhere". Every other file contributes to the "(no sector)" row instead: a
    cross-cutting file (DCC/DIC/FIC, market share limits, a policy) repeats the Sector of
    whatever node it targets, so counting it would report a sector as covered even when
    the folder that defines it never loaded.

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
            df = pl.scan_csv(path, infer_schema_length=0, null_values=[""]).select(COL.region, COL.sector).collect()
        except Exception:
            continue
        if path in sector_paths:
            non_null = df.drop_nulls()
            pairs.update(zip(non_null[COL.region].to_list(), non_null[COL.sector].to_list()))
            no_sector = df.filter(pl.col(COL.sector).is_null() & pl.col(COL.region).is_not_null())
            region_only.update(no_sector[COL.region].to_list())
        else:
            # Cross-cutting file: every region it touches counts as "has data",
            # none of its sectors count as defined.
            regions = df.filter(pl.col(COL.region).is_not_null())
            region_only.update(regions[COL.region].to_list())
    return pairs, region_only


def print_coverage_matrix(
    paths: Iterable[str],
    region_list: Iterable[str],
    sector_list: Iterable[str],
    sector_paths: Iterable[str] = None,
):
    """Print a region x sector matrix of which requested combinations have at least one row.

    Built from the actual Region/Sector column values found in the sector-defining
    files, not from file or folder names — a single file may contain rows for any mix
    of regions/sectors, so coverage can only be determined from the data.

    sector_paths names the files that are allowed to define a sector; pass None to treat
    every path as sector-defining (the pre-filtering behaviour).
    """
    regions = list(region_list)
    sectors = list(sector_list)
    if not regions or not sectors:
        return

    paths = list(paths)
    sector_paths = set(paths) if sector_paths is None else set(sector_paths)
    pairs, region_only = _coverage_data(paths, sector_paths)
    # Blank for a combination with nothing found
    table = [["(no sector)"] + ["x" if region in region_only else "" for region in regions]]
    table.extend(
        [sector] + ["x" if (region, sector) in pairs else "" for region in regions]
        for sector in sectors
    )
    print("\n    Region x Sector coverage (x = sector defined, blank = not defined):\n")
    print("      Only the folders listed as sector files count — a sector named by a")
    print("      cross-cutting file (DCC/DIC/FIC, limits, policies) is not defined by it.")
    print("      '(no sector)' row: does the region have any data at all, ignoring sector\n")
    lines = tabulate(table, headers=["Sector"] + regions, tablefmt="simple").splitlines()
    for line in lines:
        print(f"      {line}")

    covered_sectors = sum(1 for sector in sectors if any((region, sector) in pairs for region in regions))
    print(f"\n      Summary: {covered_sectors} of {len(sectors)} requested sectors are defined in at least one region.\n")


def collect_files(
    model_path: str,
    base_model: str,
    region_list: Iterable[str],
    sector_list: Iterable[str],
    update_files: Mapping[str, Iterable[str]],
    sector_folders: Iterable[str] = None,
    verbose: bool = False,
) -> Tuple[List[str], List[str], List[dict]]:
    """
    Collect base and update paths, print a summary, and return found lists plus health rows.
    Returns (base_found, update_found, health_rows); health_rows can be re-printed later
    via print_file_health(health_rows, verbose=True) for the full per-entry breakdown.

    sector_folders names the update entries that define sectors (e.g., the notebook's
    sector_req). Only files collected from those folders count toward the coverage
    matrix; everything else is cross-cutting and feeds the "(no sector)" row. A folder
    named here that does not exist contributes no files, so a typo excludes its sector
    from the matrix instead of leaving it falsely covered. Pass None to treat every
    collected path as sector-defining.
    """
    base_found, base_rows = collect_base_files(model_path, base_model)
    update_found, update_rows = collect_update_files(update_files)
    health_rows = base_rows + update_rows

    if sector_folders is None:
        sector_paths = None
    else:
        # Match on the immediate parent directory name: collect_update_files globbed
        # each entry's files straight out of <dir>/<entry>.
        wanted = {str(name) for name in sector_folders}
        sector_paths = set(base_found)
        sector_paths.update(p for p in update_found if Path(p).parent.name in wanted)

    print_coverage_matrix(base_found + update_found, region_list, sector_list, sector_paths)
    print_file_health(health_rows, verbose=verbose)
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
                pl.scan_csv(path, infer_schema_length=0, null_values=[""])
                .filter(pl.col(COL.parameter) == PARAM.competition_type)
                .select(COL.branch, COL.region, COL.value)
                .collect()
            )
        except Exception:
            continue
        region_rows = df.filter(pl.col(COL.value) == REGION_COMPETITION_TYPE)
        region_of_branch.update(zip(region_rows[COL.branch].to_list(), region_rows[COL.region].to_list()))

    return {branch for branch, region in region_of_branch.items() if region not in regions}


def _collect_frames(lazy_frames: List) -> pl.DataFrame:
    """Concatenate and materialise the per-file lazy frames."""
    return (
        pl.concat(lazy_frames, how="diagonal_relaxed")
        .collect()
        .unique(maintain_order=True)
    )


def _partition_loadable(lazy_frames: List, frame_paths: List[str]) -> Tuple[List, List[Tuple[str, Exception]]]:
    """Split frames into those that collect cleanly and those that raise.

    Only called after a batch collect has already failed, so the cost of collecting
    each file separately is paid once, on the error path.
    """
    good, bad = [], []
    for lf, path in zip(lazy_frames, frame_paths):
        try:
            lf.collect()
        except Exception as exc:
            bad.append((path, exc))
        else:
            good.append(lf)
    return good, bad


def _report_unloadable(unloadable: List[Tuple[str, Exception]]) -> None:
    """Name the files that could not be parsed, and why, without raising."""
    if not unloadable:
        return
    print(f"\n    ! {len(unloadable)} file(s) passed the readability check but failed to parse.")
    print("      They were EXCLUDED from the model:\n")
    for path, exc in unloadable:
        reason = " ".join(str(exc).split())
        if len(reason) > 140:
            reason = reason[:137] + "..."
        print(f"        ! {Path(path).as_posix()}")
        print(f"            {type(exc).__name__}: {reason}")


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
    after filtering. Files that failed the readability check are skipped — already
    reported by print_file_health/collect_files at collection time. A file that passes
    that check but still fails to parse (damage below SCAN_PROBE_ROWS) is dropped here
    and named on stdout, rather than aborting the whole load with a bare ComputeError.

    See _excluded_region_branches for the one exception to per-column independence: a
    structural parent's unconditional request to an excluded region node is also dropped.
    """
    year_strs = [str(y) for y in year_list]
    sectors = list(sector_list) if sector_list else None
    regions = list(region_list) if region_list else None

    excluded_region_branches = list(_excluded_region_branches(paths, regions)) if regions else []

    lazy_frames = []
    frame_paths = []
    for path in paths:
        if not _scannable(path):
            continue
        lf = pl.scan_csv(path, infer_schema_length=0, null_values=[""])
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
        frame_paths.append(path)

    if not lazy_frames:
        return pd.DataFrame(columns=list(col_list) + [COL.year, COL.value])

    try:
        collected = _collect_frames(lazy_frames)
    except Exception:
        # One bad file stops the whole concat and polars cannot say which. Retry
        # file by file to attribute the failure, drop the offenders, and carry on
        # with the rest rather than failing model instantiation outright.
        lazy_frames, unloadable = _partition_loadable(lazy_frames, frame_paths)
        _report_unloadable(unloadable)
        if not lazy_frames:
            return pd.DataFrame(columns=list(col_list) + [COL.year, COL.value])
        collected = _collect_frames(lazy_frames)

    df = collected.to_pandas().replace({np.nan: None, "": None})

    meta_cols = [c for c in df.columns if c not in (COL.year, COL.value) and c in col_list]
    return df[meta_cols + [COL.year, COL.value]]
