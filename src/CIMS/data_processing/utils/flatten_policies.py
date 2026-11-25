"""
Build the long, annual policy files the model reads from policy source files.

Input:  paths.RAW / policies/<source folder>/*.csv
        Wide files with 5-year columns (2000, 2005, ... 2050), or files already
        in long format (Year and Value columns). Source folders keep the names
        they were authored with (for example Off_OBPS_Fed), or already use the
        output name (for example carbon-tax_ref).
Output: paths.POLICIES / <policy folder>/<policy folder>_<region>.csv
        Long files with Year and Value columns, one row per year, in
        kebab-snake names (for example obps_federal_off/obps_federal_off_ab.csv).

For each file, the script:
  A) drops the navigation-hint row (",<-- Navigate ...") if present;
  B) splits Atlantic (AT) files into NB, NS, PE, and NL files, substituting the
     province in the Branch and Region columns;
  C) writes one row per year from the first year with a value to YEAR_MAX:
     numeric values are interpolated linearly between the given years, literal
     values (such as "None", which turns a policy off) carry forward until the
     next given year, and the last given value holds to YEAR_MAX. Years before
     the first given value get no row, so a policy that starts in 2030 stays
     unset before 2030;
  D) passes rows with no year values through with a blank Year and Value;
  E) finalises currency units with utils/currency.py (for example "$/tCO2e"
     becomes "2020_CAD/tCO2e", and blank fic and dic_min units get the
     untagged currency).

Files already in long format (with Year and Value columns) are passed through
as they are, apart from the AT split (B) and the currency step (E).

Reconciles two rewrites (plan, phase 0 step 4): the wide-file handling and
naming come from emh_mmcf_2026; the long-file pass-through and the
utils/currency.py step from data-processing-calibration.
"""

import csv
import re
from pathlib import Path

from CIMS.data_processing.common import paths
from CIMS.data_processing.utils.currency import finalise_currency_csv
from CIMS.data_processing.utils.flatten_fixed_data import BLANK_LITERAL_SENTINELS

# ── configuration ──────────────────────────────────────────────────────────────
INPUT_DIR  = paths.RAW / 'policies'
OUTPUT_DIR = paths.POLICIES

AT_PROVINCES = ['NB', 'NS', 'PE', 'NL']

YEAR_MIN = 2000
YEAR_MAX = 2050

# Source folder name -> output policy name. Folders not listed here are named
# by policy_name() below.
NAME_MAP = {
    'Off_OBPS_Fed': 'obps_federal_off',
    'Off_carbon tax': 'carbon-tax_off',
}

# Whole components that are spelled out in output names.
COMPONENT_MAP = {
    'fed': 'federal',
    'prov': 'provincial',
}


# ── naming ─────────────────────────────────────────────────────────────────────

def policy_name(folder: str) -> str:
    """
    Return the kebab-snake output name for a source folder.

    Hyphens join words within a component and underscores separate components.
    A leading "ref_" is dropped and a leading "Off_" becomes a trailing "_off".
    """
    if folder in NAME_MAP:
        return NAME_MAP[folder]

    suffix = ''
    if folder.lower().startswith('off_'):
        folder, suffix = folder[4:], '_off'
    elif folder.lower().startswith('ref_'):
        folder = folder[4:]

    components = []
    for component in folder.split('_'):
        component = re.sub(r'\s+', '-', component.strip().lower())
        components.append(COMPONENT_MAP.get(component, component))
    return '_'.join(c for c in components if c) + suffix


# ── reading ────────────────────────────────────────────────────────────────────

def _read_csv_skip_nav(path: Path) -> tuple[list[str], list[list[str]]]:
    """
    Read a policy CSV, dropping the leading navigation-hint row.

    The hint row is identified by any cell containing '<--'. After skipping it,
    the first remaining row is the real header.
    """
    rows: list[list[str]] = []
    for encoding in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            with path.open(newline='', encoding=encoding) as f:
                rows = list(csv.reader(f))
            break
        except UnicodeDecodeError:
            continue

    if rows and any('<--' in cell for cell in rows[0]):
        rows = rows[1:]

    if not rows:
        return [], []

    return rows[0], rows[1:]


def _year_columns(header: list[str]) -> list[tuple[int, int]]:
    """Return (year, column index) for header cells that are years in range."""
    years = []
    for idx, cell in enumerate(header):
        cell = cell.strip()
        if cell.isdigit() and YEAR_MIN <= int(cell) <= YEAR_MAX:
            years.append((int(cell), idx))
    return years


def _is_long(header: list[str]) -> bool:
    norm = {h.strip().lower() for h in header}
    return {'year', 'value'} <= norm


# ── transforming ───────────────────────────────────────────────────────────────

def _split_at(header: list[str], rows: list[list[str]]) -> dict[str, list[list[str]]]:
    """
    Produce one row-list per Atlantic province by substituting AT → province.

    Replacements are made in the Branch and Region columns only.
    """
    norm = [h.strip().lower() for h in header]
    branch_idx = next((i for i, h in enumerate(norm) if h == 'branch'), None)
    region_idx = next((i for i, h in enumerate(norm) if h == 'region'), None)

    result: dict[str, list[list[str]]] = {p: [] for p in AT_PROVINCES}
    for row in rows:
        for prov in AT_PROVINCES:
            r = list(row)
            if branch_idx is not None and branch_idx < len(r):
                b = r[branch_idx]
                if b == 'CIMS.CAN.AT':
                    r[branch_idx] = f'CIMS.CAN.{prov}'
                else:
                    r[branch_idx] = b.replace('CIMS.CAN.AT.', f'CIMS.CAN.{prov}.')
            if region_idx is not None and region_idx < len(r):
                if r[region_idx].strip() == 'AT':
                    r[region_idx] = prov
            result[prov].append(r)
    return result


def _as_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _annual(points: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """
    Expand (year, value) points to one value per year, first point to YEAR_MAX.

    Linear between two numeric points; otherwise the earlier value carries forward.
    """
    out = []
    for (y0, v0), (y1, v1) in zip(points, points[1:] + [(YEAR_MAX + 1, None)]):
        n0 = _as_number(v0)
        n1 = _as_number(v1) if v1 is not None else None
        for year in range(y0, min(y1, YEAR_MAX + 1)):
            if year == y0 or n0 is None or n1 is None:
                out.append((year, v0))
            else:
                value = n0 + (n1 - n0) * (year - y0) / (y1 - y0)
                out.append((year, repr(round(value, 10))))
    return out


def flatten(header: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Convert wide rows to long annual rows. Drops the Comments column."""
    year_cols = _year_columns(header)
    year_idx = {idx for _, idx in year_cols}
    keep = [i for i, h in enumerate(header)
            if i not in year_idx and h.strip().lower() != 'comments']
    out_header = [header[i].strip() for i in keep] + ['Year', 'Value']

    out_rows = []
    for row in rows:
        row = [cell.strip() for cell in row] + [''] * (len(header) - len(row))
        if not any(row):
            continue
        fixed = [row[i] for i in keep]
        points = sorted((year, row[idx]) for year, idx in year_cols
                        if row[idx] != '' and row[idx].lower() not in BLANK_LITERAL_SENTINELS)
        if not points:
            out_rows.append(fixed + ['', ''])
            continue
        for year, value in _annual(points):
            out_rows.append(fixed + [str(year), value])
    return out_header, out_rows


# ── main processing ────────────────────────────────────────────────────────────

def process_all(input_dir: Path, output_dir: Path) -> None:
    total_in = total_out = 0

    for subfolder in sorted(input_dir.iterdir()):
        if not subfolder.is_dir():
            continue

        name = policy_name(subfolder.name)
        clash = [d.name for d in input_dir.iterdir()
                 if d.is_dir() and d != subfolder and policy_name(d.name) == name]
        if clash:
            raise ValueError(f'{subfolder.name} and {clash[0]} both map to policy name {name!r}')
        out_sub = output_dir / name

        for csv_file in sorted(subfolder.glob('*.csv')):
            header, data_rows = _read_csv_skip_nav(csv_file)
            if not header:
                print(f'  SKIP {csv_file.name}: unreadable')
                continue
            long_format = _is_long(header)
            if not long_format and not _year_columns(header):
                print(f'  SKIP {csv_file.name}: no year columns')
                continue
            total_in += 1

            region = csv_file.stem.rsplit('_', 1)[-1]
            if region.upper() == 'AT':
                targets = _split_at(header, data_rows)
            else:
                targets = {region: data_rows}

            out_sub.mkdir(parents=True, exist_ok=True)
            for reg, rows in targets.items():
                if long_format:
                    out_header = [h.strip() for h in header]
                    out_rows = [[cell.strip() for cell in row] for row in rows if any(cell.strip() for cell in row)]
                else:
                    out_header, out_rows = flatten(header, rows)
                out_path = out_sub / f'{name}_{reg.lower()}.csv'
                with out_path.open('w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f, lineterminator='\n')
                    w.writerow(out_header)
                    w.writerows(out_rows)
                finalise_currency_csv(out_path)
                total_out += 1
                note = ', long format' if long_format else ''
                print(f'  {subfolder.name}/{csv_file.name} → {name}/{out_path.name} ({len(out_rows)} rows{note})')

    print()
    print(f'Input files:  {total_in}')
    print(f'Output files: {total_out}')


def main() -> None:
    print('=' * 60)
    print('FLATTEN POLICY FILES')
    print('=' * 60)
    print(f'Input:  {INPUT_DIR}')
    print(f'Output: {OUTPUT_DIR}')
    print(f'Years:  first given year to {YEAR_MAX} (annual)')
    print()

    if not INPUT_DIR.is_dir():
        print(f'No policy source folder at {INPUT_DIR}; nothing to do.')
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process_all(INPUT_DIR, OUTPUT_DIR)

    print()
    print('Done.')


if __name__ == '__main__':
    main()
