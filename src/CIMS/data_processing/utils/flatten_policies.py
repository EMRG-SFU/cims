"""
Build model-ready policy CSVs from policy source files:
  A) Split AT (Atlantic aggregate) files into province files (NB, NS, PE, NL)
  B) Flatten wide 5-year columns (2000, 2005, … 2050) to annual long format;
     files already in long format (Year and Value columns) pass through as-is
  C) Finalise currency units (utils/currency.py), e.g. '$/tCO2e' -> '2020_CAD/tCO2e'

Input:  BASE_PATH / raw_data/policies/<policy>/<policy>_<region>.csv
Output: BASE_PATH / model_inputs/policies/<policy>/<policy>_<region>.csv

Both sides use one folder per policy (e.g. carbon-tax_ref/, obps_federal/), the
layout scenarios/Reference.py reads from data/model_inputs/policies/. Output file
names use a lower-case region suffix (carbon-tax_ref_bc.csv); an AT file
(<policy>_AT.csv or _at.csv) becomes <policy>_nb.csv, _ns.csv, _pe.csv and _nl.csv.
Existing output files with the same name are overwritten.

A policy CSV may have a navigation-hint row as line 1 (e.g. ",<-- Navigate …").
This row is stripped before flattening so process_file sees the real header.

AT rows are duplicated four times with CIMS.CAN.AT.… → CIMS.CAN.{prov}.…
and the Region field replaced accordingly.

5-year values are linearly interpolated to annual 2000–2050.
Rows with no numeric year values (e.g. market_share_class placeholders) are
passed through with blank Year/Value (flatten_fixed_data default behaviour).
"""

import csv
import tempfile
from pathlib import Path

import CIMS.data_processing.utils.flatten_fixed_data as _flat_mod

from CIMS.data_processing.utils.controls_conversions import BASE_PATH
from CIMS.data_processing.utils.currency import finalise_currency_csv

# ── configuration ──────────────────────────────────────────────────────────────
INPUT_DIR  = BASE_PATH / 'raw_data/policies'
OUTPUT_DIR = BASE_PATH / 'model_inputs/policies'

AT_PROVINCES = ['NB', 'NS', 'PE', 'NL']

YEAR_MIN     = 2000
YEAR_MAX     = 2050
TARGET_START = 2000
TARGET_END   = 2050
TARGET_STEP  = 1


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_csv_skip_nav(path: Path) -> tuple[list[str], list[list[str]]]:
    """
    Read a policy CSV, dropping the leading navigation-hint row.

    The hint row is identified by any cell containing '<--'. After skipping it,
    the first remaining row is the real header.
    """
    for encoding in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            with path.open(newline='', encoding=encoding) as f:
                rows = list(csv.reader(f))
            break
        except UnicodeDecodeError:
            continue

    if not rows:
        return [], []

    if any('<--' in cell for cell in rows[0]):
        rows = rows[1:]

    if not rows:
        return [], []

    return rows[0], rows[1:]


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


def _write_temp(header: list[str], rows: list[list[str]]) -> Path:
    """Write header + rows to a NamedTemporaryFile and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False, encoding='utf-8', newline=''
    )
    w = csv.writer(tmp)
    w.writerow(header)
    w.writerows(rows)
    tmp.close()
    return Path(tmp.name)


# ── main processing ────────────────────────────────────────────────────────────

def _is_long(header: list[str]) -> bool:
    """True when the file is already annual long format (Year and Value columns)."""
    norm = {h.strip().lower() for h in header}
    return {'year', 'value'} <= norm


def _write_rows(header: list[str], rows: list[list[str]], out_path: Path) -> None:
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow([h.strip() for h in header])
        w.writerows(rows)


def process_all(input_dir: Path, output_dir: Path) -> None:
    total_in = total_out = 0

    for subfolder in sorted(input_dir.iterdir()):
        if not subfolder.is_dir():
            continue

        out_sub = output_dir / subfolder.name
        out_sub.mkdir(parents=True, exist_ok=True)

        for csv_file in sorted(subfolder.glob('*.csv')):
            total_in += 1
            header, data_rows = _read_csv_skip_nav(csv_file)
            if not header:
                print(f'  SKIP {csv_file.name} — unreadable')
                continue

            stem = csv_file.stem
            is_at = stem.lower().endswith('_at')

            if is_at:
                province_data = _split_at(header, data_rows)
                targets = {
                    f'{stem[:-3]}_{prov.lower()}': rows
                    for prov, rows in province_data.items()
                }
            else:
                targets = {stem: data_rows}

            long_format = _is_long(header)
            for out_stem, rows in targets.items():
                out_path = out_sub / f'{out_stem}.csv'
                try:
                    if long_format:
                        _write_rows(header, rows, out_path)
                    else:
                        tmp = _write_temp(header, rows)
                        try:
                            _flat_mod.process_file(
                                input_path=tmp,
                                output_path=out_path,
                                year_min=YEAR_MIN,
                                year_max=YEAR_MAX,
                                target_start=TARGET_START,
                                target_end=TARGET_END,
                                target_step=TARGET_STEP,
                            )
                        finally:
                            tmp.unlink(missing_ok=True)
                    finalise_currency_csv(out_path)
                    total_out += 1
                    print(f'  {subfolder.name}/{csv_file.name} → {out_stem}.csv'
                          f'{" (long format)" if long_format else ""}')
                except Exception as exc:
                    print(f'  ERROR {csv_file.name} → {out_stem}.csv: {exc}')

    print()
    print(f'Input files:  {total_in}')
    print(f'Output files: {total_out}')


def main() -> None:
    print('=' * 60)
    print('BUILD POLICY FILES')
    print('=' * 60)
    print(f'Input:  {INPUT_DIR}')
    print(f'Output: {OUTPUT_DIR}')
    print(f'Years:  {TARGET_START}–{TARGET_END} (annual, linear interpolation)')
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
