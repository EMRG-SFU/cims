"""
Process reference policy CSVs:
  A) Split AT (Atlantic aggregate) files into province files (NB, NS, PE, NL)
  B) Flatten wide 5-year columns (2000, 2005, … 2050) to annual long format

Input:  BASE_PATH / model_inputs/policies/reference/
Output: BASE_PATH / model_inputs/policies/reference_annual/

Every policy CSV has a navigation-hint row as line 1 (e.g. ",<-- Navigate …").
This row is stripped before flattening so process_file sees the real header.

AT rows are duplicated four times with CIMS.CAN.AT.… → CIMS.CAN.{prov}.…
and the Region field replaced accordingly.

5-year values are linearly interpolated to annual 2000–2050.
Rows with no numeric year values (e.g. market_share_class placeholders) are
passed through with blank Year/Value (flatten_fixed_data default behaviour).
"""

import sys
import csv
import tempfile
import importlib.util
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

_flat_spec = importlib.util.spec_from_file_location(
    'flatten_fixed_data',
    _PIPELINE_ROOT / 'utils' / 'flatten_fixed_data.py',
)
_flat_mod = importlib.util.module_from_spec(_flat_spec)
_flat_spec.loader.exec_module(_flat_mod)

from utils.controls_conversions import BASE_PATH

# ── configuration ──────────────────────────────────────────────────────────────
INPUT_DIR  = BASE_PATH / 'model_inputs/policies/reference'
OUTPUT_DIR = BASE_PATH / 'model_inputs/policies/reference_annual'

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
                r[branch_idx] = r[branch_idx].replace(f'CIMS.CAN.AT.', f'CIMS.CAN.{prov}.')
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

            is_at = csv_file.stem.endswith('_AT')

            if is_at:
                province_data = _split_at(header, data_rows)
                targets = {
                    csv_file.stem.removesuffix('_AT') + '_' + prov: rows
                    for prov, rows in province_data.items()
                }
            else:
                targets = {csv_file.stem: data_rows}

            for out_stem, rows in targets.items():
                tmp = _write_temp(header, rows)
                out_path = out_sub / f'{out_stem}.csv'
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
                    total_out += 1
                    label = f'  {csv_file.name} → {out_stem}.csv'
                    print(label)
                except Exception as exc:
                    print(f'  ERROR {csv_file.name} → {out_stem}.csv: {exc}')
                finally:
                    tmp.unlink(missing_ok=True)

    print()
    print(f'Input files:  {total_in}')
    print(f'Output files: {total_out}  '
          f'(AT files × 4 = {sum(1 for f in input_dir.rglob("*_AT.csv"))} × 4 extra)')


def main() -> None:
    print('=' * 60)
    print('FLATTEN REFERENCE POLICY FILES')
    print('=' * 60)
    print(f'Input:  {INPUT_DIR}')
    print(f'Output: {OUTPUT_DIR}')
    print(f'Years:  {TARGET_START}–{TARGET_END} (annual, linear interpolation)')
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process_all(INPUT_DIR, OUTPUT_DIR)

    print()
    print('Done.')


if __name__ == '__main__':
    main()
