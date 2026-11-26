"""
CIMS root and regions — Model Inputs

Writes the top of the model tree: the root node (CIMS) with its model-wide settings
to model/root/, and the region nodes (CAN, RoW, and each province and territory)
to model/region/. Flattens the JCIMS CIMS_base fixed-data CSVs from wide year-column format to long
format, then appends Population and GDP attribute rows (from the pop_gdp
pipeline) to each provincial/territorial file.

CIMS, CAN, and RoW files are flattened only — no attribute rows added.

The root's requests to CIMS.CAN and CIMS.RoW go in those regions' files, the way a
region's file carries its parent's request to it (CIMS.CAN -> CIMS.CAN.BC in
region_bc.csv). The model reads model/root/ as its base and model/region/ as scenario
files, so the base holds the root alone and every request sits beside the node it asks
for (which the excluded-region filter needs: readers/helpers.filter_model_data).

Inputs:
    extracted/legacy/fixed-data/cims_base/CIMS_base_{suffix}.csv
    processed/gov-can/statcan/pop-gdp.csv  (via pop_gdp.main())

Output:
    model/root/root_cims.csv, and model/region/region_{suffix}.csv
    Columns: Branch, Type, Region, Sector, Service, Technology,
             Parameter, Context, Sub_Context, Target, Source, Unit,
             Year, Value
"""

import tempfile
from pathlib import Path

import polars as pl

# ── path setup ──────────────────────────────────────────────────────────────────
from CIMS.data_processing.utils.flatten_fixed_data import read_fixed_data_file

import CIMS.data_processing.source.stats_can.pop_gdp as _pg_mod

from CIMS.data_processing.common import paths
from CIMS.data_processing.utils.controls_conversions import PROJECTION_END
from CIMS.data_processing.utils.currency import write_model_csv
from CIMS.data_processing.utils.collapse_constant_years import collapse_constant_years

# ── configuration ───────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = paths.FIXED_DATA / 'cims_base'
ROOT_DIR        = paths.MODEL_ROOT     # the root node (CIMS) and its model-wide settings
REGION_DIR      = paths.MODEL_REGION   # the region nodes: CAN, RoW, and the provinces and territories

PROVINCES    = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']
FLATTEN_ONLY = ['CIMS', 'CAN', 'RoW']

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]


# ── helpers ─────────────────────────────────────────────────────────────────────

def _clean_csv(src: Path, dst: Path) -> None:
    """Write src to dst, skipping any rows before the Branch header line.

    Some CIMS_base CSVs have a navigation/comment row as the first line
    (e.g. ",<-- Navigate by typing to search,..."). process_file requires
    the header row to be first.
    """
    for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            text = src.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith('Branch')),
        0,
    )
    dst.write_text('\n'.join(lines[header_idx:]), encoding='utf-8')


def _flatten_fixed(stem: str) -> pl.DataFrame:
    """Flatten one CIMS_base CSV to long format.

    Also strips any pre-existing attribute Population/GDP rows so they can
    be replaced with fresh pipeline data.
    """
    input_path = FIXED_INPUT_DIR / f'{stem}.csv'
    with tempfile.TemporaryDirectory() as tmp:
        cleaned = Path(tmp) / f'{stem}.csv'
        _clean_csv(input_path, cleaned)
        df = read_fixed_data_file(cleaned, year_max=PROJECTION_END)
    df = df.filter(
        ~(
            (pl.col('Parameter') == 'attribute') &
            pl.col('Context').is_in(['Population', 'GDP'])
        )
    )
    _currency = pl.col('Parameter') == 'currency'
    return df.with_columns(
        pl.when(_currency)
        .then(
            pl.col('Value').cast(pl.Float64, strict=False).cast(pl.Int64).cast(pl.String)
            + pl.lit('_') + pl.col('Unit')
        )
        .otherwise(pl.col('Value'))
        .alias('Value'),
        pl.when(_currency).then(pl.lit('')).otherwise(pl.col('Unit')).alias('Unit'),
        pl.when(pl.col('Source') == 'n/a').then(pl.lit('')).otherwise(pl.col('Source')).alias('Source'),
    )


def _build_attribute_rows(pop_gdp: pl.DataFrame, region: str) -> pl.DataFrame:
    """Build Population then GDP attribute rows for a province/territory."""
    data = (
        pop_gdp
        .filter(pl.col('region') == region)
        # Population before GDP, years ascending
        .sort(['variable', 'year'], descending=[True, False])
    )
    if len(data) == 0:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return data.select([
        pl.lit(f'CIMS.CAN.{region}').alias('Branch'),
        pl.lit('Region').alias('Type'),
        pl.lit(region).alias('Region'),
        pl.lit('').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('attribute').alias('Parameter'),
        pl.col('variable').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        pl.lit('').alias('Target'),
        pl.col('source').alias('Source'),
        pl.col('unit').alias('Unit'),
        pl.col('year').cast(pl.String).alias('Year'),
        pl.col('value').cast(pl.String).alias('Value'),
    ])


def _assemble_region(region: str, pop_gdp: pl.DataFrame) -> pl.DataFrame:
    """Flatten fixed data and append population/GDP attribute rows."""
    fixed = _flatten_fixed(f'cims_base_{region.lower()}')
    attrs = _build_attribute_rows(pop_gdp, region)
    return pl.concat([fixed.select(OUTPUT_COLS), attrs], how='diagonal_relaxed')


# ── entry point ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("Running pop_gdp pipeline...")
    pop_gdp = pl.from_pandas(_pg_mod.main())

    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    REGION_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFlattening CIMS / CAN / RoW files...")
    flat = {suffix: _flatten_fixed(f'cims_base_{suffix.lower()}').select(OUTPUT_COLS) for suffix in FLATTEN_ONLY}
    is_region_request = (pl.col('Branch') == 'CIMS') & (pl.col('Parameter') == 'service_request')
    root_requests = flat['CIMS'].filter(is_region_request)
    flat['CIMS'] = flat['CIMS'].filter(~is_region_request)
    for suffix in FLATTEN_ONLY:
        stem = f'cims_base_{suffix.lower()}'
        fixed = flat[suffix]
        if suffix != 'CIMS':
            # the root's request to this region comes first, as in the provincial files
            fixed = pl.concat([root_requests.filter(pl.col('Target') == f'CIMS.{suffix}'), fixed])
        fixed_out = collapse_constant_years(fixed)
        out = ROOT_DIR / 'root_cims.csv' if suffix == 'CIMS' else REGION_DIR / f'region_{paths.region_file_code(suffix)}.csv'
        write_model_csv(fixed_out, out)
        print(f"  {stem}: {len(fixed_out)} rows")
    moved = root_requests.filter(pl.col('Target').is_in([f'CIMS.{s}' for s in FLATTEN_ONLY if s != 'CIMS']))
    if len(moved) != len(root_requests):
        raise SystemExit(f'cims_base_cims.csv: a root service_request targets no CAN or RoW file: '
                         f'{sorted(set(root_requests["Target"]) - set(moved["Target"]))}')

    print("\nAssembling provincial files...")
    for region in PROVINCES:
        stem  = f'cims_base_{region.lower()}'
        df    = _assemble_region(region, pop_gdp)
        df    = collapse_constant_years(df)
        n_attr = len(df.filter(pl.col('Parameter') == 'attribute'))
        write_model_csv(df, REGION_DIR / f'region_{paths.region_file_code(region)}.csv')
        print(f"  {stem}: {len(df)} rows  ({n_attr} attribute)")

    total = len(FLATTEN_ONLY) + len(PROVINCES)
    print(f"\nDone. {total} files written to {ROOT_DIR} and {REGION_DIR}")


if __name__ == "__main__":
    main()
