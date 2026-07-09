"""
CIMS Base — Model Inputs

Flattens raw CIMS_base fixed-data CSVs from wide year-column format to long
format, then appends Population and GDP attribute rows (from the pop_gdp
pipeline) to each provincial/territorial file.

CIMS, CAN, and RoW files are flattened only — no attribute rows added.

Inputs:
    raw_data/fixed_data/cims_base/CIMS_base_{suffix}.csv
    processed_data/stats_can/pop_gdp.csv  (via pop_gdp.main())

Output:
    model_inputs/model/cims_base/CIMS_base_{suffix}.csv
    Columns: Branch, Type, Region, Sector, Service, Technology,
             Parameter, Context, Sub_Context, Target, Source, Unit,
             Year, Value
"""

import sys
import tempfile
import importlib.util
from pathlib import Path

import polars as pl

# ── path setup ──────────────────────────────────────────────────────────────────
_PIPELINE_ROOT = Path(__file__).parent.parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

_flatten_spec = importlib.util.spec_from_file_location(
    'flatten_fixed_data',
    _PIPELINE_ROOT / 'utils' / 'flatten_fixed_data.py',
)
_flatten_mod = importlib.util.module_from_spec(_flatten_spec)
_flatten_spec.loader.exec_module(_flatten_mod)

_pg_spec = importlib.util.spec_from_file_location(
    'pop_gdp',
    _PIPELINE_ROOT / 'source' / 'stats_can' / 'pop_gdp.py',
)
_pg_mod = importlib.util.module_from_spec(_pg_spec)
_pg_spec.loader.exec_module(_pg_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END

# ── configuration ───────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/cims_base'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/cims_base'

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
        tmp = Path(tmp)
        cleaned  = tmp / f'{stem}_clean.csv'
        out_file = tmp / f'{stem}.csv'
        _clean_csv(input_path, cleaned)
        _flatten_mod.process_file(
            input_path=cleaned,
            output_path=out_file,
            year_min=DATA_START,
            year_max=PROJECTION_END,
            target_start=DATA_START,
            target_end=PROJECTION_END,
            target_step=1,
        )
        df = pl.read_csv(out_file, infer_schema_length=0)
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFlattening CIMS / CAN / RoW files...")
    for suffix in FLATTEN_ONLY:
        stem = f'cims_base_{suffix.lower()}'
        fixed = _flatten_fixed(stem)
        fixed.select(OUTPUT_COLS).write_csv(OUTPUT_DIR / f'{stem}.csv')
        print(f"  {stem}: {len(fixed)} rows")

    print("\nAssembling provincial files...")
    for region in PROVINCES:
        stem  = f'cims_base_{region.lower()}'
        df    = _assemble_region(region, pop_gdp)
        n_attr = len(df.filter(pl.col('Parameter') == 'attribute'))
        df.write_csv(OUTPUT_DIR / f'{stem}.csv')
        print(f"  {stem}: {len(df)} rows  ({n_attr} attribute)")

    total = len(FLATTEN_ONLY) + len(PROVINCES)
    print(f"\nDone. {total} files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
