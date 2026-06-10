"""
Metal Smelting Pipeline — Model Inputs

Combines fixed structural parameters with pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/Metal Smelting/metal_smelting_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Encodes the full Metal Smelting hierarchy:
      Metal Smelting (Sector)
        ├── Primary Metals (Service, Fixed Ratio)
        │     ├── Aluminum (Service, Fixed Ratio)
        │     │     ├── Primary Aluminum
        │     │     │     ├── Alumina
        │     │     │     └── Electrolysis
        │     │     │           ├── Pre-bake Anode
        │     │     │           └── Soderberg Anode
        │     │     └── Recycle
        │     ├── Copper (Service, Fixed Ratio)
        │     │     ├── Metallurgy
        │     │     │     ├── Hydro Metallurgy
        │     │     │     └── Pyro Metallurgy
        │     │     │           ├── Smelting Process
        │     │     │           │     ├── Conventional Smelting
        │     │     │           │     └── Continuous Smelting
        │     │     │           └── Refine
        │     │     └── Melt Cast
        │     ├── Lead, Magnesium, Nickel, Zinc  (same structure as Copper)
        │     └── [Titanium — QC, NB, NL only]
        ├── Methane Blend, Steam, Oxygen  (shared utility services)
        ├── HVAC, Lighting, Ventilation  (building services)
        └── Compression, Conveyance, Direct Drive, Machine Drive,
            Pumping  (motor services)

Total production  (service_request at sector level)
    pipeline/source/activity/heavy_industry.py  (called directly via main())
    Variable: 'Metal Smelting'  (tonnes, per province, 2000–2100)

Subproduct shares  (service_request rows at Primary Metals service level)
    pipeline/source/activity/heavy_industry.py  (called directly via main())
    Variables: 'Metal Smelting.{subproduct}'  (% of tonnes)
    Subproducts: Aluminum, Copper, Lead, Magnesium, Nickel, Zinc
    Target format: CIMS.CAN.{region}.Metal Smelting.Primary Metals.{subproduct}

Energy price multipliers  (multiplier_price rows)
    pipeline/source/energy_prices/energy_price_multipliers.py  (called directly via main())

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value

Output order per region
-----------------------
1. service_request  — total production (from heavy_industry)
2. competition      — from fixed data (Sector level)
3. is_supply        — generated (TRUE)
4. multiplier_price — from energy_price_multipliers
5. service_request  — Primary Metals → each subproduct (from heavy_industry)
6. rest of fixed data
"""

import sys
import tempfile
import importlib.util
from pathlib import Path

import polars as pl

# ── path setup ─────────────────────────────────────────────────────────────────
_PIPELINE_ROOT = Path(__file__).parent.parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

_spec = importlib.util.spec_from_file_location(
    'flatten_fixed_data',
    _PIPELINE_ROOT / 'utils' / 'flatten_fixed_data.py',
)
_flatten_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_flatten_mod)

_hi_spec = importlib.util.spec_from_file_location(
    'heavy_industry',
    _PIPELINE_ROOT / 'source' / 'activity' / 'heavy_industry.py',
)
_heavy_industry_mod = importlib.util.module_from_spec(_hi_spec)
_hi_spec.loader.exec_module(_heavy_industry_mod)

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

from utils.controls_conversions import DATA_START, PROJECTION_END, LAST_DATA_YEAR

# ── configuration ──────────────────────────────────────────────────────────────
BASE_PATH       = Path('C:/cims/data')
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/Metal Smelting'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/metal smelting'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

SUBPRODUCTS = [
    'Aluminum',
    'Copper',
    'Lead',
    'Magnesium',
    'Nickel',
    'Zinc',
]

REGION_SPECIFIC_ENERGIES = {
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen', 'Renewable Natural Gas',
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed() -> pl.DataFrame:
    """Flatten all Metal Smelting fixed CSVs and return combined DataFrame."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _flatten_mod.main(
            input_folder=FIXED_INPUT_DIR,
            output_folder=tmp_path,
            year_min=DATA_START,
            year_max=LAST_DATA_YEAR["cer"],
            target_start=DATA_START,
            target_end=PROJECTION_END,
            target_step=1,
        )
        frames = [
            pl.read_csv(f, infer_schema_length=0)
            for f in sorted(tmp_path.rglob('*.csv'))
        ]
    return pl.concat(frames, how='diagonal_relaxed').cast(pl.String)


def _build_total_rows(heavy_ind: pl.DataFrame) -> pl.DataFrame:
    """
    service_request row for the Metal Smelting sector.
    Value = total Metal Smelting tonnes from heavy_industry.
    """
    df = heavy_ind.filter(pl.col('Variable') == 'Metal Smelting')
    return df.select([
        (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Metal Smelting')).alias('Branch'),
        pl.lit('Sector').alias('Type'),
        pl.col('Region'),
        pl.lit('Metal Smelting').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        pl.lit('').alias('Target'),
        pl.col('Source'),
        pl.lit('tonne').alias('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Value').cast(pl.String).alias('Value'),
    ])


def _build_is_supply_rows(regions: list[str]) -> pl.DataFrame:
    """Generate a single is_supply=TRUE row per region for the sector header."""
    return pl.DataFrame([
        {
            'Branch':      f'CIMS.CAN.{r}.Metal Smelting',
            'Type':        'Sector',
            'Region':      r,
            'Sector':      'Metal Smelting',
            'Service':     '',
            'Technology':  '',
            'Parameter':   'is_supply',
            'Context':     'TRUE',
            'Sub_Context': '',
            'Target':      '',
            'Source':      '',
            'Unit':        '',
            'Year':        '',
            'Value':       '',
        }
        for r in regions
    ])


def _build_price_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """multiplier_price rows for the Metal Smelting sector."""
    return (
        multipliers
        .filter(pl.col('Sector') == 'Metal Smelting')
        .select([
            (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Metal Smelting')).alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Metal Smelting').alias('Sector'),
            pl.lit('').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('multiplier_price').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            pl.when(pl.col('Energy').is_in(list(REGION_SPECIFIC_ENERGIES)))
            .then(pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.') + pl.col('Energy'))
            .otherwise(pl.lit('CIMS.Generic Fuels.') + pl.col('Energy'))
            .alias('Target'),
            pl.col('Source').alias('Source'),
            pl.lit('').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Multiplier').cast(pl.String).alias('Value'),
        ])
    )


def _build_subproduct_rows(heavy_ind: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows at the Primary Metals service level.
    One block per subproduct; value = ratio (% of tonnes) from heavy_industry.
    Target: CIMS.CAN.{region}.Metal Smelting.Primary Metals.{subproduct}
    """
    parts = []
    for sub in SUBPRODUCTS:
        df = heavy_ind.filter(pl.col('Variable') == f'Metal Smelting.{sub}')
        if df.is_empty():
            continue
        parts.append(df.select([
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit('.Metal Smelting.Primary Metals')).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.col('Region'),
            pl.lit('Metal Smelting').alias('Sector'),
            pl.lit('Primary Metals').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit(f'.Metal Smelting.Primary Metals.{sub}')).alias('Target'),
            pl.col('Source'),
            pl.lit('% of tonnes').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Value').cast(pl.String).alias('Value'),
        ]))
    if not parts:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return pl.concat(parts)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble Metal Smelting model inputs and write one CSV per region."""
    print('=' * 60)
    print('METAL SMELTING MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed()
    print(f'  Rows: {len(fixed):,}')

    print('Loading heavy industry activity data...')
    heavy_ind = _heavy_industry_mod.main()

    print('Building total production rows...')
    total_rows = _build_total_rows(heavy_ind)
    print(f'  Rows: {len(total_rows):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Building subproduct service request rows...')
    subprod_rows = _build_subproduct_rows(heavy_ind)
    print(f'  Rows: {len(subprod_rows):,}')

    print('Combining...')

    # Sector-level branch: ends with '.Metal Smelting' (no further sub-path)
    _sector_branch = pl.col('Branch').str.ends_with('.Metal Smelting')

    # Keep only competition from fixed data at sector level;
    # service_provide rows remain null (fixed data); activity data inserted as service_request.
    fixed_sector_competition = fixed.filter(
        _sector_branch & (pl.col('Parameter').fill_null('') == 'competition')
    )

    # Everything from fixed except sector-level competition (service_provide rows remain null).
    # This includes the full Primary Metals sub-tree (with all fixed splits),
    # Methane Blend, Steam, Oxygen, HVAC, Lighting, Ventilation, and all
    # motor services — but the sector-level service_request (linking Metal
    # Smelting → Primary Metals) is also retained here since it is structural.
    fixed_rest = fixed.filter(
        ~(_sector_branch & pl.col('Parameter').fill_null('').is_in(['competition']))
    )

    regions = total_rows['Region'].unique().sort().to_list()
    is_supply_rows = _build_is_supply_rows(regions)

    output = (
        pl.concat(
            [total_rows, fixed_sector_competition, is_supply_rows,
             price_rows, subprod_rows, fixed_rest],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'metal_smelting_{region}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Metal Smelting model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
