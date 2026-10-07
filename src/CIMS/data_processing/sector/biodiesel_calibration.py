"""
Extract biodiesel calibration data and save to CIMS-formatted CSV files.

Sources
-------
Emissions  (calibration_emissions_total from crosswalk; calibration_emissions_by_type from nir_to_cims)
    nir_crosswalk_tables_cims.py  → total tCO2e per biodiesel CIMS branch
                                    5-year intervals (2000–2020);
                                    abbreviation regions (AB, BC, …)
    nir_to_cims.py                → per-gas kt per biodiesel CIMS branch,
                                    annual resolution (2000–latest NIR year);
                                    summed to tCO2e using AR5 GWP100 factors;
                                    full province names mapped to abbreviations

Energy demand  (calibration_quantity_requested)
    cer_resd_demand.py            → energy demand in PJ by fuel and CIMS node;
                                    abbreviation regions

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""

from pathlib import Path

import polars as pl

# ── path setup ────────────────────────────────────────────────────────────────


import CIMS.data_processing.source.eccc.nir.nir_crosswalk_tables_cims as _crosswalk_mod
import CIMS.data_processing.source.eccc.nir.nir_to_cims as _nir_mod
import CIMS.data_processing.source.cer.cer_resd_demand as _cer_mod
from CIMS.data_processing.utils.controls_conversions import BASE_PATH, load_sector_regions, filter_excluded_branches
from CIMS.data_processing.utils.calibration_helpers import (
    OUTPUT_COLS,
    build_cer_energy,
    build_crosswalk_emissions,
    build_nir_emissions,
)
from CIMS.data_processing.utils.output_builder import write_per_region_csvs

# ── configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = BASE_PATH / 'calibration/biodiesel'

SECTOR_NAME = 'Biodiesel'


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble biodiesel calibration data and write one CSV per region."""
    print('=' * 60)
    print('BIODIESEL CALIBRATION')
    print('=' * 60)

    print('\nRunning NIR crosswalk (nir_crosswalk_tables_cims)...')
    crosswalk_df = pl.from_pandas(_crosswalk_mod.main())

    print('\nRunning NIR to CIMS (nir_to_cims)...')
    nir_df = _nir_mod.main()

    print('\nRunning CER demand (cer_resd_demand)...')
    cer_df = _cer_mod.main()

    print('\nBuilding CER energy demand rows...')
    cer_rows = build_cer_energy(cer_df, SECTOR_NAME)
    print(f'  Rows: {len(cer_rows):,}')

    print('Building crosswalk emission rows...')
    crosswalk_rows = build_crosswalk_emissions(crosswalk_df, SECTOR_NAME)
    print(f'  Rows: {len(crosswalk_rows):,}')

    print('Building NIR annual emission rows (tCO2e via AR5 GWP100)...')
    nir_rows = build_nir_emissions(nir_df, SECTOR_NAME)
    print(f'  Rows: {len(nir_rows):,}')

    print('Combining...')
    output = (
        pl.concat([cer_rows, crosswalk_rows, nir_rows], how='diagonal_relaxed')
        .select(OUTPUT_COLS)
    )

    print('Filtering to regions with Biodiesel (see sector_region_map.csv)...')
    allowed_regions = load_sector_regions().get(SECTOR_NAME)
    if allowed_regions:
        before_count = len(output)
        output = output.filter(pl.col('Region').is_in(list(allowed_regions)))
        dropped_count = before_count - len(output)
        if dropped_count:
            print(f'  Dropped {dropped_count:,} rows for regions without Biodiesel')

    output = filter_excluded_branches(output)

    regions = write_per_region_csvs(output, OUTPUT_DIR, 'biodiesel', skip_if_all_zero=True)

    print(f'\n✅ Biodiesel calibration complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
