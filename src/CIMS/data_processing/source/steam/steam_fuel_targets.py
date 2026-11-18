"""
Steam fuel targets for the six steam sectors.

Builds the .Steam calibration_quantity_requested rows (GJ of fuel by region,
year, and fuel) for every sector in techdb_steam.SECTORS, and writes them to
one table. The steam code (steam_chp_shares and techdb_steam) reads this table
instead of reading the sectors' calibration files back from data/calibration/.

Why a separate stage
--------------------
Each steam sector's calibration script needs the CHP shares, and the CHP
shares need every steam sector's steam fuel targets. When those targets were
read from data/calibration/, each script read all six sectors' files from the
previous run, including its own, before writing its new file. A stale file
could not be replaced, and the results depended on which files were on disk
and the order the scripts ran. The targets only need source data (CER demand
and the MECS steam shares), so this stage builds them first.

Method
------
For each sector, the rows come from exactly the steps the sector's
*_calibration.py applies to its .Steam rows:

    1. mecs_steam_calibration.build_mecs_steam_rows('.{sector}')
       (what apply_mecs_steam puts in place of the CER steam rows)
    2. calibration_helpers.build_cer_energy (row by row, no aggregation)
    3. keep regions listed for the sector in sector_region_map.csv
    4. controls_conversions.filter_excluded_branches
    5. keep the main .Steam node's calibration_quantity_requested rows

The sector calibration scripts still write these same rows to their own
calibration files; this table must match them. Any change to those steps in
the calibration scripts must be made here too.

Reads
-----
processed_data/cer/, processed_data/eia/ and the StatCan biomass and refinery
fuel tables, through mecs_steam_calibration (run process_source_data first).

Writes
------
processed_data/steam/steam_fuel_targets.csv
    Calibration output columns (calibration_helpers.OUTPUT_COLS), one row per
    sector, region, year, and fuel target.

Usage
-----
python -m CIMS.data_processing.source.steam.steam_fuel_targets
"""

from functools import lru_cache

import pandas as pd
import polars as pl

from CIMS.data_processing.utils.controls_conversions import (
    BASE_PATH, filter_excluded_branches, load_sector_regions,
)
from CIMS.data_processing.utils.calibration_helpers import OUTPUT_COLS, build_cer_energy
from CIMS.data_processing.utils.mecs_steam_calibration import build_mecs_steam_rows

OUTPUT_DIR  = BASE_PATH / 'processed_data/steam'
OUTPUT_FILE = OUTPUT_DIR / 'steam_fuel_targets.csv'


def _sectors() -> list[str]:
    # Imported here: techdb_steam reads this module's output, so a top-level
    # import would be circular.
    from CIMS.data_processing.utils.techdb_steam import SECTORS
    return list(SECTORS)


def build_sector(sector: str) -> pl.DataFrame:
    """The .Steam fuel target rows for one sector, as its calibration script builds them."""
    rows = build_cer_energy(build_mecs_steam_rows(f'.{sector}'), sector)
    allowed = load_sector_regions().get(sector)
    if allowed:
        rows = rows.filter(pl.col('Region').is_in(list(allowed)))
    rows = filter_excluded_branches(rows)
    return rows.filter(
        (pl.col('Branch') == pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit(f'.{sector}.Steam'))
        & (pl.col('Parameter') == 'calibration_quantity_requested')
    ).select(OUTPUT_COLS)


def build() -> pl.DataFrame:
    """The .Steam fuel target rows for every steam sector."""
    return pl.concat([build_sector(sector) for sector in _sectors()], how='vertical_relaxed')


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    """The steam fuel targets table, as written by main()."""
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f'{OUTPUT_FILE} not found. Run: python -m CIMS.data_processing.source.steam.steam_fuel_targets '
            '(process_source_data runs it last).'
        )
    return pd.read_csv(OUTPUT_FILE, dtype={'Year': 'Int64'}, encoding='utf-8-sig')


def regions(sector: str) -> list[str]:
    """Regions with steam fuel targets for one sector."""
    df = load()
    return sorted(df.loc[df['Sector'] == sector, 'Region'].dropna().unique())


def main() -> pl.DataFrame:
    print('=' * 60)
    print('STEAM FUEL TARGETS')
    print('=' * 60)
    parts = []
    for sector in _sectors():
        part = build_sector(sector)
        print(f'  {sector}: {len(part):,} rows, {part["Region"].n_unique()} regions')
        parts.append(part)
    out = pl.concat(parts, how='vertical_relaxed')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_pandas().to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', lineterminator='\n')
    load.cache_clear()
    print(f'\n  Wrote {len(out):,} rows → {OUTPUT_FILE}')
    return out


if __name__ == '__main__':
    main()
