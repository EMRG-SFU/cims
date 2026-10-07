"""
Extract transportation fuel blend calibration data and save to CIMS-formatted CSV files.

Sources
-------
Technology market shares  (calibration_market_share_total)
    cer_resd_demand.py            → raw CER transport demand (vTrDmd-CIMS.csv), restricted
                                    to the domestic Passenger and Freight CER sectors and
                                    mapped to CIMS fuel names / region abbreviations.
                                    Each fuel's share of its blend pool, by region and year:
                                      Fuel Blends.Diesel_Transportation:
                                        Diesel, Biodiesel, Renewable Diesel (CER HDRD)
                                      Fuel Blends.Gasoline_Transportation:
                                        Gasoline, Ethanol, Renewable Gasoline
                                    CER reports no renewable gasoline, so it gets a 0 target.
                                    Off-Road (gasoline only, no ethanol) and Foreign
                                    Passenger/Freight (international bunkers) carry no
                                    biofuel in CER and are left out of the pool.
                                    Blend technologies each provide 1 GJ of blend per GJ of
                                    fuel, so energy shares are the technology market shares.

    Biofuel volumes CER reports before the blend technology's `available` year in
    fixed_data/fuels (e.g. HDRD before 2020) are moved into the fossil technology, so every
    target is reachable by the model.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""

import pandas as pd
import polars as pl

import CIMS.data_processing.source.cer.cer_resd_demand as _cer_mod
from CIMS.data_processing.utils.controls_conversions import BASE_PATH

# ── configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = BASE_PATH / 'calibration/fuels'
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/fuels'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

# CER sectors (lower-cased, as load_cer_data() returns them) whose demand forms the pools
CER_POOL_SECTORS = {'passenger', 'freight'}

# blend service → (fossil technology, {CIMS fuel name: blend technology})
BLEND_POOLS: dict[str, tuple[str, dict[str, str]]] = {
    'Diesel_Transportation': ('Diesel', {
        'Diesel':           'Diesel',
        'Biodiesel':        'Biodiesel',
        'Renewable Diesel': 'Renewable Diesel',
    }),
    'Gasoline_Transportation': ('Gasoline', {
        'Gasoline':           'Gasoline',
        'Ethanol':            'Ethanol',
        'Renewable Gasoline': 'Renewable Gasoline',
    }),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _empty_df() -> pl.DataFrame:
    return pl.DataFrame(schema={c: pl.Utf8 for c in OUTPUT_COLS})


def _blend_branch(region: str, service: str) -> str:
    return f'CIMS.CAN.{region}.Fuel Blends.{service}'


def _available_years() -> dict[tuple[str, str], int]:
    """(Branch, Technology) → first available year, from fixed_data/fuels."""
    available: dict[tuple[str, str], int] = {}
    for path in sorted(FIXED_INPUT_DIR.glob('fuels_*.csv')):
        fixed = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
        rows = fixed[
            (fixed['Parameter'] == 'available')
            & fixed['Branch'].str.contains('.Fuel Blends.', regex=False, na=False)
        ]
        years = pd.to_numeric(rows['2000'], errors='coerce')
        for branch, tech, year in zip(rows['Branch'], rows['Technology'], years):
            if pd.notna(year):
                available[(branch, tech)] = int(year)
    return available


# ── pool demand ───────────────────────────────────────────────────────────────

def _load_pool_demand() -> pd.DataFrame:
    """CER transport demand for the blend-pool fuels: Region, Service, Technology, Year, Data (TJ)."""
    cer = _cer_mod.load_cer_data()
    cer = cer[(cer['demand_type'] == 'transport') & cer['cer_sector'].isin(CER_POOL_SECTORS)]
    mapping, energy_map, region_map = _cer_mod._load_mapping_tables()
    merged = _cer_mod._map_to_cims(cer, mapping, energy_map, region_map, verbose=False)

    frames = []
    for service, (_, fuel_to_tech) in BLEND_POOLS.items():
        pool = merged[merged['Fuel'].isin(fuel_to_tech)].copy()
        pool['Service'] = service
        pool['Technology'] = pool['Fuel'].map(fuel_to_tech)
        frames.append(pool)
    pool = pd.concat(frames, ignore_index=True)
    return pool.groupby(['Region', 'Service', 'Technology', 'Year'], as_index=False)['Data'].sum()


def _shift_unavailable_to_fossil(pool: pd.DataFrame) -> pd.DataFrame:
    """Move biofuel volumes from years before a technology is available into the fossil tech."""
    available = _available_years()
    pool = pool.copy()
    pool['Branch'] = [_blend_branch(r, s) for r, s in zip(pool['Region'], pool['Service'])]
    first_year = pd.Series(
        [available.get((b, t)) for b, t in zip(pool['Branch'], pool['Technology'])],
        index=pool.index, dtype='float',
    )
    early = first_year.notna() & (pool['Year'] < first_year) & (pool['Data'] != 0)
    if early.any():
        summary = (
            pool[early].groupby(['Technology', 'Region'])['Year']
            .agg(['min', 'max']).reset_index()
        )
        print('  Warning: CER reports biofuel before the technology is available in '
              'fixed_data/fuels; volumes moved to the fossil technology:')
        for r in summary.itertuples(index=False):
            print(f'    {r.Technology:<18} {r.Region}  {r.min}–{r.max}')

        moved = pool[early].copy()
        moved['Technology'] = moved['Service'].map(lambda s: BLEND_POOLS[s][0])
        pool.loc[early, 'Data'] = 0.0
        pool = (
            pd.concat([pool, moved], ignore_index=True)
            .groupby(['Region', 'Service', 'Technology', 'Year', 'Branch'], as_index=False)['Data'].sum()
        )
    return pool


# ── market share builder ──────────────────────────────────────────────────────

def _build_blend_shares(pool: pd.DataFrame) -> pl.DataFrame:
    """Share of each blend technology in its pool, with a 0 target for every technology
    the pool defines but CER has no volume for (e.g. Renewable Gasoline)."""
    totals = pool.groupby(['Region', 'Service', 'Year'])['Data'].transform('sum')
    pool = pool[totals > 0].assign(share=pool['Data'] / totals)

    # Complete the Region/Service/Year × Technology grid with 0 shares
    keys = pool[['Region', 'Service', 'Year']].drop_duplicates()
    techs = pd.DataFrame(
        [(s, t) for s, (_, f2t) in BLEND_POOLS.items() for t in dict.fromkeys(f2t.values())],
        columns=['Service', 'Technology'],
    )
    grid = keys.merge(techs, on='Service')
    shares = grid.merge(pool[['Region', 'Service', 'Year', 'Technology', 'share']],
                        on=['Region', 'Service', 'Year', 'Technology'], how='left')
    shares['share'] = shares['share'].fillna(0.0)

    rows = [{
        'Branch':      _blend_branch(r.Region, r.Service),
        'Type':        'Service',
        'Region':      r.Region,
        'Sector':      '',
        'Service':     r.Service,
        'Technology':  r.Technology,
        'Parameter':   'calibration_market_share_total',
        'Context':     '',
        'Sub_Context': '',
        'Target':      '',
        'Source':      'CER',
        'Unit':        '%',
        'Year':        str(int(r.Year)),
        'Value':       f'{r.share:.12g}',
    } for r in shares.sort_values(['Region', 'Service', 'Technology', 'Year']).itertuples(index=False)]
    if not rows:
        return _empty_df()
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


def blend_shares() -> pl.DataFrame:
    """CER blend market shares for every region/year, as calibration_market_share_total rows.

    Also used by fuels_model_inputs to set the base-year market_share_total.
    """
    pool = _load_pool_demand()
    print(f'  CER blend-pool rows: {len(pool):,}')
    pool = _shift_unavailable_to_fossil(pool)
    return _build_blend_shares(pool)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble transportation fuel blend calibration data and write one CSV per region."""
    print('=' * 60)
    print('FUELS (TRANSPORTATION BLENDS) CALIBRATION')
    print('=' * 60)

    print('\nBuilding blend market share rows from CER Passenger/Freight demand...')
    output = blend_shares()
    print(f'  Rows: {len(output):,}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = output['Region'].drop_nulls().unique().sort().to_list()
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'fuels_{region.lower()}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Fuels calibration complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
