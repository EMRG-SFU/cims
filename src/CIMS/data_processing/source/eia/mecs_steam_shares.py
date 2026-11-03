"""
MECS Steam Shares by CIMS Sector

Computes the share of each CIMS industrial sector's energy demand that goes to
raising steam (conventional boilers + CHP/cogeneration), from the EMRG
harmonised MECS / DOE footprint table, and fills it to an annual series.

Input
-----
raw_data/eia/mecs/emrg_2026_mecs-footprint-metrics.csv   (or .xlsx, first sheet)
    One row per energy flow; see emrg_2026_mecs-footprint-metrics_metadata.md.

Method
------
Rows used: flow == 'input' and data_source != 'footprint'
    i.e. MECS Table 5.4 + residual rows only. The footprint rows describe the
    same energy a second time and must never be added to them.

Numerator: end_use_2 in {'boilers', 'cogeneration'}
    All CHP fuel is counted, because CIMS Steam nodes hold their CHP
    technologies and book all CHP fuel there (crediting electricity back).
    Split into:
      steam_share_fuel         — combustion fuels to boilers + CHP
      steam_share_electricity  — electricity to boilers (electric boilers)

Denominator: all remaining rows for the sector (fuel + net electricity,
    excludes feedstock).

Sector aggregation: PJ summed across MECS sectors before dividing
    (energy-weighted). Subsectors that duplicate a parent are not used.

Year fill (MECS surveys: 2002, 2006, 2010, 2014, 2018, 2022)
    before the first survey  → first survey value
    between surveys          → most recent survey at or before the year
    after the last survey    → last survey value

Output: processed_data/eia/mecs_steam_shares.csv
    Columns: Node | Variable | Unit | Source | Year | Value
"""

import pandas as pd

from CIMS.data_processing.utils.controls_conversions import (
    BASE_PATH, DATA_START, LAST_DATA_YEAR,
)

RAW_DIR     = BASE_PATH / 'raw_data/eia/mecs'
RAW_STEM    = 'emrg_2026_mecs-footprint-metrics'
OUTPUT_DIR  = BASE_PATH / 'processed_data/eia'
OUTPUT_FILE = OUTPUT_DIR / 'mecs_steam_shares.csv'

STEAM_END_USES = {'boilers', 'cogeneration'}

# CIMS sector node → MECS sector names (non-overlapping NAICS only).
# Industrial Minerals is left out: its energy is direct-fired kiln/furnace heat
# (boilers + CHP ~1% in recent MECS), so it has no Steam node in CIMS.
CIMS_TO_MECS: dict[str, list[str]] = {
    '.Chemical Products':  ['chemicals'],                        # 325
    '.Iron and Steel':     ['iron_steel'],                       # 33111, 3312
    '.Light Industrial':   [
        'food_beverage',                                         # 311, 312
        'textiles',                                              # 313-316
        'plastics_rubber',                                       # 326
        'machinery',                                             # 333
        'transportation_equipment',                              # 336
        'computers_electronics_electrical',                      # 334-335 (2018+)
        'computers_electronics_electrical_equipment',            # 334-335 (to 2014)
    ],
    '.Metal Smelting':     ['alumina_aluminum'],                 # 3313 (foundries 3315 excluded: castings, not smelting)
    '.Petroleum Refining': ['petroleum_refining'],               # 32411
    '.Pulp and Paper':     ['forest_products'],                  # 321, 322
}


def load_mecs() -> pd.DataFrame:
    """Read the harmonised MECS table (CSV preferred, xlsx fallback)."""
    csv_path, xlsx_path = RAW_DIR / f'{RAW_STEM}.csv', RAW_DIR / f'{RAW_STEM}.xlsx'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif xlsx_path.exists():
        df = pd.read_excel(xlsx_path, sheet_name=0)
    else:
        raise FileNotFoundError(f'No MECS file found at {csv_path} or {xlsx_path}')
    for col in ('name', 'flow', 'data_source', 'end_use_2', 'fuel_type'):
        df[col] = df[col].fillna('').astype(str).str.strip().str.lower()
    df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0.0)
    return df


def survey_shares(mecs: pd.DataFrame) -> pd.DataFrame:
    """Steam shares by CIMS node for each MECS survey year."""
    rows = mecs[(mecs['flow'] == 'input') & (mecs['data_source'] != 'footprint')]
    is_steam = rows['end_use_2'].isin(STEAM_END_USES)
    is_elec  = rows['fuel_type'] == 'electricity'

    out = []
    for node, names in CIMS_TO_MECS.items():
        sub = rows[rows['name'].isin(names)]
        for year, g in sub.groupby('mecs_year'):
            total = g['value'].sum()
            if total <= 0:
                continue
            steam = g.loc[is_steam.loc[g.index], 'value']
            elec  = g.loc[(is_steam & is_elec).loc[g.index], 'value'].sum()
            out.append({
                'Node': node, 'Year': int(year),
                'steam_share_fuel':        (steam.sum() - elec) / total,
                'steam_share_electricity': elec / total,
            })
    return pd.DataFrame(out)


def fill_years(shares: pd.DataFrame, years: range) -> pd.DataFrame:
    """Step-fill survey values to every year (carry back first, carry forward latest)."""
    out = []
    for node, g in shares.groupby('Node'):
        g = g.sort_values('Year').set_index('Year')
        surveys = list(g.index)
        for y in years:
            src = max([s for s in surveys if s <= y], default=surveys[0])
            for var in ('steam_share_fuel', 'steam_share_electricity'):
                out.append({'Node': node, 'Variable': var, 'Year': y,
                            'Value': float(g.at[src, var]), 'Survey_Year': src})
    return pd.DataFrame(out)


def main() -> pd.DataFrame:
    print('Loading MECS footprint metrics...')
    mecs = load_mecs()

    print('Computing steam shares by CIMS sector and survey year...')
    shares = survey_shares(mecs)
    print(shares.assign(total=shares['steam_share_fuel'] + shares['steam_share_electricity'])
                .pivot(index='Node', columns='Year', values='total').round(3).to_string())

    last_year = max(LAST_DATA_YEAR['cer'], int(shares['Year'].max()))
    filled = fill_years(shares, range(DATA_START, last_year + 1))
    result = filled.assign(Unit='fraction', Source='EIA MECS (EMRG harmonised)')[
        ['Node', 'Variable', 'Unit', 'Source', 'Year', 'Value', 'Survey_Year']
    ].sort_values(['Node', 'Variable', 'Year']).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f'\nDone. {len(result):,} rows → {OUTPUT_FILE}')
    return result


if __name__ == '__main__':
    main()
