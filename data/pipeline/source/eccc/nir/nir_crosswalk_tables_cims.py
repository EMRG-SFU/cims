"""
Canadian GHG Emissions — Regional Estimation Script (CIMS Branch Output)
=========================================================================

Same estimation logic as the original nir_crosswalk_tables.py, but the
output CSV maps NIR data to CIMS branch paths instead of NIR
Sector / Sub_Sector / Category / Sub_Category columns.

Output columns:
  CIMS_Branch, Region, Year, Value

Where CIMS_Branch looks like:
  CIMS.CAN.<REGION>.<CIMSSector>.<CIMSSubsector>...

Mapping logic:
  Each NIR (Sub_Sector, Sub_Sub_Sector, Category, Sub_Category) combination
  is mapped to one or more CIMS branch paths in NIR_TO_CIMS_MAP below.
  When a NIR row maps to multiple CIMS branches, the value is split equally.

Usage:
    python nir_crosswalk_tables_cims.py

Outputs:
    nir_crosswalk_cims.csv

Suppression handling
--------------------
The NIR contains 'x' cells where ECCC suppresses emissions data for
confidentiality, primarily in BC, AB, ON, and QC for certain sectors.
find_x_locations() identifies these cells by region, year, and column
offset. impute_x_values() then recovers them using proportional allocation
from Canada-level totals and the known non-suppressed regional values —
the same residual approach used in nir_to_cims.py. Cells are only imputed
where a Canada-level total and at least one non-suppressed regional value
are available; otherwise they remain as zero.
"""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline.utils.controls_conversions import parse_cell, BASE_PATH


# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = BASE_PATH / 'raw_data/eccc/nir'
OUTPUT   = BASE_PATH / 'processed_data/eccc/nir/nir_crossswalk_cims.csv'

YEARS     = [2000, 2005, 2010, 2015, 2020]
YEAR_ROWS = [0, 59, 118, 177, 236]

FULL_REGIONS   = ['BC', 'AB', 'ON', 'QC']
SPARSE_REGIONS = ['SK', 'MB', 'NB', 'NS', 'PE', 'NL', 'YT', 'NT', 'NU']
ALL_REGIONS    = FULL_REGIONS + SPARSE_REGIONS

# ── NIR → CIMS Branch Mapping ─────────────────────────────────────────────────
# Loaded from an external CSV so the mapping can be shared across scripts and
# edited without touching pipeline code.
#
# CSV columns: CIMS nodes, Source, Sector, Sub-sector, Sub-sub-sector
#
# Each NIR combination can map to multiple CIMS nodes rows (one row per target).
# When a NIR row has multiple targets its value is split equally across them.

MAPPING_FILE = BASE_PATH / 'mappings_conversions/nir_crosswalk_map.csv'

def load_nir_to_cims_map(path):
    """
    Read the mapping CSV and return a dict keyed by
    (Source, Sector, Sub-sector, Sub-sub-sector)
    with a list of CIMS node path strings as values.
    """
    df_map = pd.read_csv(path, dtype=str).fillna('')
    mapping = {}
    for _, row in df_map.iterrows():
        key = (
            row['Source'],
            row['Sector'],
            row['Sub-sector'],
            row['Sub-sub-sector'],
        )
        mapping.setdefault(key, []).append(row['CIMS nodes'])
    return mapping


# ── Row offset → (Source, Sector, Sub-sector, Sub-sub-sector) ─────────────────
# Tuples match the four key columns in nir_crosswalk_map.csv exactly.
SUBSECTOR_MAP = {
    3:  ('Oil and Gas', 'Upstream Oil and Gas',   'Natural Gas Production and Processing',        ''),
    5:  ('Oil and Gas', 'Upstream Oil and Gas',   'Conventional Oil Production',                  'Conventional Light Oil Production'),
    6:  ('Oil and Gas', 'Upstream Oil and Gas',   'Conventional Oil Production',                  'Conventional Heavy Oil Production'),
    7:  ('Oil and Gas', 'Upstream Oil and Gas',   'Conventional Oil Production',                  'Frontier Oil Production'),
    9:  ('Oil and Gas', 'Upstream Oil and Gas',   'Oil Sands and Thermal Heavy Oil Production',   'Mining and Extraction'),
    10: ('Oil and Gas', 'Upstream Oil and Gas',   'Oil Sands and Thermal Heavy Oil Production',   'In-Situ Thermal Extraction'),
    11: ('Oil and Gas', 'Upstream Oil and Gas',   'Oil Sands and Thermal Heavy Oil Production',   'Upgrading'),
    12: ('Oil and Gas', 'Upstream Oil and Gas',   'Oil, Natural Gas and CO2  Transmission',       ''),
    14: ('Oil and Gas', 'Downstream Oil and Gas', 'Petroleum Refining',                           ''),
    15: ('Oil and Gas', 'Downstream Oil and Gas', 'Natural Gas Distribution',                     ''),
    16: ('Electricity', '', '', ''),
    19: ('Transport', 'Passenger Transport',                           '', ''),
    20: ('Transport', 'Passenger Transport',                           '', ''),
    22: ('Transport', 'Freight Transport',                             '', ''),
    23: ('Transport', 'Freight Transport',                             '', ''),
    24: ('Transport', 'Other: Recreational, Commercial and Residential', '', ''),
    26: ('Heavy Industry', 'Mining',                                              '', ''),
    27: ('Heavy Industry', 'Smelting and Refining (Non-Ferrous Metals)',          '', ''),
    28: ('Heavy Industry', 'Pulp and Paper',                                      '', ''),
    29: ('Heavy Industry', 'Iron and Steel',                                      '', ''),
    30: ('Heavy Industry', 'Cement',                                              '', ''),
    31: ('Heavy Industry', 'Lime and Gypsum',                                     '', ''),
    32: ('Heavy Industry', 'Chemicals, Fertilizers and Biofuel Manufacturing',    '', ''),
    34: ('Buildings', 'Service Industry', '', ''),
    35: ('Buildings', 'Residential',      '', ''),
    37: ('Agriculture', 'On Farm Fuel Use',  '', ''),
    38: ('Agriculture', 'Crop Production',   '', ''),
    39: ('Agriculture', 'Animal Production', '', ''),
    41: ('Waste', '', '', ''),
    42: ('Waste', '', '', ''),
    43: ('Waste', '', '', ''),
    44: ('Coal Production', '', '', ''),
    46: ('Light Manufacturing, Construction and Forest Resources', 'Light Manufacturing', '', ''),
    47: ('Light Manufacturing, Construction and Forest Resources', 'Construction',        '', ''),
    48: ('Light Manufacturing, Construction and Forest Resources', 'Forest Resources',    '', ''),
}

# ── Column → (Category, Sub_Category[, Sub_Sub_Category]) ────────────────────
COL_INFO = {
    3:  ('Energy', 'Stationary'),
    4:  ('Energy', 'Industrial', 'Electricity'),
    5:  ('Energy', 'Industrial', 'Steam'),
    6:  ('Energy', 'Transport'),
    7:  ('Energy', 'Fugitive'),
    8:  ('Energy', 'Flaring'),
    9:  ('Energy', 'Venting'),
    11: ('Industrial Processes and Product Use', 'Minerals'),
    12: ('Industrial Processes and Product Use', 'Chemicals'),
    13: ('Industrial Processes and Product Use', 'Metals'),
    14: ('Industrial Processes and Product Use', 'Halocarbons'),
    15: ('Industrial Processes and Product Use', 'Non-Energy'),
    16: ('Industrial Processes and Product Use', 'Other Mfg'),
    18: ('Agriculture', 'Manure'),
    19: ('Agriculture', 'Enteric Ferm'),
    20: ('Agriculture', 'Agri Soils'),
    22: ('Waste', 'Solid Waste'),
    23: ('Waste', 'Bio Treatment'),
    24: ('Waste', 'Wastewater'),
    25: ('Waste', 'Incineration'),
    26: ('Waste', 'Wood Waste'),
    28: ('CO2 Captured', 'CO2 Captured'),
    29: ('LULUCF',       'LULUCF'),
}

N_DATA_ROWS = 49
TOTAL_COL   = 2
ALL_COLS    = list(range(2, 30))
DATA_COLS   = list(range(3, 30))

NIC_GROUPS = {
    'energy':  {'subcols': [3, 4, 5, 6, 7, 8, 9],   'total_col': 10},
    'ippu':    {'subcols': [11, 12, 13, 14, 15, 16], 'total_col': 17},
    'agri':    {'subcols': [18, 19, 20],              'total_col': 21},
    'waste':   {'subcols': [22, 23, 24, 25, 26],      'total_col': 27},
    'co2':     {'subcols': [28],                      'total_col': 28},
    'lulucf':  {'subcols': [29],                      'total_col': 29},
}
NIC_TOTAL_COLS = [g['total_col'] for g in NIC_GROUPS.values()]

COL_TO_GROUP = {
    sc: g
    for g in NIC_GROUPS.values()
    for sc in g['subcols']
}


# ── Data extraction ────────────────────────────────────────────────────────────

def extract(region, raw):
    df = raw[region]
    out = {}
    for year, yr in zip(YEARS, YEAR_ROWS):
        block = {}
        for offset in range(N_DATA_ROWS):
            row_idx = yr + 7 + offset
            block[offset] = {
                c: parse_cell(df.iloc[row_idx, c]) if row_idx < len(df) else 0.0
                for c in ALL_COLS
            }
        out[year] = block
    return out

def find_x_locations(region, raw):
    df = raw[region]
    locs = {}
    for year, yr in zip(YEARS, YEAR_ROWS):
        locs[year] = {}
        for offset in range(N_DATA_ROWS):
            row_idx = yr + 7 + offset
            x_cols = [
                c for c in ALL_COLS
                if row_idx < len(df) and str(df.iloc[row_idx, c]).strip().lower() == 'x'
            ]
            if x_cols:
                locs[year][offset] = x_cols
    return locs


# ── Suppression imputation ─────────────────────────────────────────────────────

def impute_x_values(data, x_locs):
    print("\nImputing suppressed (x) values in full regions...")
    for region in FULL_REGIONS:
        n_imputed = 0
        for year in YEARS:
            for offset, x_cols in x_locs[region].get(year, {}).items():
                groups_with_x = {}
                for xc in x_cols:
                    g = COL_TO_GROUP.get(xc)
                    if g is None:
                        continue
                    tc = g['total_col']
                    groups_with_x.setdefault(tc, []).append(xc)

                for tc, x_in_group in groups_with_x.items():
                    g = NIC_GROUPS[[
                        gn for gn, gv in NIC_GROUPS.items()
                        if gv['total_col'] == tc
                    ][0]]

                    cat_total = data[region][year][offset][tc]
                    known_sum = sum(
                        data[region][year][offset][sc]
                        for sc in g['subcols']
                        if sc not in x_in_group
                    )
                    residual = round(cat_total - known_sum, 4)

                    if len(x_in_group) == 1:
                        data[region][year][offset][x_in_group[0]] = residual
                        n_imputed += 1
                    else:
                        can_weights = {
                            xc: data['CAN'][year][offset][xc]
                            for xc in x_in_group
                        }
                        weight_sum = sum(can_weights.values())

                        if weight_sum == 0:
                            for xc in x_in_group:
                                data[region][year][offset][xc] = 0.0
                                n_imputed += 1
                        else:
                            assigned = 0.0
                            for xc in x_in_group[:-1]:
                                share = round(residual * (can_weights[xc] / weight_sum), 4)
                                data[region][year][offset][xc] = share
                                assigned += share
                                n_imputed += 1
                            data[region][year][offset][x_in_group[-1]] = round(
                                residual - assigned, 4
                            )
                            n_imputed += 1

        print(f"  {region}: imputed {n_imputed} values")


# ── Constrained two-level proportional solver ─────────────────────────────────

def solve_constrained(year, data, estimates):
    cat_share = {}
    sub_share = {}

    for offset in range(N_DATA_ROWS):
        full_col_c = sum(data[r][year][offset][TOTAL_COL] for r in FULL_REGIONS)

        raw_shares = {}
        for g in NIC_GROUPS.values():
            tc = g['total_col']
            full_tc = sum(data[r][year][offset][tc] for r in FULL_REGIONS)
            raw_shares[tc] = (full_tc / full_col_c) if full_col_c != 0 else 0.0

        total_share = sum(raw_shares.values())
        cat_share[offset] = {
            tc: (s / total_share if total_share != 0 else 0.0)
            for tc, s in raw_shares.items()
        }

        sub_share[offset] = {}
        for g in NIC_GROUPS.values():
            full_tc = sum(data[r][year][offset][g['total_col']] for r in FULL_REGIONS)
            raw_sub = {}
            for sc in g['subcols']:
                full_sc = sum(data[r][year][offset][sc] for r in FULL_REGIONS)
                raw_sub[sc] = (full_sc / full_tc) if full_tc != 0 else 0.0
            sub_total = sum(raw_sub.values())
            for sc in g['subcols']:
                sub_share[offset][sc] = (
                    raw_sub[sc] / sub_total if sub_total != 0 else 0.0
                )

    can_col_c  = data['CAN'][year][0][TOTAL_COL]
    full_col_c = sum(estimates[r][year][0][TOTAL_COL] for r in FULL_REGIONS)
    sparse_col_c_sum = sum(
        data[r][year][0][TOTAL_COL] for r in SPARSE_REGIONS
    )
    col_c_scale = (
        (can_col_c - full_col_c) / sparse_col_c_sum
        if sparse_col_c_sum != 0 else 1.0
    )

    for region in SPARSE_REGIONS:
        for offset in range(N_DATA_ROWS):
            raw_col_c = data[region][year][offset][TOTAL_COL]
            col_c = round(raw_col_c * col_c_scale, 4)
            estimates[region][year][offset][TOTAL_COL] = col_c

            for g in NIC_GROUPS.values():
                tc = g['total_col']
                cat_total = round(col_c * cat_share[offset][tc], 4)
                estimates[region][year][offset][tc] = cat_total

                for sc in g['subcols']:
                    estimates[region][year][offset][sc] = round(
                        cat_total * sub_share[offset][sc], 4
                    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading NIR→CIMS mapping from {MAPPING_FILE} ...")
    nir_to_cims_map = load_nir_to_cims_map(str(MAPPING_FILE))
    print(f"  Loaded {len(nir_to_cims_map)} mapping entries.")

    print("Loading CSVs...")
    raw = {}
    for region in ['CAN'] + ALL_REGIONS:
        fname = DATA_DIR / f'{region.lower()}_crosswalk_nir.csv'
        raw[region] = pd.read_csv(fname, header=None)
        print(f"  {region}: {raw[region].shape}")

    print("\nExtracting data...")
    data   = {r: extract(r, raw)          for r in ['CAN'] + ALL_REGIONS}
    x_locs = {r: find_x_locations(r, raw) for r in FULL_REGIONS}

    total_x = sum(
        len(cols)
        for r in FULL_REGIONS
        for yr in x_locs[r].values()
        for cols in yr.values()
    )
    print(f"  Found {total_x} suppressed (x) values across BC/AB/ON/QC")

    impute_x_values(data, x_locs)

    print("Setting up estimates for full regions...")
    estimates = {}
    for region in FULL_REGIONS:
        estimates[region] = {
            year: {
                offset: dict(data[region][year][offset])
                for offset in range(N_DATA_ROWS)
            }
            for year in YEARS
        }
    for region in SPARSE_REGIONS:
        estimates[region] = {
            year: {offset: {c: 0.0 for c in ALL_COLS} for offset in range(N_DATA_ROWS)}
            for year in YEARS
        }

    print("Running constrained two-level proportional solver...")
    for year in YEARS:
        solve_constrained(year, data, estimates)

    # ── Build flat tidy CSV with CIMS branches ────────────────────────────────

    print("\nBuilding CIMS-branch output DataFrame...")

    unmapped_combos = set()
    records = []

    for region in ALL_REGIONS:
        for year in YEARS:
            for offset, subsector_info in SUBSECTOR_MAP.items():
                sector, sub_sector, sub_sub_sector, sub_sub_sub_sector = subsector_info

                row_data = estimates[region][year][offset]

                for col, col_info in COL_INFO.items():
                    category, sub_cat, *rest2 = col_info
                    sub_sub_cat = rest2[0] if rest2 else ''

                    value = row_data.get(col, 0.0)
                    if value == 0.0:
                        continue

                    cims_paths = nir_to_cims_map.get(
                        (sector, sub_sector, sub_sub_sector, sub_sub_sub_sector)
                    )

                    if cims_paths is None:
                        unmapped_combos.add(
                            (sector, sub_sector, sub_sub_sector, sub_sub_sub_sector)
                        )
                        continue

                    split_value = round(value / len(cims_paths), 4)

                    for cims_path in cims_paths:
                        branch = f"CIMS.CAN.{region}.{cims_path.lstrip('.')}"
                        records.append({
                            'CIMS_Branch': branch,
                            'Region':      region,
                            'Year':        year,
                            'Value':       split_value,
                        })

    df_out = pd.DataFrame(records)

    df_out = (
        df_out
        .groupby(['CIMS_Branch', 'Region', 'Year'], as_index=False)['Value']
        .sum()
    )
    df_out['Value'] = df_out['Value'].round(4)

    df_out = df_out.sort_values(['Region', 'Year', 'CIMS_Branch']).reset_index(drop=True)

    print(f"  Total rows:      {len(df_out):,}")
    print(f"  Regions:         {df_out['Region'].nunique()}  ({', '.join(sorted(df_out['Region'].unique()))})")
    print(f"  Years:           {sorted(df_out['Year'].unique())}")
    print(f"  Unique branches: {df_out['CIMS_Branch'].nunique()}")

    if unmapped_combos:
        print(f"\n  ⚠️  {len(unmapped_combos)} NIR combos had no CIMS mapping and were skipped:")
        for combo in sorted(unmapped_combos):
            print(f"     {combo}")
    else:
        print("\n  ✅  All NIR combos mapped successfully.")

    # ── Save CSV ──────────────────────────────────────────────────────────────

    df_out = df_out[df_out['Value'] != 0].reset_index(drop=True)

    from pipeline.utils.add_cims_totals import add_totals
    df_out = add_totals(df_out)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT, index=False)
    print(f"\n✅  Saved: {OUTPUT}")

    # ── Quick spot-check ──────────────────────────────────────────────────────

    print("\n── Spot-check: AB 2000, Agriculture branches ──")
    check = df_out[
        (df_out['Region'] == 'AB') &
        (df_out['Year']   == 2000) &
        (df_out['CIMS_Branch'].str.contains('Agriculture'))
    ][['CIMS_Branch', 'Value']]
    print(check.to_string(index=False))

    print("\n── Spot-check: SK 2000, Natural Gas Production branches ──")
    check2 = df_out[
        (df_out['Region'] == 'SK') &
        (df_out['Year']   == 2000) &
        (df_out['CIMS_Branch'].str.contains('Natural Gas.Production'))
    ][['CIMS_Branch', 'Value']]
    print(check2.to_string(index=False))


if __name__ == '__main__':
    main()
