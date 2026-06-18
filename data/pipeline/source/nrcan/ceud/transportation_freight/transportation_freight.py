"""
Transportation Freight Pipeline

Extracts freight tonne-kilometres (k*tkm = thousands of tonne-km) from the
NRCan CEUD (Comprehensive Energy Use Database) for all Canadian provinces,
projects activity to 2100, and produces historical technology market shares.

Key behavioural notes
---------------------
- BC uses tran_bct_e.xls (BC + Territories combined in CEUD).
- Territories (YT, NT, NU) are split from BC using population shares.
- AT is not used; Atlantic provinces (NB, NS, PE, NL) are processed individually.
- All output is in long format: (province, variable, category, parameter, unit, year, value)
- Unit for all tkm variables: 'k*tkm' (thousands of tonne-km).
  CEUD activity is in millions of tonne-km (M·tkm); values are × 1000 on output.
- Technology market shares end at LAST_HIST_YEAR (from CONTROLS).
- k*tkm activity and mode-level shares extend to 2100 via CAGR projection.
- Marine and off-road require IPCC/external data; output as zero if unavailable.

Variables extracted / derived
------------------------------
Activity & mode shares (to 2100):
  total_ktkm (service_request, k*tkm)          — Land + Marine + Air only
  Freight.Off-Road (service_request, k*tkm)     — separate absolute; NOT included in total
  Freight.Land, Freight.Marine, Freight.Air (service_request, % of k*tkm)  — sum to 1
  Freight.Land.Light Medium, Freight.Land.Heavy (service_request, % of Land k*tkm)

Technology market shares (to LAST_HIST_YEAR, held flat to 2100):
  Light Medium: fuel-based tech shares (% of Light Medium k*tkm)
  Freight.Land.Heavy: Trucks vs Rail (% of Heavy k*tkm)

Outputs:
  Light Medium: tech-specific output (k*tkm, historical)
  Heavy Trucks: total truck output (k*tkm, historical)

Equations sourced from trans_freight_ceud_old.py (Sprint 1).
"""

from pathlib import Path
from typing import Optional
import sys

import polars as pl
import pandas as pd
import numpy as np

_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from mappings_conversions.control import CONTROLS
from pipeline.utils.extractors.nrcan_ceud import row_to_series
from pipeline.utils.controls_conversions import BASE_PATH as _CIMS_BASE

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_PATH        = _CIMS_BASE / 'raw_data/nrcan/ceud/transportation'
CAN_FILE         = BASE_PATH / 'tran_ca_e.xls'
ASSUMPTIONS_CSV  = _CIMS_BASE / 'raw_data/assumptions/transport_freight_assumptions.csv'
STATSCAN_POP_PATH = _CIMS_BASE / 'raw_data/stats_can/population/1710000901.csv'
OUTPUT_DIR       = _CIMS_BASE / 'processed_data/nrcan/ceud'
LAST_HIST_YEAR   = CONTROLS["last_data_year"]["ceud"]

FUEL_SCALE = 1000.0  # CEUD reports energy in PJ; × 1000 → TJ

IPCC_FILE               = _CIMS_BASE / 'raw_data/eccc/nir/GHG_IPCC_Can_Prov_Terr.csv'
OFFROAD_MJ_PER_TKM      = 7.82     # TJ per M·tkm (TJ→MJ and M·tkm cancel); same for all regions
GASOLINE_EF_KT_CO2EQ_TJ = 0.064   # kt CO2eq per TJ for motor gasoline
HFO_EF_KT_CO2EQ_TJ      = 0.074737 # kt CO2eq per TJ for heavy fuel oil (marine)
MARINE_TKM_DOMESTIC_SHARE = 0.46   # fraction of CAN total marine M·tkm that is domestic

# IPCC region name → province abbreviation(s)
# "Northwest Territories and Nunavut" is combined in IPCC; NT and NU each get 50%.
# BC key maps to "British Columbia" (territories are handled separately after BCT split).
_IPCC_PROV_MAP: dict[str, str] = {
    'NL': 'Newfoundland and Labrador',
    'PE': 'Prince Edward Island',
    'NS': 'Nova Scotia',
    'NB': 'New Brunswick',
    'QC': 'Quebec',
    'ON': 'Ontario',
    'MB': 'Manitoba',
    'SK': 'Saskatchewan',
    'AB': 'Alberta',
    'BC': 'British Columbia',
    'YT': 'Yukon',
}
_IPCC_NT_NU_REGION = 'Northwest Territories and Nunavut'

PROVINCES = {
    'AB': 'Alberta',
    'BC': 'British Columbia and Territories',
    'MB': 'Manitoba',
    'NB': 'New Brunswick',
    'NL': 'Newfoundland and Labrador',
    'NS': 'Nova Scotia',
    'ON': 'Ontario',
    'PE': 'Prince Edward Island',
    'QC': 'Quebec',
    'SK': 'Saskatchewan',
}

# BC maps to the BCT CEUD file; all others map 1:1 by lowercase code.
_PROV_FILE_CODE = {'BC': 'bct'}

PROJ_HORIZON = 2100

# BCT split regions (same as passenger)
BCT_REGIONS = ('BC', 'YT', 'NT', 'NU')

# ==============================================================================
# CAGR OVERRIDES
# Key: (province, mode) where mode matches assumption CSV names
#   ('Light Trucks', 'Medium Trucks', 'Heavy Trucks', 'Rail', 'Marine',
#    'Aviation', 'Off-Road')
# Value: (cagr_2023_to_2050, cagr_2051_to_2100)
# When present, replaces the rates read from transport_freight_assumptions.csv.
# ==============================================================================

CAGR_OVERRIDES: dict[tuple[str, str], tuple[float, float]] = {
    # Examples — uncomment or add entries to override specific province+mode rates:
    # ("SK", "Medium Trucks"): (0.015, 0.008),
    # ("SK", "Rail"):          (0.012, 0.006),
    # ("MB", "Rail"):          (0.012, 0.006),
    # ("BC", "Rail"):          (0.010, 0.005),
}

# ==============================================================================
# CEUD PROVINCIAL TABLE NUMBERS
# ==============================================================================

TABLE_FREIGHT_AIR   = 15   # Freight air fuel by source (PJ)
TABLE_FREIGHT_RAIL  = 18   # Freight rail energy use (PJ)
TABLE_FREIGHT_LT    = 35   # Freight light truck fuel by source (PJ)
TABLE_FREIGHT_MHVT  = 36   # Freight medium & heavy truck fuel + tkm
TABLE_TRUCK_EXPL    = 37   # Truck explanatory variables (stock, avg km)

# ==============================================================================
# CAN FILE ROW MAPS  (tran_ca_e.xls, 0-indexed rows — verified from file)
# ==============================================================================

# Table 27 (freight rail): row 5 = total PJ, row 29 = total M·tkm
CAN_T27_RAIL_PJ  = 5
CAN_T27_RAIL_TKM = 29

# Table 21 (freight air): row 5 = total PJ, row 13 = total M·tkm
CAN_T21_AIR_PJ  = 5
CAN_T21_AIR_TKM = 13

# Table 29 (marine): row 29 = CAN total M·tkm (domestic + international combined)
# Domestic share = MARINE_TKM_DOMESTIC_SHARE (0.46); provincial split via IPCC Domestic Navigation.
CAN_T29_MARINE_TKM = 29

# ==============================================================================
# PROVINCIAL TABLE ROW MAPS  (0-indexed — verified from tran_ab_e.xls)
# ==============================================================================

# Table 35 (Freight Light Truck energy + activity)
T35_LT_TKM_ROW = 21     # Freight Light Truck Tonne-kilometres (M·tkm)

# Table 36 (Medium and Heavy Truck energy + activity)
T36_MT_TKM_ROW = 18     # Medium Truck Tonne-kilometres (M·tkm)
T36_HT_TKM_ROW = 35     # Heavy Truck Tonne-kilometres (M·tkm)

# Table 37 (Truck explanatory variables) — used for output calculation
# Row indices verified for tran_ab_e.xls and tran_bct_e.xls (0-indexed)
T37_LT_STOCK_ROW = 17   # Freight Light Trucks stock (thousands)
T37_MT_STOCK_ROW = 18   # Medium Trucks stock (thousands)
T37_HT_STOCK_ROW = 19   # Heavy Trucks stock (thousands)
T37_LT_AVGKM_ROW = 27   # Freight Light Trucks avg distance (km/year)
T37_MT_AVGKM_ROW = 28   # Medium Trucks avg distance (km/year)
T37_HT_AVGKM_ROW = 29   # Heavy Trucks avg distance (km/year)

# CAN average load factors (tonnes/vehicle) from old-script national estimates
LF_LIGHT_TRUCK  = 1.250652177
LF_MEDIUM_TRUCK = 1.250652177
LF_HEAVY_TRUCK  = 6.861117667

# Table 18 (Freight Rail — activity not available by region, allocate from CAN)
T18_RAIL_PJ_ROW = 5     # Freight Rail Transportation Energy Use (PJ)

# Table 15 (Freight Air — activity not available by region, allocate from CAN)
T15_AIR_PJ_ROW  = 5     # Freight Air Transportation Energy Use (PJ)


# ==============================================================================
# HELPERS
# ==============================================================================


def _long(province: str, variable: str, category: str, parameter: str,
          unit: str, series: pd.Series, source: str = 'CEUD') -> pl.DataFrame:
    """Convert a year-indexed pd.Series to a Polars long-format DataFrame."""
    years  = [int(y)   for y, v in series.items() if pd.notna(v)]
    values = [float(v) for _, v in series.items() if pd.notna(v)]
    n = len(years)
    return pl.DataFrame(
        {
            'province':  [province]  * n,
            'variable':  [variable]  * n,
            'category':  [category]  * n,
            'parameter': [parameter] * n,
            'unit':      [unit]      * n,
            'source':    [source]    * n,
            'year':      years,
            'value':     values,
        },
        schema_overrides={'year': pl.Int32, 'value': pl.Float64},
    )


def _series_from_df(df: pl.DataFrame, variable: str, category: str = '') -> pd.Series:
    """Extract a year-sorted pd.Series from the long-format DataFrame."""
    mask = pl.col('variable') == variable
    if category:
        mask = mask & (pl.col('category') == category)
    sub = df.filter(mask).sort('year')
    if len(sub) == 0:
        return pd.Series(dtype=float)
    return pd.Series(sub['value'].to_list(), index=sub['year'].to_list())


def _try_labels(table: pl.DataFrame, candidates: list[str], match_n: int = 0) -> pd.Series:
    """Try multiple label variants; return the first non-empty Series."""
    for label in candidates:
        try:
            s = row_to_series(table, label, match_n=match_n)
            if not s.dropna().empty:
                return s
        except Exception:
            continue
    return pd.Series(dtype=float)


def _load_prov_table(province_code: str, table_num: int) -> Optional[pl.DataFrame]:
    """Load a single numbered table from the provincial CEUD file."""
    file_code = _PROV_FILE_CODE.get(province_code.upper(), province_code.lower())
    file_path = BASE_PATH / f"tran_{file_code}_e.xls"
    if not file_path.exists():
        return None
    sheet_name = f"Table {table_num}"
    try:
        df = pl.read_excel(str(file_path), sheet_name=sheet_name, has_header=False)
        cast_exprs = [
            pl.col(c).str.strip_chars().cast(pl.Float64, strict=False)
            for c in df.columns[2:]
            if df[c].dtype in (pl.String, pl.Utf8)
        ]
        return df.with_columns(cast_exprs) if cast_exprs else df
    except Exception:
        return None


def _load_can_sheet(sheet_name: str) -> Optional[pl.DataFrame]:
    """Load a named sheet from the national CEUD file."""
    if not CAN_FILE.exists():
        return None
    try:
        df = pl.read_excel(str(CAN_FILE), sheet_name=sheet_name, has_header=False)
        cast_exprs = [
            pl.col(c).str.strip_chars().cast(pl.Float64, strict=False)
            for c in df.columns[2:]
            if df[c].dtype in (pl.String, pl.Utf8)
        ]
        return df.with_columns(cast_exprs) if cast_exprs else df
    except Exception:
        return None


def _read_row(df: pl.DataFrame, row_idx: int) -> pd.Series:
    """
    Extract a year-indexed pd.Series from a Polars DataFrame row.
    Scans the first few rows to locate the year header, then reads data row.
    """
    arr = df.to_numpy()
    yr_cols: dict[int, int] = {}
    for header_row in range(min(10, arr.shape[0])):
        try:
            yr = int(float(arr[header_row, 2]))
            if 1990 <= yr <= 2040:
                for c in range(2, arr.shape[1]):
                    try:
                        y = int(float(arr[header_row, c]))
                        if 1990 <= y <= 2040:
                            yr_cols[y] = c
                    except (ValueError, TypeError):
                        continue
                break
        except (ValueError, TypeError):
            continue

    def _f(v) -> float:
        try:
            return float(v)
        except (ValueError, TypeError):
            return np.nan

    if row_idx >= arr.shape[0]:
        return pd.Series(dtype=float)
    return pd.Series({yr: _f(arr[row_idx, c]) for yr, c in yr_cols.items()}, dtype=float)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division replacing 0/NaN denominators with NaN (no inf)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = num / den.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _extend_flat(s: pd.Series, proj_start: int, horizon: int) -> pd.Series:
    """Extend a historical series to horizon by holding the last value constant."""
    if s.dropna().empty:
        return s
    last = float(s.dropna().iloc[-1])
    proj_years = list(range(proj_start, horizon + 1))
    return pd.concat([s, pd.Series(last, index=proj_years)]).sort_index()


# ==============================================================================
# CAN DATA LOADING
# ==============================================================================


def load_can_freight_params() -> dict:
    """
    Load CAN-level totals needed for proportional allocation to provinces:
      - can_rail_pj / can_rail_M_tkm: CAN freight rail totals (Table 27)
      - can_air_pj  / can_air_M_tkm:  CAN freight air totals  (Table 21)
      - prov_marine_M_tkm: dict of province → year-indexed M·tkm series,
            derived from Table 29 (provincial marine energy × CAN intensity).
            Key 'BC' represents BCT (BC + Territories combined in CEUD).
            CEUD footnote: all marine is freight except recreational boating.
    """
    params: dict = {}

    t27 = _load_can_sheet('Table 27')
    t21 = _load_can_sheet('Table 21')
    t29 = _load_can_sheet('Table 29')

    if t27 is not None:
        try:
            params['can_rail_pj']    = _read_row(t27, CAN_T27_RAIL_PJ).dropna()
            params['can_rail_M_tkm'] = _read_row(t27, CAN_T27_RAIL_TKM).dropna()
        except Exception:
            pass

    if t21 is not None:
        try:
            params['can_air_pj']    = _read_row(t21, CAN_T21_AIR_PJ).dropna()
            params['can_air_M_tkm'] = _read_row(t21, CAN_T21_AIR_TKM).dropna()
        except Exception:
            pass

    if t29 is not None:
        try:
            can_M_tkm_total = _read_row(t29, CAN_T29_MARINE_TKM).dropna()
            if not can_M_tkm_total.empty:
                params['can_dom_marine_M_tkm'] = (
                    can_M_tkm_total * MARINE_TKM_DOMESTIC_SHARE
                ).dropna()
        except Exception:
            pass

    return params


def load_ipcc_offroad(ipcc_file: Path = IPCC_FILE) -> dict[str, pd.Series]:
    """
    Load provincial Off-Road Other Transportation CO2eq from the ECCC IPCC file
    and convert to M·tkm using a fixed gasoline emission factor and efficiency.

    Formula (per old script):
      gasoline_TJ = CO2eq_kt / GASOLINE_EF_KT_CO2EQ_TJ
      offroad_M_tkm = gasoline_TJ / OFFROAD_MJ_PER_TKM
      (TJ→MJ and tkm→M·tkm cancel so no ×1000 needed)

    Returns dict keyed by province abbreviation → year-indexed pd.Series (M·tkm).
    NT and NU each receive 50% of "Northwest Territories and Nunavut".
    BC maps to "British Columbia"; territories are corrected after the BCT split
    if their IPCC data differs significantly.
    """
    if not ipcc_file.exists():
        return {}

    try:
        raw = pd.read_csv(str(ipcc_file))
    except Exception:
        return {}

    mask = (
        (raw['Source'].str.strip() == 'Energy')
        & (raw['Category'].str.strip() == 'Transport')
        & (raw['Sub-category'].str.strip() == 'Other Transportation')
        & (raw['Sub-sub-category'].str.strip() == 'Off-Road Other Transportation')
    )
    sub = raw.loc[mask, ['Year', 'Region', 'CO2eq']].copy()
    sub['Year']   = pd.to_numeric(sub['Year'],   errors='coerce')
    sub['CO2eq']  = pd.to_numeric(sub['CO2eq'],  errors='coerce')
    sub = sub.dropna(subset=['Year', 'CO2eq'])
    sub['Year'] = sub['Year'].astype(int)

    result: dict[str, pd.Series] = {}

    for abbr, region_name in _IPCC_PROV_MAP.items():
        rows = sub[sub['Region'].str.strip() == region_name]
        if rows.empty:
            continue
        co2eq = rows.set_index('Year')['CO2eq'].sort_index().astype(float)
        gasoline_tj = co2eq / GASOLINE_EF_KT_CO2EQ_TJ
        result[abbr] = (gasoline_tj / OFFROAD_MJ_PER_TKM).dropna()

    # NT and NU: separate from 1999 onward; combined (÷2) before 1999
    def _nt_nu(region_post99: str) -> pd.Series:
        combined = sub[sub['Region'].str.strip() == _IPCC_NT_NU_REGION]
        separate = sub[sub['Region'].str.strip() == region_post99]
        parts = []
        if not combined.empty:
            co2eq = combined.set_index('Year')['CO2eq'].sort_index().astype(float)
            parts.append((co2eq / GASOLINE_EF_KT_CO2EQ_TJ / OFFROAD_MJ_PER_TKM / 2.0).dropna())
        if not separate.empty:
            co2eq = separate.set_index('Year')['CO2eq'].sort_index().astype(float)
            parts.append((co2eq / GASOLINE_EF_KT_CO2EQ_TJ / OFFROAD_MJ_PER_TKM).dropna())
        if not parts:
            return pd.Series(dtype=float)
        return pd.concat(parts).sort_index()

    result['NT'] = _nt_nu('Northwest Territories')
    result['NU'] = _nt_nu('Nunavut')

    return result


def load_ipcc_domestic_marine(ipcc_file: Path = IPCC_FILE) -> dict[str, pd.Series]:
    """
    Load provincial Domestic Navigation CO2eq (kt) from the ECCC IPCC file.

    Returns dict keyed by province abbreviation (plus 'CAN') → year-indexed
    pd.Series of CO2eq in kt.  Used by extract_marine_ktkm to allocate domestic
    marine M·tkm proportionally:
        prov_dom_M_tkm = (prov_co2eq / can_co2eq) × can_dom_M_tkm
    The HFO emission factor cancels in the ratio so no conversion is needed here.
    """
    if not ipcc_file.exists():
        return {}

    try:
        raw = pd.read_csv(str(ipcc_file))
    except Exception:
        return {}

    mask = (
        (raw['Source'].str.strip() == 'Energy')
        & (raw['Category'].str.strip() == 'Transport')
        & (raw['Sub-category'].str.strip() == 'Marine')
        & (raw['Sub-sub-category'].str.strip() == 'Domestic Navigation')
    )
    sub = raw.loc[mask, ['Year', 'Region', 'CO2eq']].copy()
    sub['Year']  = pd.to_numeric(sub['Year'],  errors='coerce')
    sub['CO2eq'] = pd.to_numeric(sub['CO2eq'], errors='coerce')
    sub = sub.dropna(subset=['Year', 'CO2eq'])
    sub['Year'] = sub['Year'].astype(int)

    result: dict[str, pd.Series] = {}

    # CAN-level total
    can_rows = sub[sub['Region'].str.strip() == 'Canada']
    if not can_rows.empty:
        result['CAN'] = can_rows.set_index('Year')['CO2eq'].sort_index().astype(float).dropna()

    for abbr, region_name in _IPCC_PROV_MAP.items():
        rows = sub[sub['Region'].str.strip() == region_name]
        if rows.empty:
            continue
        result[abbr] = rows.set_index('Year')['CO2eq'].sort_index().astype(float).dropna()

    return result


# ==============================================================================
# PROJECTION PARAMETERS
# ==============================================================================


def load_projection_params(assumptions_csv: Path = ASSUMPTIONS_CSV) -> dict:
    """
    Load k*tkm CAGR projection assumptions from the freight assumptions CSV.

    CSV structure (0-indexed columns):
      col 2: 'k*tkm' (mode section marker)
      col 3: mode name ('Light Trucks', 'Medium Trucks', 'Heavy Trucks',
                        'Rail', 'Marine', 'Aviation', 'Off-Road')
      col 4: province code (2–3 uppercase chars)
      col 6: hist_cagr (not used in projection)
      col 8: share_of_2019 (2022 COVID multiplier relative to 2019)
      col 9: ref_cagr_2023 (CAGR for 2023–2050)
      col 10: ref_cagr_2051 (CAGR for 2051–2100)

    Returns
    -------
    dict with key 'ktkm_assumptions':
        { mode_name: { prov_code: { 'share_of_2019': float,
                                    'ref_cagr_2023':  float,
                                    'ref_cagr_2051':  float } } }
    """
    params: dict = {}
    _KTKM_MODES = {
        'Light Trucks', 'Medium Trucks', 'Heavy Trucks',
        'Rail', 'Marine', 'Aviation', 'Off-Road',
    }

    try:
        raw = pd.read_csv(assumptions_csv, header=None, dtype=str)
    except FileNotFoundError:
        return params

    arr = raw.fillna('').to_numpy()

    def cell(row, col: int) -> str:
        return str(row[col]).strip() if col < len(row) else ''

    def _float(v: str) -> Optional[float]:
        try:
            return float(v)
        except ValueError:
            return None

    ktkm: dict = {}
    cur_mode: Optional[str] = None
    for row in arr:
        c2, c3, c4 = cell(row, 2), cell(row, 3), cell(row, 4)
        if c2 == 'k*tkm' and c3 in _KTKM_MODES:
            cur_mode = c3
            ktkm.setdefault(cur_mode, {})
            continue
        if cur_mode and c4 and c4.isupper() and 2 <= len(c4) <= 3:
            # Skip aggregate codes AT and TR
            if c4 in ('AT', 'TR'):
                continue
            s2019 = _float(cell(row, 8))
            c23   = _float(cell(row, 9))
            c51   = _float(cell(row, 10))
            ktkm[cur_mode][c4] = {
                'share_of_2019': s2019 if s2019 is not None else 0.9,
                'ref_cagr_2023': c23   if c23   is not None else 0.0,
                'ref_cagr_2051': c51   if c51   is not None else 0.0,
            }

    if ktkm:
        params['ktkm_assumptions'] = ktkm
    return params


# ==============================================================================
# TABLE LOADING
# ==============================================================================


def load_tables(province_code: str) -> dict:
    """Load freight CEUD tables for a province. Returns dict 'Table N' → pl.DataFrame."""
    table_numbers = [
        TABLE_FREIGHT_AIR,
        TABLE_FREIGHT_RAIL,
        TABLE_FREIGHT_LT,
        TABLE_FREIGHT_MHVT,
        TABLE_TRUCK_EXPL,
    ]
    tables = {}
    for n in sorted(set(table_numbers)):
        df = _load_prov_table(province_code, n)
        if df is not None:
            tables[f'Table {n}'] = df
    return tables


# ==============================================================================
# EXTRACTION — TRUCK ACTIVITY (M·tkm direct from provincial tables)
# ==============================================================================


def extract_truck_ktkm(province: str, tables: dict) -> list[pl.DataFrame]:
    """
    Freight truck activity (M·tkm) read directly from provincial CEUD tables:
      Table 35 row 21: Freight Light Truck Tonne-kilometres (M·tkm)
      Table 36 row 18: Medium Truck Tonne-kilometres (M·tkm)
      Table 36 row 35: Heavy Truck Tonne-kilometres (M·tkm)

    These values are already freight-only and available at the provincial level.
    """
    t35 = tables.get(f'Table {TABLE_FREIGHT_LT}')
    t36 = tables.get(f'Table {TABLE_FREIGHT_MHVT}')
    frames = []

    if t35 is not None:
        lt_tkm = _read_row(t35, T35_LT_TKM_ROW).dropna()
        if not lt_tkm.empty:
            frames.append(_long(province, 'lt_ktkm', '', 'intermediate', 'M_tkm', lt_tkm))

    if t36 is not None:
        mt_tkm = _read_row(t36, T36_MT_TKM_ROW).dropna()
        if not mt_tkm.empty:
            frames.append(_long(province, 'mt_ktkm', '', 'intermediate', 'M_tkm', mt_tkm))

        ht_tkm = _read_row(t36, T36_HT_TKM_ROW).dropna()
        if not ht_tkm.empty:
            frames.append(_long(province, 'ht_ktkm', '', 'intermediate', 'M_tkm', ht_tkm))

    return frames


def extract_truck_mvkm(province: str, tables: dict) -> list[pl.DataFrame]:
    """
    Extract freight truck vehicle-km (M·vkm) from Table 37 stock × avg_km.

    Formula: stock (thousands) × avg_km (km/year) / 1000 = M·vkm
    Multiplying by the load factor gives M·tkm for the output parameter.
    Row indices verified for tran_ab_e.xls and tran_bct_e.xls.
    """
    t37 = tables.get(f'Table {TABLE_TRUCK_EXPL}')
    if t37 is None:
        return []
    frames = []
    for stock_var, mvkm_var, stock_row, avgkm_row in [
        ('lt_stock', 'lt_m_vkm', T37_LT_STOCK_ROW, T37_LT_AVGKM_ROW),
        ('mt_stock', 'mt_m_vkm', T37_MT_STOCK_ROW, T37_MT_AVGKM_ROW),
        ('ht_stock', 'ht_m_vkm', T37_HT_STOCK_ROW, T37_HT_AVGKM_ROW),
    ]:
        stock  = _read_row(t37, stock_row).dropna()   # thousands of vehicles
        avg_km = _read_row(t37, avgkm_row).dropna()   # km/year per vehicle
        if not stock.empty:
            frames.append(_long(province, stock_var, '', 'intermediate', 'k_veh', stock))
        common = sorted(set(stock.index) & set(avg_km.index))
        if not common:
            continue
        m_vkm = (stock.reindex(common) * avg_km.reindex(common) / 1000.0).dropna()
        if not m_vkm.empty:
            frames.append(_long(province, mvkm_var, '', 'intermediate', 'M_vkm', m_vkm))
    return frames


# ==============================================================================
# EXTRACTION — RAIL (M·tkm via proportional allocation from CAN)
# ==============================================================================


def extract_rail_ktkm(province: str, tables: dict, can_params: dict) -> list[pl.DataFrame]:
    """
    Provincial freight rail M·tkm via proportional allocation:
      prov_rail_M_tkm = (prov_rail_pj / can_rail_pj) × can_rail_M_tkm

    Table 18 row 5 gives provincial freight rail energy (PJ) — already freight-only.
    CAN Table 27 rows 5/29 give national totals. Provincial activity is not available
    directly by region (per CEUD note), so CAN-proportional allocation is used.
    """
    t18 = tables.get(f'Table {TABLE_FREIGHT_RAIL}')
    if t18 is None:
        return []

    prov_pj   = _read_row(t18, T18_RAIL_PJ_ROW).dropna()
    can_pj    = can_params.get('can_rail_pj',    pd.Series(dtype=float))
    can_M_tkm = can_params.get('can_rail_M_tkm', pd.Series(dtype=float))
    if prov_pj.empty or can_pj.empty or can_M_tkm.empty:
        return []

    common = sorted(set(prov_pj.index) & set(can_pj.index) & set(can_M_tkm.index))
    if not common:
        return []

    prov_M_tkm = (
        _safe_div(prov_pj.reindex(common), can_pj.reindex(common))
        * can_M_tkm.reindex(common)
    ).dropna()
    if prov_M_tkm.empty:
        return []
    return [_long(province, 'rail_ktkm', '', 'intermediate', 'M_tkm', prov_M_tkm)]


# ==============================================================================
# EXTRACTION — AIR (M·tkm via proportional allocation from CAN)
# ==============================================================================


def extract_air_ktkm(province: str, tables: dict, can_params: dict) -> list[pl.DataFrame]:
    """
    Provincial freight air M·tkm via proportional allocation:
      prov_air_M_tkm = (prov_air_pj / can_air_pj) × can_air_M_tkm

    Table 15 row 5 gives provincial freight air energy (PJ) — already freight-only.
    CAN Table 21 rows 5/13 give national totals. Provincial activity is not available
    directly by region (per CEUD note), so CAN-proportional allocation is used.
    """
    t15 = tables.get(f'Table {TABLE_FREIGHT_AIR}')
    if t15 is None:
        return []

    prov_pj   = _read_row(t15, T15_AIR_PJ_ROW).dropna()
    can_pj    = can_params.get('can_air_pj',    pd.Series(dtype=float))
    can_M_tkm = can_params.get('can_air_M_tkm', pd.Series(dtype=float))
    if prov_pj.empty or can_pj.empty or can_M_tkm.empty:
        return []

    common = sorted(set(prov_pj.index) & set(can_pj.index) & set(can_M_tkm.index))
    if not common:
        return []

    prov_M_tkm = (
        _safe_div(prov_pj.reindex(common), can_pj.reindex(common))
        * can_M_tkm.reindex(common)
    ).dropna()
    if prov_M_tkm.empty:
        return []
    return [_long(province, 'air_ktkm', '', 'intermediate', 'M_tkm', prov_M_tkm)]


# ==============================================================================
# EXTRACTION — LIGHT MEDIUM FUEL SHARES
# ==============================================================================


def extract_marine_ktkm(
    province: str, can_params: dict, ipcc_marine: dict
) -> list[pl.DataFrame]:
    """
    Provincial domestic marine freight M·tkm via IPCC Domestic Navigation CO2eq.

    Formula (HFO emission factor cancels in the ratio):
        prov_dom_M_tkm = (prov_CO2eq / can_CO2eq) × can_dom_M_tkm

    Where can_dom_M_tkm = CAN Table 29 total M·tkm × MARINE_TKM_DOMESTIC_SHARE (0.46).
    BCT ('BC') uses "British Columbia" IPCC data; territories (YT/NT/NU) are zeroed
    in _split_bct since they have no significant domestic marine freight.
    """
    prov_co2eq    = ipcc_marine.get(province, pd.Series(dtype=float)).dropna()
    can_co2eq     = ipcc_marine.get('CAN',    pd.Series(dtype=float)).dropna()
    can_dom_M_tkm = can_params.get('can_dom_marine_M_tkm', pd.Series(dtype=float)).dropna()

    if prov_co2eq.empty or can_co2eq.empty or can_dom_M_tkm.empty:
        return []

    common = sorted(set(prov_co2eq.index) & set(can_co2eq.index) & set(can_dom_M_tkm.index))
    if not common:
        return []

    prov_M_tkm = (
        _safe_div(prov_co2eq.reindex(common), can_co2eq.reindex(common))
        * can_dom_M_tkm.reindex(common)
    ).dropna()

    if prov_M_tkm.empty:
        return []
    return [_long(province, 'marine_ktkm', '', 'intermediate', 'M_tkm', prov_M_tkm, 'IPCC')]


def extract_offroad_ktkm(province: str, ipcc_offroad: dict) -> list[pl.DataFrame]:
    """
    Provincial off-road freight M·tkm from pre-loaded IPCC off-road dict.
    Off-road is not in CEUD transportation tables; source is ECCC IPCC file
    (Off-Road Other Transportation CO2eq → gasoline TJ → M·tkm via 7.82 MJ/tkm).
    """
    series = ipcc_offroad.get(province, pd.Series(dtype=float))
    if series.dropna().empty:
        return []
    return [_long(province, 'offroad_ktkm', '', 'intermediate', 'M_tkm', series, 'IPCC')]


def extract_lm_fuel_shares(province: str, tables: dict) -> list[pl.DataFrame]:
    """
    Compute fuel energy fractions for Light Medium (LT + MT combined).
    These drive historical technology market shares.

    Sources:
      Table 35: LT fuel (NG, Gasoline, Diesel, Ethanol, Biodiesel, Propane)
      Table 36: MT fuel (Gasoline, Diesel, Ethanol, Biodiesel) — match_n=0 for MT section
    """
    t35 = tables.get(f'Table {TABLE_FREIGHT_LT}')
    t36 = tables.get(f'Table {TABLE_FREIGHT_MHVT}')

    energy: dict[str, pd.Series] = {}

    if t35 is not None:
        for key, candidates in [
            ('lt_ng',       ['Natural gas',    'Natural Gas']),
            ('lt_gasoline', ['Motor gasoline', 'Motor Gasoline']),
            ('lt_diesel',   ['Diesel fuel oil', 'Diesel Fuel Oil']),
            ('lt_ethanol',  ['Ethanol']),
            ('lt_biodiesel',['Biodiesel fuel', 'Biodiesel Fuel']),
            ('lt_propane',  ['Propane']),
        ]:
            s = _try_labels(t35, candidates, match_n=0)
            if not s.dropna().empty:
                energy[key] = s.fillna(0.0) * FUEL_SCALE

    if t36 is not None:
        for key, candidates, mn in [
            ('mt_gasoline', ['Motor gasoline', 'Motor Gasoline'], 0),
            ('mt_diesel',   ['Diesel fuel oil', 'Diesel Fuel Oil'], 0),
            ('mt_ethanol',  ['Ethanol'], 0),
            ('mt_biodiesel',['Biodiesel fuel', 'Biodiesel Fuel'], 0),
        ]:
            s = _try_labels(t36, candidates, match_n=mn)
            if not s.dropna().empty:
                energy[key] = s.fillna(0.0) * FUEL_SCALE

    if not energy:
        return []

    # Aggregate by fuel type
    fuel_agg: dict[str, pd.Series] = {}
    all_yrs = sorted(set.union(*[set(s.index) for s in energy.values()]))

    def _add(key: str, sources: list[str]) -> None:
        combined = pd.Series(0.0, index=all_yrs)
        for src in sources:
            if src in energy:
                combined = combined + energy[src].reindex(all_yrs).fillna(0.0)
        if combined.sum() > 0:
            fuel_agg[key] = combined

    _add('Diesel',      ['lt_diesel',   'mt_diesel'])
    _add('Gasoline',    ['lt_gasoline', 'mt_gasoline'])
    _add('Natural Gas', ['lt_ng'])
    _add('Ethanol',     ['lt_ethanol',  'mt_ethanol'])
    _add('Biodiesel',   ['lt_biodiesel','mt_biodiesel'])
    _add('Propane',     ['lt_propane'])

    if not fuel_agg:
        return []

    total_energy = sum(fuel_agg.values()).replace(0, np.nan)

    frames = []
    for fuel_label, fuel_s in fuel_agg.items():
        share = (fuel_s / total_energy).dropna()
        if not share.empty:
            var_name = f"lm_fuel_{fuel_label.lower().replace(' ', '_')}"
            frames.append(_long(province, var_name, '', 'intermediate', 'fraction', share))

    return frames


# ==============================================================================
# PROJECTION + DERIVED COMPUTATION
# ==============================================================================


def apply_extensions(df: pl.DataFrame, province: str, params: dict) -> pl.DataFrame:
    """
    Project all freight mode M·tkm series to 2100 via CAGR, then compute:
      - total_ktkm (k*tkm, to 2100)
      - Mode shares: Off-Road, Land, Marine, Air (% of total, to 2100)
      - Sub-mode shares: Light Medium, Heavy within Land (% of Land, to 2100)
      - Heavy tech shares: Trucks vs Rail (% of Heavy, historical → flat to 2100)
      - Light Medium fuel tech shares (% of LM, historical → flat to 2100)
      - Light Medium tech output (k*tkm, historical)
      - Heavy Trucks output (k*tkm, historical)
    """
    all_years  = list(range(2000, PROJ_HORIZON + 1))
    hist_years = list(range(2000, LAST_HIST_YEAR + 1))
    reg        = province
    proj_start = LAST_HIST_YEAR + 1

    ktkm_cfg = params.get('ktkm_assumptions', {})

    # mode_name maps to assumptions CSV key and CAGR defaults for YT/NT/NU
    _modes_meta = {
        'lt_ktkm':      'Light Trucks',
        'mt_ktkm':      'Medium Trucks',
        'ht_ktkm':      'Heavy Trucks',
        'rail_ktkm':    'Rail',
        'air_ktkm':     'Aviation',
        'marine_ktkm':  'Marine',
        'offroad_ktkm': 'Off-Road',
        # Vehicle-km from Table 37 — projected with same CAGR as M·tkm counterparts
        'lt_m_vkm':  'Light Trucks',
        'mt_m_vkm':  'Medium Trucks',
        'ht_m_vkm':  'Heavy Trucks',
        # Stock (k·vehicles) — needed for per-vehicle output calculation
        'lt_stock':  'Light Trucks',
        'mt_stock':  'Medium Trucks',
        'ht_stock':  'Heavy Trucks',
    }

    full: dict[str, pd.Series] = {}

    # For territories (YT, NT, NU), fall back to BC assumptions
    cfg_region = reg if reg not in ('YT', 'NT', 'NU') else 'BC'

    for var, mode_name in _modes_meta.items():
        hist = _series_from_df(df, var)
        cfg  = ktkm_cfg.get(mode_name, {}).get(cfg_region, {})
        s2019 = cfg.get('share_of_2019', 0.9)
        c23   = cfg.get('ref_cagr_2023',  0.0)
        c51   = cfg.get('ref_cagr_2051',  0.0)

        ov = CAGR_OVERRIDES.get((reg, mode_name))
        if ov is not None:
            c23, c51 = ov

        s = pd.Series(np.nan, index=all_years, dtype=float)
        for y in hist.index:
            if y in s.index:
                s[y] = hist[y]

        # 2022 COVID adjustment
        base_2019 = s.get(2019, np.nan)
        if pd.notna(base_2019) and s2019 > 0 and LAST_HIST_YEAR <= 2022 and 2022 in s.index:
            s[2022] = base_2019 * s2019

        # Project 2023–2050
        for y in range(proj_start, 2051):
            prev = s.get(y - 1, np.nan)
            s[y] = prev * (1.0 + c23) if pd.notna(prev) else np.nan

        # Project 2051–2100
        for y in range(2051, PROJ_HORIZON + 1):
            prev = s.get(y - 1, np.nan)
            s[y] = prev * (1.0 + c51) if pd.notna(prev) else np.nan

        full[var] = s

    # --- Compute aggregate mode series ---
    lt_s      = full['lt_ktkm']
    mt_s      = full['mt_ktkm']
    ht_s      = full['ht_ktkm']
    rail_s    = full['rail_ktkm']
    air_s     = full['air_ktkm']
    marine_s  = full['marine_ktkm']
    offroad_s = full['offroad_ktkm']

    lm_s    = lt_s.add(mt_s, fill_value=0).where(lt_s.notna() | mt_s.notna())
    land_s  = (lm_s.fillna(0) + ht_s.fillna(0) + rail_s.fillna(0))
    land_s  = land_s.where(lm_s.notna() | ht_s.notna() | rail_s.notna())
    heavy_s = (ht_s.fillna(0) + rail_s.fillna(0)).where(ht_s.notna() | rail_s.notna())
    # Off-road excluded from total so that Land + Marine + Air = 1
    total_s = (land_s.fillna(0) + air_s.fillna(0) + marine_s.fillna(0))
    total_s = total_s.where(land_s.notna() | air_s.notna() | marine_s.notna())

    total_nn  = total_s.replace(0, np.nan)
    land_nn   = land_s.replace(0, np.nan)
    heavy_nn  = heavy_s.replace(0, np.nan)

    # --- Mode shares (Land + Marine + Air sum to 1) ---
    land_share   = (_safe_div(land_s.reindex(all_years).fillna(0),   total_nn)).dropna()
    marine_share = (_safe_div(marine_s.reindex(all_years).fillna(0), total_nn)).dropna()
    air_share    = (_safe_div(air_s.reindex(all_years).fillna(0),    total_nn)).dropna()

    lm_share_land    = (_safe_div(lm_s.reindex(all_years).fillna(0),  land_nn)).dropna()
    heavy_share_land = (_safe_div(heavy_s.reindex(all_years).fillna(0), land_nn)).dropna()

    truck_share_heavy = (_safe_div(ht_s.reindex(all_years).fillna(0), heavy_nn)).dropna()
    rail_share_heavy  = (_safe_div(rail_s.reindex(all_years).fillna(0), heavy_nn)).dropna()

    frames: list[pl.DataFrame] = []
    source_tag = 'CEUD'

    # --- total k*tkm (Land + Marine + Air only; M·tkm × 1000 = k*tkm) ---
    total_ktkm_out = (total_s * 1000.0).dropna()
    if not total_ktkm_out.empty:
        frames.append(_long(province, 'total_ktkm', '', 'service_request', 'k*tkm',
                            total_ktkm_out, source_tag))

    # --- Off-road: ratio relative to (Land + Marine + Air) total, same base as other mode shares ---
    offroad_share_out = (_safe_div(offroad_s.reindex(all_years), total_nn)).dropna()
    if not offroad_share_out.empty:
        frames.append(_long(province, 'Freight.Off-Road', '', 'service_request', '% of k*tkm',
                            offroad_share_out, source_tag))

    # --- Top-level mode shares (Land + Marine + Air = 1) ---
    for var, s in [
        ('Freight.Land',   land_share),
        ('Freight.Marine', marine_share),
        ('Freight.Air',    air_share),
    ]:
        if not s.empty:
            frames.append(_long(province, var, '', 'service_request', '% of k*tkm', s, source_tag))

    # --- Land sub-mode shares ---
    for var, s in [
        ('Freight.Land.Light Medium', lm_share_land),
        ('Freight.Land.Heavy',        heavy_share_land),
    ]:
        if not s.empty:
            frames.append(_long(province, var, '', 'service_request', '% of Land k*tkm', s, source_tag))

    # --- Heavy: Trucks vs Rail market shares (historical → flat to 2100) ---
    for cat, s in [('Trucks', truck_share_heavy), ('Rail', rail_share_heavy)]:
        s_hist = s.reindex(hist_years).dropna()
        s_ext  = _extend_flat(s_hist, proj_start, PROJ_HORIZON)
        if not s_ext.empty:
            frames.append(_long(province, 'Freight.Land.Heavy', cat,
                                'market_share_total', '% of Heavy k*tkm', s_ext, source_tag))

    # --- Per-vehicle output: M·tkm (Table 35/36) / stock (Table 37) = k*tkm/vehicle ---
    lt_stock_s = full['lt_stock']
    mt_stock_s = full['mt_stock']
    ht_stock_s = full['ht_stock']

    lm_stock_s = lt_stock_s.add(mt_stock_s, fill_value=0).where(
        lt_stock_s.notna() | mt_stock_s.notna()
    )

    lm_per_veh = _safe_div(lm_s, lm_stock_s.replace(0, np.nan))   # k*tkm/vehicle
    ht_per_veh = _safe_div(full['ht_ktkm'], ht_stock_s.replace(0, np.nan))  # k*tkm/vehicle

    # --- Light Medium fuel tech market shares (historical → flat to 2100) ---
    for fuel_label in ['Diesel', 'Gasoline', 'Natural Gas', 'Ethanol', 'Biodiesel', 'Propane']:
        var_name = f"lm_fuel_{fuel_label.lower().replace(' ', '_')}"
        fuel_share_hist = _series_from_df(df, var_name).reindex(hist_years).dropna()
        if fuel_share_hist.empty:
            continue

        # Market share (historical → flat to 2100)
        share_ext = _extend_flat(fuel_share_hist, proj_start, PROJ_HORIZON)
        frames.append(_long(province, 'Light Medium', fuel_label,
                            'market_share_total', '% of Light Medium k*tkm', share_ext, source_tag))


    # --- Light Medium total output (k*tkm/vehicle, all years to 2100) ---
    lm_out = lm_per_veh.reindex(all_years).dropna()
    if not lm_out.empty:
        frames.append(_long(province, 'Light Medium', '', 'output', 'k*tkm',
                            lm_out, source_tag))

    # --- Heavy Trucks total output (k*tkm/vehicle, all years to 2100) ---
    ht_out = ht_per_veh.reindex(all_years).dropna()
    if not ht_out.empty:
        frames.append(_long(province, 'Heavy Trucks', '', 'output', 'k*tkm',
                            ht_out, source_tag))

    if not frames:
        return df
    return pl.concat(frames, how='diagonal_relaxed').sort(['province', 'variable', 'category', 'year'])


# ==============================================================================
# MAIN EXTRACTION FUNCTION
# ==============================================================================


def extract_all_data(
    province_code: str,
    can_params: Optional[dict] = None,
    projection_params: Optional[dict] = None,
    ipcc_offroad: Optional[dict] = None,
    ipcc_marine: Optional[dict] = None,
) -> pl.DataFrame:
    """
    Extract all freight k*tkm for a province into a single long-format DataFrame.

    Returns
    -------
    pl.DataFrame with columns:
        province, variable, category, parameter, unit, source, year, value
    """
    province = province_code.upper()
    if province not in PROVINCES:
        raise ValueError(
            f"Invalid province code: {province_code!r}. "
            f"Valid codes: {list(PROVINCES.keys())}"
        )

    if can_params is None:
        can_params = load_can_freight_params()
    if projection_params is None:
        projection_params = load_projection_params()
    if ipcc_offroad is None:
        ipcc_offroad = load_ipcc_offroad()
    if ipcc_marine is None:
        ipcc_marine = load_ipcc_domestic_marine()

    tables = load_tables(province)

    frames: list[pl.DataFrame] = []
    frames += extract_truck_ktkm(province, tables)
    frames += extract_truck_mvkm(province, tables)
    frames += extract_rail_ktkm(province, tables, can_params)
    frames += extract_air_ktkm(province, tables, can_params)
    frames += extract_marine_ktkm(province, can_params, ipcc_marine)
    frames += extract_offroad_ktkm(province, ipcc_offroad)
    frames += extract_lm_fuel_shares(province, tables)

    if not frames:
        raise RuntimeError(f"No freight data extracted for {province}.")

    df = pl.concat(frames, how='diagonal_relaxed')
    df = apply_extensions(df, province, projection_params)

    # Strip intermediate variables from final output
    df = df.filter(pl.col('parameter') != 'intermediate')

    return df.sort(['province', 'variable', 'category', 'year'])


# ==============================================================================
# BCT SPLIT  (same logic as transportation_passenger.py)
# ==============================================================================


def _load_bct_population_shares(
    pop_path: Path = STATSCAN_POP_PATH,
) -> dict[str, pd.Series]:
    """
    Compute annual population shares of BC, YT, NT, NU within the BCT region.
    Source: Statistics Canada table 17-10-0009-01 (quarterly, persons).
    Shares hold the last observed value constant for projection years.
    """
    _GEO_MAP = {
        'British Columbia':     'BC',
        'Yukon':                'YT',
        'Northwest Territories':'NT',
        'Nunavut':              'NU',
    }

    df = (
        pl.read_csv(str(pop_path))
        .filter(pl.col('GEO').is_in(list(_GEO_MAP.keys())))
        .filter(pl.col('VALUE').is_not_null())
        .select(['REF_DATE', 'GEO', 'VALUE'])
        .with_columns([
            pl.col('REF_DATE').str.slice(0, 4).cast(pl.Int32).alias('year'),
            pl.col('GEO').replace(_GEO_MAP).alias('prov'),
            pl.col('VALUE').cast(pl.Float64),
        ])
    )

    annual = (
        df.group_by(['year', 'prov'])
        .agg([
            pl.col('VALUE').mean().alias('pop'),
            pl.col('VALUE').count().alias('n_quarters'),
        ])
        .filter(pl.col('n_quarters') == 4)
        .sort(['prov', 'year'])
    )

    pop: dict[str, pd.Series] = {}
    for prov in BCT_REGIONS:
        sub = annual.filter(pl.col('prov') == prov)
        pop[prov] = pd.Series(
            sub['pop'].to_list(), index=sub['year'].to_list(), dtype=float
        )

    hist_years = sorted(set.union(*[set(s.index) for s in pop.values()]))
    all_years  = list(range(min(hist_years), PROJ_HORIZON + 1))

    bct_total = sum(
        pop[p].reindex(hist_years).fillna(0) for p in BCT_REGIONS
    ).replace(0, np.nan)

    shares_hist = {
        prov: (pop[prov].reindex(hist_years) / bct_total).dropna()
        for prov in BCT_REGIONS
    }

    shares_full: dict[str, pd.Series] = {}
    for prov in BCT_REGIONS:
        s = pd.Series(np.nan, index=all_years, dtype=float)
        s.update(shares_hist[prov])
        last_val = shares_hist[prov].iloc[-1] if not shares_hist[prov].empty else np.nan
        s = s.fillna(last_val)
        shares_full[prov] = s.dropna()

    return shares_full


def _split_bct(bct_df: pl.DataFrame,
               shares: dict[str, pd.Series]) -> dict[str, pl.DataFrame]:
    """
    Split a BCT-combined DataFrame into separate BC, YT, NT, NU DataFrames.
    Absolute quantities (unit 'k*tkm') are scaled by population share.
    Fractions and market shares are copied unchanged.

    Territories (YT, NT, NU) have no marine freight.  After the population-share
    split, their marine share is zeroed, total_ktkm is reduced by the marine
    component, and the remaining mode shares are renormalized to sum to 1.
    """
    # Per-vehicle output rows (parameter='output') must NOT be pop-scaled —
    # they are a rate (k*tkm/vehicle), not a total.
    scale_mask = pl.col('unit').is_in(['k*tkm']) & (pl.col('parameter') != 'output')
    _RENORM_VARS = ('Freight.Land', 'Freight.Air')
    _TERRITORY_REGIONS = ('YT', 'NT', 'NU')

    # BCT marine share per year — needed to correct territory totals/shares
    marine_share_df = (
        bct_df
        .filter(pl.col('variable') == 'Freight.Marine')
        .select(['year', 'value'])
        .rename({'value': 'marine_share'})
    )
    has_marine = len(marine_share_df) > 0

    def _share_frame(prov: str) -> pl.DataFrame:
        s = shares[prov]
        return pl.DataFrame({
            'year':      [int(y) for y in s.index],
            'pop_share': s.values.tolist(),
        }).with_columns(pl.col('year').cast(pl.Int32))

    results: dict[str, pl.DataFrame] = {}
    for prov in BCT_REGIONS:
        share_pl = _share_frame(prov)

        activity_df = (
            bct_df.filter(scale_mask)
            .join(share_pl, on='year', how='left')
            .with_columns(
                (pl.col('value') * pl.col('pop_share').fill_null(0.0)).alias('value'),
                pl.lit(prov).alias('province'),
            )
            .drop('pop_share')
        )

        non_activity_df = (
            bct_df.filter(~scale_mask)
            .with_columns(pl.lit(prov).alias('province'))
        )

        combined = pl.concat(
            [activity_df, non_activity_df], how='diagonal_relaxed'
        )

        if prov in _TERRITORY_REGIONS and has_marine:
            # non_marine_frac = 1 - BCT_marine_share  (per year)
            correction = marine_share_df.with_columns(
                (1.0 - pl.col('marine_share')).alias('non_marine_frac')
            ).select(['year', 'non_marine_frac'])

            # total_ktkm: remove marine component (value × non_marine_frac)
            tkm_fixed = (
                combined.filter(pl.col('variable') == 'total_ktkm')
                .join(correction, on='year', how='left')
                .with_columns(
                    (pl.col('value') * pl.col('non_marine_frac').fill_null(1.0)).alias('value')
                )
                .drop('non_marine_frac')
            )

            # Freight.Marine: set to 0
            marine_zeroed = (
                combined.filter(pl.col('variable') == 'Freight.Marine')
                .with_columns(pl.lit(0.0).alias('value'))
            )

            # Remaining mode shares: renormalize by dividing by non_marine_frac
            # so that Land + Air + Off-Road sum to 1 after marine is removed
            shares_renormed = (
                combined.filter(pl.col('variable').is_in(list(_RENORM_VARS)))
                .join(correction, on='year', how='left')
                .with_columns(
                    (pl.col('value') / pl.col('non_marine_frac').fill_null(1.0)).alias('value')
                )
                .drop('non_marine_frac')
            )

            other = combined.filter(
                ~pl.col('variable').is_in(
                    ['total_ktkm', 'Freight.Marine'] + list(_RENORM_VARS)
                )
            )

            combined = pl.concat(
                [tkm_fixed, marine_zeroed, shares_renormed, other],
                how='diagonal_relaxed',
            )

        results[prov] = combined.sort(['variable', 'category', 'year'])

    return results


# ==============================================================================
# BATCH EXTRACTION
# ==============================================================================


def extract_all_provinces(
    province_codes: Optional[list[str]] = None,
) -> dict[str, pl.DataFrame]:
    """
    Extract freight data for all provinces, then split BC (BCT) into
    BC, YT, NT, NU using population shares.

    Returns dict keyed by province code: AB BC MB NB NL NS ON PE QC SK YT NT NU.
    """
    if province_codes is None:
        province_codes = list(PROVINCES.keys())

    can_params   = load_can_freight_params()
    proj_params  = load_projection_params()
    ipcc_offroad = load_ipcc_offroad()
    ipcc_marine  = load_ipcc_domestic_marine()
    results      = {}
    failed       = []

    for prov in province_codes:
        try:
            results[prov] = extract_all_data(prov, can_params, proj_params, ipcc_offroad, ipcc_marine)
        except Exception as exc:
            failed.append((prov, str(exc)))

    # Split BC (BCT) into BC + YT + NT + NU via population proxy
    if 'BC' in results:
        try:
            bct_shares = _load_bct_population_shares()
            split = _split_bct(results['BC'], bct_shares)
            results.update(split)
        except Exception as exc:
            failed.append(('BCT-split', str(exc)))

    for prov, err in failed:
        print(f'Warning: {prov} failed: {err}')

    return results


# ==============================================================================
# MAIN
# ==============================================================================


def main(
    province_codes: Optional[list[str]] = None,
    output_dir: Path = OUTPUT_DIR,
    export_csv: bool = True,
) -> dict[str, pl.DataFrame]:
    """Run the full transportation freight pipeline and optionally export CSV."""
    if province_codes is None:
        province_codes = list(PROVINCES.keys())

    if export_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = extract_all_provinces(province_codes)

    if export_csv and results:
        all_frames = list(results.values())
        combined   = pl.concat(all_frames, how='diagonal_relaxed')
        combined   = combined.with_columns(
            pl.when(pl.col('year') <= LAST_HIST_YEAR)
            .then(pl.lit('CEUD'))
            .otherwise(pl.lit('Assumptions'))
            .alias('source')
        )
        combined = combined.filter(
            ~((pl.col('parameter') == 'market_share_total')
              & (pl.col('year') > LAST_HIST_YEAR))
        )
        combined = combined.sort(['province', 'variable', 'category', 'year'])

        output_file = output_dir / 'transportation_freight.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f'\nTransportation freight extraction complete')
        print(f'   Total rows:          {combined.height:,}')
        print(f'   Provinces processed: {combined["province"].n_unique()}')
        print(f'   Variables:           {sorted(combined["variable"].unique().to_list())}')
        print(f'   Years covered:       {combined["year"].min()} -- {combined["year"].max()}')
        print(f'   Saved to:            {output_file}')
        combined = combined.rename({
            'province': 'Region',  'variable': 'Variable',  'category': 'Category',
            'parameter': 'Parameter', 'unit': 'Unit', 'source': 'Source',
            'year': 'Year', 'value': 'Value',
        })
        combined.write_csv(str(output_file))

    return results


if __name__ == '__main__':
    main()
