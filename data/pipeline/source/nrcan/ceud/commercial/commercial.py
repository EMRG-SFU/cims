"""
Commercial Pipeline

Extracts and processes commercial building data from the NRCan CEUD
(Comprehensive Energy Use Database) for all Canadian regions.

Key behavioural notes
---------------------
- AT (Atlantic) groups NB + NS + PE + NL together.
- BC groups BC + Territories together.
- BC has Marine and Cold climate HVAC (80 % Marine / 20 % Cold floorspace split).
- NG heating is split by efficiency tier using ng_eff_assumptions_commercial.csv.
- Hot water technologies are NOT split by building shell type.
- All intermediate data is held as Polars DataFrames in long format:
      (region, variable, category, parameter, unit, year, value)

Suppression handling
--------------------
CER shares (vDmd-CIMS.csv commercial Space Heating / Water Heating demand):
    Loaded via _build_comm_cer_shares(), which sums CER's per-activity-sector
    demand by fuel/geo/year -- no suppression codes to handle (CER's export
    is already clean numeric). A zero group total (no measured demand for a
    fuel/geo/year) still yields a real 0.0 share rather than an undefined
    one; a geo missing from the CSV entirely yields NaN, handled by the
    population-proxy fallback in the disaggregation step.

CEUD technology splits (NRCan Excel):
    Steam energy shares are occasionally suppressed for small territories.
    When a steam share exists for some years but is null for others, the
    null years are recovered as 1 − sum(all other known fuel shares),
    ensuring fuel-type shares sum to 1 for every year.
"""

from pathlib import Path
from typing import Optional
import csv
import sys

import polars as pl
import pandas as pd
import numpy as np

_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from mappings_conversions.control import CONTROLS
from pipeline.utils.extractors.nrcan_ceud import (
    get_row_series, row_to_series, pct_series, find_row_indices, extract_year_cols, _to_float,
)
from pipeline.utils.output_builder import pl_to_series, pl_get_scalar
from pipeline.utils.extractors.stats_can import build_population_shares
from pipeline.utils.data_extensions import (
    extend_series_trend_dampener,
    load_cagr_assumptions,
    compute_cagr,
    extend_cagr_periods,
)
from pipeline.utils.controls_conversions import DATA_START, PROJECTION_END, BASE_PATH as _CIMS_BASE

# ==============================================================================
# CONFIGURATION
# ==============================================================================

FIXED_DATA_DIR   = _CIMS_BASE / 'raw_data/fixed_data/commercial'
CALIBRATION_DIR  = _CIMS_BASE / 'calibration/commercial'
BASE_PATH        = _CIMS_BASE / 'raw_data/nrcan/ceud/commercial'
ASSUMPTIONS_CSV  = _CIMS_BASE / 'raw_data/assumptions/commercial_assumptions.csv'
ACTIVITY_CAGR_CSV = _CIMS_BASE / 'raw_data/assumptions/activity_cagr_projections.csv'
NG_EFFICIENCY_CSV = _CIMS_BASE / 'raw_data/assumptions/ng_eff_assumptions_commercial.csv'
OUTPUT_DIR       = _CIMS_BASE / 'processed_data/nrcan/ceud'
CER_DEMAND_CSV = _CIMS_BASE / 'raw_data/cer/cer_demand_data_update/vDmd-CIMS.csv'
POP_CSV        = _CIMS_BASE / 'raw_data/stats_can/population/1710000901.csv'
EFFICIENCY_XLS = _CIMS_BASE / 'raw_data/nrcan/ceud/residential/res_ca_e_32.xls'
YEARS            = list(range(DATA_START, PROJECTION_END + 1))
LAST_HIST_YEAR   = CONTROLS["last_data_year"]["ceud"]

# Per-region overrides — explicit annual rate per period, bypasses computed CAGR.
CAGR_OVERRIDES: dict[str, tuple[float, ...]] = {
}

REGIONS = {
    'AB': 'Alberta',
    'AT': 'Atlantic',
    'BC': 'British Columbia',
    'MB': 'Manitoba',
    'ON': 'Ontario',
    'QC': 'Quebec',
    'SK': 'Saskatchewan',
}

ACTIVITY_MAPPING = {
    'Wholesale Trade':                      'Wholesale',
    'Retail Trade':                         'Retail',
    'Transportation and Warehousing':       'Transportation and Warehousing',
    'Information and Cultural Industries':  'Information and Cultural',
    'Offices':                              'Offices',
    'Educational Services':                 'Educational',
    'Health Care and Social Assistance':    'Healthcare and Social Assistance',
    'Arts, Entertainment and Recreation':   'Arts Entertainment and Recreation',
    'Accommodation and Food Services':      'Accommodation and Food Services',
    'Other Services':                       'Other Services',
}

TOTAL_FLOORSPACE_TABLE = 1
END_USE_TABLE     = 2
FLOORSPACE_TABLES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
HVAC_TABLE       = 24
HOT_WATER_TABLE   = 26

FILE_NAME_MAP = {'BC': 'bct', 'AT': 'atl'}

# ==============================================================================
# END-USE INTENSITY CONSTANTS -- Lighting / Hot Water / Auxiliary Equipment
# ==============================================================================
#
# CEUD Table 2 reports region-wide totals for Lighting and Water Heating
# secondary energy use, and for "Auxiliary Equipment" -- a single lumped
# category that lumps Refrigeration, Cooking and Plug Load together with no
# further public breakdown. There is no CEUD source that reports these three
# separately.
#
# The constants below convert those CEUD energy totals (PJ) into the
# Buildings -> {Lighting, Hot Water, Refrigeration, Cooking, Plug Load}
# service_request intensities (m2 or "unit" per m2 of floorspace) that
# raw_data/fixed_data/commercial previously hardcoded. They come from the
# technology-level service_request (GJ per unit/m2) and year-2000
# market_share_total values already in that fixed data, which are identical
# across all 13 regions for every end-use except Hot Water (whose technology
# mix is itself CEUD-derived per region -- see hot_water_tech below) --
# so they're captured here once as the shared national engineering factors,
# and used to back the top-level intensity out of each region's own CEUD
# energy total: intensity = energy / (floorspace * weighted_gj_per_unit).

# Weighted GJ per service_request "unit", from each end-use's year-2000
# technology market_share_total x its own GJ/unit (Refrigeration: 100%
# Existing; Cooking: 76.6% Electric + 23.4% NG; Plug Load: 100% Std Eff).
AUX_TECH_GJ_PER_UNIT: dict[str, float] = {
    'refrigeration': 1.818181818,
    'cooking':       0.766 * 1.428571429 + 0.234 * 1.818181818,
    'plug_load':     0.014043,
}

# Lighting sub-service shares of total lit floorspace and each sub-service's
# weighted GJ/m2 (100% "Existing" technology at year 2000), combined into one
# weighted GJ/m2 factor for the whole Lighting end-use.
LIGHTING_SUBSERVICE_SHARES: dict[str, float] = {
    'General Area':    0.746420765,
    'Service Lighting': 0.214826958,
    'High Bay':        0.038752277,
}
LIGHTING_SUBSERVICE_GJ_PER_M2: dict[str, float] = {
    'General Area':    0.266470436,
    'Service Lighting': 0.271183964,
    'High Bay':        0.20330031,
}
LIGHTING_GJ_PER_M2: float = sum(
    LIGHTING_SUBSERVICE_SHARES[s] * LIGHTING_SUBSERVICE_GJ_PER_M2[s]
    for s in LIGHTING_SUBSERVICE_SHARES
)

# Hot Water technology total GJ per unit -- summed across every energy target
# a technology serves (e.g. an NG boiler burns Methane Blend for heat *and*
# draws Motive Power for pumps). Combined at runtime with each region's own
# (CEUD-derived) hot_water_tech market shares, since -- unlike the other
# end-uses -- Hot Water's technology mix already varies by region.
HOT_WATER_TECH_GJ_PER_UNIT: dict[str, float] = {
    'Natural Gas_Boiler_Medium Efficiency':    0.124 + 0.003435932,
    'Natural Gas_Boiler_High Efficiency':      0.104296772 + 0.003435932,
    'Electricity_Boiler_High Efficiency':      0.065185482 + 0.003435932,
    'Electricity_ASHP':                        0.003 + 0.003435932,
    'Light Fuel Oil_Boiler_Medium Efficiency': 0.124 + 0.003435932,
    'Heavy Fuel Oil_Boiler_Medium Efficiency': 0.124 + 0.003435932,
    'Propane_Boiler_Medium Efficiency':        0.124 + 0.003435932,
}

# National split of "Auxiliary Equipment" energy across Refrigeration /
# Cooking / Plug Load. Backed out from the ratios previously applied
# identically (as a fixed, unit-level split) across 12 of the 13 regions in
# raw_data/fixed_data/commercial -- BC's fixed file diverged from this shared
# pattern. Cooking:Refrigeration is constant; Plug Load:Refrigeration grows
# over time (plug load intensity was assumed to triple from 2000-2040, flat
# after) and is interpolated between these checkpoint years.
AUX_SPLIT_YEARS: list[int] = [2000, 2005, 2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
COOK_TO_REFRIG_UNIT_RATIO: float = 0.033538333 / 0.034414309
PLUG_TO_REFRIG_UNIT_RATIO: list[float] = [
    v / 0.034414309 for v in
    (4.301788657, 6.063233959, 7.417562032, 8.560236378, 9.567396056,
     10.47819105, 11.31591393, 12.09575663, 12.82827926, 12.82827926, 12.82827926)
]


SPACE_HEATING_TABLE = 25

# Universal (region-independent) year-2000 Buildings.Shell.<Activity> -> HVAC
# (Cold) service_request rates (GJ per m2 of that activity's floorspace),
# from raw_data/fixed_data/commercial -- identical across all 13 regions.
# 100% of year-2000 market share sits on "Std" (LEED Silver/Platinum only
# become available from 2010/2015 on), so Std alone anchors the weighted
# average, exactly like AUX_TECH_GJ_PER_UNIT. compute_hvac_service_requests()
# rescales all three tiers by the same per-activity/region/year factor to
# match CEUD Table 25 (Space Heating by Activity Type), preserving each
# tier's relative envelope-efficiency improvement over Std. Keyed by the
# same CIMS activity name used elsewhere (ACTIVITY_MAPPING values).
SHELL_TECH_GJ_PER_M2: dict[str, dict[str, float]] = {
    'Wholesale':                          {'Std': 0.967671128, 'LEED Silver': 0.783813613, 'LEED Platinum': 0.53221912},
    'Retail':                             {'Std': 0.943654332, 'LEED Silver': 0.764360009, 'LEED Platinum': 0.519009883},
    'Transportation and Warehousing':     {'Std': 0.811187758, 'LEED Silver': 0.657062083, 'LEED Platinum': 0.446153267},
    'Information and Cultural':           {'Std': 0.766898664, 'LEED Silver': 0.621187918, 'LEED Platinum': 0.421794266},
    'Offices':                            {'Std': 0.8558116,   'LEED Silver': 0.693207398, 'LEED Platinum': 0.47069638},
    'Educational':                        {'Std': 0.927642918, 'LEED Silver': 0.751390765, 'LEED Platinum': 0.510203605},
    'Healthcare and Social Assistance':   {'Std': 1.429507969, 'LEED Silver': 1.157901455, 'LEED Platinum': 0.786229383},
    'Arts Entertainment and Recreation':  {'Std': 1.1343701,   'LEED Silver': 0.918839781, 'LEED Platinum': 0.623903555},
    'Accommodation and Food Services':    {'Std': 1.512157371, 'LEED Silver': 1.224847471, 'LEED Platinum': 0.831686554},
    'Other Services':                     {'Std': 0.981626655, 'LEED Silver': 0.795117592, 'LEED Platinum': 0.539894659},
}

# HVAC (Marine) (BC only) = HVAC (Cold) x this ratio -- a fixed,
# activity/technology-independent national climate correction already
# embedded identically across every Marine rate in raw_data/fixed_data.
# CEUD does not distinguish climate zones within a region, so there's no
# CEUD-based way to recover this independently -- it's carried forward as-is.
MARINE_TO_COLD_RATIO: float = 0.572456460


def _aux_energy_shares(year: int) -> dict[str, float]:
    """
    Fraction of Auxiliary Equipment energy attributed to each of
    Refrigeration / Cooking / Plug Load for a given year, per the national
    split above. Interpolates PLUG_TO_REFRIG_UNIT_RATIO between its
    checkpoint years (flat outside [2000, 2050]) and converts the resulting
    unit-level split into an energy-level split via AUX_TECH_GJ_PER_UNIT.
    """
    plug_ratio = float(np.interp(year, AUX_SPLIT_YEARS, PLUG_TO_REFRIG_UNIT_RATIO))

    e_refrig = 1.0 * AUX_TECH_GJ_PER_UNIT['refrigeration']
    e_cook   = COOK_TO_REFRIG_UNIT_RATIO * AUX_TECH_GJ_PER_UNIT['cooking']
    e_plug   = plug_ratio * AUX_TECH_GJ_PER_UNIT['plug_load']
    total = e_refrig + e_cook + e_plug

    return {
        'refrigeration': e_refrig / total,
        'cooking':       e_cook / total,
        'plug_load':     e_plug / total,
    }


# ==============================================================================
# HELPERS
# ==============================================================================


def _long(region: str, variable: str, category: str, parameter: str,
          unit: str, series: pd.Series, source: str = 'CEUD') -> pl.DataFrame:
    """
    Convert a year-indexed pd.Series to a Polars long-format DataFrame.

    Builds directly from Python lists to avoid the pyarrow dependency that
    pl.from_pandas() requires for object-dtype string columns.
    """
    years  = [int(y)   for y, v in series.items() if pd.notna(v)]
    values = [float(v) for y, v in series.items() if pd.notna(v)]
    n = len(years)
    return pl.DataFrame({
        'region':    [region]    * n,
        'variable':  [variable]  * n,
        'category':  [category]  * n,
        'parameter': [parameter] * n,
        'unit':      [unit]      * n,
        'source':    [source]    * n,
        'year':      years,
        'value':     values,
    }).with_columns(pl.col('year').cast(pl.Int32))


# ==============================================================================
# PARAMETER LOADING
# ==============================================================================

def load_projection_params(assumptions_csv: Path = ASSUMPTIONS_CSV,
                           activity_cagr_csv: Path = ACTIVITY_CAGR_CSV) -> dict:
    """
    Load commercial projection parameters.

    Floorspace uses the shared activity CAGR file (activity_cagr_projections.csv),
    sector 'Commercial': CAGR Period, dampener periods — same pattern as activity scripts.

    Building shell uses commercial_assumptions.csv.

    Returns
    -------
    dict with keys 'floorspace' and 'building_shell_shares'.
    Empty dict if files not found.
    """
    def parse_pct(val) -> float | None:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val).strip()
        if s.lower() in ('trend', 'remainder', ''):
            return None
        try:
            return float(s.replace('%', '').strip()) / 100.0
        except ValueError:
            return None

    def parse_year(val) -> int | None:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val).strip().lower()
        if s == 'first':
            return LAST_HIST_YEAR + 1
        try:
            return int(float(s))
        except ValueError:
            return None

    def parse_period(row, prefix) -> tuple[int, int] | None:
        start = parse_year(row.get(f'{prefix} start'))
        end   = parse_year(row.get(f'{prefix} end'))
        if start is None or end is None:
            return None
        return start, end + 1

    try:
        raw = pd.read_csv(assumptions_csv)
    except FileNotFoundError:
        return {}

    params = {}

    # -- 1. Floorspace — activity-style CAGR (activity_cagr_projections.csv)
    cagr_start, cagr_end, periods = load_cagr_assumptions('Commercial', activity_cagr_csv)
    params['floorspace'] = {
        'cagr_start': cagr_start,
        'cagr_end':   cagr_end,
        'periods':    periods,
    }

    # -- 2. Building shell shares — trend then dampener, per activity ----------
    bs_rows = raw[raw['Variable'] == 'Building Shell']
    first_bs     = bs_rows.iloc[0]
    trend_yrs    = parse_period(first_bs, '1st period') or (LAST_HIST_YEAR + 1, 2031)
    decline1_yrs = parse_period(first_bs, '2nd period') or (2031, 2051)
    decline2_yrs = parse_period(first_bs, '3rd period') or (2051, 2101)

    activity_rates = {}
    for _, row in bs_rows.iterrows():
        activity = str(row['Variable.1']).strip()
        r1 = parse_pct(row['2nd period rate'])
        r2 = parse_pct(row['3rd period rate'])
        activity_rates[activity] = (
            r1 if r1 is not None else -0.10,
            r2 if r2 is not None else -0.10,
        )

    params['building_shell_shares'] = {
        'method': 'trend_dampener',
        'trend_start': 2000,
        'trend_end': LAST_HIST_YEAR,
        'trend_period': trend_yrs,
        'activity_rates': activity_rates,
        'decline1_years': decline1_yrs,
        'decline2_years': decline2_yrs,
    }

    return params
def load_ng_efficiency_splits(ng_efficiency_csv: Path = NG_EFFICIENCY_CSV) -> dict:
    """
    Load NG efficiency tier splits from the assumptions CSV.

    Returns
    -------
    dict  { year: {'low': float, 'medium': float, 'high': float} }
    Falls back to equal thirds if the file cannot be read.
    """
    TECH_KEY_MAP = {
        'Natural Gas_Furnace_Low Efficiency':    'low',
        'Natural Gas_Furnace_Medium Efficiency': 'medium',
        'Natural Gas_Furnace_High Efficiency':   'high',
    }
    default = {year: {'low': 1/3, 'medium': 1/3, 'high': 1/3} for year in YEARS}

    try:
        df = pl.read_csv(str(ng_efficiency_csv), has_header=True)
    except Exception:
        return default

    tech_col = df.columns[0]
    year_cols = df.columns[1:]
    splits: dict[int, dict[str, float]] = {}

    for row in df.iter_rows(named=True):
        tech_name = str(row[tech_col]).strip().lstrip('\ufeff')
        tier = TECH_KEY_MAP.get(tech_name)
        if tier is None:
            continue
        for yr_str in year_cols:
            try:
                yr = int(yr_str)
                val = row[yr_str]
                if val is not None:
                    splits.setdefault(yr, {})[tier] = float(val)
            except (ValueError, TypeError):
                continue

    if not splits:
        return default

    last_year = max(splits.keys())
    last_vals  = splits[last_year]
    for year in YEARS:
        if year not in splits:
            splits[year] = last_vals.copy()

    return splits


# ==============================================================================
# TABLE LOADING
# ==============================================================================

def load_tables(region_code: str) -> dict:
    """Load and clean CEUD Excel sheets for a commercial region."""
    region_lower = FILE_NAME_MAP.get(region_code.upper(), region_code.lower())
    file_path = BASE_PATH / f"com_{region_lower}_e.xls"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    table_numbers = ([TOTAL_FLOORSPACE_TABLE] + [END_USE_TABLE] + FLOORSPACE_TABLES +
                      [HVAC_TABLE] + [HOT_WATER_TABLE] + [SPACE_HEATING_TABLE])
    table_names = [f"Table {n}" for n in sorted(set(table_numbers))]

    def load_and_clean(sheet_name: str) -> pl.DataFrame:
        df = pl.read_excel(str(file_path), sheet_name=sheet_name, has_header=False)
        data_cols = df.columns[2:]
        cast_exprs = []
        for c in data_cols:
            if df[c].dtype in (pl.String, pl.Utf8):
                cast_exprs.append(
                    pl.col(c)
                    .str.strip_chars()
                    .cast(pl.Float64, strict=False)
                )
        if cast_exprs:
            df = df.with_columns(cast_exprs)
        return df

    return {name: load_and_clean(name) for name in table_names}


# ==============================================================================
# EXTRACTION — FLOORSPACE
# ==============================================================================

def extract_floorspace(region: str, tables: dict) -> list[pl.DataFrame]:
    """
    Extract total commercial floorspace and per-activity floorspace shares.

    Returns
    -------
    list of pl.DataFrame in long format.
    """
    t1 = tables[f"Table {TOTAL_FLOORSPACE_TABLE}"]
    total_raw = row_to_series(t1, "Total Floor Space (million m2)") * 1e6
    frames = [_long(region, 'total_floorspace', '', 'service_request', 'm2', total_raw)]

    cims_activities = list(ACTIVITY_MAPPING.values())
    for tbl_num, cims_name in zip(FLOORSPACE_TABLES, cims_activities):
        tbl = tables[f"Table {tbl_num}"]
        raw = row_to_series(tbl, "Floor Space (million m2)") * 1e6
        frames.append(_long(region, 'floorspace_by_activity', cims_name,
                            'service_request', 'm2', raw))

    return frames


# ==============================================================================
# EXTRACTION — END-USE ENERGY (LIGHTING / WATER HEATING / AUXILIARY EQUIPMENT)
# ==============================================================================

def extract_end_use_energy(region: str, tables: dict) -> list[pl.DataFrame]:
    """
    Extract region-wide Lighting, Water Heating, and Auxiliary Equipment
    secondary energy use (all activity types combined) from Table 2.

    These feed compute_enduse_service_requests(), which converts them into
    the Buildings -> {Lighting, Hot Water, Refrigeration, Cooking, Plug Load}
    service_request intensities.

    Returns
    -------
    list of pl.DataFrame in long format, values in GJ.
    """
    t2 = tables[f"Table {END_USE_TABLE}"]
    lighting = row_to_series(t2, "Lighting", match_n=0) * 1e6
    water    = row_to_series(t2, "Water Heating", match_n=0) * 1e6
    aux      = row_to_series(t2, "Auxiliary Equipment", match_n=0) * 1e6
    heating  = row_to_series(t2, "Space Heating", match_n=0) * 1e6
    cooling  = row_to_series(t2, "Space Cooling", match_n=0) * 1e6

    return [
        _long(region, 'lighting_energy', '', 'service_request', 'GJ', lighting),
        _long(region, 'water_heating_energy', '', 'service_request', 'GJ', water),
        _long(region, 'aux_equipment_energy', '', 'service_request', 'GJ', aux),
        _long(region, 'space_heating_energy', '', 'service_request', 'GJ', heating),
        _long(region, 'space_cooling_energy', '', 'service_request', 'GJ', cooling),
    ]


def _row_to_series_by_prefix(table: pl.DataFrame, label: str, match_n: int = 0) -> pd.Series:
    """
    Like row_to_series(), but falls back to a prefix match when no row label
    equals `label` exactly. CEUD activity-type row labels occasionally carry
    a trailing footnote digit that varies by table (e.g. "Offices2" in one
    table, "Offices3" in another), so an exact match is tried first and a
    startswith() match is used only if that fails.
    """
    idxs = find_row_indices(table, label)
    if len(idxs) <= match_n:
        label_col = table.columns[1]
        idxs = (
            table.select(pl.col(label_col).cast(pl.Utf8).str.strip_chars().alias('lab'))
                 .with_row_index('row')
                 .filter(pl.col('lab').str.starts_with(label))
                 .select('row')['row'].to_list()
        )
    if len(idxs) <= match_n:
        raise KeyError(f"Label '{label}' not found (need #{match_n + 1}, got {len(idxs)})")

    arr = table.to_numpy()
    years = extract_year_cols(table)
    return pd.Series({y: _to_float(arr[idxs[match_n], c]) for y, c in years})


def extract_space_heating_by_activity(region: str, tables: dict) -> list[pl.DataFrame]:
    """
    Extract Space Heating secondary energy use by activity type (Table 25) --
    the CEUD source compute_hvac_service_requests() uses to calibrate
    Buildings.Shell.<Activity> -> HVAC service_request rates.
    """
    tbl = tables[f"Table {SPACE_HEATING_TABLE}"]
    frames = []
    for ceud_label, cims_name in ACTIVITY_MAPPING.items():
        try:
            series = _row_to_series_by_prefix(tbl, ceud_label) * 1e6
        except KeyError:
            continue
        frames.append(_long(region, 'space_heating_by_activity', cims_name,
                            'service_request', 'GJ', series))
    return frames


# ==============================================================================
# EXTRACTION — BUILDING SHELL SHARES
# ==============================================================================

def extract_building_shell_shares(region: str,
                                   floorspace_df: pl.DataFrame) -> list[pl.DataFrame]:
    """
    Compute each activity's fraction of total floorspace.

    Parameters
    ----------
    floorspace_df : pl.DataFrame
        Output of extract_floorspace — must contain both 'total_floorspace'
        and 'floorspace_by_activity' rows.

    Returns
    -------
    list of pl.DataFrame
    """
    total_s = pl_to_series(
        floorspace_df.filter(pl.col('variable') == 'total_floorspace')
    )

    frames = []
    for cims_name in ACTIVITY_MAPPING.values():
        act_s = pl_to_series(
            floorspace_df.filter(
                (pl.col('variable') == 'floorspace_by_activity') &
                (pl.col('category') == cims_name)
            )
        )
        share = act_s / total_s.replace(0, np.nan)
        frames.append(_long(region, 'building_shell_shares', cims_name,
                            'market_share_total', '%', share))

    return frames


# ==============================================================================
# EXTRACTION — HVAC TECHNOLOGIES
# ==============================================================================

def extract_hvac_technologies(region: str, tables: dict,
                               ng_splits: dict) -> list[pl.DataFrame]:
    """
    Extract HVAC technology shares from Table 24, which provides the
    region-level weighted average shares directly.

    NG total share is split into Low / Medium / High efficiency tiers
    using ng_splits. Steam is treated as NG Cogeneration.
    BC gets both Cold (Table 24) and Marine (also Table 24) climate frames.

    Parameters
    ----------
    ng_splits : dict
        Output of load_ng_efficiency_splits().

    Returns
    -------
    list of pl.DataFrame
    """
    is_bc = region.upper() == 'BC'
    tbl = tables[f"Table {HVAC_TABLE}"]

    def _safe_pct(label, n=0):
        try:
            return pct_series(tbl, label, match_n=n)
        except KeyError:
            return pd.Series(dtype=float)

    def _split_ng(ng_total: pd.Series):
        ng_low, ng_med, ng_high = {}, {}, {}
        for year, val in ng_total.items():
            if pd.isna(val):
                ng_low[year] = ng_med[year] = ng_high[year] = np.nan
            else:
                sp = ng_splits.get(int(year), {'low': 1/3, 'medium': 1/3, 'high': 1/3})
                ng_low[year]  = val * sp['low']
                ng_med[year]  = val * sp['medium']
                ng_high[year] = val * sp['high']
        return pd.Series(ng_low), pd.Series(ng_med), pd.Series(ng_high)

    # Table 24 Shares (%) section — match_n=1 selects the Shares rows
    elec  = _safe_pct("Electricity",                n=1)
    ng_t  = _safe_pct("Natural Gas",                n=1)
    lfo   = _safe_pct("Light Fuel Oil and Kerosene", n=1)
    hfo   = _safe_pct("Heavy Fuel Oil",              n=1)
    steam = _safe_pct("Steam",                       n=1)
    other = _safe_pct("Other2",                      n=1)

    # If steam exists but has suppressed (NaN) years, recover those years as
    # the remainder so shares sum to 1.
    if not steam.dropna().empty:
        known = elec.add(ng_t, fill_value=0).add(lfo, fill_value=0) \
                    .add(hfo, fill_value=0).add(other, fill_value=0)
        steam = steam.fillna((1.0 - known).clip(lower=0))

    # Steam → NG Cogeneration; Other → Propane
    ng_lo, ng_md, ng_hi = _split_ng(ng_t)

    tech_shares = {
        'Natural Gas_Furnace_Low Efficiency':       ng_lo,
        'Natural Gas_Furnace_Medium Efficiency':    ng_md,
        'Natural Gas_Furnace_High Efficiency':      ng_hi,
        'Natural Gas_Cogeneration':                 steam,
        'Light Fuel Oil_Furnace_Low Efficiency':    lfo,
        'Light Fuel Oil_Furnace_Medium Efficiency': pd.Series(dtype=float),
        'Heavy Fuel Oil_Furnace_Low Efficiency':    hfo,
        'Heavy Fuel Oil_Furnace_Medium Efficiency': pd.Series(dtype=float),
        'Propane_Furnace_Medium Efficiency':        other,
        'Propane_Furnace_High Efficiency':          pd.Series(dtype=float),
        'Electricity_Furnace_High Efficiency':      elec,
        'Electricity_GSHP':                         pd.Series(dtype=float),
        'Electricity_ASHP_Natural Gas_Backup':      pd.Series(dtype=float),
        'Electricity_ASHP_Electricity_Backup':      pd.Series(dtype=float),
        'Natural Gas_ASHP_Natural Gas_Backup':      pd.Series(dtype=float),
    }

    frames = []
    for tech, s in tech_shares.items():
        if s.dropna().empty:
            continue
        frames.append(_long(region, 'hvac_cold', tech,
                            'market_share_total', '%', s))

    if is_bc:
        marine_shares = {
            'Natural Gas_Furnace_Low Efficiency':       ng_lo,
            'Natural Gas_Furnace_Medium Efficiency':    ng_md,
            'Natural Gas_Furnace_High Efficiency':      ng_hi,
            'Natural Gas_Cogeneration':                 steam,
            'Light Fuel Oil_Furnace_Low Efficiency':    lfo,
            'Light Fuel Oil_Furnace_Medium Efficiency': pd.Series(dtype=float),
            'Heavy Fuel Oil_Furnace_Low Efficiency':    hfo,
            'Heavy Fuel Oil_Furnace_Medium Efficiency': pd.Series(dtype=float),
            'Propane_Furnace_Medium Efficiency':        other,
            'Propane_Furnace_High Efficiency':          pd.Series(dtype=float),
            'Electricity_Furnace_High Efficiency':      elec,
            'Electricity_GSHP':                         pd.Series(dtype=float),
            'Electricity_ASHP':                         pd.Series(dtype=float),
            'Natural Gas_ASHP':                         pd.Series(dtype=float),
        }
        for tech, s in marine_shares.items():
            if s.dropna().empty:
                continue
            frames.append(_long(region, 'hvac_marine', tech,
                                'market_share_total', '%', s))

    return frames


# ==============================================================================
# EXTRACTION — HOT WATER
# ==============================================================================

def extract_hot_water(region: str, tables: dict) -> list[pl.DataFrame]:
    """
    Extract commercial hot water technology shares from Table 26.

    Steam is treated as a proxy for NG medium-efficiency boilers.

    Returns
    -------
    list of pl.DataFrame
    """
    tbl = tables[f"Table {HOT_WATER_TABLE}"]

    def _hw(label, n=1):
        try:
            return pct_series(tbl, label, match_n=n)
        except KeyError:
            return pd.Series(dtype=float)

    elec  = _hw("Electricity")
    ng    = _hw("Natural Gas")
    lfo   = _hw("Light Fuel Oil and Kerosene")
    hfo   = _hw("Heavy Fuel Oil")
    steam = _hw("Steam")
    other = _hw("Other2")

    if not steam.dropna().empty:
        known = elec.add(ng, fill_value=0).add(lfo, fill_value=0) \
                    .add(hfo, fill_value=0).add(other, fill_value=0)
        steam = steam.fillna((1.0 - known).clip(lower=0))

    # Steam → NG medium efficiency proxy
    ng_med = ng.add(steam, fill_value=0)

    hot_water_tech = {
        'Electricity_Boiler_High Efficiency':       elec,
        'Natural Gas_Boiler_Medium Efficiency':     ng_med,
        'Light Fuel Oil_Boiler_Medium Efficiency':  lfo,
        'Heavy Fuel Oil_Boiler_Medium Efficiency':  hfo,
        'Propane_Boiler_Medium Efficiency':         other,
        'Electricity_ASHP':                         pd.Series(dtype=float),
    }

    frames = []
    for tech, s in hot_water_tech.items():
        if not s.dropna().empty:
            frames.append(_long(region, 'hot_water_tech', tech,
                                'market_share_total', '%', s))

    return frames


# ==============================================================================
# PROJECTION EXTENSIONS
# ==============================================================================

def apply_extensions(df: pl.DataFrame, region: str, params: dict) -> pl.DataFrame:
    """
    Apply projection extensions to a region's long-format DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Region data in long format (historical only).
    region : str
        Upper-case region code.
    params : dict
        Output of load_projection_params().

    Returns
    -------
    pl.DataFrame
        Historical data combined with projected rows through 2100.
    """
    if not params:
        return df

    frames = [df]

    def _series(variable: str, category: str) -> pd.Series:
        subset = df.filter(
            (pl.col('variable') == variable) & (pl.col('category') == category)
        )
        return pl_to_series(subset).sort_index()

    def _apply(variable: str, category: str, fn, **kwargs) -> pl.DataFrame:
        s = _series(variable, category)
        if s.dropna().empty:
            return pl.DataFrame()
        max_hist = int(s.dropna().index.max())
        extended = fn(s, base_year=max_hist, **kwargs)
        new = extended[extended.index > max_hist]
        if new.empty:
            return pl.DataFrame()
        meta = df.filter(
            (pl.col('variable') == variable) & (pl.col('category') == category)
        ).head(1)
        if len(meta) == 0:
            return pl.DataFrame()
        parameter = pl_get_scalar(meta, 'parameter')
        unit      = pl_get_scalar(meta, 'unit')
        return _long(region, variable, category, parameter, unit, new)

    # 1. Total floorspace — activity-style CAGR (same pattern as activity scripts)
    fs_params = params.get('floorspace', {})
    if fs_params:
        _cagr_start = fs_params['cagr_start']
        _cagr_end   = fs_params['cagr_end']
        _periods    = fs_params['periods']

        def _extend_floorspace(series, base_year,
                               cs=_cagr_start, ce=_cagr_end, p=_periods):
            raw_cagr = compute_cagr(series, cs, min(ce, base_year))
            base_val = float(series[base_year])
            # anchor_year=base_year: some regions' raw CEUD source lags a
            # year behind the sector-wide cutoff `p` was written against
            # (e.g. Atlantic stops at 2022 while `p` assumes history through
            # 2023) — anchoring here starts the projection right after this
            # region's own last historical year instead of leaving a gap.
            projected = extend_cagr_periods(
                base_val, raw_cagr, p, CAGR_OVERRIDES.get(region),
                anchor_year=base_year,
            )
            return pd.concat([series, projected]).sort_index()

        frames.append(_apply('total_floorspace', '', _extend_floorspace))

    # 2. Lighting / Water Heating / Auxiliary Equipment energy — hold energy
    #    intensity (GJ per m2 of floorspace) constant beyond the last
    #    historical year; there's no published growth assumption for these
    #    end-uses, so floorspace growth alone drives their projection.
    floorspace_full = pl.concat([f for f in frames if len(f) > 0], how='diagonal_relaxed')
    floorspace_series = pl_to_series(
        floorspace_full.filter(pl.col('variable') == 'total_floorspace')
    ).sort_index()

    def _extend_flat_intensity(series, base_year, floorspace=floorspace_series):
        base_fs = float(floorspace[base_year])
        intensity = float(series[base_year]) / base_fs if base_fs else 0.0
        future_years = [y for y in floorspace.index if y > base_year]
        projected = pd.Series({y: intensity * float(floorspace[y]) for y in future_years})
        return pd.concat([series, projected]).sort_index()

    for var in ('lighting_energy', 'water_heating_energy', 'aux_equipment_energy',
                'space_heating_energy', 'space_cooling_energy'):
        frames.append(_apply(var, '', _extend_flat_intensity))

    # 3. Building shell shares — trend then dampener, per activity
    bs_params = params.get('building_shell_shares', {})
    if bs_params:
        trend_kwargs_base = dict(
            trend_start  = bs_params.get('trend_start', 2000),
            trend_end    = bs_params.get('trend_end', LAST_HIST_YEAR),
            trend_period = bs_params.get('trend_period', (LAST_HIST_YEAR + 1, 2031)),
        )
        activity_rates = bs_params.get('activity_rates', {})
        decline1_yrs   = bs_params.get('decline1_years', (2031, 2051))
        decline2_yrs   = bs_params.get('decline2_years', (2051, 2101))

        activities = list(ACTIVITY_MAPPING.values())
        declining  = [a for a in activities if a != 'Other Services']

        projected_declining: dict[str, pd.Series] = {}
        for activity in declining:
            s = _series('building_shell_shares', activity)
            if s.dropna().empty:
                continue
            r1, r2 = activity_rates.get(activity, (-0.10, -0.10))
            td_kwargs = {
                **trend_kwargs_base,
                'decline_periods': [
                    (decline1_yrs[0], decline1_yrs[1], r1),
                    (decline2_yrs[0], decline2_yrs[1], r2),
                ],
            }
            max_hist_bs = int(s.dropna().index.max())
            ext = extend_series_trend_dampener(s, base_year=max_hist_bs, **td_kwargs)
            projected_declining[activity] = ext
            frames.append(_apply('building_shell_shares', activity,
                                 extend_series_trend_dampener, **td_kwargs))

        # Other Services = 1 - sum(all other activities)
        if projected_declining:
            max_hist = int(df.filter(pl.col('variable') == 'building_shell_shares')
                           ['year'].max())
            other_vals: dict[int, float] = {}
            for yr in range(max_hist + 1, PROJECTION_END + 1):
                total_other = 0.0
                for s in projected_declining.values():
                    v = s.get(yr) if yr in s.index else np.nan
                    if pd.notna(v):
                        total_other += float(v)
                other_vals[yr] = max(0.0, 1.0 - total_other)
            other_series = pd.Series(other_vals)
            meta = df.filter(
                (pl.col('variable') == 'building_shell_shares') &
                (pl.col('category') == 'Other Services')
            ).head(1)
            if len(meta) > 0:
                frames.append(_long(region, 'building_shell_shares', 'Other Services',
                                    pl_get_scalar(meta, 'parameter'),
                                    pl_get_scalar(meta, 'unit'),
                                    other_series))

    # 4. Space Heating by activity — hold energy intensity (GJ per m2 of that
    #    activity's own floorspace) constant beyond the last historical year,
    #    same rationale as step 2. Activity floorspace = total_floorspace x
    #    building_shell_shares(activity), using the (now projected) series
    #    from steps 1 and 3 above.
    shares_and_floorspace = pl.concat([f for f in frames if len(f) > 0], how='diagonal_relaxed')
    floorspace_series = pl_to_series(
        shares_and_floorspace.filter(pl.col('variable') == 'total_floorspace')
    ).sort_index()

    for activity in ACTIVITY_MAPPING.values():
        share_series = pl_to_series(
            shares_and_floorspace.filter(
                (pl.col('variable') == 'building_shell_shares') & (pl.col('category') == activity)
            )
        ).sort_index()
        if share_series.dropna().empty:
            continue
        activity_floorspace = (floorspace_series * share_series).dropna()

        def _extend_activity_flat_intensity(series, base_year, fs=activity_floorspace):
            base_fs = float(fs[base_year]) if base_year in fs.index else 0.0
            intensity = float(series[base_year]) / base_fs if base_fs else 0.0
            future_years = [y for y in fs.index if y > base_year]
            projected = pd.Series({y: intensity * float(fs[y]) for y in future_years})
            return pd.concat([series, projected]).sort_index()

        frames.append(_apply('space_heating_by_activity', activity, _extend_activity_flat_intensity))

    return pl.concat([f for f in frames if f is not None and len(f) > 0],
                     how='diagonal_relaxed')


# ==============================================================================
# COMBINE — BUILDINGS -> {LIGHTING, HOT WATER, REFRIGERATION, COOKING,
#           PLUG LOAD} SERVICE_REQUEST INTENSITIES
# ==============================================================================

def _year_indexed(df: pl.DataFrame, variable: str, category: str = '') -> pd.Series:
    subset = df.filter((pl.col('variable') == variable) & (pl.col('category') == category))
    return pl_to_series(subset).sort_index()


def compute_enduse_service_requests(df: pl.DataFrame, region: str) -> pl.DataFrame:
    """
    Combine CEUD end-use energy (lighting_energy / water_heating_energy /
    aux_equipment_energy) with the fixed national technology energy factors
    (AUX_TECH_GJ_PER_UNIT, LIGHTING_GJ_PER_M2) and this region's own
    hot_water_tech mix to back out the Buildings -> {Lighting, Hot Water,
    Refrigeration, Cooking, Plug Load} service_request intensities (m2 or
    "unit" per m2 of floorspace) that raw_data/fixed_data/commercial
    previously hardcoded as constants.

    Must be called on a region's FULLY DISAGGREGATED frame (i.e. after
    disaggregate_commercial()) so that Hot Water reflects that specific
    region's own fuel mix rather than its parent AT/BCT aggregate.
    """
    floorspace = _year_indexed(df, 'total_floorspace')
    lighting_e = _year_indexed(df, 'lighting_energy')
    water_e    = _year_indexed(df, 'water_heating_energy')
    aux_e      = _year_indexed(df, 'aux_equipment_energy')

    hw_shares = {
        r['category']: r['value']
        for r in df.filter((pl.col('variable') == 'hot_water_tech') & (pl.col('year') == 2000))
                  .select(['category', 'value']).iter_rows(named=True)
        if r['value'] is not None and r['value'] == r['value']  # drop null/NaN shares
    }
    hw_gj_per_unit = sum(
        share * HOT_WATER_TECH_GJ_PER_UNIT.get(tech, 0.0)
        for tech, share in hw_shares.items()
    )

    years = sorted(
        set(floorspace.dropna().index) & set(lighting_e.dropna().index) &
        set(water_e.dropna().index) & set(aux_e.dropna().index)
    )

    lighting_vals, hw_vals, refrig_vals, cook_vals, plug_vals = {}, {}, {}, {}, {}
    for y in years:
        fs = float(floorspace[y])
        if not fs:
            continue
        lighting_vals[y] = float(lighting_e[y]) / (fs * LIGHTING_GJ_PER_M2)

        shares = _aux_energy_shares(y)
        aux_gj = float(aux_e[y])
        refrig_vals[y] = shares['refrigeration'] * aux_gj / (fs * AUX_TECH_GJ_PER_UNIT['refrigeration'])
        cook_vals[y]   = shares['cooking']       * aux_gj / (fs * AUX_TECH_GJ_PER_UNIT['cooking'])
        plug_vals[y]   = shares['plug_load']     * aux_gj / (fs * AUX_TECH_GJ_PER_UNIT['plug_load'])

        if hw_gj_per_unit > 0:
            hw_vals[y] = float(water_e[y]) / (fs * hw_gj_per_unit)

    frames = [
        _long(region, 'lighting_service_request', '', 'service_request', 'm2', pd.Series(lighting_vals)),
        _long(region, 'refrigeration_service_request', '', 'service_request', 'unit', pd.Series(refrig_vals)),
        _long(region, 'cooking_service_request', '', 'service_request', 'unit', pd.Series(cook_vals)),
        _long(region, 'plug_load_service_request', '', 'service_request', 'unit', pd.Series(plug_vals)),
    ]
    if hw_vals:
        frames.append(_long(region, 'hot_water_service_request', '', 'service_request', 'unit',
                            pd.Series(hw_vals)))

    return pl.concat(frames, how='diagonal_relaxed')


# ==============================================================================
# COMBINE — SHELL -> HVAC AND HVAC -> COOLING SERVICE_REQUEST INTENSITIES
# ==============================================================================

# First year Shell's own technology-mix (Std vs LEED Silver/Platinum)
# drives HVAC's assessed_demand, replacing the CEUD-historical direct
# request below. Independent of fixed_data 'available' -- LEED can be (and
# currently is) available for the stock competition earlier than this; that
# only decides which technology wins share, not which demand source HVAC
# reads from. Before this year, Buildings -> HVAC carries exact historical
# demand instead (see compute_buildings_hvac_direct_request), regardless of
# what the Shell competition's technology mix would otherwise imply.
SHELL_HVAC_COMPETITION_START_YEAR = 2025
BUILDINGS_HVAC_HISTORICAL_CUTOFF = SHELL_HVAC_COMPETITION_START_YEAR - 1


def compute_buildings_hvac_direct_request(df: pl.DataFrame, region: str) -> pl.DataFrame:
    """
    Buildings -> HVAC (Cold)/(Marine) direct service_request, years 2000
    through BUILDINGS_HVAC_HISTORICAL_CUTOFF only.

    Unlike Shell -> HVAC (per shell technology, vintage-weighted), this is a
    node-level request from 'Buildings' -- no Technology field, matching
    the existing Buildings -> {Lighting, Hot Water, ...} pattern. CIMS only
    vintage-weights service_request when it can find a stock_total to split
    by vintage (see _get_vintage_weights); a non-technology node has none,
    so vintage_weights collapses to {year: 1} and the current year's value
    applies to the entire assessed_demand uniformly -- old floorspace and
    new alike, not diluted toward older vintages the way Shell -> HVAC's
    per-technology rate was. This reproduces CEUD's actual historical heat
    demand exactly.

    Buildings' assessed_demand equals total_floorspace exactly (Commercial
    -> Buildings is a literal 1:1 service_request in fixed_data), so this
    rate is GJ of heat per m2 of the region's total floorspace.

    Only covers history: Shell's own technology competition needs to keep
    driving demand for projection years (see compute_hvac_service_requests),
    since there's no CEUD ground truth for the future to match, and Shell's
    competition needs a live, non-zero operating-cost signal once LEED
    options become available to choose from.

    Divides by _hvac_fuel_conversion_factor for the same reason Shell ->
    HVAC does: CEUD's space_heating_energy is secondary energy (fuel
    already consumed), but this feeds HVAC (Cold)'s assessed_demand, a
    pre-efficiency quantity HVAC's own technologies then convert to fuel --
    skipping this division would reintroduce the original double-counted-
    efficiency bug for the historical period.
    """
    floorspace = _year_indexed(df, 'total_floorspace')
    heat = _year_indexed(df, 'space_heating_energy')
    conversion_factor = _hvac_fuel_conversion_factor(region)
    is_bc = region.upper() == 'BC'

    years = sorted(
        y for y in (set(floorspace.dropna().index) & set(heat.dropna().index))
        if y <= BUILDINGS_HVAC_HISTORICAL_CUTOFF
    )

    vals_cold = {
        y: (float(heat[y]) / float(conversion_factor[y])) / float(floorspace[y])
        for y in years if float(floorspace[y])
    }

    frames = [_long(region, 'buildings_hvac_service_request', 'Cold',
                    'service_request', 'GJ', pd.Series(vals_cold))]
    if is_bc:
        vals_marine = {y: v * MARINE_TO_COLD_RATIO for y, v in vals_cold.items()}
        frames.append(_long(region, 'buildings_hvac_service_request', 'Marine',
                            'service_request', 'GJ', pd.Series(vals_marine)))
    return pl.concat(frames, how='diagonal_relaxed')


def _hvac_technologies(region: str, climate: str) -> list[str]:
    """
    Technology names competing at HVAC (<climate>), read from fixed_data
    (one row per technology, service_request targeting Cooling). Cold and
    Marine (BC only) carry slightly different technology sets (e.g. Marine's
    'Electricity_ASHP' vs Cold's 'Electricity_ASHP_Natural Gas_Backup' /
    'Electricity_ASHP_Electricity_Backup'), so this must be read per climate,
    not shared.
    """
    fixed_path = FIXED_DATA_DIR / f'commercial_{region.lower()}.csv'
    techs, seen = [], set()
    with open(fixed_path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if (r['Branch'].endswith(f'.HVAC ({climate})') and r['Parameter'] == 'service_request'
                    and r['Target'].endswith('.Cooling') and r['Technology'] not in seen):
                seen.add(r['Technology'])
                techs.append(r['Technology'])
    return techs


def _cooling_own_conversion_rate(region: str) -> float:
    """
    The Cooling node's own (single) technology -- 'Std', 100% market share,
    no competing techs -- service_request rate to Electricity. Constant
    across years in fixed_data. Needed to back out cooling *assessed_demand*
    (a pre-efficiency 'cooling service' quantity) from CEUD's
    space_cooling_energy, which is already actual electricity consumed --
    see _hvac_cooling_to_heat_ratio.
    """
    fixed_path = FIXED_DATA_DIR / f'commercial_{region.lower()}.csv'
    with open(fixed_path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['Branch'].endswith('.Commercial.Cooling') and r['Parameter'] == 'service_request':
                return float(r['2000'])
    return 1.0


def _hvac_cooling_to_heat_ratio(df: pl.DataFrame, region: str) -> pd.Series:
    """
    GJ-cooling-per-GJ-heat ratio for HVAC (Cold)/(Marine) technologies' own
    service_request to Cooling, replacing the flat "1" every technology used
    previously (which made Cooling demand track heat demand 1:1, regardless
    of the real, much smaller, cooling load).

    Uses CEUD's region-wide Table 2 totals (space_cooling_energy /
    space_heating_energy) -- CEUD doesn't break Space Cooling out per
    activity the way Table 25 does for heating, so this is one ratio per
    region/year, applied uniformly across every HVAC technology and every
    activity (there's no CEUD-based way to differentiate cooling load by
    heating technology, and architecturally there's no reason to expect one
    -- cooling is a separate system from whatever's providing the heat).

    Divides by _cooling_own_conversion_rate for the same reason
    _hvac_fuel_conversion_factor does: CEUD's space_cooling_energy is
    secondary energy (actual electricity already consumed), but this feeds
    HVAC's service_request to Cooling, which sets Cooling's
    assessed_demand -- a pre-efficiency quantity Cooling's own 'Std'
    technology then converts to actual electricity. Skipping the division
    would double-count that conversion.

    Zero for years before SHELL_HVAC_COMPETITION_START_YEAR: the actual
    historical cooling demand is carried instead by the Buildings -> Cooling
    direct request (see compute_buildings_cooling_direct_request), which
    isn't vintage-weighted (Buildings has no competing technologies) and so
    reproduces CEUD's real annual variation exactly, including sharp
    single-year spikes (e.g. QC 2005) that any smoothed or back-solved
    per-technology rate would otherwise flatten out. Zeroing this uniformly
    across every HVAC technology for those years doesn't distort HVAC's own
    technology competition (Natural Gas Furnace vs Electric Furnace, etc.,
    which must stay live and economically meaningful throughout history) --
    it's the same value subtracted from every competing technology's
    operating cost equally, so relative ranking is unaffected either way.
    Shell's own technology competition (Std vs LEED) is zeroed the same way
    over this range for the same reason, even though it may itself be live
    during history -- see SHELL_HVAC_COMPETITION_START_YEAR.

    For projection years, raw_ratio is used directly with no smoothing --
    naturally flat by construction (both space_cooling_energy and
    space_heating_energy hold their last historical value constant beyond
    CEUD's data), and there's no CEUD ground truth to track for the future
    anyway.
    """
    heat = _year_indexed(df, 'space_heating_energy')
    cool = _year_indexed(df, 'space_cooling_energy')
    years = sorted(set(heat.dropna().index) & set(cool.dropna().index))

    cooling_conversion = _cooling_own_conversion_rate(region)
    demand = pd.Series({y: float(heat[y]) for y in years if float(heat[y])})
    raw_ratio = pd.Series({
        y: (float(cool[y]) / cooling_conversion) / demand[y] for y in demand.index
    })

    return pd.Series({
        y: (float(raw_ratio[y]) if y >= SHELL_HVAC_COMPETITION_START_YEAR else 0.0)
        for y in raw_ratio.index
    })


def compute_buildings_cooling_direct_request(df: pl.DataFrame, region: str) -> pl.DataFrame:
    """
    Buildings -> Cooling direct service_request, years 2000 through
    BUILDINGS_HVAC_HISTORICAL_CUTOFF only.

    Unlike HVAC (Cold)/(Marine), which are genuinely separate model nodes,
    Cooling is a single node region-wide -- both HVAC (Cold)'s and HVAC
    (Marine)'s technologies (BC only) target the same CIMS.CAN.<region>.
    Commercial.Cooling. So this produces exactly one value, not a Cold/
    Marine split the way compute_buildings_hvac_direct_request does;
    splitting it (e.g. Cold + Cold*MARINE_TO_COLD_RATIO) would create two
    rows targeting the same key, and the model reader would silently keep
    only the last one, discarding the other's contribution -- the same
    last-row-wins collision collapse_constant_years was fixed for.

    Mirrors compute_buildings_hvac_direct_request otherwise, for the same
    reason: a node-level request from 'Buildings' (no Technology field)
    isn't vintage-weighted (CIMS finds no stock_total to split by vintage
    for a non-technology node -- see _get_vintage_weights), so the current
    year's value applies to all existing floorspace uniformly, reproducing
    CEUD's actual historical cooling demand exactly -- including sharp
    single-year spikes a per-technology vintage-weighted or smoothed rate
    can't track.

    Buildings' assessed_demand equals total_floorspace exactly (Commercial
    -> Buildings is a literal 1:1 service_request in fixed_data), so this
    rate is GJ of cooling per m2 of the region's total floorspace -- cooling
    demand scales with floorspace being cooled directly, with no need to
    route it through heat demand the way HVAC (Cold)/(Marine)'s own
    technologies' service_request to Cooling does for projection years.

    Divides by _cooling_own_conversion_rate for the same reason
    _hvac_cooling_to_heat_ratio does (CEUD's figure is already actual
    electricity consumed, not the pre-efficiency assessed_demand level
    Cooling's own technology then converts).
    """
    floorspace = _year_indexed(df, 'total_floorspace')
    cool = _year_indexed(df, 'space_cooling_energy')
    cooling_conversion = _cooling_own_conversion_rate(region)

    years = sorted(
        y for y in (set(floorspace.dropna().index) & set(cool.dropna().index))
        if y <= BUILDINGS_HVAC_HISTORICAL_CUTOFF
    )

    vals = {
        y: (float(cool[y]) / cooling_conversion) / float(floorspace[y])
        for y in years if float(floorspace[y])
    }

    return _long(region, 'buildings_cooling_service_request', '',
                'service_request', 'GJ', pd.Series(vals))


_AUX_HVAC_TARGETS = {'Motive Power', 'Cooling'}


def _hvac_fuel_conversion_factor(region: str) -> pd.Series:
    """
    Weighted-average 'fuel-in per useful-heat-out' factor for HVAC (Cold)'s
    own technology mix, indexed by year.

    CEUD's Space Heating figures (space_heating_by_activity) are secondary
    energy -- actual fuel burned -- but the Shell -> HVAC (Cold)
    service_request feeds assessed_demand, a pre-efficiency 'useful heat'
    quantity in the model's internal accounting. HVAC (Cold)'s own
    technologies then multiply assessed_demand by their own service_request
    rate (>1 for anything below 100% efficient) to get actual fuel
    consumption. Calibrating the Shell -> HVAC rate directly against CEUD's
    fuel figure therefore double-counts efficiency: once implicitly (CEUD
    already reflects real-world efficiency), once again via HVAC's own
    technology conversion. Dividing target_energy by this factor before
    rescaling backs out the useful-heat level so HVAC's own conversion
    reproduces CEUD's fuel total instead of overshooting it.

    Sourced from fixed_data (constant across all years there) and
    calibration_market_share_total (available 2000 through the last
    calibrated year; held constant for years beyond that): each HVAC
    technology's own primary-fuel service_request rate (summed across all
    its targets except Motive Power/Cooling, so a cogen technology's
    negative Electricity credit nets in correctly, and so Motive Power/
    Cooling's own auxiliary energy -- not part of CEUD's heat-fuel-only
    figure -- isn't folded in), weighted by its calibration market share.
    """
    region_lower = region.lower()
    fixed_path = FIXED_DATA_DIR / f'commercial_{region_lower}.csv'
    calib_path = CALIBRATION_DIR / f'commercial_{region_lower}.csv'

    with open(fixed_path, encoding='utf-8-sig') as f:
        fixed_rows = list(csv.DictReader(f))

    own_fuel_rate: dict[str, float] = {}
    for r in fixed_rows:
        if not r['Branch'].endswith('.HVAC (Cold)') or r['Parameter'] != 'service_request':
            continue
        target_leaf = r['Target'].split('.')[-1]
        if target_leaf in _AUX_HVAC_TARGETS:
            continue
        tech = r['Technology']
        own_fuel_rate[tech] = own_fuel_rate.get(tech, 0.0) + float(r['2000'])

    shares_by_year: dict[int, dict[str, float]] = {}
    if calib_path.exists():
        with open(calib_path, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r['Parameter'] != 'calibration_market_share_total':
                    continue
                if not r['Branch'].endswith('.HVAC (Cold)'):
                    continue
                try:
                    year = int(r['Year'])
                    value = float(r['Value'])
                except (ValueError, TypeError):
                    continue
                shares_by_year.setdefault(year, {})[r['Technology']] = value

    factor_by_year = {
        year: sum(share * own_fuel_rate.get(tech, 0.0) for tech, share in shares.items())
        for year, shares in shares_by_year.items()
    }

    if not factor_by_year:
        return pd.Series({y: 1.0 for y in YEARS})

    last_factor = factor_by_year[max(factor_by_year)]
    return pd.Series({y: factor_by_year.get(y, last_factor) for y in YEARS})


def compute_hvac_service_requests(df: pl.DataFrame, region: str) -> pl.DataFrame:
    """
    Combine CEUD Space Heating/Cooling energy with the fixed national
    shell-technology GJ/m2 rates (SHELL_TECH_GJ_PER_M2) to compute HVAC
    (Cold)/(Marine)'s service_request rates. Two demand sources cover
    mutually exclusive year ranges to avoid double-counting (see
    SHELL_HVAC_COMPETITION_START_YEAR):
      - Historical years: Buildings -> HVAC/Cooling direct requests
        (compute_buildings_hvac_direct_request /
        compute_buildings_cooling_direct_request) reproduce CEUD's actual
        demand exactly, since they aren't vintage-weighted. Shell's own
        per-technology service_request to HVAC, and HVAC's own
        service_request to Cooling, are both zero for these years.
      - Projection years: Shell's own technology competition (Std vs LEED
        Silver/Platinum -- see fixed_data 'available' for when each becomes
        eligible to compete) takes over, rescaling SHELL_TECH_GJ_PER_M2 per
        region/year and preserving each shell tier's relative improvement
        over Std. HVAC (Marine) (BC only) is derived as Cold x
        MARINE_TO_COLD_RATIO, the same fixed relationship
        raw_data/fixed_data/commercial already applied uniformly (CEUD
        doesn't distinguish climate zones within a region).

    The Cooling node's own 'Std' technology's service_request to
    Electricity (its own efficiency conversion) is untouched fixed_data,
    not computed here.

    Must be called on a region's FULLY DISAGGREGATED frame, like
    compute_enduse_service_requests().

    Returns
    -------
    pl.DataFrame with variables:
      - 'hvac_service_request', category f'{activity}|{Cold|Marine}|{tech}'
        (zero for years before SHELL_HVAC_COMPETITION_START_YEAR)
      - 'hvac_cooling_service_request', category f'{Cold|Marine}|{tech}'
        (zero for years before SHELL_HVAC_COMPETITION_START_YEAR)
      - 'buildings_hvac_service_request', category '{Cold|Marine}' (years
        through BUILDINGS_HVAC_HISTORICAL_CUTOFF only)
      - 'buildings_cooling_service_request', category '{Cold|Marine}' (years
        through BUILDINGS_HVAC_HISTORICAL_CUTOFF only)
    """
    floorspace  = _year_indexed(df, 'total_floorspace')
    is_bc = region.upper() == 'BC'
    conversion_factor = _hvac_fuel_conversion_factor(region)

    frames = []

    # Per-activity Shell -> HVAC rates
    for activity, tech_rates in SHELL_TECH_GJ_PER_M2.items():
        target_energy = _year_indexed(df, 'space_heating_by_activity', activity)
        shell_share   = _year_indexed(df, 'building_shell_shares', activity)
        years = sorted(
            set(floorspace.dropna().index) & set(target_energy.dropna().index) &
            set(shell_share.dropna().index)
        )
        std_rate = tech_rates['Std']

        demand = pd.Series({
            y: float(floorspace[y]) * float(shell_share[y])
            for y in years if float(floorspace[y]) * float(shell_share[y])
        })
        raw_scale = pd.Series({
            y: (float(target_energy[y]) / float(conversion_factor[y])) / demand[y] / std_rate
            for y in demand.index
        })

        # Shell -> HVAC now only drives HVAC's assessed_demand for
        # projection years (>= SHELL_HVAC_COMPETITION_START_YEAR). Historical
        # years are driven entirely by the Buildings -> HVAC direct request
        # instead (see compute_buildings_hvac_direct_request), which isn't
        # vintage-weighted at all (Buildings has no competing technologies,
        # so CIMS's vintage-weighting finds no stock_total to split by
        # vintage and applies the current year's value to all existing
        # floorspace uniformly -- exactly reproducing CEUD's real historical
        # demand, instead of the ~98% diluted-toward-old-vintages result
        # Shell's own per-technology service_request produced).
        #
        # Feeding Shell's own (per-technology, vintage-weighted) rate for
        # historical years too would double-count that demand on top of the
        # direct request, so it's set to exactly 0 for years before
        # SHELL_HVAC_COMPETITION_START_YEAR, regardless of which shell
        # technologies are available to compete. LEED Silver/Platinum can be
        # (and currently are) available before this year, so Shell's stock
        # competition is live during history -- but with every technology's
        # operating-cost signal zeroed alike, it has no efficiency-based
        # ranking to go on; calibrated FICs carry the full market-share
        # differentiation for those years instead.
        #
        # For projection years, raw_scale is used directly with no
        # smoothing or back-solving -- it's naturally flat by construction
        # (both target_energy and conversion_factor hold their last
        # historical value constant beyond CEUD's data), and there's no
        # CEUD ground truth for future years to track anyway.
        scale = pd.Series({
            y: (float(raw_scale[y]) if y >= SHELL_HVAC_COMPETITION_START_YEAR else 0.0)
            for y in raw_scale.index
        })

        for tech, fixed_rate in tech_rates.items():
            vals_cold, vals_marine = {}, {}
            for y in scale.index:
                cold_rate = fixed_rate * float(scale[y])
                vals_cold[y] = cold_rate
                if is_bc:
                    vals_marine[y] = cold_rate * MARINE_TO_COLD_RATIO

            if vals_cold:
                frames.append(_long(region, 'hvac_service_request', f'{activity}|Cold|{tech}',
                                    'service_request', 'GJ', pd.Series(vals_cold)))
            if vals_marine:
                frames.append(_long(region, 'hvac_service_request', f'{activity}|Marine|{tech}',
                                    'service_request', 'GJ', pd.Series(vals_marine)))

    # HVAC (Cold)/(Marine) technologies' own service_request to Cooling --
    # replaces the flat "1" fixed_data constant (zero before
    # SHELL_HVAC_COMPETITION_START_YEAR, real CEUD-derived ratio for
    # projection years -- see _hvac_cooling_to_heat_ratio).
    cooling_ratio = _hvac_cooling_to_heat_ratio(df, region)
    if len(cooling_ratio):
        cooling_vals = {y: float(cooling_ratio[y]) for y in cooling_ratio.index}
        for tech in _hvac_technologies(region, 'Cold'):
            frames.append(_long(region, 'hvac_cooling_service_request', f'Cold|{tech}',
                                'service_request', 'ratio', pd.Series(cooling_vals)))
        if is_bc:
            for tech in _hvac_technologies(region, 'Marine'):
                frames.append(_long(region, 'hvac_cooling_service_request', f'Marine|{tech}',
                                    'service_request', 'ratio', pd.Series(cooling_vals)))

    # Buildings -> HVAC (Cold)/(Marine) direct historical request -- carries
    # HVAC's assessed_demand for years before Shell's own competition takes
    # over (see compute_buildings_hvac_direct_request and the SHELL_HVAC_
    # COMPETITION_START_YEAR split above).
    frames.append(compute_buildings_hvac_direct_request(df, region))

    # Buildings -> Cooling (Cold)/(Marine) direct historical request -- same
    # pattern as Buildings -> HVAC above, see compute_buildings_cooling_
    # direct_request.
    frames.append(compute_buildings_cooling_direct_request(df, region))

    return pl.concat(frames, how='diagonal_relaxed') if frames else pl.DataFrame()


# ==============================================================================
# MAIN EXTRACTION FUNCTION
# ==============================================================================

def extract_all_data(
    region_code: str,
    apply_projections: bool = True,
    projection_params: Optional[dict] = None,
    ng_efficiency_splits: Optional[dict] = None,
) -> pl.DataFrame:
    """
    Extract all commercial data for a region into a single long-format DataFrame.

    Parameters
    ----------
    region_code : str
        Region identifier (e.g. 'AB', 'BC').
    apply_projections : bool
    projection_params : dict, optional
    ng_efficiency_splits : dict, optional

    Returns
    -------
    pl.DataFrame
        Columns: region, variable, category, parameter, unit, source, year, value.
    """
    region = region_code.upper()
    if region not in REGIONS:
        raise ValueError(f"Invalid region code: {region_code}. "
                         f"Valid codes: {list(REGIONS.keys())}")

    is_bc = region == 'BC'

    if ng_efficiency_splits is None:
        ng_efficiency_splits = load_ng_efficiency_splits()

    tables = load_tables(region)

    # -- Floorspace & shell shares -------------------------------------------
    floorspace_frames = extract_floorspace(region, tables)
    floorspace_df = pl.concat(floorspace_frames)

    shell_frames = extract_building_shell_shares(region, floorspace_df)
    shell_df = pl.concat(shell_frames)

    # -- HVAC ----------------------------------------------------------------
    hvac_frames = extract_hvac_technologies(region, tables, ng_efficiency_splits)

    # -- Hot water -----------------------------------------------------------
    hw_frames = extract_hot_water(region, tables)

    # -- End-use energy (Lighting / Water Heating / Auxiliary Equipment / -----
    # -- Space Heating / Space Cooling totals, and Space Heating by activity)
    end_use_frames = extract_end_use_energy(region, tables)
    space_heating_frames = extract_space_heating_by_activity(region, tables)

    # Assemble — floorspace_by_activity was only needed to derive shell shares.
    # Activity-level floorspace for HVAC calibration is instead derived on
    # the fly (total_floorspace x building_shell_shares) in
    # compute_hvac_service_requests(), so it doesn't need to be carried
    # through as its own projected/disaggregated variable.
    total_floorspace_frames = [f for f in floorspace_frames
                               if f['variable'][0] == 'total_floorspace']
    all_frames = (total_floorspace_frames + shell_frames + hvac_frames + hw_frames +
                  end_use_frames + space_heating_frames)
    df = pl.concat(all_frames, how='diagonal_relaxed')

    if apply_projections:
        if projection_params is None:
            projection_params = load_projection_params()
        df = apply_extensions(df, region, projection_params)

    return df.sort(['region', 'variable', 'category', 'year'])


# ==============================================================================
# BATCH EXTRACTION
# ==============================================================================

def extract_all_regions(
    region_codes: Optional[list[str]] = None,
    apply_projections: bool = True,
) -> dict[str, pl.DataFrame]:
    """
    Extract data for multiple commercial regions.

    Returns
    -------
    dict mapping region code → pl.DataFrame
    """
    if region_codes is None:
        region_codes = list(REGIONS.keys())

    params    = load_projection_params() if apply_projections else None
    ng_splits = load_ng_efficiency_splits()
    results, failed = {}, []

    for region in region_codes:
        try:
            results[region] = extract_all_data(region, apply_projections,
                                               params, ng_splits)
        except Exception as exc:
            failed.append((region, str(exc)))

    return results

# ==============================================================================
# COMMERCIAL REGION DISAGGREGATION
# ==============================================================================
#
# Splits two aggregated CEUD regions into individual provinces/territories:
#
#   AT (Atlantic) → NL, PE, NS, NB
#   BC (BC + Territories) → BC, YT, NT, NU
#
# Rules per variable:
#   total_floorspace          → split by population share
#   floorspace_by_activity    → identical shares within group (copy parent value
#                               scaled by total_floorspace split)
#   building_shell_shares     → identical across all sub-regions
#   hvac_cold                 → split by CER Space Heating fuel-demand share,
#                               efficiency-corrected, renormalized
#   hvac_marine               → BC only (dropped for YT/NT/NU and AT provinces)
#   hot_water_tech            → split by CER Water Heating fuel-demand share,
#                               efficiency-corrected, renormalized
#
# ==============================================================================

# CIMS commercial activity sector names as they appear in CER's vDmd-CIMS.csv
# 'Sector' column. Overlaps ACTIVITY_MAPPING's CEUD-side keys except for two
# sectors CER names differently (Transportation and Warehousing → Warehousing
# and Storage; Other Services → Other Commercial). Summing demand across
# these mirrors the CEUD-derived hvac_cold/hot_water_tech shares themselves,
# which are already region-wide totals across every commercial activity.
CER_COMMERCIAL_SECTORS = {
    'Wholesale Trade', 'Retail Trade', 'Warehousing and Storage',
    'Information and Cultural Industries', 'Offices', 'Educational Services',
    'Health Care and Social Assistance', 'Arts, Entertainment and Recreation',
    'Accommodation and Food Services', 'Other Commercial',
}

# CER's vDmd-CIMS.csv 'Area' names that differ from the Stats Can population
# geo names already used as COMM_REGION_MAP keys elsewhere in this module.
# Names not listed here (PEI, Nova Scotia, New Brunswick, Nunavut) match as-is.
CER_AREA_NAME: dict[str, str] = {
    'Newfoundland and Labrador': 'Newfoundland',
    'Yukon':                     'Yukon Territory',
    'Northwest Territories':     'Northwest Territory',
}

# CER's raw 'Fuel' values folded into a canonical group before computing
# shares. 'steam' and 'kerosene' are minor fuels (~0.1% and ~1.5% of national
# commercial space heat) with no CEUD technology category of their own --
# folded into Natural Gas / Light Fuel Oil respectively, mirroring the
# steam-as-NG-proxy and kerosene/oil grouping already used elsewhere in this
# file for CEUD Table 24/26 (see extract_hvac_technologies, extract_hot_water).
CER_RAW_FUEL_GROUP: dict[str, str] = {
    'Natural Gas':    'Natural Gas',
    'steam':          'Natural Gas',
    'Electric':       'Electric',
    'Light Fuel Oil': 'Light Fuel Oil',
    'kerosene':       'Light Fuel Oil',
    'Heavy Fuel Oil': 'Heavy Fuel Oil',
    'LPG':            'LPG',
}

# CER fuel group → CEUD commercial category names
COMM_CER_FUEL_TO_CEUD = {
    'Natural Gas': [
        'Natural Gas_Furnace_Low Efficiency',
        'Natural Gas_Furnace_Medium Efficiency',
        'Natural Gas_Furnace_High Efficiency',
        'Natural Gas_Cogeneration',
        'Natural Gas_ASHP_Natural Gas_Backup',
        'Natural Gas_ASHP',
        'Natural Gas_Boiler_Medium Efficiency',
    ],
    'Light Fuel Oil': [
        'Light Fuel Oil_Furnace_Low Efficiency',
        'Light Fuel Oil_Furnace_Medium Efficiency',
        'Light Fuel Oil_Boiler_Medium Efficiency',
    ],
    'Heavy Fuel Oil': [
        'Heavy Fuel Oil_Furnace_Low Efficiency',
        'Heavy Fuel Oil_Furnace_Medium Efficiency',
        'Heavy Fuel Oil_Boiler_Medium Efficiency',
    ],
    'LPG': [
        'Propane_Furnace_Medium Efficiency',
        'Propane_Furnace_High Efficiency',
        'Propane_Boiler_Medium Efficiency',
    ],
    'Electric': [
        'Electricity_Furnace_High Efficiency',
        'Electricity_ASHP_Natural Gas_Backup',
        'Electricity_ASHP_Electricity_Backup',
        'Electricity_ASHP',
        'Electricity_GSHP',
        'Electricity_Boiler_High Efficiency',
    ],
}

# Efficiency key for each commercial CEUD category (same Table 32 efficiencies)
COMM_CATEGORY_EFFICIENCY_KEY = {
    'Natural Gas_Furnace_Low Efficiency':       'ng_low',
    'Natural Gas_Furnace_Medium Efficiency':    'ng_med',
    'Natural Gas_Furnace_High Efficiency':      'ng_high',
    'Natural Gas_Cogeneration':                 'ng_med',
    'Natural Gas_ASHP_Natural Gas_Backup':      'ng_med',
    'Natural Gas_ASHP':                         'ng_med',
    'Natural Gas_Boiler_Medium Efficiency':     'ng_med',
    'Light Fuel Oil_Furnace_Low Efficiency':    'oil_low',
    'Light Fuel Oil_Furnace_Medium Efficiency': 'oil_med',
    'Light Fuel Oil_Boiler_Medium Efficiency':  'oil_med',
    'Heavy Fuel Oil_Furnace_Low Efficiency':    'oil_low',
    'Heavy Fuel Oil_Furnace_Medium Efficiency': 'oil_med',
    'Heavy Fuel Oil_Boiler_Medium Efficiency':  'oil_med',
    'Propane_Furnace_Medium Efficiency':        'other',
    'Propane_Furnace_High Efficiency':          'other',
    'Propane_Boiler_Medium Efficiency':         'other',
    'Electricity_Furnace_High Efficiency':      'elec',
    'Electricity_ASHP_Natural Gas_Backup':      'heat_pump',
    'Electricity_ASHP_Electricity_Backup':      'heat_pump',
    'Electricity_ASHP':                         'heat_pump',
    'Electricity_GSHP':                         'heat_pump',
    'Electricity_Boiler_High Efficiency':       'elec',
}

# Variables identical across sub-regions (shares copied as-is)
COMM_IDENTICAL_VARIABLES = {
    'building_shell_shares',
}

# Absolute quantities split by population share when disaggregating AT/BCT
# into individual provinces/territories. lighting_energy / water_heating_energy
# / aux_equipment_energy are CEUD region-wide totals (like total_floorspace),
# not technology shares, so they're apportioned the same way.
COMM_POP_SPLIT_VARIABLES = {
    'total_floorspace', 'floorspace_by_activity',
    'lighting_energy', 'water_heating_energy', 'aux_equipment_energy',
    'space_heating_energy', 'space_cooling_energy', 'space_heating_by_activity',
}

# Market share variables needing efficiency correction + renormalization
COMM_MARKET_SHARE_VARIABLES = {
    'hvac_cold',
    'hot_water_tech',
}

# hvac_marine is BC-only — dropped for all other sub-regions
MARINE_ONLY_REGIONS = {'BC'}

# Maps aggregated CEUD region → individual sub-region codes and their
# full names as they appear in the Stats Can population CSV (CER_AREA_NAME
# translates these to the corresponding vDmd-CIMS.csv 'Area' spelling)
COMM_REGION_MAP = {
    'AT': {
        'Newfoundland and Labrador': 'NL',
        'Prince Edward Island':      'PE',
        'Nova Scotia':               'NS',
        'New Brunswick':             'NB',
    },
    'BC': {
        'Yukon':                 'YT',
        'Northwest Territories': 'NT',
        'Nunavut':               'NU',
        # BC itself stays as BC — handled separately
    },
}

COMM_PROJECTION_END = 2100


def _build_comm_cer_shares(cer_csv: Path, enduse: str) -> pd.DataFrame:
    """
    Load CER's vDmd-CIMS.csv commercial demand for one enduse ('Space
    Heating' or 'Water Heating'), aggregate across CIMS commercial activity
    sectors (CER_COMMERCIAL_SECTORS) and fold minor fuels into their
    canonical group (CER_RAW_FUEL_GROUP), and return each sub-region's fuel
    share within its parent AT/BC-territory disaggregation group.

    Returns
    -------
    pd.DataFrame with columns: year, geo, fuel, group, share
        geo   -- CER Area name (see CER_AREA_NAME for the mapping from the
                 Stats Can population geo names used elsewhere in this module)
        fuel  -- canonical fuel group name (COMM_CER_FUEL_TO_CEUD key)
        share = this geo's fraction of the group (AT or BC-territories) total
        for this fuel/year.
    """
    df = pd.read_csv(cer_csv, sep=';')
    df = df[
        df['Sector'].isin(CER_COMMERCIAL_SECTORS) & (df['Enduse'] == enduse)
    ].copy()
    df['fuel'] = df['Fuel'].map(CER_RAW_FUEL_GROUP)
    df = df[df['fuel'].notna()]

    # Aggregate across activity sectors and folded fuels
    agg = (
        df.groupby(['Year', 'Area', 'fuel'])['Data']
        .sum()
        .reset_index()
        .rename(columns={'Year': 'year', 'Area': 'geo', 'Data': 'demand_TJ'})
    )

    # Assign each geo to its parent group
    at_geos   = {CER_AREA_NAME.get(g, g) for g in COMM_REGION_MAP['AT']}
    terr_geos = {CER_AREA_NAME.get(g, g) for g in COMM_REGION_MAP['BC']}

    def get_group(geo):
        if geo in at_geos:
            return 'AT'
        if geo in terr_geos:
            return 'BC_terr'
        return None

    agg['group'] = agg['geo'].apply(get_group)
    agg = agg[agg['group'].notna()].copy()

    # Group total per fuel/year within each parent group
    group_total = (
        agg.groupby(['year', 'fuel', 'group'])['demand_TJ']
        .sum()
        .rename('group_total')
        .reset_index()
    )
    agg = agg.merge(group_total, on=['year', 'fuel', 'group'])
    agg['share'] = agg['demand_TJ'] / agg['group_total'].replace(0, np.nan)
    # A zero group total means nobody in the group had measurable demand for
    # this fuel/year (e.g. Natural Gas across Atlantic Canada in 2000, before
    # any province's distribution network existed) -- that's a real 0% share
    # for every geo, not an undefined one.
    agg.loc[agg['group_total'] == 0, 'share'] = 0.0

    return agg[['year', 'geo', 'fuel', 'group', 'share']].copy()


def _get_comm_efficiency(efficiencies: dict, category: str, year: int) -> float:
    """Look up efficiency for a commercial category/year."""
    key = COMM_CATEGORY_EFFICIENCY_KEY.get(category)
    if key is None:
        return 1.0
    series = efficiencies.get(key, {})
    if not series:
        return 1.0
    if year in series:
        return series[year]
    return series[max(series.keys())]


def _build_comm_cat_to_fuel_map() -> dict:
    """Invert COMM_CER_FUEL_TO_CEUD into {category: fuel}."""
    cat_to_fuel = {}
    for fuel, cats in COMM_CER_FUEL_TO_CEUD.items():
        for cat in cats:
            cat_to_fuel[cat] = fuel
    return cat_to_fuel

def _load_efficiencies(efficiency_xls: Path) -> dict:
    """
    Load heating system efficiencies from CEUD Table 32.

    Returns
    -------
    dict mapping efficiency_key → {year: efficiency_as_fraction}
    e.g. {'ng_low': {2000: 0.62, 2001: 0.62, ...}, ...}
    """
    df = pd.read_excel(str(efficiency_xls), sheet_name='Table 32', header=None)

    years_row = df.iloc[10]
    year_cols = {
        int(v): j for j, v in enumerate(years_row)
        if pd.notna(v) and str(v).replace('.0', '').isdigit()
    }

    def get_series(row_idx: int) -> dict[int, float]:
        row = df.iloc[row_idx]
        return {
            yr: float(row.iloc[col]) / 100.0
            for yr, col in year_cols.items()
            if pd.notna(row.iloc[col])
        }

    return {
        'ng_low':    get_series(16),
        'ng_med':    get_series(17),
        'ng_high':   get_series(18),
        'oil_low':   get_series(13),
        'oil_med':   get_series(14),
        'oil_high':  get_series(15),
        'elec':      get_series(19),
        'heat_pump': get_series(20),
        'other':     get_series(21),
        'wood':      get_series(22),
    }

def disaggregate_commercial(
    region_dfs: dict,
    cer_csv: Path,
    pop_csv: Path,
    efficiency_xls: Path,
) -> dict:
    """
    Split AT and BC CEUD commercial data into individual provinces/territories.

    Parameters
    ----------
    region_dfs : dict
        Output of extract_all_data for each region — keys are region codes
        ('AB', 'AT', 'BC', etc.), values are pl.DataFrames.
    cer_csv : Path
        Path to CER's vDmd-CIMS.csv (commercial Space Heating / Water Heating
        demand by fuel, activity sector, and province/territory).
    pop_csv : Path
        Path to population CSV (Statistics Canada table 17-10-0009-01).
    efficiency_xls : Path
        Path to CEUD national res_ca_e_32.xls (Table 32 efficiencies).

    Returns
    -------
    dict
        Same structure as region_dfs but with AT and BC replaced by their
        constituent provinces/territories. BC itself is kept; only the
        territory sub-regions are added.
        Keys: 'AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'QC',
              'PE', 'SK', 'YT'
    """
    cer_heat_shares  = _build_comm_cer_shares(cer_csv, 'Space Heating')
    cer_water_shares = _build_comm_cer_shares(cer_csv, 'Water Heating')
    cer_shares_by_variable = {
        'hvac_cold':      cer_heat_shares,
        'hot_water_tech': cer_water_shares,
    }
    efficiencies = _load_efficiencies(efficiency_xls)
    cat_to_fuel  = _build_comm_cat_to_fuel_map()

    # Build population shares for each group
    at_pop = build_population_shares(
        pop_csv,
        regions=list(COMM_REGION_MAP['AT'].keys()),
        projection_end=COMM_PROJECTION_END,
    )
    terr_pop = build_population_shares(
        pop_csv,
        regions=list(COMM_REGION_MAP['BC'].keys()),
        projection_end=COMM_PROJECTION_END,
    )

    results = {}

    # Keep non-AT, non-BC regions unchanged
    for code, df in region_dfs.items():
        if code not in ('AT', 'BC'):
            results[code] = df

    # ------------------------------------------------------------------
    # Split AT → NL, PE, NS, NB
    # ------------------------------------------------------------------
    if 'AT' in region_dfs:
        at_df = region_dfs['AT']
        at_pd = at_df.to_pandas()
        at_pd = at_pd[at_pd['year'] <= COMM_PROJECTION_END].copy()

        at_cer_shares = {
            var: shares[shares['group'] == 'AT'].copy()
            for var, shares in cer_shares_by_variable.items()
        }

        for geo_name, prov_code in COMM_REGION_MAP['AT'].items():
            sub_df = at_pd.copy()
            sub_df['region'] = prov_code
            cer_geo = CER_AREA_NAME.get(geo_name, geo_name)

            def get_cer_share_at(row, geo=cer_geo, shares_by_var=at_cer_shares):
                fuel = cat_to_fuel.get(row['category'])
                shares = shares_by_var.get(row['variable'])
                if fuel is None or shares is None:
                    return np.nan
                mask = (
                    (shares['geo'] == geo) &
                    (shares['fuel'] == fuel)
                )
                available = shares[mask].sort_values('year')
                if available.empty:
                    return np.nan
                yr_match = available[available['year'] == row['year']]
                if not yr_match.empty:
                    return float(yr_match.iloc[0]['share'])
                return float(available.iloc[-1]['share'])

            def get_pop_share_at(row, geo=geo_name):
                try:
                    return at_pop.loc[
                        (at_pop['territory'] == geo) &
                        (at_pop['year'] == row['year']),
                        'pop_share'
                    ].iloc[0]
                except (IndexError, KeyError):
                    avail = at_pop[at_pop['territory'] == geo]
                    return float(avail.iloc[-1]['pop_share']) if not avail.empty else np.nan

            def apply_share_at(row, _get_pop=get_pop_share_at, _get_cer=get_cer_share_at):
                var = row['variable']
                if var in COMM_IDENTICAL_VARIABLES:
                    return row['value']
                if var in COMM_POP_SPLIT_VARIABLES:
                    return row['value'] * _get_pop(row)
                if var == 'hvac_marine':
                    return np.nan
                return row['value'] * _get_cer(row)

            sub_df['value'] = sub_df.apply(apply_share_at, axis=1)
            # Drop marine rows for Atlantic
            sub_df = sub_df[sub_df['variable'] != 'hvac_marine']
            results[prov_code] = pl.from_pandas(sub_df)

    # ------------------------------------------------------------------
    # Split BC territories → YT, NT, NU  (BC itself kept unchanged)
    # ------------------------------------------------------------------
    if 'BC' in region_dfs:
        bc_df = region_dfs['BC']
        bc_pd = bc_df.to_pandas()
        bc_pd = bc_pd[bc_pd['year'] <= COMM_PROJECTION_END].copy()

        terr_cer_shares = {
            var: shares[shares['group'] == 'BC_terr'].copy()
            for var, shares in cer_shares_by_variable.items()
        }

        # Build BC-only population share from full population CSV
        # BC's share of (BC + territories) total
        bc_and_terr = ['British Columbia'] + list(COMM_REGION_MAP['BC'].keys())
        bc_terr_pop = build_population_shares(
            pop_csv,
            regions=bc_and_terr,
            projection_end=COMM_PROJECTION_END,
        )

        def get_pop_share_bc_province(row):
            try:
                return bc_terr_pop.loc[
                    (bc_terr_pop['territory'] == 'British Columbia') &
                    (bc_terr_pop['year'] == row['year']),
                    'pop_share'
                ].iloc[0]
            except (IndexError, KeyError):
                avail = bc_terr_pop[bc_terr_pop['territory'] == 'British Columbia']
                return float(avail.iloc[-1]['pop_share']) if not avail.empty else np.nan

        def apply_share_bc_province(row):
            var = row['variable']
            if var in COMM_IDENTICAL_VARIABLES:
                return row['value']
            if var in COMM_POP_SPLIT_VARIABLES:
                return row['value'] * get_pop_share_bc_province(row)
            # BC keeps marine, hvac_cold and hot_water_tech stay as BCT values
            # (territories are cold-only; BC retains its marine/cold split)
            return row['value']

        bc_split = bc_pd.copy()
        bc_split['region'] = 'BC'
        bc_split['value'] = bc_split.apply(apply_share_bc_province, axis=1)
        results['BC'] = pl.from_pandas(bc_split)

        # Territory population shares relative to (BC + territories) total
        # so each territory gets its fraction of the full BCT floorspace
        terr_names = list(COMM_REGION_MAP['BC'].keys())
        terr_of_bct_pop = build_population_shares(
            pop_csv,
            regions=bc_and_terr,
            projection_end=COMM_PROJECTION_END,
        )

        for geo_name, terr_code in COMM_REGION_MAP['BC'].items():
            sub_df = bc_pd.copy()
            sub_df['region'] = terr_code
            cer_geo = CER_AREA_NAME.get(geo_name, geo_name)

            def get_cer_share_bc(row, geo=cer_geo, shares_by_var=terr_cer_shares):
                fuel = cat_to_fuel.get(row['category'])
                shares = shares_by_var.get(row['variable'])
                if fuel is None or shares is None:
                    return np.nan
                mask = (
                    (shares['geo'] == geo) &
                    (shares['fuel'] == fuel)
                )
                available = shares[mask].sort_values('year')
                if available.empty:
                    return np.nan
                yr_match = available[available['year'] == row['year']]
                if not yr_match.empty:
                    return float(yr_match.iloc[0]['share'])
                return float(available.iloc[-1]['share'])

            def get_pop_share_bc(row, geo=geo_name):
                try:
                    return terr_of_bct_pop.loc[
                        (terr_of_bct_pop['territory'] == geo) &
                        (terr_of_bct_pop['year'] == row['year']),
                        'pop_share'
                    ].iloc[0]
                except (IndexError, KeyError):
                    avail = terr_of_bct_pop[terr_of_bct_pop['territory'] == geo]
                    return float(avail.iloc[-1]['pop_share']) if not avail.empty else np.nan
              
            def apply_share_bc(row, _get_pop=get_pop_share_bc, _get_cer=get_cer_share_bc):
                var = row['variable']
                if var in COMM_IDENTICAL_VARIABLES:
                    return row['value']
                if var in COMM_POP_SPLIT_VARIABLES:
                    return row['value'] * _get_pop(row)
                if var == 'hvac_marine':
                    return np.nan
                return row['value'] * _get_cer(row)

            sub_df['value'] = sub_df.apply(apply_share_bc, axis=1)
            # Drop marine rows for territories
            sub_df = sub_df[sub_df['variable'] != 'hvac_marine']
            results[terr_code] = pl.from_pandas(sub_df)

    # ------------------------------------------------------------------
    # Efficiency correction + renormalization for market share variables
    # ------------------------------------------------------------------
    # Efficiency correction + renormalization for disaggregated regions only.
    # AT and BC sub-regions were split using CER fuel-demand shares, which are
    # fuel energy shares — these need efficiency correction to convert to
    # technology market shares. Non-split regions (AB, MB, ON, QC, SK) already
    # have correct market shares from Table 24 and must NOT be modified.
    disaggregated_codes = (
        set(COMM_REGION_MAP['AT'].values()) |
        set(COMM_REGION_MAP['BC'].values()) |
        {'BC'}
    )
    for code in list(results.keys()):
        if code not in disaggregated_codes:
            continue
        df_pd = results[code].to_pandas()
        changed = False

        for var in COMM_MARKET_SHARE_VARIABLES:
            var_mask = df_pd['variable'] == var
            if not var_mask.any():
                continue
            changed = True

            # Divide by efficiency to convert fuel share → useful energy share
            def eff_correct(row):
                eff = _get_comm_efficiency(efficiencies, row['category'], row['year'])
                return row['value'] / eff if eff > 0 else row['value']

            df_pd.loc[var_mask, 'value'] = df_pd[var_mask].apply(eff_correct, axis=1)

            # Renormalize per region/year
            totals = (
                df_pd[var_mask]
                .groupby(['region', 'year'])['value']
                .sum()
                .rename('total')
                .reset_index()
            )
            df_pd = df_pd.merge(totals, on=['region', 'year'], how='left')
            df_pd.loc[var_mask, 'value'] = (
                df_pd.loc[var_mask, 'value'] /
                df_pd.loc[var_mask, 'total'].replace(0, np.nan)
            )
            df_pd = df_pd.drop(columns='total')

        if changed:
            results[code] = pl.from_pandas(df_pd)

    return results

# ==============================================================================
# MAIN
# ==============================================================================

def main(
    region_codes: Optional[list[str]] = None,
    output_dir: Path = OUTPUT_DIR,
    apply_projections: bool = True,
    export_csv: bool = True,
) -> dict[str, pl.DataFrame]:
    """
    Run the full commercial pipeline and optionally export a combined CSV.
    """
    if region_codes is None:
        region_codes = list(REGIONS.keys())

    if export_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    params    = load_projection_params() if apply_projections else None
    ng_splits = load_ng_efficiency_splits()
    results, failed, all_frames = {}, [], []

    for region in region_codes:
        try:
            df = extract_all_data(region, apply_projections, params, ng_splits)
            results[region] = df
            all_frames.append(df)
        except Exception as exc:
            failed.append((region, str(exc)))

    # Disaggregate AT → NL/PE/NS/NB and BC territories → YT/NT/NU
    if not failed or any(r not in [f[0] for f in failed] for r in ['AT', 'BC']):
        results = disaggregate_commercial(results, CER_DEMAND_CSV, POP_CSV, EFFICIENCY_XLS)

    # Buildings -> {Lighting, Hot Water, Refrigeration, Cooking, Plug Load}
    # and Shell -> HVAC / HVAC -> Cooling service_request intensities,
    # computed per final (post-disaggregation) region so Hot Water and HVAC
    # (Marine) reflect each region's own hot_water_tech mix / climate split.
    for region, region_df in list(results.items()):
        enduse_rows = compute_enduse_service_requests(region_df, region)
        hvac_rows = compute_hvac_service_requests(region_df, region)
        results[region] = pl.concat([region_df, enduse_rows, hvac_rows], how='diagonal_relaxed')

    all_frames = list(results.values())

    if export_csv and all_frames:
        combined = pl.concat(all_frames, how='diagonal_relaxed')
        combined = combined.with_columns(
            pl.when(pl.col('year') <= LAST_HIST_YEAR)
            .then(pl.lit('CEUD'))
            .otherwise(pl.lit('Assumptions'))
            .alias('source')
        )
        combined = combined.sort(['region', 'variable', 'category', 'year'])
        output_file = output_dir / "commercial.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n✅ Commercial extraction complete")
        print(f"   Total rows:          {combined.height:,}")
        print(f"   Regions processed:   {combined['region'].n_unique()}")
        print(f"   Variables:           {sorted(combined['variable'].unique().to_list())}")
        print(f"   Years covered:       {combined['year'].min()} – {combined['year'].max()}")
        print(f"   Saved to:            {output_file}")
        combined = combined.rename({
            'region': 'Region', 'variable': 'Variable', 'category': 'Category',
            'parameter': 'Parameter', 'unit': 'Unit', 'source': 'Source',
            'year': 'Year', 'value': 'Value',
        })
        combined.write_csv(str(output_file))

    if failed:
        print(f"\n⚠️  Failed regions ({len(failed)}):")
        for region, err in failed:
            print(f"   • {region}: {err}")

    return results


if __name__ == "__main__":
    main()
