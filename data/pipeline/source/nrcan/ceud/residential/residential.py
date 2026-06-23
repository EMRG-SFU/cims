"""
Residential Pipeline

Extracts and processes residential building data from the NRCan CEUD
(Comprehensive Energy Use Database) for all Canadian provinces and territories.

Key behavioural notes
---------------------
- BC has Marine climate heating technologies (heating only, not water heating).
- All provinces use Cold climate for water heating technologies.
- MB (Manitoba) uses "2022_after" as the last vintage bin; all others use
  "2021_after".
- All intermediate data is held as Polars DataFrames in long format:
      (province, variable, category, parameter, unit, year, value)
  This matches the shape produced by the pipeline's other modules.

Suppression handling
--------------------
RESD shares (Stats Can 2510002901-residential):
    Loaded via load_resd(), which reads all columns as strings and casts
    VALUE to Float64 with strict=False — suppressed 'x' cells become null
    (NaN in pandas). NaN demand values propagate through groupby sums and
    share divisions, producing NaN shares for suppressed territory-fuel
    pairs. Because territories drive disaggregation of Canada-level CEUD
    totals rather than contributing to them, a suppressed territory share
    gracefully falls back to the population proxy (_wood_proxy) in the
    disaggregation step.
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
from pipeline.utils.extractors.nrcan_ceud import get_row_series, row_to_series, pct_series
from pipeline.utils.output_builder import pl_to_series, pl_get_scalar
from pipeline.utils.extractors.stats_can import build_population_shares
from pipeline.utils.data_extensions import (
    extend_series_constant,
    extend_series_trend_dampener,
    load_cagr_assumptions,
    compute_cagr,
    extend_cagr_periods,
)
from pipeline.utils.extractors.stats_can import load_resd
from pipeline.utils.controls_conversions import BASE_PATH as _CIMS_BASE

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_PATH      = _CIMS_BASE / 'raw_data/nrcan/ceud/residential'
ASSUMPTIONS_CSV = _CIMS_BASE / 'raw_data/assumptions/residential_assumptions.csv'
OUTPUT_DIR     = _CIMS_BASE / 'processed_data/nrcan/ceud'
RESD_CSV = _CIMS_BASE / 'raw_data/stats_can/resd/25100029.csv'
POP_CSV  = _CIMS_BASE / 'raw_data/stats_can/population/1710000901.csv'
EFFICIENCY_XLS = _CIMS_BASE / 'raw_data/nrcan/ceud/residential/res_ca_e_32.xls'
ACTIVITY_CAGR_CSV = _CIMS_BASE / 'raw_data/assumptions/activity_cagr_projections.csv'
LAST_HIST_YEAR = CONTROLS["last_data_year"]["ceud"]

# Per-province overrides — explicit annual rate per period, bypasses computed CAGR.
CAGR_OVERRIDES: dict[str, tuple[float, ...]] = {
}

PROVINCES = {
    'AB': 'Alberta',
    'BC': 'British Columbia',
    'MB': 'Manitoba',
    'NB': 'New Brunswick',
    'NL': 'Newfoundland and Labrador',
    'NS': 'Nova Scotia',
    'ON': 'Ontario',
    'PE': 'Prince Edward Island',
    'QC': 'Quebec',
    'SK': 'Saskatchewan',
    'TR': 'Territories',
}

# Columns in the final output DataFrame
_SCHEMA = ['province', 'variable', 'category', 'parameter', 'unit', 'year', 'value']

# Maps internal variable keys to output metadata
VARIABLE_CONFIGS = [
    {'key': 'housing_thousand',       'name': 'housing_thousand',       'parameter': 'service_request',    'unit': 'household'},
    {'key': 'building_shares',        'name': 'building_shares',        'parameter': 'market_share_total', 'unit': '%'},
    {'key': 'floorspace_per_building','name': 'floorspace_per_building','parameter': 'service_request',    'unit': 'm2/building'},
    {'key': 'appliances_per_household','name': 'appliances_per_household','parameter': 'service_request', 'unit': 'unit/building'},
    {'key': 'vintage_bins_lowmed',    'name': 'vintage_bins_low_med',   'parameter': 'market_share_total', 'unit': '%'},
    {'key': 'vintage_bins_high',      'name': 'vintage_bins_high',      'parameter': 'market_share_total', 'unit': '%'},
    {'key': 'heating_lowmed_cold',    'name': 'heating_data_lowmed_cold','parameter': 'market_share_total','unit': '%'},
    {'key': 'heating_high_cold',      'name': 'heating_data_high_cold', 'parameter': 'market_share_total', 'unit': '%'},
    {'key': 'heating_lowmed_marine',  'name': 'heating_data_lowmed_marine','parameter': 'market_share_total','unit': '%'},
    {'key': 'heating_high_marine',    'name': 'heating_data_high_marine','parameter': 'market_share_total','unit': '%'},
    {'key': 'cooling_share_data',     'name': 'cooling_share_data',     'parameter': 'service_request',    'unit': 'GJ/GJ'},
    {'key': 'wh_lowmed',              'name': 'wh_lowmed',              'parameter': 'service_request',    'unit': 'GJ/GJ'},
    {'key': 'wh_high',                'name': 'wh_high',                'parameter': 'service_request',    'unit': 'GJ/GJ'},
    {'key': 'wh_tech_lowmed',         'name': 'wh_tech_lowmed',         'parameter': 'market_share_total', 'unit': '%'},
    {'key': 'wh_tech_high',           'name': 'wh_tech_high',           'parameter': 'market_share_total', 'unit': '%'},
]


# ==============================================================================
# HELPERS
# ==============================================================================



def _long(province: str, variable: str, category: str, parameter: str,
          unit: str, series: pd.Series, source: str = 'CEUD') -> pl.DataFrame:
    """
    Convert a year-indexed pd.Series to a Polars long-format DataFrame.

    Builds directly from Python lists to avoid the pyarrow dependency that
    pl.from_pandas() requires for object-dtype string columns.

    Parameters
    ----------
    province : str
    variable : str
    category : str
        Sub-dimension (e.g. building type, tech name). Empty string for scalars.
    parameter : str
    unit : str
    series : pd.Series
        Year-indexed values. NaN rows are dropped.
    source : str

    Returns
    -------
    pl.DataFrame with columns: province, variable, category, parameter,
    unit, source, year, value.
    """
    years  = [int(y)   for y, v in series.items() if pd.notna(v)]
    values = [float(v) for y, v in series.items() if pd.notna(v)]
    n = len(years)
    return pl.DataFrame({
        'province':  [province]  * n,
        'variable':  [variable]  * n,
        'category':  [category]  * n,
        'parameter': [parameter] * n,
        'unit':      [unit]      * n,
        'source':    [source]    * n,
        'year':      years,
        'value':     values,
    }).with_columns(pl.col('year').cast(pl.Int32))


# ==============================================================================
# PROJECTION PARAMETER LOADING
# ==============================================================================

def load_projection_params(assumptions_csv: Path = ASSUMPTIONS_CSV,
                           activity_cagr_csv: Path = ACTIVITY_CAGR_CSV) -> dict:
    """
    Parse the flat residential assumptions CSV and return projection parameters.

    Expected CSV columns:
        Variable, Method, Variable.1,
        1st period rate, 1st period start, 1st period end,
        2nd period rate, 2nd period start, 2nd period end,
        3rd period rate, 3rd period start, 3rd period end

    'first' in a start column means LAST_HIST_YEAR + 1.
    End values are stored inclusive and made exclusive (+1) here.

    Returns
    -------
    dict with keys 'housing_stock', 'building_shares', 'floorspace_per_building'.
    Empty dict if the file is not found.
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
        """Parse a single year value. 'first' -> LAST_HIST_YEAR + 1."""
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
        """Read start/end columns, return (start, end+1) exclusive."""
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

    # -- 1. Housing stock — activity-style CAGR (activity_cagr_projections.csv)
    cagr_start, cagr_end, periods = load_cagr_assumptions('Residential', activity_cagr_csv)
    params['housing_stock'] = {
        'cagr_start': cagr_start,
        'cagr_end':   cagr_end,
        'periods':    periods,
    }

    # -- 2. Building shares — trend then dampener ------------------------------
    bs_row = raw[(raw['Variable'] == 'Housing by type') &
                 (raw['Variable.1'] == 'Detached')].iloc[0]
    trend_yrs    = parse_period(bs_row, '1st period') or (LAST_HIST_YEAR + 1, 2031)
    decline1_yrs = parse_period(bs_row, '2nd period') or (2031, 2051)
    decline2_yrs = parse_period(bs_row, '3rd period') or (2051, 2101)
    decline1_rate = parse_pct(bs_row['2nd period rate']) or -0.05
    decline2_rate = parse_pct(bs_row['3rd period rate']) or -0.10

    params['building_shares'] = {
        'method': 'trend_dampener',
        'trend_start': 2000,
        'trend_end': LAST_HIST_YEAR,
        'trend_period': trend_yrs,
        'decline_periods': [
            (decline1_yrs[0], decline1_yrs[1], decline1_rate),
            (decline2_yrs[0], decline2_yrs[1], decline2_rate),
        ],
    }

    # -- 3. Floorspace — trend then dampener -----------------------------------
    fs_row = raw[raw['Variable'] == 'Floorspace'].iloc[0]
    fs_trend_yrs    = parse_period(fs_row, '1st period') or (LAST_HIST_YEAR + 1, 2031)
    fs_decline1_yrs = parse_period(fs_row, '2nd period') or (2031, 2051)
    fs_decline2_yrs = parse_period(fs_row, '3rd period') or (2051, 2101)
    fs_rate1 = parse_pct(fs_row['2nd period rate']) or -0.05
    fs_rate2 = parse_pct(fs_row['3rd period rate']) or -0.10

    params['floorspace_per_building'] = {
        'method': 'trend_dampener',
        'trend_start': 2000,
        'trend_end': LAST_HIST_YEAR,
        'trend_period': fs_trend_yrs,
        'decline_periods': [
            (fs_decline1_yrs[0], fs_decline1_yrs[1], fs_rate1),
            (fs_decline2_yrs[0], fs_decline2_yrs[1], fs_rate2),
        ],
    }

    return params


# ==============================================================================
# TABLE LOADING
# ==============================================================================

def load_tables(province_code: str) -> dict:
    """
    Load CEUD Excel sheets for a province and return them as Polars DataFrames.

    Parameters
    ----------
    province_code : str
        2-letter code (e.g. 'BC', 'ON').

    Returns
    -------
    dict mapping sheet names to pl.DataFrame.

    Raises
    ------
    FileNotFoundError if the province's data file does not exist.
    """
    file_path = BASE_PATH / f"res_{province_code.lower()}_e.xls"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    table_names = [
        "Table 4", "Table 10", "Table 11", "Table 15", "Table 18",
        "Table 19", "Table 20", "Table 22", "Table 23", "Table 24",
        "Table 25", "Table 31",
    ]
    return {name: pl.read_excel(str(file_path), sheet_name=name, has_header=False)
            for name in table_names}


# ==============================================================================
# EXTRACTION — HOUSING STOCK
# ==============================================================================

def extract_housing_stock(province: str, tables: dict) -> list[pl.DataFrame]:
    """
    Extract total housing stock (thousands → units) and per-type stock.

    Returns
    -------
    list of pl.DataFrame
        One long-format DataFrame per variable extracted.
    """
    t15 = tables["Table 15"]
    building_types = ["Single Detached", "Single Attached", "Apartments", "Mobile Homes"]

    total_raw = row_to_series(t15, "Total Housing Stock (thousands)") * 1000
    frames = [_long(province, 'housing_thousand', '', 'service_request', 'household', total_raw)]

    for bt in building_types:
        s = row_to_series(t15, bt, match_n=0)
        frames.append(_long(province, 'housing_by_type', bt, 'service_request', 'household', s))

    return frames


# ==============================================================================
# EXTRACTION — BUILDING SHARES
# ==============================================================================

def extract_building_shares(province: str, tables: dict) -> list[pl.DataFrame]:
    """
    Extract building share percentages as fractions (0–1).

    Returns
    -------
    list of pl.DataFrame
    """
    t15 = tables["Table 15"]
    building_types = ["Single Detached", "Single Attached", "Apartments", "Mobile Homes"]

    frames = []
    for bt in building_types:
        s = pct_series(t15, bt, match_n=1)
        frames.append(_long(province, 'building_shares', bt, 'market_share_total', '%', s))

    return frames


# ==============================================================================
# EXTRACTION — FLOORSPACE PER BUILDING
# ==============================================================================

def extract_floorspace_per_building(province: str, tables: dict) -> list[pl.DataFrame]:
    """
    Compute floor space per building (m²/building) = total_fs_m2 / stock_units.

    Returns
    -------
    list of pl.DataFrame
    """
    t15, t18 = tables["Table 15"], tables["Table 18"]
    building_types = ["Single Detached", "Single Attached", "Apartments", "Mobile Homes"]

    frames = []
    for bt in building_types:
        fs_m2   = row_to_series(t18, bt, match_n=0) * 1e6   # million m² → m²
        stock   = row_to_series(t15, bt, match_n=0) * 1000  # thousands → units
        ratio   = fs_m2 / stock.replace(0, np.nan)
        frames.append(_long(province, 'floorspace_per_building', bt,
                            'service_request', 'm2/building', ratio))

    return frames


# ==============================================================================
# EXTRACTION — APPLIANCES
# ==============================================================================

def extract_appliances(province: str, tables: dict) -> list[pl.DataFrame]:
    """
    Extract appliances per household for each appliance type.

    Returns
    -------
    list of pl.DataFrame
    """
    t31 = tables["Table 31"]

    appliance_mapping = {
        "Refrigerator":      "Refrigerators",
        "Freezer":           "Freezers",
        "Range":             "Ranges",
        "Dishwasher":        "Dishwashing",
        "Clothes Washer":    "Clothes Washing",
        "Other Appliances1": "Minor Appliances",
    }

    frames = []
    for excel_name, cims_name in appliance_mapping.items():
        try:
            s = row_to_series(t31, excel_name, match_n=1)
            frames.append(_long(province, 'appliances_per_household', cims_name,
                                'service_request', 'unit/building', s))
        except KeyError:
            pass

    return frames


# ==============================================================================
# EXTRACTION — VINTAGE BINS
# ==============================================================================

def extract_vintages(province: str, tables: dict,
                     building_shares_df: pl.DataFrame) -> list[pl.DataFrame]:
    """
    Aggregate vintage (age) bins by density type.

    Low/med density = weighted average of Single Detached, Single Attached,
    Mobile Homes using their building shares as weights.
    High density = Apartments.

    Parameters
    ----------
    building_shares_df : pl.DataFrame
        Output of extract_building_shares — used for weighting.

    Returns
    -------
    list of pl.DataFrame
    """
    t19, t20 = tables["Table 19"], tables["Table 20"]
    is_mb = province.upper() == 'MB'
    last_label = "2022_after" if is_mb else "2021_after"

    vintage_labels = [
        "Before 1946", "1946–1960", "1961–1977", "1978–1983", "1984–1995",
        "1996–2000", "2001–2005", "2006–2010", "2011–2015", "2016_2020",
        last_label,
    ]

    # Pull all raw vintage fractions into a dict of {label: {bt: series}}
    raw = {
        "Single Detached": {lbl: pct_series(t19, lbl, match_n=1) for lbl in vintage_labels},
        "Single Attached":  {lbl: pct_series(t19, lbl, match_n=3) for lbl in vintage_labels},
        "Apartments":       {lbl: pct_series(t20, lbl, match_n=1) for lbl in vintage_labels},
        "Mobile Homes":     {lbl: pct_series(t20, lbl, match_n=3) for lbl in vintage_labels},
    }

    # Build weight series for the three low/med types from building_shares_df
    def _weight(bt: str) -> pd.Series:
        return pl_to_series(building_shares_df.filter(pl.col('category') == bt))

    w_det  = _weight("Single Detached")
    w_att  = _weight("Single Attached")
    w_mob  = _weight("Mobile Homes")
    w_tot  = w_det.add(w_att, fill_value=0).add(w_mob, fill_value=0).replace(0, np.nan)

    def _weighted_avg(label: str) -> pd.Series:
        """Weighted average of Detached, Attached, Mobile for one vintage label."""
        numerator = (
            raw["Single Detached"][label].mul(w_det, fill_value=0)
            .add(raw["Single Attached"][label].mul(w_att, fill_value=0), fill_value=0)
            .add(raw["Mobile Homes"][label].mul(w_mob, fill_value=0), fill_value=0)
        )
        return numerator / w_tot

    # Aggregate into CIMS vintage bins
    bin_map_lowmed = {
        "<1960":     ["Before 1946", "1946–1960"],
        "1961-1980": ["1961–1977", "1978–1983"],
        "1981-2000": ["1984–1995", "1996–2000"],
        "2001-2020": ["2001–2005", "2006–2010", "2011–2015", "2016_2020"],
        "2021-2035": [last_label],
    }

    frames = []
    for bin_name, labels in bin_map_lowmed.items():
        s = sum(_weighted_avg(lbl) for lbl in labels)
        frames.append(_long(province, 'vintage_bins_lowmed', bin_name,
                            'market_share_total', '%', s))

    bin_map_high = {
        "<1960":     ["Before 1946", "1946–1960"],
        "1961-1980": ["1961–1977", "1978–1983"],
        "1981-2000": ["1984–1995", "1996–2000"],
        "2001-2020": ["2001–2005", "2006–2010", "2011–2015", "2016_2020"],
        "2021-2035": [last_label],
    }
    for bin_name, labels in bin_map_high.items():
        s = sum(raw["Apartments"][lbl] for lbl in labels)
        frames.append(_long(province, 'vintage_bins_high', bin_name,
                            'market_share_total', '%', s))

    return frames


# ==============================================================================
# EXTRACTION — HEATING TECHNOLOGIES
# ==============================================================================

def _extract_heating_bucket(table: pl.DataFrame, tech_rows: list[str]) -> pd.Series:
    """Sum one or more CEUD heating rows (already converted to fractions)."""
    if not tech_rows:
        return pd.Series(dtype=float)
    parts = []
    for row_label in tech_rows:
        try:
            parts.append(pct_series(table, row_label, match_n=1))
        except KeyError:
            pass
    if not parts:
        return pd.Series(dtype=float)
    total = parts[0]
    for p in parts[1:]:
        total = total.add(p, fill_value=0)
    return total


def _weighted_heat(tech_series: dict[str, pd.Series],
                   w_det: pd.Series, w_att: pd.Series,
                   w_mob: pd.Series) -> dict[str, pd.Series]:
    """
    Weighted average of Detached, Attached, Mobile heating shares.

    Parameters
    ----------
    tech_series : dict mapping tech_name → {bt: pd.Series}
        Outer key = CIMS tech name. Inner key = building type.
    w_det, w_att, w_mob : pd.Series
        Building-share weights for each of the three low/med types.

    Returns
    -------
    dict mapping tech_name → weighted pd.Series
    """
    w_tot = w_det.add(w_att, fill_value=0).add(w_mob, fill_value=0).replace(0, np.nan)
    out = {}
    for tech, bt_map in tech_series.items():
        num = (
            bt_map.get('det', pd.Series(dtype=float)).mul(w_det, fill_value=0)
            .add(bt_map.get('att', pd.Series(dtype=float)).mul(w_att, fill_value=0), fill_value=0)
            .add(bt_map.get('mob', pd.Series(dtype=float)).mul(w_mob, fill_value=0), fill_value=0)
        )
        out[tech] = num / w_tot
    return out


def extract_heating_technologies(province: str, tables: dict,
                                  building_shares_df: pl.DataFrame) -> list[pl.DataFrame]:
    """
    Extract heating technology market shares by density type and climate zone.

    BC receives both Cold and Marine data.  All other provinces get Cold only.

    Parameters
    ----------
    building_shares_df : pl.DataFrame
        Output of extract_building_shares — used for weighting.

    Returns
    -------
    list of pl.DataFrame
    """
    t22, t23, t24, t25 = (tables["Table 22"], tables["Table 23"],
                           tables["Table 24"], tables["Table 25"])
    is_bc = province.upper() == 'BC'

    def _weight(bt: str) -> pd.Series:
        return pl_to_series(building_shares_df.filter(pl.col('category') == bt))

    w_det = _weight("Single Detached")
    w_att = _weight("Single Attached")
    w_mob = _weight("Mobile Homes")

    # -- COLD climate ----------------------------------------------------------
    cold_buckets = {
        "Natural Gas_Furnace_Low Efficiency":      ["Natural Gas – Normal Efficiency", "Natural Gas/Electric"],
        "Natural Gas_Furnace_Medium Efficiency":   ["Natural Gas – Medium Efficiency"],
        "Natural Gas_Furnace_High Efficiency":     ["Natural Gas – High Efficiency"],
        "Propane_Furnace_Medium Efficiency":       ["Other1"],  
        "Electricity_Resistance_High Efficiency":  ["Electric"],
        "Light Fuel Oil_Furnace_Low Efficiency":   ["Heating Oil – Normal Efficiency", "Heating Oil/Electric"],
        "Light Fuel Oil_Furnace_Medium Efficiency":["Heating Oil – Medium Efficiency"],
        "Light Fuel Oil_Furnace_High Efficiency":  ["Heating Oil – High Efficiency"],
        "Wood_Furnace_Low Efficiency":             ["Wood", "Wood/Electric", "Wood/Heating Oil"],
        'Natural Gas_ASHP_Natural Gas_Backup':     [],
        "Electricity_ASHP_Natural Gas_Backup":     ["Heat Pump"],
        "Electricity_ASHP_Electricity_Backup":     [],
    }

    # For each tech, collect per-building-type series then weighted average
    cold_by_tech_lowmed = {}
    cold_high = {}
    for tech, row_labels in cold_buckets.items():
        cold_by_tech_lowmed[tech] = {
            'det': _extract_heating_bucket(t22, row_labels),
            'att': _extract_heating_bucket(t23, row_labels),
            'mob': _extract_heating_bucket(t25, row_labels),
        }
        cold_high[tech] = _extract_heating_bucket(t24, row_labels)

    lowmed_cold = _weighted_heat(cold_by_tech_lowmed, w_det, w_att, w_mob)

    frames = []
    for tech, s in lowmed_cold.items():
        frames.append(_long(province, 'heating_lowmed_cold', tech,
                            'market_share_total', '%', s))
    for tech, s in cold_high.items():
        frames.append(_long(province, 'heating_high_cold', tech,
                            'market_share_total', '%', s))

    # -- MARINE climate (BC only) ----------------------------------------------
    if is_bc:
        marine_buckets = {
            "Natural Gas_Furnace_Low Efficiency":      ["Natural Gas – Normal Efficiency", "Natural Gas/Electric"],
            "Natural Gas_Furnace_Medium Efficiency":   ["Natural Gas – Medium Efficiency"],
            "Natural Gas_Furnace_High Efficiency":     ["Natural Gas – High Efficiency"],
            "Propane_Furnace_Medium Efficiency":       ["Other1"],  
            "Electricity_Resistance_High Efficiency":  ["Electric"],
            "Light Fuel Oil_Furnace_Low Efficiency":   ["Heating Oil – Normal Efficiency", "Heating Oil/Electric"],
            "Light Fuel Oil_Furnace_Medium Efficiency":["Heating Oil – Medium Efficiency"],
            "Light Fuel Oil_Furnace_High Efficiency":  ["Heating Oil – High Efficiency"],
            "Wood_Furnace_Low Efficiency":             ["Wood", "Wood/Electric", "Wood/Heating Oil"],
            "Natural Gas_ASHP":                        [],
            "Electricity_ASHP":                        ["Heat Pump"],
        }

        marine_by_tech_lowmed = {}
        marine_high = {}
        for tech, row_labels in marine_buckets.items():
            marine_by_tech_lowmed[tech] = {
                'det': _extract_heating_bucket(t22, row_labels),
                'att': _extract_heating_bucket(t23, row_labels),
                'mob': _extract_heating_bucket(t25, row_labels),
            }
            marine_high[tech] = _extract_heating_bucket(t24, row_labels)

        lowmed_marine = _weighted_heat(marine_by_tech_lowmed, w_det, w_att, w_mob)

        for tech, s in lowmed_marine.items():
            frames.append(_long(province, 'heating_lowmed_marine', tech,
                                'market_share_total', '%', s))
        for tech, s in marine_high.items():
            frames.append(_long(province, 'heating_high_marine', tech,
                                'market_share_total', '%', s))

    return frames


# ==============================================================================
# EXTRACTION — COOLING TECHNOLOGIES
# ==============================================================================

def extract_cooling_technologies(province: str, tables: dict) -> list[pl.DataFrame]:
    """
    Extract cooling technology market shares as fractions.

    For Territories (TR), missing 2002 and 2014 values are filled from the
    adjacent year.

    Returns
    -------
    list of pl.DataFrame
    """
    t4 = tables["Table 4"]

    frames = []
    for cooling_type in ["Room", "Central"]:
        s = pct_series(t4, cooling_type, match_n=1)

        # Territories: patch known missing years
        if province.upper() == 'TR':
            for miss, fill in [(2002, 2001), (2014, 2013)]:
                if fill in s.index and (
                    miss not in s.index or
                    pd.isna(s.get(miss, np.nan)) or
                    s.get(miss, 0) == 0
                ):
                    s[miss] = s[fill]

        frames.append(_long(province, 'cooling_share_data', cooling_type,
                            'service_request', 'GJ/GJ', s))

    return frames


# ==============================================================================
# EXTRACTION — WATER HEATING
# ==============================================================================

def extract_water_heating(province: str, tables: dict,
                           building_shares_df: pl.DataFrame,
                           heating_df: pl.DataFrame) -> list[pl.DataFrame]:
    """
    Derive water heating intensity and technology shares.

    CEUD does not break water heating by efficiency tier, so we borrow the
    NG and oil efficiency split from the heating technology data.

    Parameters
    ----------
    building_shares_df : pl.DataFrame
        Output of extract_building_shares.
    heating_df : pl.DataFrame
        Output of extract_heating_technologies (cold climate only needed here).

    Returns
    -------
    list of pl.DataFrame
    """
    t10, t11 = tables["Table 10"], tables["Table 11"]

    def _weight(bt: str) -> pd.Series:
        return pl_to_series(building_shares_df.filter(pl.col('category') == bt))

    def _heat_tech(variable: str, category: str) -> pd.Series:
        """Retrieve a heating tech series from the heating DataFrame."""
        subset = heating_df.filter(
            (pl.col('variable') == variable) & (pl.col('category') == category)
        )
        return pl_to_series(subset) if len(subset) > 0 else pd.Series(dtype=float)

    # Water heating shares by building type — from Table 11 Shares (%) section
    # match_n=1 picks the second occurrence of each label (the % rows, not PJ rows)
    wh_det  = pct_series(t11, "Single Detached", match_n=1)
    wh_att  = pct_series(t11, "Single Attached",  match_n=1)
    wh_apt  = pct_series(t11, "Apartments",       match_n=1)
    wh_mob  = pct_series(t11, "Mobile Homes",     match_n=1)

    # wh_lowmed = sum of detached + attached + mobile shares
    wh_lowmed = wh_det.add(wh_att, fill_value=0).add(wh_mob, fill_value=0)

    # wh_high = apartments share
    wh_high = wh_apt

    # Water heating fuel shares (aggregate from t10)
    ng_wh   = pct_series(t10, "Natural Gas", match_n=1).add(
                pct_series(t10, "Other2", match_n=1), fill_value=0)
    oil_wh  = pct_series(t10, "Heating Oil", match_n=1)
    elec_wh = (pct_series(t10, "Electricity", match_n=1)
               .add(pct_series(t10, "Wood", match_n=1), fill_value=0))
    
    # NG efficiency split from cold heating data
    def _ng_total(var: str) -> pd.Series:
        lo = _heat_tech(var, "Natural Gas_Furnace_Low Efficiency")
        md = _heat_tech(var, "Natural Gas_Furnace_Medium Efficiency")
        hi = _heat_tech(var, "Natural Gas_Furnace_High Efficiency")
        return lo.add(md, fill_value=0).add(hi, fill_value=0).replace(0, np.nan)

    def _oil_total(var: str) -> pd.Series:
        lo = _heat_tech(var, "Light Fuel Oil_Furnace_Low Efficiency")
        md = _heat_tech(var, "Light Fuel Oil_Furnace_Medium Efficiency")
        return lo.add(md, fill_value=0).replace(0, np.nan)

    # Low/med density
    ng_tot_lm  = _ng_total('heating_lowmed_cold')
    oil_tot_lm = _oil_total('heating_lowmed_cold')

    wh_tech_lowmed = {
        "Natural Gas_Boiler_Low Efficiency":      ng_wh * (_heat_tech('heating_lowmed_cold', "Natural Gas_Furnace_Low Efficiency")     / ng_tot_lm),
        "Natural Gas_Boiler_Medium Efficiency":   ng_wh * (_heat_tech('heating_lowmed_cold', "Natural Gas_Furnace_Medium Efficiency")  / ng_tot_lm),
        "Natural Gas_Boiler_High Efficiency":     ng_wh * (_heat_tech('heating_lowmed_cold', "Natural Gas_Furnace_High Efficiency")    / ng_tot_lm),
        "Light Fuel Oil_Boiler_Low Efficiency":   oil_wh * (_heat_tech('heating_lowmed_cold', "Light Fuel Oil_Furnace_Low Efficiency") / oil_tot_lm),
        "Light Fuel Oil_Boiler_Medium Efficiency":oil_wh * (_heat_tech('heating_lowmed_cold', "Light Fuel Oil_Furnace_Medium Efficiency") / oil_tot_lm),
        "Electricity_Boiler_High Efficiency":     elec_wh,
    }

    # High density
    ng_tot_h  = _ng_total('heating_high_cold')
    oil_tot_h = _oil_total('heating_high_cold')

    wh_tech_high = {
        "Natural Gas_Boiler_Low Efficiency":      ng_wh * (_heat_tech('heating_high_cold', "Natural Gas_Furnace_Low Efficiency")     / ng_tot_h),
        "Natural Gas_Boiler_Medium Efficiency":   ng_wh * (_heat_tech('heating_high_cold', "Natural Gas_Furnace_Medium Efficiency")  / ng_tot_h),
        "Natural Gas_Boiler_High Efficiency":     ng_wh * (_heat_tech('heating_high_cold', "Natural Gas_Furnace_High Efficiency")    / ng_tot_h),
        "Light Fuel Oil_Boiler_Low Efficiency":   oil_wh * (_heat_tech('heating_high_cold', "Light Fuel Oil_Furnace_Low Efficiency") / oil_tot_h),
        "Light Fuel Oil_Boiler_Medium Efficiency":oil_wh * (_heat_tech('heating_high_cold', "Light Fuel Oil_Furnace_Medium Efficiency") / oil_tot_h),
        "Electricity_Boiler_High Efficiency":     elec_wh,
    }

    frames = [
        _long(province, 'wh_lowmed', '', 'service_request', 'GJ/GJ', wh_lowmed),
        _long(province, 'wh_high',   '', 'service_request', 'GJ/GJ', wh_high),
    ]
    for tech, s in wh_tech_lowmed.items():
        frames.append(_long(province, 'wh_tech_lowmed', tech,
                            'market_share_total', '%', s))
    for tech, s in wh_tech_high.items():
        frames.append(_long(province, 'wh_tech_high', tech,
                            'market_share_total', '%', s))

    return frames


# ==============================================================================
# PROJECTION EXTENSIONS
# ==============================================================================

def apply_extensions(df: pl.DataFrame, province: str, params: dict) -> pl.DataFrame:
    """
    Apply projection extensions to a province's long-format DataFrame.
 
    Parameters
    ----------
    df : pl.DataFrame
        Province data in long format (output of extract_all_data before extension).
    province : str
        Upper-case province code.
    params : dict
        Output of load_projection_params().
 
    Returns
    -------
    pl.DataFrame
        Same schema as input, extended through 2100.
    """
    if not params:
        return df

    frames = [df]
 
    def _apply_to_variable(variable: str, category: str,
                           series: pd.Series, fn, **kwargs) -> pl.DataFrame:
        """Apply fn to series and return a long-format Polars frame for new years only."""
        max_hist = int(series.dropna().index.max()) if not series.dropna().empty else LAST_HIST_YEAR
        extended = fn(series, base_year=max_hist, **kwargs)
        new = extended[extended.index > max_hist]
        meta = df.filter(
            (pl.col('variable') == variable) & (pl.col('category') == category)
        ).head(1)
        if len(meta) == 0 or new.empty:
            return pl.DataFrame()
        return _long(province, variable, category,
                     pl_get_scalar(meta, 'parameter'),
                     pl_get_scalar(meta, 'unit'),
                     new,
                     pl_get_scalar(meta, 'source') if 'source' in meta.columns else 'CEUD')
 
    def _series(variable: str, category: str) -> pd.Series:
        """Extract a year-indexed pd.Series for one variable/category combination."""
        return pl_to_series(
            df.filter((pl.col('variable') == variable) & (pl.col('category') == category))
        ).sort_index()
 
    # -- 1. Housing stock — activity-style CAGR --------------------------------
    hs_params = params.get('housing_stock', {})
    if hs_params:
        _cs = hs_params['cagr_start']
        _ce = hs_params['cagr_end']
        _p  = hs_params['periods']

        def _extend_households(series, base_year,
                               cs=_cs, ce=_ce, p=_p):
            raw_cagr = compute_cagr(series, cs, min(ce, base_year))
            base_val = float(series[base_year])
            projected = extend_cagr_periods(
                base_val, raw_cagr, p, CAGR_OVERRIDES.get(province)
            )
            return pd.concat([series, projected]).sort_index()

        s = _series('housing_thousand', '')
        frames.append(_apply_to_variable('housing_thousand', '', s, _extend_households))
 
    # -- 2. Building shares — trend then dampener; Apartments = remainder ------
    bs_params = params.get('building_shares', {})
    if bs_params:
        trend_kwargs = dict(
            trend_start=bs_params['trend_start'],
            trend_end=bs_params['trend_end'],
            trend_period=bs_params['trend_period'],
            decline_periods=bs_params['decline_periods'],
        )
 
        # Extend Detached, Attached, Mobile with dampener
        projected_other = {}
        for bt in ['Single Detached', 'Single Attached', 'Mobile Homes']:
            s = _series('building_shares', bt)
            max_hist_bs = int(s.dropna().index.max()) if not s.dropna().empty else LAST_HIST_YEAR
            ext = extend_series_trend_dampener(s, base_year=max_hist_bs, **trend_kwargs)
            projected_other[bt] = ext
            frames.append(_apply_to_variable('building_shares', bt, s,
                                             extend_series_trend_dampener, **trend_kwargs))
 
        # Apartments = 1 - sum(others) for every projected year
        max_hist = int(df.filter(pl.col('variable') == 'building_shares')['year'].max())
        apts_vals = {}
        for yr in range(max_hist + 1, 2101):
            total_other = sum(
                float(projected_other[bt].loc[yr])
                if yr in projected_other[bt].index and pd.notna(projected_other[bt].loc[yr])
                else 0.0
                for bt in projected_other
            )
            apts_vals[yr] = max(1.0 - total_other, 0.0)
        apts_series = pd.Series(apts_vals)
 
        meta = df.filter(
            (pl.col('variable') == 'building_shares') &
            (pl.col('category') == 'Apartments')
        ).head(1)
        if len(meta) > 0:
            frames.append(_long(province, 'building_shares', 'Apartments',
                                pl_get_scalar(meta, 'parameter'),
                                pl_get_scalar(meta, 'unit'),
                                apts_series))

 
    # -- 3. Floorspace — trend then dampener -----------------------------------
    fs_params = params.get('floorspace_per_building', {})
    if fs_params:
        trend_kwargs = dict(
            trend_start=fs_params['trend_start'],
            trend_end=fs_params['trend_end'],
            trend_period=fs_params['trend_period'],
            decline_periods=fs_params['decline_periods'],
        )
        for bt in ['Single Detached', 'Single Attached', 'Apartments', 'Mobile Homes']:
            s = _series('floorspace_per_building', bt)
            frames.append(_apply_to_variable('floorspace_per_building', bt, s,
                                             extend_series_trend_dampener, **trend_kwargs))
    
    # -- 4. Appliances per household — constant (hold last historical value) --
    appliance_types = ['Refrigerators', 'Freezers', 'Ranges', 'Dishwashing',
                       'Clothes Washing', 'Minor Appliances']
    for app in appliance_types:
        s = _series('appliances_per_household', app)
        if not s.dropna().empty:
            frames.append(_apply_to_variable('appliances_per_household', app,
                                             s, extend_series_constant))
 
    # -- 5. Water heating intensity — constant (hold last historical value) ----
    for var in ['wh_lowmed', 'wh_high']:
        s = _series(var, '')
        frames.append(_apply_to_variable(var, '', s, extend_series_constant))
 
    # -- 6. Cooling — constant (hold last historical value) --------------------
    for cooling_type in ['Room', 'Central']:
        s = _series('cooling_share_data', cooling_type)
        if not s.dropna().empty:
            frames.append(_apply_to_variable('cooling_share_data', cooling_type,
                                             s, extend_series_constant))
 
    return pl.concat([f for f in frames if f is not None and len(f) > 0],
                     how='diagonal_relaxed')

# ==============================================================================
# TERRITORY DISAGGREGATION
# ==============================================================================
#
# Rules for splitting 'TR' (Territories) data into YT, NT, NU:
#
#   - Heating/WH technology market shares  → split by RESD fuel share, then
#                                            efficiency-corrected and renormalized
#   - Wood technologies (not in RESD)      → split by total RESD energy share,
#                                            efficiency-corrected and renormalized
#   - Housing stock                        → split by population share per year
#   - Building shares, floorspace,
#     vintage bins, appliances, cooling    → identical across all three territories
#
# ==============================================================================

# Maps RESD fuel names → CEUD category names
RESD_FUEL_TO_CEUD_CATEGORIES = {
    'Natural gas': [
        'Natural Gas_Furnace_Low Efficiency',
        'Natural Gas_Furnace_Medium Efficiency',
        'Natural Gas_Furnace_High Efficiency',
        'Natural Gas_ASHP_Natural Gas_Backup',
        'Natural Gas_Boiler_Low Efficiency',
        'Natural Gas_Boiler_Medium Efficiency',
        'Natural Gas_Boiler_High Efficiency',
    ],
    'Light fuel oil': [
        'Light Fuel Oil_Furnace_Low Efficiency',
        'Light Fuel Oil_Furnace_Medium Efficiency',
        'Light Fuel Oil_Furnace_High Efficiency',
        'Light Fuel Oil_Boiler_Low Efficiency',
        'Light Fuel Oil_Boiler_Medium Efficiency',
    ],
    'Kerosene and stove oil': [
        'Light Fuel Oil_Furnace_Low Efficiency',
        'Light Fuel Oil_Furnace_Medium Efficiency',
        'Light Fuel Oil_Furnace_High Efficiency',
        'Light Fuel Oil_Boiler_Low Efficiency',
        'Light Fuel Oil_Boiler_Medium Efficiency',
    ],
    "Gas plant natural gas liquids (NGL's)": [
        'Propane_Furnace_Medium Efficiency',
    ],
    'Primary electricity, hydro and nuclear': [
        'Electricity_Resistance_High Efficiency',
        'Electricity_ASHP_Natural Gas_Backup',
        'Electricity_ASHP_Electricity_Backup',
        'Electricity_Boiler_High Efficiency',
    ],
    '_wood_proxy': [
        'Wood_Furnace_Low Efficiency',
    ],
}

# Maps each CEUD category to its Table 32 efficiency key
# Efficiencies are expressed as % (e.g. 90 = 90%) — divided by 100 when used
CATEGORY_EFFICIENCY_KEY = {
    # Natural gas heating
    'Natural Gas_Furnace_Low Efficiency':       'ng_low',
    'Natural Gas_Furnace_Medium Efficiency':    'ng_med',
    'Natural Gas_Furnace_High Efficiency':      'ng_high',
    'Natural Gas_ASHP_Natural Gas_Backup':      'ng_med',
    # Natural gas water heating
    'Natural Gas_Boiler_Low Efficiency':        'ng_low',
    'Natural Gas_Boiler_Medium Efficiency':     'ng_med',
    'Natural Gas_Boiler_High Efficiency':       'ng_high',
    # Light fuel oil heating
    'Light Fuel Oil_Furnace_Low Efficiency':    'oil_low',
    'Light Fuel Oil_Furnace_Medium Efficiency': 'oil_med',
    'Light Fuel Oil_Furnace_High Efficiency':   'oil_high',
    # Light fuel oil water heating
    'Light Fuel Oil_Boiler_Low Efficiency':     'oil_low',
    'Light Fuel Oil_Boiler_Medium Efficiency':  'oil_med',
    # Propane
    'Propane_Furnace_Medium Efficiency':        'other',
    # Electricity
    'Electricity_Resistance_High Efficiency':   'elec',
    'Electricity_ASHP_Natural Gas_Backup':      'heat_pump',
    'Electricity_ASHP_Electricity_Backup':      'heat_pump',
    'Electricity_Boiler_High Efficiency':       'elec',
    # Wood
    'Wood_Furnace_Low Efficiency':              'wood',
}

# Variables where TR value is copied identically to all three territories
IDENTICAL_VARIABLES = {
    'building_shares',
    'floorspace_per_building',
    'vintage_bins_lowmed',
    'vintage_bins_high',
    'appliances_per_household',
    'cooling_share_data',
    'wh_lowmed',
    'wh_high',
}

# Variables split by population share
POPULATION_VARIABLES = {
    'housing_thousand',
    'housing_by_type',
}

# Market share variables that require efficiency correction + renormalization
MARKET_SHARE_VARIABLES = {
    'heating_lowmed_cold',
    'heating_high_cold',
    'heating_lowmed_marine',
    'heating_high_marine',
    'wh_tech_lowmed',
    'wh_tech_high',
}

# Territory map — keys must match GEO values in the Stats Can population CSV
TERRITORY_MAP = {
    'Yukon':                 'YT',
    'Northwest Territories': 'NT',
    'Nunavut':               'NU',
}

PROJECTION_END = 2100


def _load_efficiencies(efficiency_xls: Path) -> dict[str, dict[int, float]]:
    """
    Load heating system efficiencies from Table 32.

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
        'ng_low':    get_series(16),   # Natural Gas – Normal Efficiency
        'ng_med':    get_series(17),   # Natural Gas – Medium Efficiency
        'ng_high':   get_series(18),   # Natural Gas – High Efficiency
        'oil_low':   get_series(13),   # Heating Oil – Normal Efficiency
        'oil_med':   get_series(14),   # Heating Oil – Medium Efficiency
        'oil_high':  get_series(15),   # Heating Oil – High Efficiency
        'elec':      get_series(19),   # Electric
        'heat_pump': get_series(20),   # Heat Pump
        'other':     get_series(21),   # Other1 (propane/coal)
        'wood':      get_series(22),   # Wood
    }


def _get_efficiency(efficiencies: dict, category: str, year: int) -> float:
    """
    Look up efficiency for a category/year. Falls back to last available year.
    Returns 1.0 if category has no efficiency mapping (no correction applied).
    """
    key = CATEGORY_EFFICIENCY_KEY.get(category)
    if key is None:
        return 1.0
    series = efficiencies.get(key, {})
    if not series:
        return 1.0
    if year in series:
        return series[year]
    # Use most recent available year
    return series[max(series.keys())]


def _build_resd_shares(resd_csv: Path) -> pd.DataFrame:
    """
    Load RESD CSV and return territorial energy shares by fuel and year.
    Also computes '_wood_proxy' = each territory's share of total energy.

    Returns
    -------
    pd.DataFrame with columns: year, territory, fuel, share
    """
    resd = (
        load_resd(resd_csv)
        .filter(pl.col("characteristic") == "Residential")
        .select(["year", "geo", "fuel", "value"])
        .to_pandas()
    )
    resd.columns = ['year', 'territory', 'fuel', 'demand_TJ']

    national = (
        resd.groupby(['year', 'fuel'])['demand_TJ']
        .sum().rename('national_total').reset_index()
    )
    resd = resd.merge(national, on=['year', 'fuel'])
    resd['share'] = resd['demand_TJ'] / resd['national_total']
    fuel_shares = resd[['year', 'territory', 'fuel', 'share']].copy()

    terr_total = (
        resd.groupby(['year', 'territory'])['demand_TJ']
        .sum().rename('terr_total').reset_index()
    )
    grand_total = (
        resd.groupby('year')['demand_TJ']
        .sum().rename('grand_total').reset_index()
    )
    wood = terr_total.merge(grand_total, on='year')
    wood['share'] = wood['terr_total'] / wood['grand_total']
    wood['fuel'] = '_wood_proxy'

    return pd.concat(
        [fuel_shares, wood[['year', 'territory', 'fuel', 'share']]],
        ignore_index=True
    )


def _build_category_to_fuel_map() -> dict:
    """Invert RESD_FUEL_TO_CEUD_CATEGORIES into {category: resd_fuel}."""
    cat_to_fuel = {}
    for fuel, categories in RESD_FUEL_TO_CEUD_CATEGORIES.items():
        for cat in categories:
            cat_to_fuel[cat] = fuel
    return cat_to_fuel


def disaggregate_territories(
    tr_df: pl.DataFrame,
    resd_csv: Path,
    pop_csv: Path,
    efficiency_xls: Path,
) -> pl.DataFrame:
    """
    Split the lumped 'TR' Territories data into YT, NT, and NU.

    For technology market share variables (heating, water heating), the split
    uses RESD fuel shares adjusted for equipment efficiency differences, then
    renormalized so shares sum to 1.

    Parameters
    ----------
    tr_df : pl.DataFrame
        Output of extract_all_data('TR', apply_projections=True).
    resd_csv : Path
        Path to RESD CSV (Statistics Canada table 25-10-0029-01).
    pop_csv : Path
        Path to quarterly population CSV (Statistics Canada table 17-10-0009-01).
    efficiency_xls : Path
        Path to CEUD national res_ca_e_32.xls containing Table 32 efficiencies.

    Returns
    -------
    pl.DataFrame
        Same schema as tr_df, province replaced with 'YT', 'NT', or 'NU'.
        Rows beyond PROJECTION_END are dropped. TR rows are NOT included.
    """
    resd_shares  = _build_resd_shares(resd_csv)
    pop_shares = build_population_shares(
        pop_csv,
        regions=list(TERRITORY_MAP.keys()),  # ['Yukon', 'Northwest Territories', 'Nunavut']
        projection_end=PROJECTION_END,
    )
    cat_to_fuel  = _build_category_to_fuel_map()
    efficiencies = _load_efficiencies(efficiency_xls)

    # Drop any rows beyond the projection end
    tr_pd = tr_df.to_pandas()
    tr_pd = tr_pd[tr_pd['year'] <= PROJECTION_END].copy()

    output_frames = []

    for geo_name, prov_code in TERRITORY_MAP.items():

        terr_df = tr_pd.copy()
        terr_df['province'] = prov_code

        def get_resd_share(row, geo=geo_name):
            """Look up RESD fuel share for this territory/year/category."""
            fuel = cat_to_fuel.get(row['category'], '_wood_proxy')
            mask = (
                (resd_shares['territory'] == geo) &
                (resd_shares['fuel'] == fuel)
            )
            available = resd_shares[mask].sort_values('year')

            if available.empty:
                mask_proxy = (
                    (resd_shares['territory'] == geo) &
                    (resd_shares['fuel'] == '_wood_proxy')
                )
                available = resd_shares[mask_proxy].sort_values('year')
                if available.empty:
                    return np.nan

            year_match = available[available['year'] == row['year']]
            if not year_match.empty:
                return float(year_match.iloc[0]['share'])
            return float(available.iloc[-1]['share'])

        def get_pop_share(row, geo=geo_name):
            try:
                result = pop_shares.loc[
                    (pop_shares['territory'] == geo) &
                    (pop_shares['year'] == row['year']),
                    'pop_share'
                ].iloc[0]
                return result
            except (IndexError, KeyError):
                available = pop_shares[pop_shares['territory'] == geo]
                if available.empty:
                    return np.nan
                return float(available.iloc[-1]['pop_share'])

        def apply_share(row):
            var = row['variable']
            if var in IDENTICAL_VARIABLES:
                return row['value']
            if var in POPULATION_VARIABLES:
                return row['value'] * get_pop_share(row)
            # For market share variables and everything else: apply RESD fuel share
            # Efficiency correction is applied as a post-processing step below
            return row['value'] * get_resd_share(row)

        terr_df['value'] = terr_df.apply(apply_share, axis=1)
        output_frames.append(terr_df)

    combined = pd.concat(output_frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Efficiency correction + renormalization for market share variables
    # ------------------------------------------------------------------
    # For each market share variable, the raw split used fuel energy shares.
    # We now convert to useful heat shares by dividing each tech's value by
    # its efficiency, then renormalize so shares sum to 1.
    #
    # Formula:
    #   useful_heat_share_i = (fuel_share_i / eff_i) / sum(fuel_share_j / eff_j)
    #
    # This is done per province/variable/year group.

    for var in MARKET_SHARE_VARIABLES:
        var_mask = combined['variable'] == var
        if not var_mask.any():
            continue

        # Apply efficiency correction: divide value by efficiency
        def efficiency_correct(row):
            eff = _get_efficiency(efficiencies, row['category'], row['year'])
            return row['value'] / eff if eff > 0 else row['value']

        combined.loc[var_mask, 'value'] = (
            combined[var_mask].apply(efficiency_correct, axis=1)
        )

        # Renormalize per province/year so shares sum to 1
        totals = (
            combined[var_mask]
            .groupby(['province', 'year'])['value']
            .sum()
            .rename('total')
            .reset_index()
        )
        combined = combined.merge(totals, on=['province', 'year'], how='left')
        combined.loc[var_mask, 'value'] = (
            combined.loc[var_mask, 'value'] /
            combined.loc[var_mask, 'total'].replace(0, np.nan)
        )
        combined = combined.drop(columns='total')

    return pl.from_pandas(combined)

# ==============================================================================
# MAIN EXTRACTION FUNCTION
# ==============================================================================

def extract_all_data(
    province_code: str,
    apply_projections: bool = True,
    projection_params: Optional[dict] = None,
) -> pl.DataFrame:
    """
    Extract all residential data for a province and return a single long-format
    Polars DataFrame.

    Parameters
    ----------
    province_code : str
        2-letter code (e.g. 'BC', 'ON').
    apply_projections : bool
        Whether to extend data beyond 2022 (default True).
    projection_params : dict, optional
        Pre-loaded params from load_projection_params(). Loaded from file if None.

    Returns
    -------
    pl.DataFrame
        Columns: province, variable, category, parameter, unit, source, year, value.

    Raises
    ------
    ValueError if province_code is not in PROVINCES.
    """
    province = province_code.upper()
    if province not in PROVINCES:
        raise ValueError(f"Invalid province code: {province_code}. "
                         f"Valid codes: {list(PROVINCES.keys())}")

    tables = load_tables(province)

    # Extract historical data — each function returns list[pl.DataFrame]
    frames: list[pl.DataFrame] = []
    frames += extract_housing_stock(province, tables)

    building_shares_frames = extract_building_shares(province, tables)
    frames += building_shares_frames
    building_shares_df = pl.concat(building_shares_frames)

    frames += extract_floorspace_per_building(province, tables)
    frames += extract_appliances(province, tables)
    frames += extract_vintages(province, tables, building_shares_df)

    heating_frames = extract_heating_technologies(province, tables, building_shares_df)
    frames += heating_frames
    heating_df = pl.concat(heating_frames)

    frames += extract_cooling_technologies(province, tables)
    frames += extract_water_heating(province, tables, building_shares_df, heating_df)

    df = pl.concat(frames, how='diagonal_relaxed')

    if apply_projections:
        if projection_params is None:
            projection_params = load_projection_params()
        df = apply_extensions(df, province, projection_params)

    return df.sort(['province', 'variable', 'category', 'year'])


# ==============================================================================
# BATCH EXTRACTION
# ==============================================================================

def extract_all_provinces(
    province_codes: Optional[list[str]] = None,
    apply_projections: bool = True,
) -> dict[str, pl.DataFrame]:
    """
    Extract data for multiple provinces.

    Parameters
    ----------
    province_codes : list of str, optional
        Codes to extract. Defaults to all provinces in PROVINCES.
    apply_projections : bool

    Returns
    -------
    dict mapping province code → pl.DataFrame
    """
    if province_codes is None:
        province_codes = list(PROVINCES.keys())

    params = load_projection_params() if apply_projections else None
    results, failed = {}, []

    for prov in province_codes:
        try:
            results[prov] = extract_all_data(prov, apply_projections, params)
        except Exception as exc:
            failed.append((prov, str(exc)))

    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main(
    province_codes: Optional[list[str]] = None,
    output_dir: Path = OUTPUT_DIR,
    apply_projections: bool = True,
    export_csv: bool = True,
) -> dict[str, pl.DataFrame]:
    """
    Run the full residential pipeline and optionally export a combined CSV.

    Parameters
    ----------
    province_codes : list of str, optional
    output_dir : Path
    apply_projections : bool
    export_csv : bool

    Returns
    -------
    dict mapping province code → pl.DataFrame
    """
    if province_codes is None:
        province_codes = list(PROVINCES.keys())

    if export_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    params = load_projection_params() if apply_projections else None
    results, failed, all_frames = {}, [], []

    for prov in province_codes:
        try:
            df = extract_all_data(prov, apply_projections, params)

            if prov.upper() == 'TR':
                territory_df = disaggregate_territories(df, RESD_CSV, POP_CSV, EFFICIENCY_XLS)
                for terr_code in ['YT', 'NT', 'NU']:
                    terr_data = territory_df.filter(pl.col('province') == terr_code)
                    results[terr_code] = terr_data
                    all_frames.append(terr_data)
            else:
                results[prov] = df
                all_frames.append(df)

        except Exception as exc:
            failed.append((prov, str(exc)))

    if export_csv and all_frames:
        combined = pl.concat(all_frames, how='diagonal_relaxed')
        combined = combined.with_columns(
            pl.when(pl.col('year') <= LAST_HIST_YEAR)
            .then(pl.lit('CEUD'))
            .otherwise(pl.lit('Assumptions'))
            .alias('source')
        )
        combined = combined.sort(['province', 'variable', 'category', 'year'])
        output_file = output_dir / "residential.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n✅ Residential extraction complete")
        print(f"   Total rows:          {combined.height:,}")
        print(f"   Provinces processed: {combined['province'].n_unique()}")
        print(f"   Variables:           {sorted(combined['variable'].unique().to_list())}")
        print(f"   Years covered:       {combined['year'].min()} – {combined['year'].max()}")
        print(f"   Saved to:            {output_file}")
        combined = combined.rename({
            'province': 'Region', 'variable': 'Variable', 'category': 'Category',
            'parameter': 'Parameter', 'unit': 'Unit', 'source': 'Source',
            'year': 'Year', 'value': 'Value',
        })
        combined.write_csv(str(output_file))

    if failed:
        print(f"\n⚠️  Failed provinces ({len(failed)}):")
        for prov, err in failed:
            print(f"   • {prov}: {err}")

    return results


if __name__ == "__main__":
    main()
