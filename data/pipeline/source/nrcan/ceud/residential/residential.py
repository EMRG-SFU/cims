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
"""

from pathlib import Path
from typing import Optional
import sys

import polars as pl
import pandas as pd
import numpy as np

_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.extractors.nrcan_ceud import get_row_series
from utils.extensions.data_extensions import (
    extend_series_constant,
    extend_series_linear,
    extend_series_trend_decline,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_PATH      = Path('C:/cims/data/raw_data/nrcan/ceud/residential')
ASSUMPTIONS_CSV = Path('C:/cims/data/raw_data/assumptions/residential_assumptions.csv')
OUTPUT_DIR     = Path('C:/cims/data/processed_data/nrcan/ceud')

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
    {'key': 'vintage_bins_lowmed',    'name': 'vintage_bins_low_med',   'parameter': 'market_share_total', 'unit': '% of m2'},
    {'key': 'vintage_bins_high',      'name': 'vintage_bins_high',      'parameter': 'market_share_total', 'unit': '% of m2'},
    {'key': 'heating_lowmed_cold',    'name': 'heating_data_lowmed_cold','parameter': 'market_share_total','unit': '% of GJ of heat'},
    {'key': 'heating_high_cold',      'name': 'heating_data_high_cold', 'parameter': 'market_share_total', 'unit': '% of GJ of heat'},
    {'key': 'heating_lowmed_marine',  'name': 'heating_data_lowmed_marine','parameter': 'market_share_total','unit': '% of GJ of heat'},
    {'key': 'heating_high_marine',    'name': 'heating_data_high_marine','parameter': 'market_share_total','unit': '% of GJ of heat'},
    {'key': 'cooling_share_data',     'name': 'cooling_share_data',     'parameter': 'service_request',    'unit': 'GJ/GJ'},
    {'key': 'wh_lowmed',              'name': 'wh_lowmed',              'parameter': 'service_request',    'unit': 'GJ/GJ'},
    {'key': 'wh_high',                'name': 'wh_high',                'parameter': 'service_request',    'unit': 'GJ/GJ'},
    {'key': 'wh_tech_lowmed',         'name': 'wh_tech_lowmed',         'parameter': 'market_share_total', 'unit': '% of GJ of water heat'},
    {'key': 'wh_tech_high',           'name': 'wh_tech_high',           'parameter': 'market_share_total', 'unit': '% of GJ of water heat'},
]


# ==============================================================================
# HELPERS
# ==============================================================================

def _row_to_series(table: pl.DataFrame, label: str, match_n: int = 0) -> pd.Series:
    """
    Thin wrapper around get_row_series that returns a year-indexed pd.Series.
    Values of None are preserved as NaN so pandas can work with them.
    """
    raw = get_row_series(table, label, match_n)
    return pd.Series({int(k): (float(v) if v is not None else np.nan)
                      for k, v in raw.items()})


def _pct_series(table: pl.DataFrame, label: str, match_n: int = 0) -> pd.Series:
    """Return a fraction (0–1) Series from a percentage row in a CEUD table."""
    return _row_to_series(table, label, match_n) / 100.0


def _pl_to_series(df: pl.DataFrame) -> pd.Series:
    """
    Extract year→value from a long-format Polars DataFrame without pyarrow.

    Selects only the integer 'year' and float 'value' columns (no strings),
    so Polars can convert them to numpy arrays directly.
    """
    years  = df.get_column('year').cast(pl.Int64).to_list()
    values = df.get_column('value').cast(pl.Float64).to_list()
    return pd.Series(values, index=years, dtype=float)


def _pl_get_scalar(df: pl.DataFrame, col: str) -> object:
    """Return the first value of a column from a one-row Polars DataFrame."""
    return df.get_column(col).to_list()[0]


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

def load_projection_params(assumptions_csv: Path = ASSUMPTIONS_CSV) -> dict:
    """
    Parse the residential assumptions CSV and return projection parameters.

    The CSV is a sparse wide-format file. We read known cell positions to
    reconstruct growth rates and decline percentages for each province.

    Parameters
    ----------
    assumptions_csv : Path

    Returns
    -------
    dict with keys 'housing_stock', 'building_shares', 'floorspace_per_building'.
    Empty dict if the file is not found.
    """
    PROV_NAME_TO_CODE = {
        'British Columbia': 'BC', 'Alberta': 'AB', 'Saskatchewan': 'SK',
        'Manitoba': 'MB', 'Ontario': 'ON', 'Quebec': 'QC',
        'New Brunswick': 'NB', 'Nova Scotia': 'NS',
        'Prince Edward Island': 'PE', 'Newfoundland and Labrador': 'NL',
        'Territories': 'TR',
        'Atlantic': None, 'Yukon': None,
        'Northwest Territories': None, 'Nunavut': None,
    }

    try:
        raw = pd.read_csv(assumptions_csv, header=None, dtype=str)
    except FileNotFoundError:
        print(f"⚠️  Assumptions CSV {assumptions_csv} not found. "
              "Extensions will not be applied.")
        return {}
    except Exception as exc:
        print(f"❌ Error reading assumptions CSV: {exc}")
        return {}

    def cell(row, col):
        try:
            v = raw.iloc[row, col]
            return None if pd.isna(v) or str(v).strip() == '' else str(v).strip()
        except (IndexError, KeyError):
            return None

    def pct(row, col):
        v = cell(row, col)
        if v is None:
            return None
        try:
            return float(v.replace('%', '').strip()) / 100.0
        except ValueError:
            return None

    params = {}

    # 1. Housing stock — linear growth, per-province rates
    params['housing_stock'] = {'method': 'linear'}
    for row_idx in range(9, 24):
        prov_name = cell(row_idx, 5)
        code = PROV_NAME_TO_CODE.get(prov_name) if prov_name else None
        if code is None:
            continue
        rate1, rate2 = pct(row_idx, 9), pct(row_idx, 10)
        if rate1 is not None and rate2 is not None:
            params['housing_stock'][code] = {
                'periods': [(2023, 2051, rate1), (2051, 2101, rate2)]
            }

    # 2. Building shares — trend decline, global values
    bs_dec1 = abs(pct(31, 9) or -0.05)
    bs_dec2 = abs(pct(31, 10) or -0.10)
    params['building_shares'] = {
        'method': 'trend_decline',
        'trend_start': 2000, 'trend_end': 2022, 'trend_period': (2023, 2031),
        'decrease_periods': [(2031, 2051, bs_dec1), (2051, 2101, bs_dec2)],
    }

    # 3. Floorspace per building — trend decline, global values
    fs_dec1 = abs(pct(38, 9) or -0.05)
    fs_dec2 = abs(pct(38, 10) or -0.10)
    params['floorspace_per_building'] = {
        'method': 'trend_decline',
        'trend_start': 2000, 'trend_end': 2022, 'trend_period': (2023, 2031),
        'decrease_periods': [(2031, 2051, fs_dec1), (2051, 2101, fs_dec2)],
    }

    print(f"✅ Loaded projection parameters from {assumptions_csv}")
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

    total_raw = _row_to_series(t15, "Total Housing Stock (thousands)") * 1000
    frames = [_long(province, 'housing_thousand', '', 'service_request', 'household', total_raw)]

    for bt in building_types:
        s = _row_to_series(t15, bt, match_n=0)
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
        s = _pct_series(t15, bt, match_n=1)
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
        fs_m2   = _row_to_series(t18, bt, match_n=0) * 1e6   # million m² → m²
        stock   = _row_to_series(t15, bt, match_n=0) * 1000  # thousands → units
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
            s = _row_to_series(t31, excel_name, match_n=1)
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
        "Single Detached": {lbl: _pct_series(t19, lbl, match_n=1) for lbl in vintage_labels},
        "Single Attached":  {lbl: _pct_series(t19, lbl, match_n=3) for lbl in vintage_labels},
        "Apartments":       {lbl: _pct_series(t20, lbl, match_n=1) for lbl in vintage_labels},
        "Mobile Homes":     {lbl: _pct_series(t20, lbl, match_n=3) for lbl in vintage_labels},
    }

    # Build weight series for the three low/med types from building_shares_df
    def _weight(bt: str) -> pd.Series:
        return _pl_to_series(building_shares_df.filter(pl.col('category') == bt))

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
                            'market_share_total', '% of m2', s))

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
                            'market_share_total', '% of m2', s))

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
            parts.append(_pct_series(table, row_label, match_n=1))
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
        return _pl_to_series(building_shares_df.filter(pl.col('category') == bt))

    w_det = _weight("Single Detached")
    w_att = _weight("Single Attached")
    w_mob = _weight("Mobile Homes")

    # -- COLD climate ----------------------------------------------------------
    cold_buckets = {
        "NG - Low Efficiency":                    ["Natural Gas – Normal Efficiency", "Other1", "Natural Gas/Electric"],
        "NG - Medium Efficiency":                 ["Natural Gas – Medium Efficiency"],
        "NG - High Efficiency":                   ["Natural Gas – High Efficiency"],
        "Electric - Resistance":                  ["Electric"],
        "Heating Oil - Low Efficiency":           ["Heating Oil – Normal Efficiency", "Heating Oil/Electric"],
        "Heating Oil - Medium Efficiency":        ["Heating Oil – Medium Efficiency", "Heating Oil – High Efficiency"],
        "Wood":                                   ["Wood", "Wood/Electric", "Wood/Heating Oil"],
        "NG - ASHP / NG - backup":               [],
        "Electric - ASHP / NG - backup":         ["Heat Pump"],
        "Electric - ASHP / Electric - backup":   [],
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
                            'market_share_total', '% of GJ of heat', s))
    for tech, s in cold_high.items():
        frames.append(_long(province, 'heating_high_cold', tech,
                            'market_share_total', '% of GJ of heat', s))

    # -- MARINE climate (BC only) ----------------------------------------------
    if is_bc:
        marine_buckets = {
            "NG - Low Efficiency":                  ["Natural Gas – Normal Efficiency", "Other1", "Natural Gas/Electric"],
            "NG - Medium Efficiency":               ["Natural Gas – Medium Efficiency"],
            "NG - High Efficiency":                 ["Natural Gas – High Efficiency"],
            "Electric - Resistance":                ["Electric"],
            "Heating Oil - Low Efficiency":         ["Heating Oil – Normal Efficiency", "Heating Oil/Electric"],
            "Heating Oil - Medium Efficiency":      ["Heating Oil – Medium Efficiency", "Heating Oil – High Efficiency"],
            "Wood":                                 ["Wood", "Wood/Electric", "Wood/Heating Oil"],
            "NG - ASHP":                            [],
            "Electric - ASHP":                      ["Heat Pump"],
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
                                'market_share_total', '% of GJ of heat', s))
        for tech, s in marine_high.items():
            frames.append(_long(province, 'heating_high_marine', tech,
                                'market_share_total', '% of GJ of heat', s))

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
        s = _pct_series(t4, cooling_type, match_n=1)

        # Territories: patch known missing years
        if province.upper() == 'TR':
            for miss, fill in [(2002, 2001), (2014, 2013)]:
                if (miss not in s.index or pd.isna(s.get(miss, np.nan))) and fill in s.index:
                    s[miss] = s[fill]
                    print(f"  → Filled {cooling_type} {miss} with {fill} value: {s[fill]:.4f}")

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
        return _pl_to_series(building_shares_df.filter(pl.col('category') == bt))

    def _heat_tech(variable: str, category: str) -> pd.Series:
        """Retrieve a heating tech series from the heating DataFrame."""
        subset = heating_df.filter(
            (pl.col('variable') == variable) & (pl.col('category') == category)
        )
        return _pl_to_series(subset) if len(subset) > 0 else pd.Series(dtype=float)

    # Water heating intensity (fraction of hot water demand met)
    wh_det  = _pct_series(t11, "Single Detached", match_n=1)
    wh_att  = _pct_series(t11, "Single Attached",  match_n=1)
    wh_apt  = _pct_series(t11, "Apartments",       match_n=1)
    wh_mob  = _pct_series(t11, "Mobile Homes",     match_n=1)

    w_det = _weight("Single Detached")
    w_att = _weight("Single Attached")
    w_mob = _weight("Mobile Homes")
    w_tot = w_det.add(w_att, fill_value=0).add(w_mob, fill_value=0).replace(0, np.nan)

    wh_lowmed = (wh_det.mul(w_det, fill_value=0)
                 .add(wh_att.mul(w_att, fill_value=0), fill_value=0)
                 .add(wh_mob.mul(w_mob, fill_value=0), fill_value=0)) / w_tot

    wh_high = wh_apt

    # Water heating fuel shares (aggregate from t10)
    ng_wh   = _pct_series(t10, "Natural Gas", match_n=1).add(
                _pct_series(t10, "Other2", match_n=1), fill_value=0)
    oil_wh  = _pct_series(t10, "Heating Oil", match_n=1)
    elec_wh = (_pct_series(t10, "Electricity", match_n=1)
               .add(_pct_series(t10, "Wood", match_n=1), fill_value=0))

    # NG efficiency split from cold heating data
    def _ng_total(var: str) -> pd.Series:
        lo = _heat_tech(var, "NG - Low Efficiency")
        md = _heat_tech(var, "NG - Medium Efficiency")
        hi = _heat_tech(var, "NG - High Efficiency")
        return lo.add(md, fill_value=0).add(hi, fill_value=0).replace(0, np.nan)

    def _oil_total(var: str) -> pd.Series:
        lo = _heat_tech(var, "Heating Oil - Low Efficiency")
        md = _heat_tech(var, "Heating Oil - Medium Efficiency")
        return lo.add(md, fill_value=0).replace(0, np.nan)

    # Low/med density
    ng_tot_lm  = _ng_total('heating_lowmed_cold')
    oil_tot_lm = _oil_total('heating_lowmed_cold')

    wh_tech_lowmed = {
        "NG - Storage - Low Efficiency":    ng_wh * (_heat_tech('heating_lowmed_cold', "NG - Low Efficiency")     / ng_tot_lm),
        "NG - Storage - Medium Efficiency": ng_wh * (_heat_tech('heating_lowmed_cold', "NG - Medium Efficiency")  / ng_tot_lm),
        "NG - Storage - High Efficiency":   ng_wh * (_heat_tech('heating_lowmed_cold', "NG - High Efficiency")    / ng_tot_lm),
        "Oil - Storage - Low Efficiency":   oil_wh * (_heat_tech('heating_lowmed_cold', "Heating Oil - Low Efficiency") / oil_tot_lm),
        "Oil - Storage - Medium Efficiency":oil_wh * (_heat_tech('heating_lowmed_cold', "Heating Oil - Medium Efficiency") / oil_tot_lm),
        "Electric - Storage - Low Efficiency": elec_wh,
    }

    # High density
    ng_tot_h  = _ng_total('heating_high_cold')
    oil_tot_h = _oil_total('heating_high_cold')

    wh_tech_high = {
        "NG - Storage - Low Efficiency":    ng_wh * (_heat_tech('heating_high_cold', "NG - Low Efficiency")     / ng_tot_h),
        "NG - Storage - Medium Efficiency": ng_wh * (_heat_tech('heating_high_cold', "NG - Medium Efficiency")  / ng_tot_h),
        "NG - Storage - High Efficiency":   ng_wh * (_heat_tech('heating_high_cold', "NG - High Efficiency")    / ng_tot_h),
        "Oil - Storage - Low Efficiency":   oil_wh * (_heat_tech('heating_high_cold', "Heating Oil - Low Efficiency") / oil_tot_h),
        "Oil - Storage - Medium Efficiency":oil_wh * (_heat_tech('heating_high_cold', "Heating Oil - Medium Efficiency") / oil_tot_h),
        "Electric - Storage - Low Efficiency": elec_wh,
    }

    frames = [
        _long(province, 'wh_lowmed', '', 'service_request', 'GJ/GJ', wh_lowmed),
        _long(province, 'wh_high',   '', 'service_request', 'GJ/GJ', wh_high),
    ]
    for tech, s in wh_tech_lowmed.items():
        frames.append(_long(province, 'wh_tech_lowmed', tech,
                            'market_share_total', '% of GJ of water heat', s))
    for tech, s in wh_tech_high.items():
        frames.append(_long(province, 'wh_tech_high', tech,
                            'market_share_total', '% of GJ of water heat', s))

    return frames


# ==============================================================================
# PROJECTION EXTENSIONS
# ==============================================================================

def apply_extensions(df: pl.DataFrame, province: str, params: dict) -> pl.DataFrame:
    """
    Apply projection extensions to a province's long-format DataFrame.

    Operates variable-by-variable, pulling each subset out as a pd.Series,
    applying the appropriate extension function, and re-assembling the result.

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
        print(f"  ⚠️  No projection parameters — skipping extensions for {province}")
        return df

    print(f"  📈 Applying extensions for {province}...")
    frames = [df]  # start with historical data; we will append projected rows

    def _apply_to_variable(variable: str, category: str,
                           series: pd.Series, fn, **kwargs) -> pl.DataFrame:
        """Apply fn to series and return a long-format Polars frame for the new years only."""
        extended = fn(series, **kwargs)
        max_hist = int(series.dropna().index.max()) if not series.dropna().empty else 2022
        new = extended[extended.index > max_hist]
        meta = df.filter(
            (pl.col('variable') == variable) & (pl.col('category') == category)
        ).head(1)
        if len(meta) == 0 or new.empty:
            return pl.DataFrame()
        return _long(province, variable, category,
                     _pl_get_scalar(meta, 'parameter'),
                     _pl_get_scalar(meta, 'unit'),
                     new,
                     _pl_get_scalar(meta, 'source') if 'source' in meta.columns else 'CEUD')

    def _series(variable: str, category: str) -> pd.Series:
        """Extract a year-indexed pd.Series for one variable/category combination."""
        return _pl_to_series(
            df.filter((pl.col('variable') == variable) & (pl.col('category') == category))
        ).sort_index()

    # 1. Housing stock — linear growth
    hs_params = params.get('housing_stock', {})
    if province in hs_params:
        s = _series('housing_thousand', '')
        periods = hs_params[province].get('periods', [(2023, 2051, 0.01), (2051, 2101, 0.005)])
        frames.append(_apply_to_variable(
            'housing_thousand', '', s, extend_series_linear, periods=periods))
        print("    ✓ Housing stock extended")

    # 2. Building shares — trend decline (Apartments = 1 - sum of others)
    bs_params = params.get('building_shares', {})
    if bs_params:
        td_kwargs = dict(
            trend_start=bs_params.get('trend_start', 2000),
            trend_end=bs_params.get('trend_end', 2022),
            trend_period=bs_params.get('trend_period', (2023, 2031)),
            decrease_periods=bs_params.get('decrease_periods', [(2031, 2051, 0.05), (2051, 2101, 0.1)]),
        )
        projected_other = {}
        for bt in ['Single Detached', 'Single Attached', 'Mobile Homes']:
            s = _series('building_shares', bt)
            ext = extend_series_trend_decline(s, **td_kwargs)
            projected_other[bt] = ext
            frames.append(_apply_to_variable('building_shares', bt, s,
                                             extend_series_trend_decline, **td_kwargs))

        # Apartments = 1 - sum(others)
        max_hist = int(df.filter(pl.col('variable') == 'building_shares')
                       ['year'].max())
        all_years = range(max_hist + 1, 2101)
        apts_vals = {}
        for yr in all_years:
            total_other = sum(float(projected_other[bt].get(yr, np.nan) or 0)
                              for bt in projected_other)
            apts_vals[yr] = 1.0 - total_other
        apts_series = pd.Series(apts_vals)
        meta = df.filter(
            (pl.col('variable') == 'building_shares') &
            (pl.col('category') == 'Apartments')
        ).head(1)
        if len(meta) > 0:
            frames.append(_long(province, 'building_shares', 'Apartments',
                                _pl_get_scalar(meta, 'parameter'),
                                _pl_get_scalar(meta, 'unit'),
                                apts_series))
        print("    ✓ Building shares extended")

    # 3. Floorspace per building — trend decline (all types decline independently)
    fs_params = params.get('floorspace_per_building', {})
    if fs_params:
        td_kwargs = dict(
            trend_start=fs_params.get('trend_start', 2000),
            trend_end=fs_params.get('trend_end', 2022),
            trend_period=fs_params.get('trend_period', (2023, 2031)),
            decrease_periods=fs_params.get('decrease_periods', [(2031, 2051, 0.05), (2051, 2101, 0.1)]),
        )
        for bt in ['Single Detached', 'Single Attached', 'Apartments', 'Mobile Homes']:
            s = _series('floorspace_per_building', bt)
            frames.append(_apply_to_variable('floorspace_per_building', bt, s,
                                             extend_series_trend_decline, **td_kwargs))
        print("    ✓ Floorspace per building extended")

    # 4. Water heating intensity — constant (hold 2022)
    for var in ['wh_lowmed', 'wh_high']:
        s = _series(var, '')
        frames.append(_apply_to_variable(var, '', s, extend_series_constant))
    print("    ✓ Water heating extended (constant)")

    # 5. Cooling — constant (hold 2022)
    for cooling_type in ['Room', 'Central']:
        s = _series('cooling_share_data', cooling_type)
        if not s.dropna().empty:
            frames.append(_apply_to_variable('cooling_share_data', cooling_type,
                                             s, extend_series_constant))
    print("    ✓ Cooling extended (constant)")

    return pl.concat([f for f in frames if f is not None and len(f) > 0],
                     how='diagonal_relaxed')


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
            print(f"Extracting data for {prov}...")
            results[prov] = extract_all_data(prov, apply_projections, params)
            print(f"✅ {prov} — {PROVINCES[prov.upper()]} complete")
        except Exception as exc:
            print(f"❌ {prov} — {PROVINCES[prov.upper()]} failed: {exc}")
            failed.append((prov, str(exc)))

    if failed:
        print(f"\n⚠️  Failed provinces: {', '.join(p for p, _ in failed)}")

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

    print("=" * 80)
    print("RESIDENTIAL DATA EXTRACTION")
    print("=" * 80)
    print(f"Provinces:         {', '.join(province_codes)}")
    print(f"Apply projections: {apply_projections}")
    print(f"Export to CSV:     {export_csv}")
    if export_csv:
        print(f"Output directory:  {output_dir}")
    print("=" * 80)

    if export_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    params = load_projection_params() if apply_projections else None
    results, failed, all_frames = {}, [], []

    for prov in province_codes:
        try:
            print(f"\n{prov} — {PROVINCES[prov.upper()]}:")
            df = extract_all_data(prov, apply_projections, params)
            results[prov] = df
            all_frames.append(df)
            print(f"  ✅ Extraction complete")
        except Exception as exc:
            print(f"  ❌ Failed: {exc}")
            failed.append((prov, str(exc)))

    if export_csv and all_frames:
        combined = pl.concat(all_frames, how='diagonal_relaxed')
        combined = combined.sort(['province', 'variable', 'category', 'year'])
        output_file = output_dir / "residential.csv"
        combined.write_csv(str(output_file))
        print(f"\n  ✅ Saved {len(combined):,} rows to {output_file}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {len(results)}/{len(province_codes)} provinces")
    if failed:
        print(f"❌ Failed: {len(failed)} provinces")
        for prov, err in failed:
            print(f"   • {prov}: {err}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
