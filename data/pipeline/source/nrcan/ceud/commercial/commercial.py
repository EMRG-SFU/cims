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
    extend_series_linear,
    extend_series_trend_decline,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_PATH        = Path('C:/cims/data/raw_data/nrcan/ceud/commercial')
ASSUMPTIONS_CSV  = Path('C:/cims/data/raw_data/assumptions/commercial_assumptions.csv')
NG_EFFICIENCY_CSV = Path('C:/cims/data/raw_data/assumptions/ng_eff_assumptions_commercial.csv')
OUTPUT_DIR       = Path('C:/cims/data/processed_data/nrcan/ceud')
YEARS            = list(range(2000, 2101))

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
FLOORSPACE_TABLES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
HVAC_TABLES       = [36, 38, 40, 42, 44, 46, 48, 50, 52, 54]
HOT_WATER_TABLE   = 26

FILE_NAME_MAP = {'BC': 'bct', 'AT': 'atl'}


# ==============================================================================
# HELPERS
# ==============================================================================

def _row_to_series(table: pl.DataFrame, label: str, match_n: int = 0) -> pd.Series:
    raw = get_row_series(table, label, match_n)
    return pd.Series({int(k): (float(v) if v is not None else np.nan)
                      for k, v in raw.items()})


def _pct_series(table: pl.DataFrame, label: str, match_n: int = 0) -> pd.Series:
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

def load_projection_params(assumptions_csv: Path = ASSUMPTIONS_CSV) -> dict:
    """
    Parse commercial assumptions CSV and return projection parameters.

    Returns
    -------
    dict with keys 'floorspace' and 'building_shell_shares'.
    Empty dict if file not found.
    """
    REGION_NAME_TO_CODE = {
        'Alberta': 'AB', 'Atlantic': 'AT', 'British Columbia': 'BC',
        'Manitoba': 'MB', 'Ontario': 'ON', 'Quebec': 'QC', 'Saskatchewan': 'SK',
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

    # 1. Floorspace — linear growth, per-region
    params['floorspace'] = {'method': 'linear'}
    for row_idx in range(9, 25):
        region_name = cell(row_idx, 5)
        code = REGION_NAME_TO_CODE.get(region_name) if region_name else None
        if code is None:
            continue
        rate1, rate2 = pct(row_idx, 9), pct(row_idx, 10)
        if rate1 is not None and rate2 is not None:
            params['floorspace'][code] = {
                'periods': [(2023, 2051, rate1), (2051, 2101, rate2)]
            }

    # 2. Building shell shares — trend decline, global
    bs_dec1 = abs(pct(23, 9) or -0.05)
    bs_dec2 = abs(pct(23, 10) or -0.10)
    params['building_shell_shares'] = {
        'method': 'trend_decline',
        'trend_start': 2000, 'trend_end': 2022, 'trend_period': (2023, 2031),
        'decrease_periods': [(2031, 2051, bs_dec1), (2051, 2101, bs_dec2)],
    }

    print(f"✅ Loaded commercial projection parameters from {assumptions_csv}")
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
    except Exception as exc:
        print(f"⚠️  NG efficiency CSV not readable: {exc}")
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

    print(f"✅ Loaded NG efficiency splits (data: {min(splits)}-{last_year}, "
          f"extrapolated to {max(YEARS)})")
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

    table_numbers = [TOTAL_FLOORSPACE_TABLE] + FLOORSPACE_TABLES + HVAC_TABLES + [HOT_WATER_TABLE]
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
                    .replace(["–", "-", "—", ""], None)
                    .replace(["X", "x"], "-1.0")
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
    total_raw = _row_to_series(t1, "Total Floor Space (million m2)") * 1e6
    frames = [_long(region, 'total_floorspace', '', 'service_request', 'm2', total_raw)]

    cims_activities = list(ACTIVITY_MAPPING.values())
    for tbl_num, cims_name in zip(FLOORSPACE_TABLES, cims_activities):
        tbl = tables[f"Table {tbl_num}"]
        raw = _row_to_series(tbl, "Floor Space (million m2)") * 1e6
        frames.append(_long(region, 'floorspace_by_activity', cims_name,
                            'service_request', 'm2', raw))

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
    total_s = _pl_to_series(
        floorspace_df.filter(pl.col('variable') == 'total_floorspace')
    )

    frames = []
    for cims_name in ACTIVITY_MAPPING.values():
        act_s = _pl_to_series(
            floorspace_df.filter(
                (pl.col('variable') == 'floorspace_by_activity') &
                (pl.col('category') == cims_name)
            )
        )
        share = act_s / total_s.replace(0, np.nan)
        frames.append(_long(region, 'building_shell_shares', cims_name,
                            'market_share_total', '% of m2', share))

    return frames


# ==============================================================================
# EXTRACTION — HVAC TECHNOLOGIES
# ==============================================================================

def extract_hvac_technologies(region: str, tables: dict,
                               ng_splits: dict) -> list[pl.DataFrame]:
    """
    Extract HVAC technology shares per activity, then produce weighted-average
    region-level shares.

    NG is split into Low / Medium / High efficiency using ng_splits.
    BC gets both Cold and Marine climate data.

    Parameters
    ----------
    ng_splits : dict
        Output of load_ng_efficiency_splits().

    Returns
    -------
    list of pl.DataFrame
    """
    is_bc = region.upper() == 'BC'

    def _safe(tbl, label, n, zero_fill=False):
        """Safely extract a percentage series, returning zeros or empty on miss."""
        try:
            return _pct_series(tbl, label, match_n=n)
        except KeyError:
            if label.startswith('Other'):
                for alt in ['Other1', 'Other2', 'Other3']:
                    try:
                        return _pct_series(tbl, alt, match_n=n)
                    except KeyError:
                        continue
            if zero_fill:
                try:
                    ref = _row_to_series(tbl, "Floor Space (million m2)", match_n=0)
                    return pd.Series(0.0, index=ref.index)
                except KeyError:
                    pass
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
        return (pd.Series(ng_low), pd.Series(ng_med), pd.Series(ng_high))

    cims_activities = list(ACTIVITY_MAPPING.values())

    # Accumulate per-activity series for weighted averaging later
    # Structure: {tech_name: {activity: pd.Series}}
    cold_per_activity:   dict[str, dict[str, pd.Series]] = {}
    marine_per_activity: dict[str, dict[str, pd.Series]] = {}

    for tbl_num, activity in zip(HVAC_TABLES, cims_activities):
        tbl = tables[f"Table {tbl_num}"]

        elec  = _safe(tbl, "Electricity",               n=3)
        ng_t  = _safe(tbl, "Natural Gas",               n=3)
        lfo   = _safe(tbl, "Light Fuel Oil and Kerosene", n=1)
        hfo   = _safe(tbl, "Heavy Fuel Oil",            n=1)
        other = _safe(tbl, "Other1",                    n=1, zero_fill=True)

        # Steam: impute X (suppressed) as remainder
        try:
            raw_steam = _row_to_series(tbl, "Steam", match_n=1)
        except KeyError:
            raw_steam = pd.Series(dtype=float)

        steam = {}
        for year, val in raw_steam.items():
            if val == -1.0:
                other_sum = sum(
                    (s.get(year) or 0.0) if isinstance(s, dict) else float(s[year] if year in s.index else 0.0)
                    for s in [elec, ng_t, lfo, hfo, other]
                )
                steam[year] = max(0.0, 1.0 - other_sum)
            elif pd.isna(val):
                steam[year] = np.nan
            else:
                steam[year] = float(val) / 100.0
        steam_s = pd.Series(steam)

        ng_lo, ng_md, ng_hi = _split_ng(ng_t)

        sector_cold = {
            'Light Fuel Oil_Furnace_Low Efficiency':    lfo,
            'Light Fuel Oil_Furnace_Medium Efficiency': pd.Series(dtype=float),
            'Heavy Fuel Oil_Furnace_Low Efficiency':    hfo,
            'Heavy Fuel Oil_Furnace_Medium Efficiency': pd.Series(dtype=float),
            'Natural Gas_Furnace_Low Efficiency':       ng_lo,
            'Natural Gas_Furnace_Medium Efficiency':    ng_md,
            'Natural Gas_Furnace_High Efficiency':      ng_hi,
            'Propane_Furnace_Medium Efficiency':        other,
            'Propane_Furnace_High Efficiency':          pd.Series(dtype=float),
            'Natural Gas_Cogeneration':                 steam_s,
            'Electricity_GSHP':                         pd.Series(dtype=float),
            'Electricity_Furnace_High Efficiency':      elec,
            'Electricity_ASHP_Natural Gas_Backup':      pd.Series(dtype=float),
            'Electricity_ASHP_Electricity_Backup':      pd.Series(dtype=float),
            'Natural Gas_ASHP_Natural Gas_Backup':      pd.Series(dtype=float),
        }
        for tech, s in sector_cold.items():
            cold_per_activity.setdefault(tech, {})[activity] = s

        if is_bc:
            sector_marine = {
                'Light Fuel Oil_Furnace_Low Efficiency':    lfo,
                'Light Fuel Oil_Furnace_Medium Efficiency': pd.Series(dtype=float),
                'Heavy Fuel Oil_Furnace_Low Efficiency':    hfo,
                'Heavy Fuel Oil_Furnace_Medium Efficiency': pd.Series(dtype=float),
                'Natural Gas_Furnace_Low Efficiency':       ng_lo,
                'Natural Gas_Furnace_Medium Efficiency':    ng_md,
                'Natural Gas_Furnace_High Efficiency':      ng_hi,
                'Propane_Furnace_Medium Efficiency':        other,
                'Propane_Furnace_High Efficiency':          pd.Series(dtype=float),
                'Natural Gas_Cogeneration':                 steam_s,
                'Electricity_GSHP':                         pd.Series(dtype=float),
                'Electricity_Furnace_High Efficiency':      elec,
                'Electricity_ASHP':                         pd.Series(dtype=float),
                'Natural Gas_ASHP':                         pd.Series(dtype=float),
            }
            for tech, s in sector_marine.items():
                marine_per_activity.setdefault(tech, {})[activity] = s

    return cold_per_activity, marine_per_activity, cims_activities


def _weighted_hvac(per_activity: dict, building_shell_df: pl.DataFrame,
                   region: str, climate_var: str) -> list[pl.DataFrame]:
    """
    Collapse per-activity HVAC shares into region-level weighted averages,
    using building shell shares as weights, and return long-format frames.
    """
    cims_activities = list(ACTIVITY_MAPPING.values())
    frames = []

    # Build a weight lookup {activity: {year: weight}} from Polars without to_pandas
    weight_lookup: dict[str, dict[int, float]] = {}
    for activity in cims_activities:
        subset = building_shell_df.filter(pl.col('category') == activity)
        weight_lookup[activity] = {
            int(yr): float(val)
            for yr, val in zip(
                subset.get_column('year').cast(pl.Int64).to_list(),
                subset.get_column('value').cast(pl.Float64).to_list(),
            )
        }

    for tech, act_dict in per_activity.items():
        # Skip techs with no data at all
        if not any(not s.dropna().empty for s in act_dict.values()):
            continue

        # Gather years common across activities with data
        all_years: set[int] = set()
        for s in act_dict.values():
            all_years.update(int(y) for y in s.dropna().index)
        if not all_years:
            continue

        weighted_vals: dict[int, float] = {}
        for yr in sorted(all_years):
            numerator = 0.0
            denominator = 0.0
            for activity in cims_activities:
                tech_s = act_dict.get(activity, pd.Series(dtype=float))
                tech_val = float(tech_s[yr]) if yr in tech_s.index and pd.notna(tech_s[yr]) else None
                wt = weight_lookup.get(activity, {}).get(yr)
                if tech_val is not None and wt is not None and wt > 0:
                    numerator   += tech_val * wt
                    denominator += wt
            if denominator > 0:
                weighted_vals[yr] = numerator / denominator

        if weighted_vals:
            s = pd.Series(weighted_vals)
            frames.append(_long(region, climate_var, tech,
                                'market_share_total', '% of GJ HVAC', s))

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
            return _pct_series(tbl, label, match_n=n)
        except KeyError:
            return pd.Series(dtype=float)

    elec  = _hw("Electricity")
    ng    = _hw("Natural Gas")
    lfo   = _hw("Light Fuel Oil and Kerosene")
    hfo   = _hw("Heavy Fuel Oil")
    steam = _hw("Steam")
    other = _hw("Other2")

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
                                'market_share_total', '% of GJ hot water', s))

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
        print(f"  ⚠️  No projection parameters — skipping extensions for {region}")
        return df

    print(f"  📈 Applying extensions for {region}...")
    frames = [df]

    def _series(variable: str, category: str) -> pd.Series:
        subset = df.filter(
            (pl.col('variable') == variable) & (pl.col('category') == category)
        )
        return _pl_to_series(subset).sort_index()

    def _apply(variable: str, category: str, fn, **kwargs) -> pl.DataFrame:
        s = _series(variable, category)
        if s.dropna().empty:
            return pl.DataFrame()
        extended = fn(s, **kwargs)
        max_hist = int(s.dropna().index.max())
        new = extended[extended.index > max_hist]
        if new.empty:
            return pl.DataFrame()
        meta = df.filter(
            (pl.col('variable') == variable) & (pl.col('category') == category)
        ).head(1)
        if len(meta) == 0:
            return pl.DataFrame()
        parameter = _pl_get_scalar(meta, 'parameter')
        unit      = _pl_get_scalar(meta, 'unit')
        source    = _pl_get_scalar(meta, 'source') if 'source' in meta.columns else 'CEUD'
        return _long(region, variable, category, parameter, unit, new, source)

    # 1. Total floorspace — linear growth
    fs_params = params.get('floorspace', {})
    if region in fs_params:
        periods = fs_params[region].get('periods', [(2023, 2051, 0.01), (2051, 2101, 0.005)])
        frames.append(_apply('total_floorspace', '', extend_series_linear,
                             periods=periods))
        print("    ✓ Total floorspace extended (linear)")

    # 2. Building shell shares — trend decline
    bs_params = params.get('building_shell_shares', {})
    if bs_params:
        td_kwargs = dict(
            trend_start=bs_params.get('trend_start', 2000),
            trend_end=bs_params.get('trend_end', 2022),
            trend_period=bs_params.get('trend_period', (2023, 2031)),
            decrease_periods=bs_params.get('decrease_periods', [(2031, 2051, 0.05), (2051, 2101, 0.1)]),
        )
        activities = list(ACTIVITY_MAPPING.values())
        declining  = [a for a in activities if a != 'Other Services']

        projected_declining: dict[str, pd.Series] = {}
        for activity in declining:
            s = _series('building_shell_shares', activity)
            if s.dropna().empty:
                continue
            ext = extend_series_trend_decline(s, **td_kwargs)
            projected_declining[activity] = ext
            frames.append(_apply('building_shell_shares', activity,
                                 extend_series_trend_decline, **td_kwargs))

        # Other Services = 1 - sum(all other activities)
        if projected_declining:
            max_hist = int(df.filter(pl.col('variable') == 'building_shell_shares')
                           ['year'].max())
            other_vals: dict[int, float] = {}
            for yr in range(max_hist + 1, 2101):
                other_vals[yr] = max(0.0, 1.0 - sum(
                    float(s.get(yr, np.nan) or 0) for s in projected_declining.values()
                ))
            other_series = pd.Series(other_vals)
            meta = df.filter(
                (pl.col('variable') == 'building_shell_shares') &
                (pl.col('category') == 'Other Services')
            ).head(1)
            if len(meta) > 0:
                frames.append(_long(region, 'building_shell_shares', 'Other Services',
                                    _pl_get_scalar(meta, 'parameter'),
                                    _pl_get_scalar(meta, 'unit'),
                                    other_series))

        print("    ✓ Building shell shares extended")

    return pl.concat([f for f in frames if f is not None and len(f) > 0],
                     how='diagonal_relaxed')


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
    cold_per_act, marine_per_act, cims_activities = extract_hvac_technologies(
        region, tables, ng_efficiency_splits
    )
    hvac_cold_frames   = _weighted_hvac(cold_per_act, shell_df, region, 'hvac_cold')
    hvac_marine_frames = _weighted_hvac(marine_per_act, shell_df, region, 'hvac_marine') if is_bc else []

    # -- Hot water -----------------------------------------------------------
    hw_frames = extract_hot_water(region, tables)

    # Assemble
    all_frames = (floorspace_frames + shell_frames +
                  hvac_cold_frames + hvac_marine_frames + hw_frames)
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
            print(f"Extracting data for {region}...")
            results[region] = extract_all_data(region, apply_projections,
                                               params, ng_splits)
            print(f"✅ {region} — {REGIONS[region.upper()]} complete")
        except Exception as exc:
            print(f"❌ {region} — {REGIONS[region.upper()]} failed: {exc}")
            failed.append((region, str(exc)))

    if failed:
        print(f"\n⚠️  Failed regions: {', '.join(r for r, _ in failed)}")

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

    print("=" * 80)
    print("COMMERCIAL DATA EXTRACTION")
    print("=" * 80)
    print(f"Regions:           {', '.join(region_codes)}")
    print(f"Apply projections: {apply_projections}")
    print(f"Export to CSV:     {export_csv}")
    if export_csv:
        print(f"Output directory:  {output_dir}")
    print("=" * 80)

    if export_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    params    = load_projection_params() if apply_projections else None
    ng_splits = load_ng_efficiency_splits()
    results, failed, all_frames = {}, [], []

    for region in region_codes:
        try:
            print(f"\n{region} — {REGIONS[region.upper()]}:")
            df = extract_all_data(region, apply_projections, params, ng_splits)
            results[region] = df
            all_frames.append(df)
            print(f"  ✅ Extraction complete")
        except Exception as exc:
            print(f"  ❌ Failed: {exc}")
            failed.append((region, str(exc)))

    if export_csv and all_frames:
        combined = pl.concat(all_frames, how='diagonal_relaxed')
        combined = combined.sort(['region', 'variable', 'category', 'year'])
        output_file = output_dir / "commercial.csv"
        combined.write_csv(str(output_file))
        print(f"\n  ✅ Saved {len(combined):,} rows to {output_file}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {len(results)}/{len(region_codes)} regions")
    if failed:
        print(f"❌ Failed: {len(failed)} regions")
        for region, err in failed:
            print(f"   • {region}: {err}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
